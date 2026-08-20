"""Tests for fatal-session detection and run termination.

run_20260721_161159 spent 10 online cycles on ~300 one-second queries
that each returned only the "organization has disabled Claude
subscription access" banner: every solve attempt died instantly at
$0.00, and the restart/replan/cycle budgets absorbed them all as
ordinary no-capture attempts. ``query_fatal_error`` recognizes such
responses, ``_track_fatal_response`` counts consecutive ones across
manager instances, and ``AgentSessionFatalError`` (not an
ApproachFailure) propagates past the per-task handlers and ends the run.

The end-to-end test monkeypatches ``ClaudeSDKClient`` with a fake that
yields real SDK dataclasses; no network, no subprocess.
"""
# pylint: disable=protected-access
import asyncio
from typing import Any, List

import claude_agent_sdk
import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from predicators import utils
from predicators.agent_sdk.session_base import AgentSessionFatalError, \
    BaseAgentSessionManager, query_fatal_error
from predicators.agent_sdk.session_manager import AgentSessionManager

# The exact assistant text run_20260721_161159's queries came back with.
_AUTH_BANNER = ("Your organization has disabled Claude subscription access "
                "for Claude Code · Use an Anthropic API key instead, or ask "
                "your admin to enable access")


def _text_entry(text):
    return {"type": "assistant", "content": [{"type": "text", "text": text}]}


def _tool_use_entry():
    return {
        "type":
        "assistant",
        "content": [{
            "type": "tool_use",
            "id": "toolu_1",
            "name": "run_python",
            "input": {}
        }],
    }


def _result_entry(subtype="success", is_error=False, result=None):
    return {
        "type": "result",
        "subtype": subtype,
        "num_turns": 1,
        "total_cost_usd": 0.0,
        "is_error": is_error,
        "result": result,
    }


def _auth_response():
    return [_text_entry(_AUTH_BANNER), _result_entry()]


# ---------------------------------------------------------------------------
# query_fatal_error
# ---------------------------------------------------------------------------


def test_auth_banner_is_fatal():
    """The observed auth banner (text-only, success result) is fatal."""
    reason = query_fatal_error(_auth_response())
    assert reason == _AUTH_BANNER


def test_other_fatal_banners_matched_case_insensitively():
    """Billing/key banners match regardless of case."""
    assert query_fatal_error([_text_entry("Credit balance is too LOW")])
    assert query_fatal_error([_text_entry("Error: Invalid API key")])


def test_tool_use_gates_out_all_signals():
    """Any tool call means the agent ran: never fatal, even with an error
    result and banner-looking text later in the stream."""
    response = [
        _tool_use_entry(),
        _text_entry(_AUTH_BANNER),
        {
            "type": "error",
            "error": "transport dropped"
        },
        _result_entry(subtype="error_during_execution", is_error=True),
    ]
    assert query_fatal_error(response) is None


def test_turn_cap_result_is_not_fatal():
    """error_max_turns is the ordinary turn-cap budget end."""
    response = [_result_entry(subtype="error_max_turns", is_error=True)]
    assert query_fatal_error(response) is None


def test_error_result_is_fatal():
    """A non-turn-cap error result is fatal, reported with its text."""
    response = [
        _result_entry(subtype="error_during_execution",
                      is_error=True,
                      result="CLI crashed on startup")
    ]
    assert query_fatal_error(response) == \
        "error result: CLI crashed on startup"


def test_stream_error_entry_is_fatal():
    """A stream error before any tool call is fatal."""
    response = [{"type": "error", "error": "scripted SDK failure"}]
    assert query_fatal_error(response) == \
        "stream error: scripted SDK failure"


def test_healthy_responses_are_not_fatal():
    """Ordinary text answers and empty responses pass."""
    assert query_fatal_error(
        [_text_entry("plan: Push(green)"),
         _result_entry()]) is None
    assert query_fatal_error([]) is None


# ---------------------------------------------------------------------------
# _track_fatal_response counter
# ---------------------------------------------------------------------------


def _make_base_manager(tmp_path):
    # reset_config also zeroes the process-wide fatal-query counter.
    utils.reset_config()
    return BaseAgentSessionManager(system_prompt="You are a test agent.",
                                   log_dir=str(tmp_path),
                                   model_name="claude-fable-5")


def test_counter_raises_at_limit(tmp_path):
    """The default limit (3) raises on the third consecutive fatal query."""
    mgr = _make_base_manager(tmp_path)
    mgr._track_fatal_response(_auth_response())
    mgr._track_fatal_response(_auth_response())
    assert BaseAgentSessionManager._consecutive_fatal_queries == 2
    with pytest.raises(AgentSessionFatalError,
                       match="disabled Claude subscription access"):
        mgr._track_fatal_response(_auth_response())


def test_healthy_query_resets_counter(tmp_path):
    """One healthy query wipes the streak."""
    mgr = _make_base_manager(tmp_path)
    mgr._track_fatal_response(_auth_response())
    mgr._track_fatal_response(_auth_response())
    mgr._track_fatal_response([_text_entry("fine"), _result_entry()])
    assert BaseAgentSessionManager._consecutive_fatal_queries == 0
    mgr._track_fatal_response(_auth_response())  # streak restarts at 1


def test_counter_shared_across_manager_instances(tmp_path):
    """Fresh-context restarts recreate the manager; the streak survives."""
    mgr = _make_base_manager(tmp_path)
    mgr._track_fatal_response(_auth_response())
    other = BaseAgentSessionManager(system_prompt="restarted",
                                    log_dir=str(tmp_path),
                                    model_name="claude-fable-5")
    other._track_fatal_response(_auth_response())
    with pytest.raises(AgentSessionFatalError):
        other._track_fatal_response(_auth_response())


def test_limit_zero_disables_check(tmp_path):
    """agent_sdk_max_consecutive_fatal_queries=0 turns the check off."""
    mgr = _make_base_manager(tmp_path)
    utils.reset_config({"agent_sdk_max_consecutive_fatal_queries": 0})
    for _ in range(10):
        mgr._track_fatal_response(_auth_response())
    assert BaseAgentSessionManager._consecutive_fatal_queries == 0


def test_reset_config_zeroes_counter(tmp_path):
    """A config reset (new run/test) must not inherit an old streak."""
    mgr = _make_base_manager(tmp_path)
    mgr._track_fatal_response(_auth_response())
    mgr._track_fatal_response(_auth_response())
    utils.reset_config()
    assert BaseAgentSessionManager._consecutive_fatal_queries == 0


# ---------------------------------------------------------------------------
# End-to-end through the streamed query path
# ---------------------------------------------------------------------------


def _make_fake_client_cls():

    class _FakeClient:
        """Fake ClaudeSDKClient scripted to return the auth banner."""

        instances: List[Any] = []

        def __init__(self, options=None):
            self.options = options
            self.scripts: List[List[Any]] = []

        async def connect(self):
            """No-op connect."""

        async def disconnect(self):
            """No-op disconnect."""

        async def query(self, message):
            """No-op query."""

        async def receive_response(self):
            """Yield the next scripted message batch."""
            for msg in (self.scripts.pop(0) if self.scripts else []):
                yield msg

    def _factory(options=None):
        client = _FakeClient(options)
        _FakeClient.instances.append(client)
        return client

    _factory.instances = _FakeClient.instances  # type: ignore[attr-defined]
    return _factory


def _auth_banner_messages():
    return [
        AssistantMessage(content=[TextBlock(text=_AUTH_BANNER)],
                         model="test-model"),
        ResultMessage(subtype="success",
                      duration_ms=10,
                      duration_api_ms=8,
                      is_error=False,
                      num_turns=1,
                      session_id="sess",
                      total_cost_usd=0.0),
    ]


def test_third_dead_query_raises_through_manager(monkeypatch, tmp_path):
    """Three consecutive auth-banner queries terminate via the real streamed-
    query path (detection is hooked in _run_streamed_query)."""
    fake_cls = _make_fake_client_cls()
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", fake_cls)
    utils.reset_config()
    mgr = AgentSessionManager(system_prompt="You are a test agent.",
                              mcp_server=object(),
                              log_dir=str(tmp_path),
                              model_name="claude-fable-5",
                              allowed_tools=["mcp__predicator_tools__probe"])
    asyncio.run(mgr.start_session())
    client = fake_cls.instances[-1]
    client.scripts = [
        _auth_banner_messages(),
        _auth_banner_messages(),
        _auth_banner_messages(),
    ]
    asyncio.run(mgr.query("first"))
    asyncio.run(mgr.query("second"))
    with pytest.raises(AgentSessionFatalError,
                       match="3 consecutive agent queries"):
        asyncio.run(mgr.query("third"))
