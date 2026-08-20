"""Tests for AgentSessionManager cost accounting and session recovery.

Hermetic: ``ClaudeSDKClient`` is monkeypatched on the ``claude_agent_sdk``
module (the manager imports it inside ``start_session``) with a fake that
yields real ``ResultMessage`` dataclass instances, so ``parse_message``
isinstance checks still work. No network, no subprocess.
"""
# pylint: disable=protected-access
import asyncio
from typing import Any, List

import claude_agent_sdk
from claude_agent_sdk import ResultMessage

from predicators import utils
from predicators.agent_sdk.session_manager import AgentSessionManager, \
    run_query_sync


def _result_message(total_cost_usd, num_turns=1):
    return ResultMessage(subtype="success",
                         duration_ms=10,
                         duration_api_ms=8,
                         is_error=False,
                         num_turns=num_turns,
                         session_id="sess",
                         total_cost_usd=total_cost_usd)


def _make_fake_client_cls():
    """Build a fresh fake ClaudeSDKClient class with instance tracking."""

    class _FakeClient:
        """Fake ClaudeSDKClient: scripted responses, call counters."""

        instances: List[Any] = []

        def __init__(self, options=None):
            self.options = options
            self.connect_calls = 0
            self.disconnect_calls = 0
            self.queries: List[str] = []
            # Each query() pops one list of messages for receive_response.
            self.scripts: List[List[Any]] = []
            self.fail_next_query = False

        async def connect(self):
            """Record the connect call."""
            self.connect_calls += 1

        async def disconnect(self):
            """Record the disconnect call."""
            self.disconnect_calls += 1

        async def query(self, message):
            """Record the query, optionally raising a scripted error."""
            if self.fail_next_query:
                self.fail_next_query = False
                raise RuntimeError("scripted SDK failure")
            self.queries.append(message)

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


def _make_manager(tmp_path):
    # Populate CFG arg-defaults (log_file etc.) read by the manager.
    utils.reset_config()
    return AgentSessionManager(system_prompt="You are a test agent.",
                               mcp_server=object(),
                               log_dir=str(tmp_path),
                               model_name="claude-fable-5",
                               allowed_tools=["mcp__predicator_tools__probe"])


def test_cost_delta_accounting_across_queries(monkeypatch, tmp_path):
    """Per-solve cost is the delta of the cumulative session cost."""
    fake_cls = _make_fake_client_cls()
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", fake_cls)
    mgr = _make_manager(tmp_path)
    asyncio.run(mgr.start_session())
    client = fake_cls.instances[-1]
    client.scripts = [
        [_result_message(0.5, num_turns=3)],
        [_result_message(0.8, num_turns=2)],
    ]

    asyncio.run(mgr.query("first"))
    assert mgr._last_cost_usd == 0.5
    assert mgr._total_cost_usd == 0.5
    assert mgr._current_log_meta["solve_cost_usd"] == 0.5

    asyncio.run(mgr.query("second"))
    # Cumulative went 0.5 -> 0.8, so this solve cost the 0.3 delta.
    assert mgr._last_cost_usd == 0.8
    assert abs(mgr._total_cost_usd - 0.8) < 1e-9
    assert abs(mgr._current_log_meta["solve_cost_usd"] - 0.3) < 1e-9
    assert mgr._total_turns == 5
    assert client.queries == ["first", "second"]
    assert client.connect_calls == 1


def test_cost_drop_charges_full_new_cost(monkeypatch, tmp_path):
    """A cumulative-cost drop (session reset) charges the full new cost."""
    fake_cls = _make_fake_client_cls()
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", fake_cls)
    mgr = _make_manager(tmp_path)
    asyncio.run(mgr.start_session())
    client = fake_cls.instances[-1]
    client.scripts = [
        [_result_message(0.8)],
        [_result_message(0.1)],  # Drop below 0.8: the session reset.
    ]

    asyncio.run(mgr.query("first"))
    asyncio.run(mgr.query("after reset"))
    # Not a negative delta: the full 0.1 is charged.
    assert mgr._current_log_meta["solve_cost_usd"] == 0.1
    assert abs(mgr._total_cost_usd - 0.9) < 1e-9
    assert mgr._last_cost_usd == 0.1


def test_query_error_recovers_session_and_logs_error(monkeypatch, tmp_path):
    """A query error appends an error entry and reconnects a new client."""
    fake_cls = _make_fake_client_cls()
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", fake_cls)
    mgr = _make_manager(tmp_path)
    asyncio.run(mgr.start_session())
    first_client = fake_cls.instances[-1]
    first_client.fail_next_query = True

    collected = asyncio.run(mgr.query("doomed"))
    assert collected == [{
        "type": "error",
        "error": "scripted SDK failure",
    }]
    # The error entry is also tracked in the conversation log.
    assert mgr.conversation_log[-1]["query"] == "doomed"
    assert mgr.conversation_log[-1]["response"] == collected
    # Recovery disconnected the old client and connected a fresh one.
    assert first_client.disconnect_calls == 1
    assert len(fake_cls.instances) == 2
    recovered_client = fake_cls.instances[-1]
    assert recovered_client is not first_client
    assert recovered_client.connect_calls == 1
    assert mgr._started

    # The recovered client serves subsequent queries.
    recovered_client.scripts = [[_result_message(0.2)]]
    asyncio.run(mgr.query("retry"))
    assert recovered_client.queries == ["retry"]


def test_query_starts_session_lazily(monkeypatch, tmp_path):
    """query() starts the session when none has been started yet."""
    fake_cls = _make_fake_client_cls()
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", fake_cls)
    mgr = _make_manager(tmp_path)
    assert not fake_cls.instances
    asyncio.run(mgr.query("lazy"))
    assert len(fake_cls.instances) == 1
    assert fake_cls.instances[0].connect_calls == 1


def test_tool_names_strips_mcp_prefix(tmp_path):
    """tool_names strips the MCP server prefix from allowed tools."""
    mgr = AgentSessionManager(
        system_prompt="p",
        mcp_server=object(),
        log_dir=str(tmp_path),
        model_name="claude-fable-5",
        allowed_tools=["mcp__predicator_tools__probe", "Bash"])
    assert mgr.tool_names == ["probe", "Bash"]
    mgr_none = _make_manager(tmp_path)
    mgr_none._allowed_tools = None
    assert mgr_none.tool_names == []


def test_run_query_sync_without_running_loop():
    """run_query_sync executes an async query with no loop running."""

    class _FakeSession:
        """Session stub whose query records its kwargs."""

        def __init__(self):
            self.calls = []

        async def query(self, message, **kwargs):
            """Record and echo the query."""
            self.calls.append((message, kwargs))
            return [{"type": "result", "echo": message}]

    session = _FakeSession()
    out = run_query_sync(session, "hello", kind="learn")
    assert out == [{"type": "result", "echo": "hello"}]
    assert session.calls == [("hello", {"kind": "learn"})]
