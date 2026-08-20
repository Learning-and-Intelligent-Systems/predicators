"""Tests for shared Claude SDK response message parsing.

Characterizes ``parse_assistant_message``, ``parse_user_message``,
``parse_result_message``, and the ``parse_message`` dispatcher. The
parser isinstance-checks blocks against the real ``claude_agent_sdk``
dataclasses, so the recognized branches use real (pure, offline)
``TextBlock``/``ToolUseBlock``/``ToolResultBlock``/``ThinkingBlock``
instances; the unknown-block fallback branches use local stub classes.
No network or subprocess is involved: these are plain dataclasses.
"""
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, \
    ThinkingBlock, ToolResultBlock, ToolUseBlock, UserMessage

from predicators.agent_sdk.response_parser import parse_assistant_message, \
    parse_message, parse_result_message, parse_user_message


class _MysteryBlock:
    """Stub block type the parser does not recognize."""

    def __init__(self, **attrs):
        for key, val in attrs.items():
            setattr(self, key, val)


def _assistant(*blocks):
    return AssistantMessage(content=list(blocks), model="test-model")


def _result(**kwargs):
    defaults = {
        "subtype": "success",
        "duration_ms": 10,
        "duration_api_ms": 8,
        "is_error": False,
        "num_turns": 1,
        "session_id": "sess",
    }
    defaults.update(kwargs)
    return ResultMessage(**defaults)


# ---------------------------------------------------------------------------
# parse_assistant_message
# ---------------------------------------------------------------------------


def test_assistant_text_and_tool_use_blocks():
    """Text and tool_use blocks map to their dict shapes."""
    msg = _assistant(
        TextBlock(text="hello"),
        ToolUseBlock(id="toolu_1", name="run_probe", input={"x": 1}),
    )
    entry = parse_assistant_message(msg)
    assert entry == {
        "type":
        "assistant",
        "content": [
            {
                "type": "text",
                "text": "hello"
            },
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "run_probe",
                "input": {
                    "x": 1
                },
            },
        ],
    }


def test_assistant_thinking_block_falls_back_to_generic_dict():
    """ThinkingBlock is not special-cased: it hits the fallback branch."""
    msg = _assistant(ThinkingBlock(thinking="pondering...", signature="sig"))
    entry = parse_assistant_message(msg)
    assert entry["content"] == [{
        "type": "ThinkingBlock",
        "thinking": "pondering...",
    }]


def test_assistant_unknown_block_collects_probed_attrs_dropping_none():
    """Unknown blocks keep only the probed, non-None attributes."""
    block = _MysteryBlock(name="foo",
                          input={"a": 1},
                          id=None,
                          text="t",
                          tool_use_id="tid",
                          unprobed="ignored")
    entry = parse_assistant_message(_assistant(block))
    assert entry["content"] == [{
        "type": "_MysteryBlock",
        "name": "foo",
        "input": {
            "a": 1
        },
        "text": "t",
        "tool_use_id": "tid",
    }]


def test_assistant_empty_content():
    """An empty content list yields an empty parsed content list."""
    assert parse_assistant_message(_assistant()) == {
        "type": "assistant",
        "content": [],
    }


# ---------------------------------------------------------------------------
# parse_user_message
# ---------------------------------------------------------------------------


def test_user_text_and_tool_result_blocks():
    """Text and tool_result blocks map to their dict shapes."""
    msg = UserMessage(content=[
        TextBlock(text="user says"),
        ToolResultBlock(tool_use_id="toolu_1", content="ok", is_error=True),
    ])
    entry = parse_user_message(msg)
    assert entry == {
        "type":
        "user",
        "content": [
            {
                "type": "text",
                "text": "user says"
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "ok",
                "is_error": True,
            },
        ],
    }


def test_user_tool_result_default_is_error_passes_through_none():
    """An unset is_error (None) is carried through, not coerced to False."""
    msg = UserMessage(
        content=[ToolResultBlock(tool_use_id="toolu_2", content="fine")])
    entry = parse_user_message(msg)
    assert entry["content"][0]["is_error"] is None


def test_user_unknown_block_omits_thinking_attr():
    """The user fallback probes is_error but not thinking."""
    msg = UserMessage(content=[
        ThinkingBlock(thinking="hidden", signature="sig"),
        _MysteryBlock(text="t", is_error=False),
    ])
    entry = parse_user_message(msg)
    # ThinkingBlock's thinking attr is not in the user probe list.
    assert entry["content"][0] == {"type": "ThinkingBlock"}
    # is_error=False survives the "val is not None" filter.
    assert entry["content"][1] == {
        "type": "_MysteryBlock",
        "text": "t",
        "is_error": False,
    }


# ---------------------------------------------------------------------------
# parse_result_message
# ---------------------------------------------------------------------------


def test_result_message_fields():
    """subtype, num_turns, cost, and the error fields are extracted."""
    entry = parse_result_message(
        _result(subtype="error_max_turns",
                num_turns=50,
                total_cost_usd=1.25,
                is_error=True,
                result="hit the turn cap"))
    assert entry == {
        "type": "result",
        "subtype": "error_max_turns",
        "num_turns": 50,
        "total_cost_usd": 1.25,
        "is_error": True,
        "result": "hit the turn cap",
    }


def test_result_message_missing_attrs_default_to_none():
    """Objects lacking the result attrs parse to all-None fields."""
    entry = parse_result_message(object())
    assert entry == {
        "type": "result",
        "subtype": None,
        "num_turns": None,
        "total_cost_usd": None,
        "is_error": False,
        "result": None,
    }


# ---------------------------------------------------------------------------
# parse_message dispatcher
# ---------------------------------------------------------------------------


def test_parse_message_dispatches_by_type():
    """Each SDK message type routes to the matching parser."""
    assert parse_message(_assistant(
        TextBlock(text="hi")))["type"] == "assistant"
    assert parse_message(UserMessage(content=[]))["type"] == "user"
    assert parse_message(_result(total_cost_usd=0.5))["type"] == "result"


def test_parse_message_unknown_type_returns_none():
    """Unrecognized message objects yield None."""
    assert parse_message(object()) is None
    assert parse_message("just a string") is None
    assert parse_message(None) is None
