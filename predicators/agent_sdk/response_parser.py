"""Shared response message parsing for Claude Agent SDK sessions.

Converts ``claude_agent_sdk`` message types (``AssistantMessage``,
``UserMessage``, ``ResultMessage``) into plain dicts suitable for
logging and serialization.  Used by ``AgentSessionManager``,
``LocalSandboxSessionManager``, and ``docker_agent_runner``.
"""
from typing import Any, Dict, Optional


def parse_assistant_message(msg: Any) -> Dict[str, Any]:
    """Convert an ``AssistantMessage`` to a serializable dict."""
    from claude_agent_sdk import TextBlock, ToolUseBlock \
        # pylint: disable=import-outside-toplevel

    entry: Dict[str, Any] = {"type": "assistant", "content": []}
    for block in msg.content:
        if isinstance(block, TextBlock):
            entry["content"].append({
                "type": "text",
                "text": block.text,
            })
        elif isinstance(block, ToolUseBlock):
            entry["content"].append({
                "type": "tool_use",
                "id": getattr(block, "id", None),
                "name": block.name,
                "input": block.input,
            })
        else:
            block_type = type(block).__name__
            block_dict: Dict[str, Any] = {"type": block_type}
            for attr in ("name", "input", "id", "text", "content",
                         "tool_use_id", "thinking"):
                val = getattr(block, attr, None)
                if val is not None:
                    block_dict[attr] = val
            entry["content"].append(block_dict)
    return entry


def parse_user_message(msg: Any) -> Dict[str, Any]:
    """Convert a ``UserMessage`` to a serializable dict."""
    from claude_agent_sdk import TextBlock, ToolResultBlock \
        # pylint: disable=import-outside-toplevel

    entry: Dict[str, Any] = {"type": "user", "content": []}
    for block in msg.content:  # type: ignore[union-attr]
        if isinstance(block, TextBlock):
            entry["content"].append({
                "type": "text",
                "text": block.text,
            })
        elif isinstance(block, ToolResultBlock):
            entry["content"].append({
                "type":
                "tool_result",
                "tool_use_id":
                getattr(block, "tool_use_id", None),
                "content":
                getattr(block, "content", None),
                "is_error":
                getattr(block, "is_error", False),
            })
        else:
            block_dict: Dict[str, Any] = {"type": type(block).__name__}
            for attr in ("name", "input", "id", "text", "content",
                         "tool_use_id", "is_error"):
                val = getattr(block, attr, None)
                if val is not None:
                    block_dict[attr] = val
            entry["content"].append(block_dict)
    return entry


def parse_result_message(msg: Any) -> Dict[str, Any]:
    """Convert a ``ResultMessage`` to a serializable dict."""
    return {
        "type": "result",
        # "success" or an error marker such as "error_max_turns" (the
        # session ended because it hit ClaudeAgentOptions.max_turns).
        "subtype": getattr(msg, "subtype", None),
        "num_turns": getattr(msg, "num_turns", None),
        "total_cost_usd": getattr(msg, "total_cost_usd", None),
        # Error results carry the error text in "result"; both feed the
        # fatal-session check (session_base.query_fatal_error).
        "is_error": getattr(msg, "is_error", False),
        "result": getattr(msg, "result", None),
    }


def parse_message(msg: Any) -> Optional[Dict[str, Any]]:
    """Dispatch to the appropriate parser based on message type.

    Returns ``None`` for unrecognised message types.
    """
    # pylint: disable=import-outside-toplevel
    from claude_agent_sdk import AssistantMessage, ResultMessage, UserMessage

    # pylint: enable=import-outside-toplevel

    if isinstance(msg, AssistantMessage):
        return parse_assistant_message(msg)
    if isinstance(msg, UserMessage):
        return parse_user_message(msg)
    if isinstance(msg, ResultMessage):
        return parse_result_message(msg)
    return None
