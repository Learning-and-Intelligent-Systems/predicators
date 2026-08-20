"""Tests for markdown formatting of parsed agent conversations.

Characterizes ``format_conversation_markdown``: headers and meta lines,
turn numbering, tool-call input rendering (multiline values become
fenced code blocks with a language picked via ``_LANG_BY_KEY``), tool
results, result-cost lines, error entries, and the unknown-block
fallback.
"""
from predicators.agent_sdk.log_formatter import format_conversation_markdown


def test_title_meta_and_prompt_sections():
    """The document carries the title, meta lines, and prompt text."""
    md = format_conversation_markdown(
        [],
        title="Docker Query",
        meta={
            "query_number": 3,
            "timestamp": "20260718_120000",
            "session_id": "sess-1",
            "query": "solve the task",
        })
    assert md.startswith("# Docker Query\n")
    assert "- **Query Number:** 3" in md
    assert "- **Timestamp:** 20260718_120000" in md
    assert "- **Session Id:** sess-1" in md
    assert "## Prompt" in md
    assert "solve the task" in md
    assert "## Conversation" in md


def test_no_meta_defaults():
    """Without meta the prompt section is empty and no meta lines appear."""
    md = format_conversation_markdown([], title="Query")
    assert "# Query" in md
    assert "**Query Number:**" not in md


def test_assistant_turn_numbering_and_separator():
    """Each assistant entry starts a turn; later turns get a separator."""
    collected = [
        {
            "type": "assistant",
            "content": [{
                "type": "text",
                "text": "first"
            }]
        },
        {
            "type": "assistant",
            "content": [{
                "type": "text",
                "text": "second"
            }]
        },
    ]
    md = format_conversation_markdown(collected)
    assert "### Turn 1" in md
    assert "### Turn 2" in md
    assert "**Assistant:** first" in md
    assert "**Assistant:** second" in md
    # Separator appears only before turn 2.
    assert md.index("---") < md.index("### Turn 2")
    assert md.index("### Turn 1") < md.index("---")


def test_thinking_block_rendered_as_blockquote():
    """ThinkingBlock content renders as a quoted [thinking] section."""
    collected = [{
        "type":
        "assistant",
        "content": [{
            "type": "ThinkingBlock",
            "thinking": "line one\nline two"
        }],
    }]
    md = format_conversation_markdown(collected)
    assert "*[thinking]*" in md
    assert "> line one" in md
    assert "> line two" in md


def test_tool_use_multiline_code_gets_python_fence():
    """Multiline 'code' values are fenced with the python language."""
    collected = [{
        "type":
        "assistant",
        "content": [{
            "type": "tool_use",
            "name": "explore_python",
            "id": "toolu_9",
            "input": {
                "code": "x = 1\nprint(x)",
                "timeout": 5,
            },
        }],
    }]
    md = format_conversation_markdown(collected)
    assert "**Tool Call:** `explore_python` (id: `toolu_9`)" in md
    assert "*code:*" in md
    assert "```python\nx = 1\nprint(x)\n```" in md
    # The remaining scalar goes into a compact JSON block.
    assert "```json" in md
    assert '"timeout": 5' in md


def test_tool_use_multiline_command_gets_bash_fence():
    """Multiline 'command' values are fenced with the bash language."""
    collected = [{
        "type":
        "assistant",
        "content": [{
            "type": "tool_use",
            "name": "Bash",
            "id": "t",
            "input": {
                "command": "ls\npwd"
            },
        }],
    }]
    md = format_conversation_markdown(collected)
    assert "```bash\nls\npwd\n```" in md


def test_tool_use_multiline_unknown_key_gets_bare_fence():
    """Multiline values under unmapped keys get a language-less fence."""
    collected = [{
        "type":
        "assistant",
        "content": [{
            "type": "tool_use",
            "name": "Write",
            "id": "t",
            "input": {
                "notes": "a\nb"
            },
        }],
    }]
    md = format_conversation_markdown(collected)
    assert "*notes:*" in md
    assert "```\na\nb\n```" in md


def test_tool_use_scalar_only_input_is_single_json_block():
    """Inputs without multiline strings render as one JSON block."""
    collected = [{
        "type":
        "assistant",
        "content": [{
            "type": "tool_use",
            "name": "probe",
            "id": "t",
            "input": {
                "a": 1,
                "b": "one line"
            },
        }],
    }]
    md = format_conversation_markdown(collected)
    assert '"a": 1' in md
    assert '"b": "one line"' in md
    assert "*a:*" not in md and "*b:*" not in md


def test_user_tool_result_variants():
    """Tool results render string, list, and image contents."""
    collected = [{
        "type":
        "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": "plain result",
                "is_error": False,
            },
            {
                "type":
                "tool_result",
                "tool_use_id":
                "t2",
                "content": [
                    {
                        "type": "text",
                        "text": "item text"
                    },
                    {
                        "type": "image",
                        "mimeType": "image/png"
                    },
                    "bare string item",
                ],
                "is_error":
                True,
            },
        ],
    }]
    md = format_conversation_markdown(collected)
    assert "**Tool Result** (tool_use_id: `t1`):" in md
    assert "```\nplain result\n```" in md
    assert "**Tool Error** (tool_use_id: `t2`):" in md
    assert "```\nitem text\n```" in md
    assert "*[image: image/png]*" in md
    assert "```\nbare string item\n```" in md


def test_user_text_block():
    """User text blocks render with a User label."""
    collected = [{
        "type": "user",
        "content": [{
            "type": "text",
            "text": "hi there"
        }],
    }]
    assert "**User:** hi there" in format_conversation_markdown(collected)


def test_result_uses_meta_solve_total_split_when_available():
    """Meta solve/total costs override the raw cumulative cost."""
    collected = [{
        "type": "result",
        "num_turns": 7,
        "total_cost_usd": 9.99,
    }]
    md = format_conversation_markdown(collected,
                                      meta={
                                          "solve_cost_usd": 0.3,
                                          "total_cost_usd": 1.2,
                                      })
    assert "**Result:** 7 turns, $0.30 this solve, $1.20 total" in md
    assert "$9.99" not in md


def test_result_falls_back_to_raw_cumulative_cost():
    """Without the meta split, the raw cumulative cost is shown."""
    collected = [{"type": "result", "num_turns": 2, "total_cost_usd": 0.5}]
    assert "**Result:** 2 turns, $0.50" in \
        format_conversation_markdown(collected)


def test_result_missing_cost_and_turns_show_placeholders():
    """Missing turns/cost render as question marks."""
    md = format_conversation_markdown([{"type": "result"}])
    assert "**Result:** ? turns, ?" in md


def test_error_entry():
    """Error entries render with an Error label."""
    md = format_conversation_markdown([{"type": "error", "error": "boom"}])
    assert "**Error:** boom" in md


def test_unknown_blocks_render_type_header_and_json_extras():
    """Unknown block types show their type plus non-None extras as JSON."""
    collected = [
        {
            "type":
            "assistant",
            "content": [{
                "type": "MysteryBlock",
                "payload": "stuff",
                "empty": None,
            }],
        },
        {
            "type": "user",
            "content": [{
                "type": "OtherBlock"
            }],
        },
    ]
    md = format_conversation_markdown(collected)
    assert "**MysteryBlock:**" in md
    assert '"payload": "stuff"' in md
    assert '"empty"' not in md
    # A user-side unknown block with no extras gets only the header.
    assert "**OtherBlock:**" in md


def test_unknown_entry_type_is_skipped():
    """Entries with unrecognized top-level types are ignored."""
    md = format_conversation_markdown([{"type": "telemetry", "x": 1}])
    assert "telemetry" not in md
