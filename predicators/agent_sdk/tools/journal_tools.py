"""The record_journal solve-journal tool."""
from typing import Any, Callable, Dict

from predicators.agent_sdk.config import ValidationConfig
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.results import _error_result


def _build_journal_tools(ctx: ToolContext, _text_result: Callable,
                         tool: Callable) -> Dict[str, Any]:
    """``record_journal`` - agent-authored entries in the run's solve
    journal.

    Built only when the journal is enabled. The journal is the curated
    cross-attempt/cross-task memory channel for fresh-context solve
    sessions, so the tool guidance insists on facts and measurements:
    recorded verdicts ("X is impossible") from a failed attempt would
    re-import exactly the anchoring a restart is meant to shed
    (run_20260717_230436 seed1 concluded a "hard collision boundary"
    its sibling run placed through minutes later).
    """
    if not ValidationConfig.from_cfg().use_journal:
        return {}
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk import journal as journal_mod

    @tool(
        "record_journal",
        ("Append a short entry to the run's persistent solve journal "
         "(journal.md), which future solve attempts - starting with FRESH "
         "context - read in their prompt. Record durable, transferable "
         "facts: what you tried with exact parameters, what you measured, "
         "what worked (and its load-bearing values), and what a fresh "
         "attempt should try differently. Facts and measurements ONLY - do "
         "NOT record conclusions like 'X is impossible' or 'the task "
         "requires Y' (a wrong verdict anchors every later attempt; the "
         "evidence lets them re-judge). State every negative result as the "
         "exact family swept - parameters, orientations, regions, and any "
         "formula the sweep assumed - plus what remains untested: 'X never "
         "works' generalized from a partial sweep has buried the correct "
         "mechanism for entire runs. Keep it skimmable: a few bullets, "
         f"under {journal_mod.MAX_ENTRY_CHARS} chars."),
        {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "The journal entry (markdown bullets).",
                }
            },
            "required": ["entry"],
        },
    )
    async def record_journal(args: Dict[str, Any]) -> Dict[str, Any]:
        entry = (args.get("entry") or "").strip()
        if not entry:
            return _error_result("`entry` is required.")
        if not ctx.sandbox_dir:
            return _error_result("No sandbox directory in this session.")
        if ctx.test_task_idx is not None:
            where = f"test task {ctx.test_task_idx}"
        else:
            where = "pre-test phase"
        attempt = f", attempt {ctx.attempt_index}" if ctx.attempt_index else ""
        note = journal_mod.append_entry(ctx.sandbox_dir,
                                        f"Agent notes ({where}{attempt})",
                                        entry)
        msg = "Recorded to the solve journal."
        if note is not None:
            msg += f" NOTE: {note}."
        return _text_result(msg)

    return {"record_journal": record_journal}
