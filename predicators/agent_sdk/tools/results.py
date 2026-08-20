"""Tool-result formatting helpers and small sandbox-file utilities."""
import copy
import functools
import os
from typing import Any, Callable, Dict, Optional

from predicators.agent_sdk.config import RefinementConfig
from predicators.agent_sdk.tools.context import ToolContext


def session_log_filename(query_count: int,
                         kind: str,
                         timestamp: str,
                         test_task_idx: Optional[int] = None,
                         ext: str = "md") -> str:
    """Build the session-log filename shared by the sandbox backends.

    Layout: ``NNN_<kind>[_task<idx>]_<timestamp>.<ext>``. The counter comes
    first so alphabetical sort matches chronological order; for test queries
    the ``_task<idx>`` segment ties the file to ``main.py``'s test task index.
    """
    suffix = ""
    if kind == "test" and test_task_idx is not None:
        suffix = f"_task{test_task_idx}"
    return f"{query_count:03d}_{kind}{suffix}_{timestamp}.{ext}"


def _text_result(text: str) -> Dict[str, Any]:
    """Helper to format a successful text result."""
    return {"content": [{"type": "text", "text": text}]}


def _error_result(text: str) -> Dict[str, Any]:
    """Helper to format an error result."""
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _region_syntax_blurb() -> str:
    """The `~ [w]` region mention for tool descriptions, flag-aware.

    Advertising the syntax while ``agent_bilevel_ground_samplers`` is
    off sent every audited run through 1-3 turns of syntax guessing
    against a feature that could not work.
    """
    if RefinementConfig.from_cfg().ground_samplers:
        return ", incl. `~ [w]` half-width regions after a step's params"
    return (" - note `~ [w]` regions are DISABLED in this configuration "
            "and are ignored if given")


def _make_coercing_tool(tool: Callable) -> Callable:
    """Wrap the SDK ``tool`` decorator with numeric-string coercion.

    Harness-side JSON-schema validation rejects ``"0"`` for an
    ``integer`` property before the handler ever runs (agents lost
    whole tools to it - ``inspect_trajectories`` went 0-for-6 in
    run_20260717_154753 seed2), and which tools accept strings was
    inconsistent. This wrapper loosens every top-level ``integer`` /
    ``number`` property to also accept a string, then coerces the value
    back to the numeric type before the handler sees it, so handlers
    keep their exact-type assumptions.
    """

    def _coercing_tool(name: str, description: str, schema: Any) -> Callable:
        numeric_props: Dict[str, type] = {}
        loosened = schema
        if isinstance(schema, dict) and isinstance(schema.get("properties"),
                                                   dict):
            loosened = copy.deepcopy(schema)
            for prop, spec in loosened["properties"].items():
                if not isinstance(spec, dict):
                    continue
                if spec.get("type") == "integer":
                    numeric_props[prop] = int
                    spec["type"] = ["integer", "string"]
                elif spec.get("type") == "number":
                    numeric_props[prop] = float
                    spec["type"] = ["number", "string"]

        def _decorate(fn: Callable) -> Any:
            if not numeric_props:
                return tool(name, description, schema)(fn)

            @functools.wraps(fn)
            async def _wrapped(args: Dict[str, Any]) -> Dict[str, Any]:
                for prop, target in numeric_props.items():
                    val = args.get(prop)
                    if isinstance(val, str):
                        try:
                            args[prop] = target(val)
                        except ValueError:
                            return _error_result(
                                f"`{prop}` must be a "
                                f"{target.__name__}; got {val!r}.")
                return await fn(args)

            return tool(name, description, loosened)(_wrapped)

        return _decorate

    return _coercing_tool


def _make_spilling_text_result(
    sandbox_dir: Optional[str],
    *,
    subdir: str = "tool_outputs",
    agent_prefix: Optional[str] = None,
    char_limit: int = 30000,
    head_lines: int = 30,
    tail_lines: int = 30,
) -> Callable[[str], Dict[str, Any]]:
    """Build a ``_text_result``-style helper that spills oversize output.

    A tool result returned inline that exceeds the agent SDK's MCP
    tool-result token cap is truncated by the SDK and dumped to
    ``~/.claude/projects/.../tool-results/`` — *outside* the sandbox.
    The agent is then instructed to read that host path, which both
    defeats the sandbox boundary and is the only legitimate reason the
    agent ever needs to touch a path outside its sandbox.

    To remove that need, when ``sandbox_dir`` is set and ``text`` exceeds
    ``char_limit`` (kept well under the SDK cap), this writes the full
    text to ``<sandbox_dir>/<subdir>/result_NNNN.txt`` and returns a
    head/tail preview plus the in-sandbox path for the agent to
    ``Read``/``Grep``. Small results, or the no-sandbox case, are
    returned inline unchanged.

    ``agent_prefix`` is the path prefix the agent sees (``"."`` for the
    local sandbox, ``"/sandbox"`` for docker); when ``None`` a relative
    ``./<subdir>`` path is used, which resolves correctly because the
    agent's cwd is always the sandbox root.
    """
    counter = [0]
    host_dir = os.path.join(sandbox_dir, subdir) if sandbox_dir else None
    prefix = agent_prefix.rstrip("/") if agent_prefix else "."
    agent_dir = f"{prefix}/{subdir.replace(os.sep, '/')}"

    def _text(text: str) -> Dict[str, Any]:
        if host_dir is None or len(text) <= char_limit:
            return _text_result(text)
        counter[0] += 1
        os.makedirs(host_dir, exist_ok=True)
        filename = f"result_{counter[0]:04d}.txt"
        with open(os.path.join(host_dir, filename), "w",
                  encoding="utf-8") as f:
            f.write(text)
        lines = text.splitlines()
        total = len(lines)
        head = lines[:head_lines]
        tail = (lines[-tail_lines:] if total > head_lines + tail_lines else [])
        parts = [
            f"[output too large to inline: {len(text):,} chars across "
            f"{total:,} lines; full output saved to "
            f"{agent_dir}/{filename}. Use Read/Grep to inspect it.]",
            "",
            f"--- head ({len(head)} lines) ---",
            *head,
        ]
        if tail:
            omitted = total - len(head) - len(tail)
            parts.extend([
                "",
                f"... [{omitted:,} lines omitted] ...",
                "",
                f"--- tail ({len(tail)} lines) ---",
                *tail,
            ])
        return _text_result("\n".join(parts))

    return _text


def _save_option_to_sandbox(ctx: ToolContext, option_name: str,
                            code: str) -> Optional[str]:
    """Save option source code to sandbox/proposed_code/<name>.py.

    Returns the relative path (e.g. ``./proposed_code/Pick.py``) or None
    if the sandbox directory is not set.
    """
    if ctx.sandbox_dir is None:
        return None
    proposed_dir = os.path.join(ctx.sandbox_dir, "proposed_code")
    os.makedirs(proposed_dir, exist_ok=True)
    filename = f"{option_name}.py"
    filepath = os.path.join(proposed_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    return f"./proposed_code/{filename}"
