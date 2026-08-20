"""Shared python-exec tool core behind run_python and explore_python."""
import os
import time
from typing import Any, Callable, Dict, Optional

from predicators.agent_sdk.config import ToolSurfaceConfig
from predicators.agent_sdk.tools.budget import _arm_budget_watchdog, \
    _budget_footer
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.sandbox_guard import \
    _screen_text_for_sandbox_escape, _scrub_host_paths


def _make_python_exec_tool(
    tool: Callable,
    *,
    name: str,
    description: str,
    exec_ns: Dict[str, Any],
    sandbox_dir: Optional[str],
    sandbox_dir_for_agent: Optional[str] = None,
    text_result: Callable[[str], Dict[str, Any]],
    budget_ctx: Optional[ToolContext] = None,
) -> Any:
    """Build a code-execution MCP tool over a persistent namespace.

    Shared core behind the synthesis-phase ``run_python`` (namespace =
    trajectory data) and the solve-phase ``explore_python`` (namespace =
    the ``BeliefProbe`` exploration facade): sandbox-escape screening, in-
    process ``exec`` with stdout capture, and oversize-output spill to
    ``<sandbox_dir>/tool_outputs/<name>/``. The namespace persists
    across calls, so agents can define helpers once and reuse them.

    ``budget_ctx`` (the solve session's ToolContext) opts the tool into
    wall-clock budgeting: each call arms the per-call deadline
    (``agent_sdk_explore_python_call_timeout``) that probe sim calls
    enforce cooperatively, a call arriving after the attempt deadline is
    refused with a submit-now message, and every result carries a
    ``[budget]`` footer (attempt time + rollout counts) so sweeps have a
    visible price.
    """
    # pylint: disable=import-outside-toplevel
    import io
    import sys
    import traceback  # pylint: disable=redefined-outer-name,reimported

    # pylint: enable=import-outside-toplevel

    inline_char_limit = 30000
    preview_head_lines = 30
    preview_tail_lines = 30
    outputs_subdir = os.path.join("tool_outputs", name)
    outputs_dir_host: Optional[str] = (os.path.join(
        sandbox_dir, outputs_subdir) if sandbox_dir else None)
    if sandbox_dir_for_agent:
        outputs_dir_agent: Optional[str] = (
            f"{sandbox_dir_for_agent.rstrip('/')}/"
            f"{outputs_subdir.replace(os.sep, '/')}")
    else:
        outputs_dir_agent = outputs_dir_host
    # Continue numbering after any spill files already in the directory,
    # so re-created instances sharing a sandbox never overwrite earlier
    # outputs.
    count = [0]
    if outputs_dir_host and os.path.isdir(outputs_dir_host):
        count[0] = len(os.listdir(outputs_dir_host))

    @tool(
        name,
        description,
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                }
            },
            "required": ["code"],
        },
    )
    async def python_exec(args: Dict[str, Any]) -> Dict[str, Any]:
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.belief_probe import ProbeBudgetExceeded
        code = args["code"]
        # The code execs in-process with full filesystem access, and the
        # sandbox's PreToolUse file-path hook does not cover MCP tools, so
        # screen the code here for out-of-sandbox reads / source
        # introspection before executing (best-effort; see
        # _screen_text_for_sandbox_escape).
        if sandbox_dir is not None:
            reason = _screen_text_for_sandbox_escape(code, sandbox_dir)
            if reason is not None:
                return text_result(
                    f"Error: sandbox guard blocked this code - {reason}. "
                    "Read files with Read/Grep and use the MCP tools and "
                    "./reference/ files instead.")
        rollouts_before = 0
        if budget_ctx is not None:
            rollouts_before = budget_ctx.attempt_rollout_count
            attempt_dl = budget_ctx.attempt_deadline
            if (attempt_dl is not None and time.monotonic() > attempt_dl
                    and not budget_ctx.capture_best_effort_plan):
                return text_result(
                    "The attempt's wall-clock exploration budget is "
                    "exhausted - this call was not run. Submit your single "
                    "best plan NOW via evaluate_option_plan on the current "
                    "task (omit task_idx)." +
                    _budget_footer(budget_ctx, rollouts_before))
            call_timeout = ToolSurfaceConfig.from_cfg(
            ).explore_python_call_timeout
            if budget_ctx.probe_option_model_provider is not None:
                # Synthesis sessions probe the CANDIDATE simulator, whose
                # rollouts are far slower than belief-sim ones and whose
                # reset can trigger a fresh fit - a legitimate single call
                # can exceed the solve-tuned cap, so synthesis is exempt
                # from the per-call limit.
                call_timeout = 0.0
            budget_ctx.explore_call_deadline = (time.monotonic() + call_timeout
                                                if call_timeout > 0 else None)

        def _footer() -> str:
            if budget_ctx is None:
                return ""
            return _budget_footer(budget_ctx, rollouts_before)

        # Hard watchdog for pure-Python code that never reaches a probe
        # checkpoint (see _arm_budget_watchdog): armed to the nearest of
        # the per-call and attempt deadlines.
        watchdog_disarm: Optional[Callable[[], None]] = None
        if budget_ctx is not None:
            wd_deadlines = []
            if budget_ctx.explore_call_deadline is not None:
                wd_deadlines.append(budget_ctx.explore_call_deadline)
            if (budget_ctx.attempt_deadline is not None
                    and not budget_ctx.capture_best_effort_plan):
                wd_deadlines.append(budget_ctx.attempt_deadline)
            if wd_deadlines:
                remaining = min(wd_deadlines) - time.monotonic()
                if remaining > 0:
                    watchdog_disarm = _arm_budget_watchdog(remaining)

        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        try:
            exec(code, exec_ns)  # pylint: disable=exec-used
        except ProbeBudgetExceeded as e:
            partial = captured.getvalue()
            prefix = f"{partial}\n" if partial else ""
            # The watchdog's async exception carries no message; give it
            # the same actionable framing as the cooperative checks.
            msg = str(e) or (
                "this call exceeded its wall-clock budget and was stopped "
                "mid-execution; output printed so far is returned above. "
                "Split the work into smaller calls, and print intermediate "
                "results so partial progress survives a stop.")
            return text_result(f"{prefix}TIME BUDGET: {msg}{_footer()}")
        except Exception:  # pylint: disable=broad-except
            tb = _scrub_host_paths(traceback.format_exc())
            partial = captured.getvalue()
            prefix = f"{partial}\n" if partial else ""
            return text_result(f"{prefix}Error:\n{tb}{_footer()}")
        finally:
            if watchdog_disarm is not None:
                watchdog_disarm()
            sys.stdout = old_stdout
            if budget_ctx is not None:
                budget_ctx.explore_call_deadline = None

        output = captured.getvalue()
        if not output:
            return text_result(f"(no output){_footer()}")

        output += _footer()
        if len(output) <= inline_char_limit or outputs_dir_host is None:
            return text_result(output)

        count[0] += 1
        os.makedirs(outputs_dir_host, exist_ok=True)
        filename = f"call_{count[0]:04d}.txt"
        host_path = os.path.join(outputs_dir_host, filename)
        with open(host_path, "w", encoding="utf-8") as f:
            f.write(output)

        out_lines = output.splitlines()
        total_lines = len(out_lines)
        head = out_lines[:preview_head_lines]
        tail = (out_lines[-preview_tail_lines:] if total_lines >
                (preview_head_lines + preview_tail_lines) else [])
        agent_path = (f"{outputs_dir_agent}/{filename}"
                      if outputs_dir_agent else host_path)
        preview_parts = [
            f"[{name} output too large to inline: "
            f"{len(output):,} chars across {total_lines:,} lines; "
            f"full output saved to {agent_path}. Use Read/Grep to "
            f"inspect, or rerun with narrower print() to keep results "
            f"inline.]",
            "",
            f"--- head ({len(head)} lines) ---",
            *head,
        ]
        if tail:
            omitted = total_lines - len(head) - len(tail)
            preview_parts.extend([
                "",
                f"... [{omitted:,} lines omitted] ...",
                "",
                f"--- tail ({len(tail)} lines) ---",
                *tail,
            ])
        return text_result("\n".join(preview_parts))

    return python_exec
