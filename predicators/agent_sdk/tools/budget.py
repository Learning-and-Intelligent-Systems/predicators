"""Solve-attempt budget surfaces: the [budget] footer and watchdog."""
import time
from typing import Callable

from predicators.agent_sdk.tools.context import ToolContext


def _budget_footer(ctx: ToolContext, rollouts_before: int = 0) -> str:
    """``[budget]`` line appended to tool results during a solve attempt.

    Shows attempt wall-clock (elapsed, and the budget when one is set)
    and the attempt's cumulative sim-rollout count (plus this call's
    delta). Agents pace well when they can see a clock and terribly when
    they can't: the 47k-rollout single-call sweep of run_20260717_230436
    ran 7 h with zero cost feedback. Empty when no attempt is in flight.
    """
    start = ctx.attempt_start
    if start is None:
        return ""
    parts = []
    elapsed_min = (time.monotonic() - start) / 60.0
    deadline = ctx.attempt_deadline
    if deadline is not None:
        total_min = (deadline - start) / 60.0
        parts.append(f"attempt time {elapsed_min:.1f}/{total_min:.0f} min")
    else:
        parts.append(f"attempt time {elapsed_min:.1f} min")
    total_rollouts = ctx.attempt_rollout_count
    delta = total_rollouts - rollouts_before
    rollout_part = f"sim rollouts this attempt: {total_rollouts}"
    if delta > 0:
        rollout_part += f" (+{delta} this call)"
    parts.append(rollout_part)
    return "\n\n[budget] " + "; ".join(parts)


def _arm_budget_watchdog(seconds: float) -> Callable[[], None]:
    """Schedule a ProbeBudgetExceeded in the CALLING thread after ``seconds``;
    returns an idempotent disarm callable.

    ``explore_python``'s exec() runs on the event-loop thread, so
    pure-Python code that never reaches a probe checkpoint blocks every
    cooperative deadline check AND the sandbox's message-stream
    interrupt backstop - an async exception from a watchdog timer is the
    only preemption that reaches it. Delivery happens at the next
    bytecode boundary, so a long blocking C call (a physics step) defers
    it; those paths are exactly the ones the cooperative probe checks
    already cover.
    """
    # pylint: disable=import-outside-toplevel
    import ctypes
    import threading

    from predicators.agent_sdk.belief_probe import ProbeBudgetExceeded

    # pylint: enable=import-outside-toplevel
    target_id = threading.get_ident()
    lock = threading.Lock()
    armed = [True]

    def _fire() -> None:
        # The lock makes fire/disarm mutually exclusive so the async
        # exception cannot be injected after the call has already
        # returned and disarmed.
        with lock:
            if not armed[0]:
                return
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(target_id),
                ctypes.py_object(ProbeBudgetExceeded))

    timer = threading.Timer(seconds, _fire)
    timer.daemon = True
    timer.start()

    def _disarm() -> None:
        with lock:
            armed[0] = False
        timer.cancel()

    return _disarm
