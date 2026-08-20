"""Task-evaluator verdict helpers and ground-sampler loading."""
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, \
    Union

from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.config import RefinementConfig, SessionConfig
from predicators.agent_sdk.proposal_exec import build_exec_context, \
    load_ground_samplers
from predicators.agent_sdk.tools.context import ToolContext
from predicators.structs import Predicate, State, Task


class _EvalStateCollector:
    """Per-step states + option labels of one rollout, for evaluator verdicts.

    The single collector behind every surface that scores a belief-sim
    rollout (``evaluate_option_plan``'s first and validation rollouts,
    ``_belief_rollout_verdict``). The cascade certificate needs per-step
    states (topple-onset analysis); option-boundary states give garbage
    verdicts, so prefer the option model's ``last_trajectory`` and flag
    the verdict as coarse (``self.coarse``) when it is unavailable.
    """

    def __init__(self, option_model: Any, init_state: State) -> None:
        self._option_model = option_model
        self.states: List[State] = [init_state]
        self.labels: List[Any] = []
        self.coarse = False

    def collect(self, outcome: Any) -> None:
        """Record one executed option's outcome."""
        if outcome.post_state is None:
            return
        opt = outcome.option
        label = (opt.name, tuple(o.name for o in opt.objects),
                 tuple(float(p) for p in opt.params))
        step_traj = getattr(self._option_model, "last_trajectory", None)
        if step_traj is not None and len(step_traj.states) >= 2:
            self.states.extend(step_traj.states[1:])
            self.labels.extend([label] * len(step_traj.actions))
        else:
            self.states.append(outcome.post_state)
            self.labels.append(label)
            self.coarse = True

    def on_step(self, _i: int, outcome: Any) -> None:
        """``execute_plan_forward`` ``on_step`` adapter."""
        self.collect(outcome)


def evaluate_states_with(evaluator: Any,
                         states: Sequence[State],
                         step_options: Optional[Sequence[Any]],
                         sim_env: Optional[Any] = None) -> Dict[str, Any]:
    """Score a state/option-label sequence with a task's ``TaskEvaluator``.

    The single verdict surface: only booleans/scalars/reasons leave this
    function, never the evaluator object. Verdicts on belief-sim
    rollouts are exactly as trustworthy as the belief sim itself -
    including ``sim_env``, the belief env backing the rollout, passed
    through so physics-needing certificates (the domino counterfactual
    push probe) can probe with the same belief physics.
    ``legitimate``/``reason`` are HARNESS-INTERNAL (capture gating,
    logs), and ``terminated`` is agent-computable from the public goal
    atoms: agent-facing surfaces expose only the public (solved,
    reward) pair - the standard RL end-of-episode observables - so the
    agent must infer the scoring rules from the stated objective and
    the outcomes its rollouts earn.
    """
    ok, reason = evaluator._certify(states, step_options, sim_env=sim_env)  # pylint: disable=protected-access
    return {
        "terminated": evaluator.terminated(states[-1]),
        "reward": evaluator.reward(states, step_options, sim_env=sim_env),
        "solved": evaluator.solved(states, step_options, sim_env=sim_env),
        "legitimate": ok,
        "reason": reason,
    }


def make_solved_check(
    evaluator: Any,
    sim_env: Optional[Any],
    on_reject: Optional[Callable[[float], None]] = None
) -> Callable[[List[State], List[Any], bool], Tuple[bool, str]]:
    """Build the evaluator gate used inside refinement searches.

    One policy for every surface (the MCP ``refine_plan_sketch`` and
    ``BeliefProbe.refine``), so identical parameters can never get
    contradictory verdicts across tools:
    - a coarse rollout (option-boundary states only) never blocks, the
      same rule the capture path applies (a coarse certificate can
      falsely reject a legitimate cascade);
    - evaluator exceptions never block (fail-open, logged) - a flaky
      certificate must not abort a search mid-budget;
    - a non-terminated verdict never blocks (the goal-atom check is the
      caller's job; the gate only vetoes certified-non-solves).
    ``on_reject`` is called with the rejected verdict's reward.
    """

    def solved_check(states: List[State], labels: List[Any],
                     coarse: bool) -> Tuple[bool, str]:
        if coarse:
            return True, ""
        try:
            v = evaluate_states_with(evaluator,
                                     states,
                                     labels,
                                     sim_env=sim_env)
        except Exception as e:  # pylint: disable=broad-except
            logging.debug("In-search solved gate failed: %s", e)
            return True, ""
        if v["solved"] or not v["terminated"]:
            return True, ""
        if on_reject is not None:
            on_reject(v["reward"])
        return False, f"solved=False, reward={v['reward']:.2f}"

    return solved_check


def _format_evaluator_verdict(verdict: Dict[str, Any],
                              *,
                              coarse: bool = False) -> str:
    """One report line for an evaluator verdict on a belief-sim rollout.

    Emits only the public (solved, reward) pair; the certificate's
    legitimacy bool and reason stay harness-internal, and goal-atom
    termination is already reported (and agent-computable) separately.
    """
    line = (f"Task evaluator (belief-sim rollout - trustworthy only insofar "
            f"as your simulator is): solved={verdict['solved']}, "
            f"reward={verdict['reward']:.2f}")
    if coarse:
        line += ("\n  NOTE: per-step states were unavailable for part of the "
                 "rollout, so the verdict is coarse (computed on "
                 "option-boundary states only).")
    return line


def _resolve_task_evaluator(ctx: ToolContext, task_idx: Union[int, str,
                                                              None]) -> Any:
    """The evaluator of a tool's referenced task (``Task.evaluator``), or None.

    ``task_idx`` follows the tools' convention: an int indexes the train
    tasks; ``"current"``/None means the current solve/explore task.
    """
    if isinstance(task_idx, int):
        if 0 <= task_idx < len(ctx.train_tasks):
            return ctx.train_tasks[task_idx].evaluator
        return None
    if ctx.current_task is not None:
        return ctx.current_task.evaluator
    return None


def _ground_samplers_path(ctx: ToolContext) -> Optional[str]:
    """Host path of the agent-editable ``ground_samplers.py``.

    Resolves the sandbox base the same way the sampler-learning mixin
    does: the local sandbox lives under ``<log_dir>/sandbox``, the
    docker sandbox at ``ctx.sandbox_dir``, else the log dir itself.
    """
    if SessionConfig.from_cfg().use_local_sandbox and ctx.log_dir:
        base: Optional[str] = os.path.abspath(
            os.path.join(ctx.log_dir, "sandbox"))
    elif ctx.sandbox_dir:
        base = ctx.sandbox_dir
    else:
        base = ctx.log_dir
    if not base:
        return None
    return os.path.join(base, "ground_samplers.py")


def load_ground_sampler_fns(
        ctx: ToolContext) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load named ground samplers for ``~ my_sampler`` sketch references.

    Reads ``ground_samplers.py`` fresh (the agent edits it between
    calls) and validates its ``GROUND_SAMPLERS`` dict. Returns ``(fns,
    error)``: a missing file, or the feature being disabled, is simply
    ``({}, None)``; a broken file returns an error message for the agent
    so it can fix the code instead of silently sampling uniformly.
    """
    if not RefinementConfig.from_cfg().ground_samplers:
        return {}, None
    path = _ground_samplers_path(ctx)
    if path is None or not os.path.isfile(path):
        return {}, None
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    exec_ctx = build_exec_context(
        types=ctx.types,
        predicates=ctx.predicates
        | ctx.iteration_proposals.proposed_predicates,
        options=ctx.options | ctx.iteration_proposals.proposed_options)
    fns, warnings, err = load_ground_samplers(code, exec_ctx)
    if err is not None:
        return {}, f"Error loading {path}:\n{err}"
    for warning in warnings:
        logging.warning("ground_samplers.py: %s", warning)
    return fns, None


def _belief_rollout_verdict(
        ctx: ToolContext, task: Task, task_idx: Union[int, str, None],
        grounded_plan: List[Any],
        predicates: Set[Predicate]) -> Optional[Tuple[Dict[str, Any], bool]]:
    """Execute ``grounded_plan`` in the belief sim and score it with the task's
    evaluator, returning ``(verdict, coarse)`` or None.

    Used by ``refine_plan_sketch``, whose internal refinement rollouts
    don't expose per-step states; costs one extra plan rollout. Fully
    failure-tolerant: any problem returns None.
    """
    evaluator = _resolve_task_evaluator(ctx, task_idx)
    if evaluator is None or not grounded_plan or ctx.option_model is None:
        return None
    collector = _EvalStateCollector(ctx.option_model, task.init)
    try:
        bilevel_sketch.execute_plan_forward(task,
                                            grounded_plan,
                                            ctx.option_model,
                                            predicates=predicates,
                                            on_step=collector.on_step,
                                            stop_on_failure=True)
        if len(collector.states) < 2:
            return None
        verdict = evaluate_states_with(evaluator,
                                       collector.states,
                                       collector.labels,
                                       sim_env=getattr(ctx.option_model,
                                                       "sim_env", None))
        return verdict, collector.coarse
    except Exception as e:  # pylint: disable=broad-except
        logging.debug("Belief-rollout evaluator verdict failed: %s", e)
        return None


def _belief_rollout_verdict_line(ctx: ToolContext, task: Task,
                                 task_idx: Union[int, str, None],
                                 grounded_plan: List[Any],
                                 predicates: Set[Predicate]) -> Optional[str]:
    """Format the belief-rollout evaluator verdict as a report line."""
    scored = _belief_rollout_verdict(ctx, task, task_idx, grounded_plan,
                                     predicates)
    if scored is None:
        return None
    verdict, coarse = scored
    return _format_evaluator_verdict(verdict, coarse=coarse)
