"""Backtracking refinement of plan sketches over continuous parameters.

Split out of ``bilevel_sketch`` (see that module's docstring for the
full layout); holds ``refine_sketch`` (the backtracking search),
``refine_and_validate_report`` (its refine-then-forward-validate report
wrapper), and their supporting pieces: ``sample_params``, timeout
resolution, the ``_FeasiblePool`` / ``DeepestFailure`` /
``RefineOutcome`` records, and the search internals - a read-only
``_RefineContext`` plus a mutable ``_RefinementState`` threaded through
the module-level helpers (``_draw_params``, ``_sample_info_seeking``,
``_sample_step``, ``_validate_step``, ``_record_step_fail``) that
``refine_sketch`` wires into ``run_backtracking_refinement``.
"""
import dataclasses
import logging
from typing import Any, Callable, Collection, Dict, Iterator, List, Optional, \
    Set, Tuple, cast

import numpy as np

from predicators import utils
from predicators.agent_sdk.plan_execution import _fmt_state_features, \
    validate_plan_forward
from predicators.agent_sdk.sketch_parsing import _fmt_params
from predicators.agent_sdk.sketch_types import SketchStep
from predicators.option_model import _OptionModelBase
from predicators.planning import run_backtracking_refinement
from predicators.structs import GroundAtom, ParameterizedOption, \
    ParameterizedSampler, Predicate, State, Task, _Option

# Signature of an info-gain scorer: given a candidate post-state and the
# atoms whose truth the step is meant to establish, return a scalar where
# larger means more informative about the learned model (e.g. ensemble
# disagreement on those atoms). Used to turn refinement from
# feasibility-seeking into information-seeking.
InfoScorer = Callable[[State, Collection[GroundAtom]], float]


@dataclasses.dataclass
class _FeasiblePool:
    """Ranked stock of feasible candidates at one search node.

    A search node is a step under a fixed prefix of upstream choices —
    equivalently one attempt cycle of ``run_backtracking_refinement``
    (the step's try counter and its pre-state both change only when the
    step exhausts and an upstream step re-chooses). ``pre_state`` is the
    exact ``State`` object the pool was drawn from; holding the
    reference keeps the object alive, so ``is``-identity in
    ``_sample_info_seeking`` detects precisely when an upstream re-
    choice rewrote ``traj[idx]`` (new node ⇒ stale stock, fresh budget).
    ``spent`` counts pool rollouts charged against the node's budget;
    ``ranked`` holds the not-yet-proposed feasible candidates as
    ``(info_score, option)``, most informative first.
    """
    pre_state: State
    spent: int
    ranked: List[Tuple[float, _Option]]


@dataclasses.dataclass
class DeepestFailure:
    """Deepest validation failure seen during backtracking refinement.

    Captured at the moment of failure so it is a consistent record: the
    grounded option carries the exact params that produced the failing
    rollout, ``fail_reason`` is the validation message (missing subgoal
    atoms, or an unreached task goal at the final step), and
    ``post_state`` is that rollout's post-state (``None`` if it could
    not be stashed). Surfaced by ``refine_and_validate_report`` so a
    failed search returns the near-miss it got furthest with, not just
    the stuck step's name.
    """
    step_idx: int
    option: _Option
    fail_reason: str
    post_state: Optional[State] = None


@dataclasses.dataclass
class RefineOutcome:
    """Result of one ``refine_sketch`` search.

    Replaces the former ``step_samples_cumulative`` /
    ``termination_reason`` / ``elapsed_holder`` out-holder parameters.
    Iterating an outcome yields the legacy ``(plan, success,
    total_samples)`` triple, so three-way unpacks keep working.
    """
    # Refined grounded options: the full plan on success, the longest
    # refined prefix on failure (see ``refine_sketch``).
    plan: List[_Option]
    success: bool
    # Attempts across the whole search (info-seeking pool rollouts are
    # counted separately in total_pool_rollouts).
    total_samples: int
    # Cumulative attempts per step across backtracks.
    step_samples_cumulative: List[int]
    # "success" / "timeout" / "exhausted"; "" when the search never ran
    # (empty sketch).
    termination_reason: str
    # Wall-clock seconds spent in the backtracking loop.
    elapsed: float
    # The deepest validation near-miss seen during the search, if any.
    deepest_failure: Optional[DeepestFailure]
    # Model rollouts spent pooling info-seeking candidates.
    total_pool_rollouts: int = 0

    def __iter__(self) -> Iterator[Any]:
        """Support the legacy ``plan, success, total = refine_sketch(...)``
        unpack."""
        return iter((self.plan, self.success, self.total_samples))


def sample_params(option: ParameterizedOption,
                  rng: np.random.Generator) -> np.ndarray:
    """Sample continuous parameters uniformly from the option's box."""
    if option.params_space.shape[0] == 0:
        return np.array([], dtype=np.float32)
    low = option.params_space.low
    high = option.params_space.high
    return rng.uniform(low, high).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class _RefineContext:
    """Read-only inputs of one ``refine_sketch`` call.

    Bundles what the module-level search helpers would otherwise have to
    capture from an enclosing closure, so they can be plain functions
    taking ``(search, ctx, ...)``. Everything here is fixed for the
    duration of the search; all mutable search state lives in
    ``_RefinementState``. Field semantics are documented on the
    ``refine_sketch`` parameters of the same names.
    """
    task: Task
    sketch: List[SketchStep]
    option_model: _OptionModelBase
    predicates: Set[Predicate]
    max_samples_per_step: int
    check_subgoals: bool
    check_final_goal: bool
    log_state: bool
    run_id: str
    on_step_fail: Optional[Callable[[int, List[Optional[_Option]], str], None]]
    deepest_failure_holder: Optional[List[DeepestFailure]]
    info_scorer: Optional[InfoScorer]
    info_n_feasible_target: int
    parameterized_samplers: Optional[Dict[str, ParameterizedSampler]]
    solved_check: Optional[Callable[[List[State], List[Any], bool],
                                    Tuple[bool, str]]]


@dataclasses.dataclass
class _RefinementState:
    """Mutable search state of one ``refine_sketch`` call.

    One instance is built per search and threaded (with the read-only
    ``_RefineContext``) through the module-level search helpers.
    """

    # Snapshot of the deepest validation failure seen during backtracking
    # (an unmet subgoal atom, or - with check_final_goal - an unreached
    # task goal at the final step). Tracks (idx, plan_prefix_snapshot),
    # updated whenever _record_step_fail sees such a failure at a strictly
    # deeper index than before. The snapshot is taken at the moment of
    # failure, so it is a *consistent* trajectory: run_backtracking_refinement
    # has already written plan[idx] for that attempt and the prefix
    # plan[:idx+1] reflects the exact grounded options that led to it.
    # Consumed two ways: truncate_on_subgoal_fail (explorer mode) returns
    # the prefix, and deepest_failure reports the failing step's
    # near-miss params/state to the caller on any failed search.
    deepest_fail_idx: int = -1
    # Within one step, failures rank by how far the candidate got before
    # failing: an unmet subgoal (0) < an unreached final goal (1) < a
    # goal-reaching rollout the evaluator scored as a non-solve (2). A
    # same-step deeper-stage failure supersedes the record, so e.g. an
    # early subgoal miss cannot mask the far more informative
    # scored-non-solve near-miss on a single-step sketch.
    deepest_fail_stage: int = -1
    deepest_fail_prefix: List[Optional[_Option]] = dataclasses.field(
        default_factory=list)
    # The DeepestFailure record matching (deepest_fail_idx,
    # deepest_fail_stage): params, reason, post-state of the deepest
    # validation failure, mirrored into ctx.deepest_failure_holder (when
    # given) at the moment of failure.
    deepest_failure: Optional[DeepestFailure] = None

    # Options whose synthesized sampler already misbehaved once - so the
    # per-draw fallback warning fires at most once per option, not on every
    # one of the (potentially thousands of) draws during backtracking.
    sampler_warned: Set[str] = dataclasses.field(default_factory=set)

    # Step indices whose LLM-proposed initial_params have already been used --
    # tried directly on the plain path, or seeded into the info-seeking pool.
    # One-shot per step: on later attempts (resample after a failed subgoal
    # check, or re-descent after an upstream backtrack) the guess is not
    # re-proposed/re-seeded and selection falls to the ground-sampler/
    # parameterized-sampler/uniform/info-seeking path.
    llm_params_tried: Set[int] = dataclasses.field(default_factory=set)

    # Node-scoped pools for info-seeking steps: step_pools[idx] holds
    # the ranked feasible stock and rollout spend for the step's current
    # search node (see _FeasiblePool for the node-identity mechanism).
    # total_pool_rollouts accumulates across the whole search for the
    # completion log, since run_backtracking_refinement's total_samples
    # only counts attempts.
    step_pools: List[Optional[_FeasiblePool]] = dataclasses.field(
        default_factory=list)
    total_pool_rollouts: int = 0

    # Post-state of the most recent validation failure, stashed by
    # _validate_step for _record_step_fail (which planning.py calls right
    # after, in the same iteration, with the same idx - the only place the
    # failing rollout's post-state is visible is the validate callback).
    last_fail_post: Optional[Tuple[int, State]] = None

    # Per-step low-level rollout stash for the solved_check gate: entry
    # idx holds (states_after_first, per_action_labels, coarse) from the
    # step's most recent execution. A prefix step's stash stays valid
    # until the step re-executes (backtracking re-runs every step below
    # the backtrack point, overwriting stale deeper entries), so at
    # final-step acceptance entries 0..n-1 describe exactly the current
    # search path's rollout.
    step_trajs: List[Optional[Tuple[List[State], List[Any],
                                    bool]]] = dataclasses.field(
                                        default_factory=list)

    # One-element out-holders in the exact list shape planning.py's
    # run_backtracking_refinement fills (its API dictates mutable list
    # containers); read into the returned outcome after the search.
    step_samples_cumulative: List[int] = dataclasses.field(
        default_factory=list)
    termination_reason: List[str] = dataclasses.field(default_factory=list)
    elapsed: List[float] = dataclasses.field(default_factory=list)


def _draw_params(search: _RefinementState, ctx: _RefineContext,
                 step: SketchStep, state: State,
                 rng_: np.random.Generator) -> np.ndarray:
    """Draw continuous params for a step's option.

    Precedence, most specific first: the step's ground sampler (a ``~``
    annotation compiled into a ``GroundSampler``: uniform window or
    named code fn), then the option's learned parameterized sampler
    (keyed by option name), then uniform ``sample_params`` - the
    fallback also on a sampler error or wrong-shaped return (a
    misbehaving ground fn falls all the way to uniform, not to the
    parameterized sampler, mirroring the parameterized fallback).
    """
    if step.ground_sampler is not None:
        drawn = step.ground_sampler.draw(state, rng_, step.option.params_space,
                                         step.objects, step.subgoal_atoms
                                         or set())
        if drawn is not None:
            return drawn
        return sample_params(step.option, rng_)
    sampler = (ctx.parameterized_samplers.get(step.option.name)
               if ctx.parameterized_samplers else None)
    if sampler is not None:
        box = step.option.params_space
        expected = box.shape[0]
        try:
            raw = sampler(state, step.subgoal_atoms or set(), rng_,
                          list(step.objects))
            params = np.asarray(raw, dtype=np.float32).reshape(-1)
            if params.shape == (expected, ):
                return np.clip(params, box.low, box.high)
            reason = (f"returned shape {params.shape}, "
                      f"expected ({expected},)")
        except Exception as e:  # pylint: disable=broad-except
            reason = f"raised {type(e).__name__}: {e}"
        if step.option.name not in search.sampler_warned:
            search.sampler_warned.add(step.option.name)
            logging.warning(
                "[%s] synthesized sampler for %s %s; falling back to "
                "uniform sampling for this option.", ctx.run_id,
                step.option.name, reason)
    return sample_params(step.option, rng_)


def _ground(step: SketchStep, params: np.ndarray) -> _Option:
    """Ground a step's option with the given params.

    Wait steps inject ``wait_target_atoms`` / ``wait_target_neg_atoms``
    from the sketch's subgoal annotations into ``grounded.memory`` so
    that ``WaitOption`` terminates on the intended atom change rather
    than the first incidental one.
    """
    grounded = step.option.ground(list(step.objects), params)
    if grounded.name == "Wait":
        if step.subgoal_atoms is not None:
            grounded.memory["wait_target_atoms"] = step.subgoal_atoms
        if step.subgoal_neg_atoms is not None:
            grounded.memory["wait_target_neg_atoms"] = \
                step.subgoal_neg_atoms
    return grounded


def _info_seeking_applies(ctx: _RefineContext, step: SketchStep) -> bool:
    """Whether info-seeking pooled selection owns this step's params."""
    # Pooled selection only helps when there are continuous params to
    # choose among AND subgoal atoms whose truth the ensemble can
    # disagree about. Parameter-free steps (e.g. Wait) and unannotated
    # steps fall through to the plain single-sample path unchanged.
    return (ctx.info_scorer is not None and ctx.info_n_feasible_target > 1
            and step.option.params_space.shape[0] > 0
            and step.subgoal_atoms is not None)


def _is_deterministic(ctx: _RefineContext, step: SketchStep) -> bool:
    """Whether the step's sampler flags itself as returning constant params."""
    # A sampler may flag itself as returning constant params (ignoring
    # state/rng); re-drawing it yields the identical option, so its step
    # gets a single attempt -- backtracking then skips straight past it
    # instead of wasting the full budget re-descending through it.
    if step.ground_sampler is not None:
        # A ground-sampler step bypasses the parameterized sampler,
        # so a deterministic sampler flag must not collapse it to
        # one attempt. An all-zero window pins every draw to the
        # center, which IS deterministic - one attempt suffices.
        return step.ground_sampler.deterministic
    sampler = (ctx.parameterized_samplers.get(step.option.name)
               if ctx.parameterized_samplers else None)
    return bool(getattr(sampler, "deterministic", False))


def _sample_info_seeking(search: _RefinementState, ctx: _RefineContext,
                         step: SketchStep, state: State,
                         rng_: np.random.Generator, idx: int) -> _Option:
    """Propose the most informative not-yet-tried feasible candidate for the
    step's current search node.

    When the step carries LLM-proposed ``initial_params`` (and they
    have not been used yet), they are evaluated as the FIRST candidate
    of the node's pool, so the max-disagreement selection chooses among
    {LLM guess} ∪ sampled draws rather than the guess short-circuiting
    the probe.

    The first attempt at a node draws candidates — each rolled
    forward through the same option_model the backtracking loop uses
    — until ``info_n_feasible_target`` feasible ones are pooled or
    the node's rollout budget (``max_samples_per_step``) is spent,
    then proposes the max-disagreement one and banks the rest as a
    ranked stock. Later attempts at the same node (the loop retries
    after a final-goal miss or after downstream steps collapse back
    onto this one) pop the next-best from the stock with NO new
    rollouts: the candidates were already rolled out and
    subgoal-checked, and the pre-state is fixed within a node, so
    for a deterministic learned model they stay valid. (With a
    stochastic model a popped candidate may still fail the loop's
    re-execution — it just consumes an attempt, like any failure.)

    Candidates that aren't initiable, produce no actions, or fail
    to establish the subgoal consume budget but never enter the
    stock. If a draw round finds nothing feasible, the first sample
    is returned so the loop records the validation failure
    (explorer-mode truncation relies on it); an attempt arriving
    with both stock and budget exhausted gets a 1-draw minimum so
    it can fail fast until the attempt cap
    (= ``info_n_feasible_target``) exhausts the step.

    Node identity: ``traj[idx]`` is rewritten only when an upstream
    step re-executes, which can only happen after this step
    exhausts, so comparing the pre-state *object* (``is``) flips
    exactly at node boundaries — stale stock is dropped and the
    budget refreshed.
    """
    assert ctx.info_scorer is not None and step.subgoal_atoms is not None
    # Narrowed locals so the nested _consider closure (below) keeps the
    # non-None types mypy can't carry across the function boundary.
    scorer = ctx.info_scorer
    subgoal_atoms = step.subgoal_atoms
    objs = ", ".join(o.name for o in step.objects)
    pool = search.step_pools[idx]
    if pool is None or pool.pre_state is not state:
        pool = _FeasiblePool(pre_state=state, spent=0, ranked=[])
        search.step_pools[idx] = pool
    if pool.ranked:
        score, grounded = pool.ranked.pop(0)
        logging.info(
            "[%s] info-seeking %s(%s): proposing next-ranked stock "
            "candidate params %s (disagreement %.4f, %d left in "
            "stock) — no new rollouts.", ctx.run_id, step.option.name, objs,
            _fmt_params(grounded), score, len(pool.ranked))
        return grounded
    # Stock empty: first attempt at this node, or every pooled
    # candidate has been proposed. Draw from the node's remaining
    # budget (>=1 so the attempt can still fail fast when spent).
    draw_cap = max(ctx.max_samples_per_step - pool.spent, 1)
    best_score = -float("inf")
    best_nxt: Optional[State] = None
    scored: List[Tuple[float, _Option]] = []
    # Score of the first feasible candidate — what plain (greedy
    # first-feasible) backtracking would have accepted; logged as the
    # baseline so a run shows what boundary-probing bought. With LLM
    # params on, the seeded guess (below) is that first candidate.
    first_feasible_score: Optional[float] = None
    first_candidate: Optional[_Option] = None
    n_draws = 0

    def _consider(grounded: _Option) -> None:
        # Roll one grounded candidate forward and, when it establishes
        # the subgoal, fold it into the scored pool and running argmax.
        # Shared by the LLM-guess seed and the sampler/uniform draws so
        # the two stay in lockstep.
        nonlocal best_score, best_nxt, first_feasible_score
        nonlocal first_candidate
        if first_candidate is None:
            first_candidate = grounded
        if not grounded.initiable(state):
            return
        try:
            nxt, num_actions = \
                ctx.option_model.get_next_state_and_num_actions(
                    state, grounded)
        except Exception:  # pylint: disable=broad-except
            # Scoring rollout is best-effort; a model failure on this
            # candidate just removes it from contention.
            return
        if num_actions == 0:
            return
        post_atoms = utils.abstract(nxt, ctx.predicates)
        if not subgoal_atoms.issubset(post_atoms):
            return  # infeasible: subgoal not established
        score = scorer(nxt, subgoal_atoms)
        scored.append((score, grounded))
        if first_feasible_score is None:
            first_feasible_score = score
        if score > best_score:
            best_score = score
            best_nxt = nxt

    # Seed the LLM-proposed params (once per step) as the FIRST pool
    # candidate, so the disagreement argmax chooses among
    # {LLM guess} ∪ sampled draws instead of the guess short-circuiting
    # the probe. Clipping mirrors the plain branch; arity is already
    # validated by the option-plan parser.
    if step.initial_params is not None and \
            idx not in search.llm_params_tried:
        search.llm_params_tried.add(idx)
        box = step.option.params_space
        llm_grounded = _ground(
            step,
            np.clip(np.asarray(step.initial_params, dtype=np.float32), box.low,
                    box.high).astype(np.float32))
        n_draws += 1
        n_pooled_before = len(scored)
        _consider(llm_grounded)
        logging.info(
            "[%s] info-seeking %s(%s): seeded LLM-proposed params %s "
            "(%s) into the candidate pool.", ctx.run_id, step.option.name,
            objs, _fmt_params(llm_grounded), "feasible"
            if len(scored) > n_pooled_before else "infeasible — not pooled")

    while len(scored) < ctx.info_n_feasible_target and n_draws < draw_cap:
        grounded = _ground(step, _draw_params(search, ctx, step, state, rng_))
        n_draws += 1
        _consider(grounded)
    pool.spent += n_draws
    search.total_pool_rollouts += n_draws
    # Log every pick at INFO (not gated on log_state) — active-learning
    # visibility into where boundary-probing engaged and what it found.
    # All-zero scores ⇒ ensemble agrees here (uninformative).
    if not scored:
        assert first_candidate is not None
        logging.info(
            "[%s] info-seeking %s(%s): 0 feasible candidates after "
            "%d draws (%d/%d node budget spent; target %d); falling "
            "back to first sample (no boundary probe).", ctx.run_id,
            step.option.name, objs, n_draws, pool.spent,
            ctx.max_samples_per_step, ctx.info_n_feasible_target)
        return first_candidate
    # Stable sort: ties keep draw order, so among equally informative
    # candidates the first-drawn (what plain backtracking would have
    # taken) is proposed first.
    scored.sort(key=lambda t: t[0], reverse=True)
    _, best = scored[0]
    pool.ranked = scored[1:]
    # Per-atom disagreement of the chosen candidate, so the log shows
    # which subgoal atoms carry the uncertainty rather than only the
    # aggregate (mean) the selection maximized.
    assert best_nxt is not None
    assert first_feasible_score is not None
    per_atom = ", ".join(f"{a}={scorer(best_nxt, {a}):.4f}"
                         for a in sorted(subgoal_atoms, key=str))
    logging.info(
        "[%s] info-seeking %s(%s): picked params %s with disagreement "
        "%.4f vs first-feasible %.4f (%d/%d feasible in %d draws, "
        "%d banked, %d/%d node budget; per-atom: %s).", ctx.run_id,
        step.option.name, objs, _fmt_params(best), best_score,
        first_feasible_score, len(scored), ctx.info_n_feasible_target, n_draws,
        len(pool.ranked), pool.spent, ctx.max_samples_per_step, per_atom)
    return best


def _sample_step(search: _RefinementState, ctx: _RefineContext, idx: int,
                 state: State, rng_: np.random.Generator) -> _Option:
    """Propose a grounded option for step ``idx`` (the search's sample_fn)."""
    step = ctx.sketch[idx]
    if ctx.log_state:
        step_name = (f"{step.option.name}"
                     f"({', '.join(o.name for o in step.objects)})")
        logging.debug(f"[{ctx.run_id}]  State before {step_name}:\n"
                      f"{state.pretty_str()}")
    # Info-seeking (when on) owns param selection for eligible steps and
    # folds any LLM-proposed params into its scored candidate pool (see
    # _sample_info_seeking), so the disagreement argmax chooses among
    # {LLM guess} ∪ sampled draws instead of the guess pre-empting it.
    if _info_seeking_applies(ctx, step):
        return _sample_info_seeking(search, ctx, step, state, rng_, idx)
    # Plain path: on the first arrival at this step, try the LLM-proposed
    # params (if any) before any sampling. Clipping avoids ground()'s
    # out-of-box ValueError; arity is already validated by the parser.
    if step.initial_params is not None and \
            idx not in search.llm_params_tried:
        search.llm_params_tried.add(idx)
        box = step.option.params_space
        params = np.clip(np.asarray(step.initial_params, dtype=np.float32),
                         box.low, box.high).astype(np.float32)
        logging.debug("[%s] step %d %s: trying LLM-proposed params %s",
                      ctx.run_id, idx, step.option.name, params.tolist())
        return _ground(step, params)
    return _ground(step, _draw_params(search, ctx, step, state, rng_))


def _validate_step(search: _RefinementState, ctx: _RefineContext, idx: int,
                   _pre_state: State, option: _Option, post_state: State,
                   _num_actions: int) -> Tuple[bool, str]:
    """Validate one executed step (the search's validate_fn).

    Checks the step's subgoal atoms (when enabled), the task goal at the
    final step (when enabled), and - when a ``solved_check`` gate is
    threaded - runs it on the accumulated per-step rollouts of a
    goal-reaching final-step candidate. Stashes the post-state of any
    validation failure in ``search.last_fail_post`` for
    ``_record_step_fail`` (which planning.py calls right after, in the
    same iteration, with the same idx - this is the only place the
    failing rollout's post-state is visible).
    """
    n = len(ctx.sketch)
    step = ctx.sketch[idx]
    if ctx.check_subgoals and step.subgoal_atoms is not None:
        current_atoms = utils.abstract(post_state, ctx.predicates)
        if not step.subgoal_atoms.issubset(current_atoms):
            missing = step.subgoal_atoms - current_atoms
            search.last_fail_post = (idx, post_state)
            return False, (f"subgoal missing: "
                           f"{{{', '.join(str(a) for a in missing)}}}")
    if ctx.check_final_goal and idx == n - 1:
        if not ctx.task.goal_holds(post_state):
            search.last_fail_post = (idx, post_state)
            return False, "goal not reached"
    if ctx.solved_check is not None:
        # Stash the step's low-level rollout AFTER the atom checks:
        # most candidates fail those on the spot, and stashing first
        # would copy an O(rollout-length) state list (a Wait step is
        # up to 1000 states) per doomed candidate and pin it for the
        # rest of the search. The freshness check guards against an
        # option model that exposes ``last_trajectory`` without
        # refreshing it for THIS rollout (a stale trajectory would be
        # scored under the current option's label - a silent
        # franken-trajectory); staleness falls back to the coarse
        # option-boundary path.
        label = (option.name, tuple(o.name for o in option.objects),
                 tuple(float(p) for p in option.params))
        step_traj = getattr(ctx.option_model, "last_trajectory", None)
        if (step_traj is not None and len(step_traj.states) >= 2
                and step_traj.states[-1] is post_state):
            search.step_trajs[idx] = (list(step_traj.states[1:]),
                                      [label] * len(step_traj.actions), False)
        else:
            search.step_trajs[idx] = ([post_state], [label], True)
    if ctx.solved_check is not None and idx == n - 1 and \
            ctx.task.goal_holds(post_state):
        eval_states: List[State] = [ctx.task.init]
        eval_labels: List[Any] = []
        coarse = False
        for stash in search.step_trajs:
            # Every prefix step was validated (and therefore stashed)
            # on the current search path before the final step ran.
            assert stash is not None
            s_states, s_labels, s_coarse = stash
            eval_states.extend(s_states)
            eval_labels.extend(s_labels)
            coarse = coarse or s_coarse
        ok, why = ctx.solved_check(eval_states, eval_labels, coarse)
        if not ok:
            search.last_fail_post = (idx, post_state)
            return False, f"scored non-solve: {why}"
    return True, ""


def _record_step_fail(search: _RefinementState, ctx: _RefineContext, idx: int,
                      cur_plan: List[Optional[_Option]],
                      fail_reason: str) -> None:
    """Record the deepest validation failure (the search's on_step_fail)."""
    # run_backtracking_refinement calls this BEFORE clearing
    # plan[idx] (planning.py lines 592-599), so cur_plan[0..idx] is
    # still populated with the grounded options that produced this
    # exact failure trajectory. Record the deepest validation failure
    # (unmet subgoal, or unreached task goal at the final step) seen so
    # far along with a consistent snapshot of the prefix. A final-goal
    # failure is at idx==n-1, so its snapshot is the full plan — the
    # experiment we want to execute in reality. The record is kept
    # unconditionally (deepest-failure consumers read it on any failed
    # search); only the truncation RETURN in refine_sketch stays gated on
    # truncate_on_subgoal_fail. Non-validation failures (not initiable,
    # 0 actions, model errors) never update it.
    if fail_reason.startswith("scored non-solve"):
        stage: Optional[int] = 2
    elif fail_reason == "goal not reached":
        stage = 1
    elif fail_reason.startswith("subgoal missing"):
        stage = 0
    else:
        stage = None
    if stage is not None and (idx, stage) > (search.deepest_fail_idx,
                                             search.deepest_fail_stage):
        search.deepest_fail_idx = idx
        search.deepest_fail_stage = stage
        search.deepest_fail_prefix = list(cur_plan[:idx + 1])
        stash = search.last_fail_post
        post = stash[1] if stash and stash[0] == idx else None
        opt = cur_plan[idx]
        assert opt is not None
        search.deepest_failure = DeepestFailure(idx, opt, fail_reason, post)
        if ctx.deepest_failure_holder is not None:
            ctx.deepest_failure_holder.clear()
            ctx.deepest_failure_holder.append(search.deepest_failure)
    if ctx.on_step_fail is not None:
        ctx.on_step_fail(idx, cur_plan, fail_reason)


def refine_sketch(
    task: Task,
    sketch: List[SketchStep],
    option_model: _OptionModelBase,
    *,
    predicates: Set[Predicate],
    timeout: float,
    rng: np.random.Generator,
    max_samples_per_step: int,
    check_subgoals: bool,
    check_final_goal: bool = True,
    truncate_on_subgoal_fail: bool = False,
    log_state: bool = False,
    run_id: str = "bilevel",
    on_step_fail: Optional[Callable[[int, List[Optional[_Option]], str],
                                    None]] = None,
    deepest_failure_holder: Optional[List[DeepestFailure]] = None,
    info_scorer: Optional[InfoScorer] = None,
    info_n_feasible_target: int = 1,
    parameterized_samplers: Optional[Dict[str, ParameterizedSampler]] = None,
    solved_check: Optional[Callable[[List[State], List[Any], bool],
                                    Tuple[bool, str]]] = None,
) -> RefineOutcome:
    """Backtracking search over continuous parameters for a plan sketch.

    Returns a ``RefineOutcome`` (iterating it yields the legacy
    ``(refined_plan, success, total_samples)`` triple). On success the
    plan is fully refined; on failure it is the longest prefix of
    refined options (``None`` entries dropped). The outcome also carries
    what used to be out-holder parameters: per-step cumulative sample
    counts, the termination reason, the elapsed wall-clock time, and the
    deepest validation near-miss.

    ``solved_check`` is a caller-threaded task-evaluator gate (this
    module stays free of evaluator/CFG coupling): called only for a
    final-step candidate whose goal atoms already hold, with the
    rollout's accumulated per-low-level-step states, per-action
    ``(option, objects, params)`` labels, and a ``coarse`` flag (True
    when any step's low-level trajectory was unavailable and only its
    option-boundary state could be used). Returning ``(False, reason)``
    fails the candidate as ``"scored non-solve: <reason>"`` - the search
    keeps going instead of converging on parameters the evaluator would
    reject, and the deepest-failure near-miss records the exact
    goal-reaching-but-uncertified params. The gate runs only on
    goal-reaching candidates, so its (potentially expensive) certificate
    cost is bounded by how often the search actually reaches the goal.

    ``deepest_failure_holder`` is a transitional out-holder kept for
    callers that still construct one: when given, the single deepest
    validation failure seen during the search is appended as a
    ``DeepestFailure`` (params, reason, post-state), regardless of
    ``truncate_on_subgoal_fail`` - the same record returned as
    ``RefineOutcome.deepest_failure``, which new callers should read
    instead. Callers use it to report the search's best near-miss on
    failure.

    ``check_subgoals`` gates per-step subgoal-atom validation.
    ``check_final_goal`` gates the task-goal check on the final step.
    ``truncate_on_subgoal_fail`` (explorer mode) lets backtracking run
    to exhaustion with subgoal checks enabled, then — if the search
    fails — returns the consistent plan prefix captured at the deepest
    validation failure seen during backtracking (inclusive of the
    failing step). "Validation failure" covers both an unmet subgoal
    atom and, when ``check_final_goal`` is on, an unreached task goal at
    the final step; the latter captures the *whole* plan as the
    experiment (run it in reality and observe — a goal the mental model
    predicts won't hold is exactly the disagreement worth collecting).
    Use this to build *experiment* plans that probe a mental-model
    disagreement: upstream steps get their standard backtracking
    retries, but once the deepest unresolvable step is identified,
    subsequent sketch steps are dropped (they would be built on a false
    mental-model state).

    ``max_samples_per_step`` is a per-step rollout budget per *search
    node* (the step under a fixed prefix of upstream choices;
    backtracking past the step and re-descending with a new upstream
    choice starts a new node). Plain steps spend it the classic way, one
    sampled rollout per attempt. Info-seeking steps spend it pooling
    candidates at the node, and the pooled feasible candidates double as
    a ranked retry stock — the budget is spent once, never multiplied.

    With ``info_scorer`` set and ``info_n_feasible_target > 1``,
    parameter sampling at subgoal-annotated steps with continuous
    parameters becomes information-seeking: candidates are drawn until
    ``info_n_feasible_target`` feasible ones are pooled (bounded by the
    node's rollout budget) and proposed most-informative-first, one per
    attempt, with no re-drawing while the stock lasts (a retry after a
    downstream collapse or a final-goal miss pops the next-best for
    free). The step's attempt cap equals ``info_n_feasible_target``, so
    it exhausts exactly when every pooled candidate has been tried. See
    ``_sample_info_seeking``.

    Wait steps inject ``wait_target_atoms`` / ``wait_target_neg_atoms``
    from the sketch's subgoal annotations into ``grounded.memory`` so
    that ``WaitOption`` terminates on the intended atom change rather
    than the first incidental one.

    ``parameterized_samplers`` maps an option name to a parameterized
    (per-skill) sampler ``(state, subgoal_atoms, rng, objects) ->
    params`` (the NSRTSampler signature, with the step subgoal in the
    atoms slot), used on both plain and info-seeking draws to aim that
    option's parameters at the subgoal instead of drawing uniformly.
    The return is clipped to the option's box; a missing or misbehaving
    sampler falls back to uniform sampling. A step whose sketch line
    carries a ``~ [widths]`` region annotation bypasses the sampler
    entirely: after the one-shot center try, its draws come from the
    step's ``GroundSampler``, the most specific prior winning - ground
    sampler, then parameterized sampler, then uniform.
    """
    if not sketch:
        return RefineOutcome(plan=[],
                             success=False,
                             total_samples=0,
                             step_samples_cumulative=[],
                             termination_reason="",
                             elapsed=0.0,
                             deepest_failure=None)

    n = len(sketch)
    ctx = _RefineContext(task=task,
                         sketch=sketch,
                         option_model=option_model,
                         predicates=predicates,
                         max_samples_per_step=max_samples_per_step,
                         check_subgoals=check_subgoals,
                         check_final_goal=check_final_goal,
                         log_state=log_state,
                         run_id=run_id,
                         on_step_fail=on_step_fail,
                         deepest_failure_holder=deepest_failure_holder,
                         info_scorer=info_scorer,
                         info_n_feasible_target=info_n_feasible_target,
                         parameterized_samplers=parameterized_samplers,
                         solved_check=solved_check)
    search = _RefinementState(step_pools=[None] * n,
                              step_trajs=[None] * n,
                              step_samples_cumulative=[0] * n)

    # Per-step attempt caps. Plain steps spend their whole budget as
    # attempts: one sampled rollout per attempt, max_samples_per_step
    # attempts (unchanged semantics). Info-seeking steps get exactly
    # info_n_feasible_target attempts: the pooled feasible candidates
    # double as the node's retry stock, one proposed per attempt, so the
    # step exhausts precisely when every pooled candidate has been tried
    # (with 1-draw fillers for attempts left over when the pool came up
    # short of the target).
    max_tries = []
    for _step in sketch:
        if _step.option.params_space.shape[0] == 0:
            max_tries.append(1)
        elif _is_deterministic(ctx, _step):
            max_tries.append(1)
        elif _info_seeking_applies(ctx, _step):
            max_tries.append(info_n_feasible_target)
        else:
            max_tries.append(max_samples_per_step)

    def sample_fn(idx: int, state: State,
                  rng_: np.random.Generator) -> _Option:
        return _sample_step(search, ctx, idx, state, rng_)

    def validate_fn(idx: int, pre_state: State, option: _Option,
                    post_state: State, num_actions: int) -> Tuple[bool, str]:
        return _validate_step(search, ctx, idx, pre_state, option, post_state,
                              num_actions)

    def wrapped_on_step_fail(idx: int, cur_plan: List[Optional[_Option]],
                             fail_reason: str) -> None:
        _record_step_fail(search, ctx, idx, cur_plan, fail_reason)

    # One-line eligibility summary: if info-seeking is requested but no
    # step qualifies (a step needs continuous params + a subgoal
    # annotation), the per-step probe silently never fires — say so.
    if info_scorer is not None and info_n_feasible_target > 1:
        eligible = [
            i for i, s in enumerate(sketch) if _info_seeking_applies(ctx, s)
        ]
        logging.info(
            "[%s] info-seeking eligible steps: %s of %d (target %d, "
            "node budget %d).", run_id, eligible or "none", n,
            info_n_feasible_target, max_samples_per_step)

    plan, success, total_samples = run_backtracking_refinement(
        init_state=task.init,
        option_model=option_model,
        n_steps=n,
        max_tries=max_tries,
        sample_fn=sample_fn,
        validate_fn=validate_fn,
        rng=rng,
        timeout=timeout,
        on_step_fail=wrapped_on_step_fail,
        step_samples_cumulative=search.step_samples_cumulative,
        termination_reason=search.termination_reason,
        elapsed_holder=search.elapsed,
    )

    # total_samples counts attempts only; pool rollouts are the real
    # model-call cost of info-seeking steps, so surface them alongside.
    pool_note = (f" (+{search.total_pool_rollouts} info-seeking pool rollouts)"
                 if search.total_pool_rollouts else "")
    logging.info(
        f"[{run_id}] Refinement {'succeeded' if success else 'failed'}: "
        f"{total_samples} samples for {n} steps{pool_note}.")

    def _outcome(refined_plan: List[_Option], ok: bool) -> RefineOutcome:
        return RefineOutcome(
            plan=refined_plan,
            success=ok,
            total_samples=total_samples,
            step_samples_cumulative=search.step_samples_cumulative,
            termination_reason=(search.termination_reason[0]
                                if search.termination_reason else ""),
            elapsed=search.elapsed[0] if search.elapsed else 0.0,
            deepest_failure=search.deepest_failure,
            total_pool_rollouts=search.total_pool_rollouts)

    if (truncate_on_subgoal_fail and not success
            and search.deepest_fail_idx >= 0):
        snapshot = search.deepest_fail_prefix
        refined = [p for p in snapshot if p is not None]
        fail_note = ""
        if search.deepest_failure is not None:
            fail_note = (f" Deepest failure: "
                         f"{search.deepest_failure.option.simple_str()} -> "
                         f"{search.deepest_failure.fail_reason}.")
        logging.info(f"[{run_id}] Truncating at deepest validation failure "
                     f"(step {search.deepest_fail_idx}): "
                     f"{len(refined)}/{n} steps in experiment plan."
                     f"{fail_note}")
        return _outcome(cast(List[_Option], refined), False)

    refined = [p for p in plan if p is not None]
    return _outcome(cast(List[_Option], refined), success)


def resolve_refine_timeout(
    timeout: Optional[float],
    n_steps: int,
    *,
    per_step: float,
    minimum: float,
) -> Tuple[float, str]:
    """Resolve a refinement timeout, auto-scaling by sketch length.

    When ``timeout`` is None it auto-scales as
    ``max(minimum, per_step * n_steps)`` so longer sketches get more
    budget. Returns ``(timeout_seconds, source)`` where ``source`` is
    ``"auto"`` or ``"explicit"``. Config defaults are passed in (not read
    from ``CFG``) to keep this module settings-free.
    """
    if timeout is None:
        return float(max(minimum, per_step * n_steps)), "auto"
    return float(timeout), "explicit"


def refine_and_validate_report(
    task: Task,
    sketch: List[SketchStep],
    option_model: _OptionModelBase,
    *,
    predicates: Set[Predicate],
    timeout: float,
    rng: np.random.Generator,
    max_samples_per_step: int,
    check_subgoals: bool,
    log_state: bool = False,
    parameterized_samplers: Optional[Dict[str, ParameterizedSampler]] = None,
    run_id: str = "refine",
    timeout_source: str = "explicit",
    extra_summary_lines: Optional[List[str]] = None,
    solved_check: Optional[Callable[[List[State], List[Any], bool],
                                    Tuple[bool, str]]] = None,
) -> Tuple[bool, str, List[_Option]]:
    """Refine a sketch, forward-validate on success, return a report.

    ``solved_check`` is threaded to ``refine_sketch``: a task-evaluator
    gate on final-step goal-reaching candidates (see there), so the
    search rejects goal-atom-reaching-but-uncertified parameters during
    refinement instead of reporting a SUCCESS the evaluator would score
    as a non-solve.

    Runs ``refine_sketch`` (backtracking search over continuous params)
    and, when refinement succeeds, ``validate_plan_forward`` (continuous
    re-execution). Returns ``(overall_success, human_readable_report,
    plan)`` where ``overall_success`` is True only if both refinement and
    forward validation pass, and ``plan`` is the refined grounded-option
    plan (the longest refined prefix on failure). The report names the
    verdict (SUCCESS / TIMEOUT / SAMPLE_EXHAUSTED /
    FORWARD_VALIDATION_FAILED), per-step sample counts, the stuck step on
    failure, the deepest validation near-miss (the failing step's exact
    params, the missing atoms, and the post-state of that rollout's
    step objects) when refinement fails, and the forward-validation
    outcome.

    ``extra_summary_lines`` are appended verbatim after the time line
    (e.g. a caller-specific ``Post-fit SSE`` line). Config-derived knobs
    (``timeout``, ``max_samples_per_step``, ``check_subgoals``,
    ``log_state``) are passed explicitly so this module stays free of
    ``CFG``; callers read them from settings.
    """
    outcome = refine_sketch(
        task,
        sketch,
        option_model,
        predicates=predicates,
        timeout=timeout,
        rng=rng,
        max_samples_per_step=max_samples_per_step,
        check_subgoals=check_subgoals,
        log_state=log_state,
        run_id=run_id,
        parameterized_samplers=parameterized_samplers,
        solved_check=solved_check,
    )
    plan, success, n_samples = (outcome.plan, outcome.success,
                                outcome.total_samples)

    reason = outcome.termination_reason or ("success"
                                            if success else "exhausted")
    elapsed = outcome.elapsed
    if success:
        verdict = "SUCCESS"
    elif reason == "timeout":
        verdict = "FAILURE: TIMEOUT"
    elif reason == "exhausted":
        verdict = "FAILURE: SAMPLE_EXHAUSTED"
    else:
        verdict = "FAILURE"

    lines = [
        verdict,
        f"  Sketch: {len(sketch)} steps  Refined: {len(plan)} steps  "
        f"Samples: {n_samples} total",
        f"  Per-step samples: {outcome.step_samples_cumulative}  "
        f"(cap {max_samples_per_step}/step)",
        f"  Time: {elapsed:.1f}s used / {timeout:.1f}s allotted "
        f"(timeout source: {timeout_source})",
    ]
    if extra_summary_lines:
        lines.extend(extra_summary_lines)
    if not success and len(plan) < len(sketch):
        stuck_idx = len(plan)
        stuck = sketch[stuck_idx]
        objs = ", ".join(f"{o.name}:{o.type.name}" for o in stuck.objects)
        lines.append(f"  Stuck at step {stuck_idx}: "
                     f"{stuck.option.name}({objs})")
        if stuck.subgoal_atoms:
            atoms = ", ".join(str(a) for a in stuck.subgoal_atoms)
            lines.append(f"    subgoals: {atoms}")
    if not success and outcome.deepest_failure is not None:
        # The search's best near-miss: the exact params of the deepest
        # rollout that executed but failed validation, so the caller can
        # adjust the right step instead of restarting blind.
        df = outcome.deepest_failure
        df_objs = ", ".join(o.name for o in df.option.objects)
        df_params = ", ".join(f"{p:.4f}" for p in df.option.params)
        lines.append(f"  Deepest failure: step {df.step_idx} "
                     f"{df.option.name}({df_objs})[{df_params}] - "
                     f"{df.fail_reason}")
        if df.post_state is not None:
            feats = _fmt_state_features(df.post_state,
                                        objects=df.option.objects)
            lines.append(f"    post-state: {feats}")

    # Forward validation: re-execute the refined plan continuously (state
    # carries forward across all options). Refinement's per-step resets
    # and resampling can mask drift the real env will hit at test time.
    if success:
        try:
            fv_ok, fv_reason = validate_plan_forward(
                task,
                plan,
                option_model,
                predicates=predicates,
                sketch=sketch,
                run_id=run_id,
            )
        except Exception as e:  # pylint: disable=broad-except
            fv_ok = False
            fv_reason = f"forward validation raised: {e}"
        if fv_ok:
            lines.append("  Forward validation: SUCCESS")
        else:
            # Demote the headline verdict: refinement passed but the plan
            # does not survive continuous execution, which is what the
            # real env will see at test time.
            success = False
            lines[0] = "FAILURE: FORWARD_VALIDATION_FAILED"
            lines.append(f"  Forward validation: FAIL — {fv_reason}")
            lines.append(
                "    (Refinement resets state between options and "
                "resamples up to the per-step cap; forward validation "
                "runs the same plan once continuously. A divergence here "
                "means the refined plan does not survive continuous "
                "execution — accumulated drift, or (when the model is "
                "learned) a rule/threshold more permissive than the env's "
                "effective behavior. See the INFO log for the step-by-step "
                "divergence.)")

    return success, "\n".join(lines), plan
