"""Tests for info-seeking (draw-until-target) refinement in
``sketch_refinement``.

Verifies that, with an ``info_scorer`` supplied, ``refine_sketch`` no
longer accepts the first feasible continuous-parameter sample but
instead draws candidates until ``info_n_feasible_target`` feasible ones
are pooled and keeps the most *informative* one — while the default (no
scorer) path is unchanged.

Budget semantics: ``max_samples_per_step`` is a rollout budget per step
*search node* (the step under a fixed prefix of upstream choices; the
node changes only when the step exhausts and an upstream step
re-chooses). Plain steps spend it one rollout per attempt — classic
backtracking. Info-seeking steps spend it pooling candidates at the
node; the pooled feasible candidates double as a ranked retry stock
that backtracking walks best-first with no re-drawing, and the step's
attempt cap equals ``info_n_feasible_target`` so it exhausts exactly
when every pooled candidate has been tried.
"""

# pylint: disable=unused-import

import numpy as np
from gym.spaces import Box

from predicators import utils  # noqa: F401  (settles import order)
from predicators.agent_sdk.sketch_refinement import refine_sketch, \
    sample_params
from predicators.agent_sdk.sketch_types import SketchStep
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


def _true(_s, _m, _o, _p):
    return True


def _false(_s, _m, _o, _p):
    return False


# A 1-D option whose parameter is the post-state x-coordinate of the block.
_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_true,
    terminal=_false,
)

# A parameter-free option (max_tries=1 in refinement; never info-eligible).
_Noop = ParameterizedOption(
    "Noop",
    types=[_block_type],
    params_space=Box(low=np.zeros(0, dtype=np.float32),
                     high=np.zeros(0, dtype=np.float32)),
    policy=_noop_policy,
    initiable=_true,
    terminal=_false,
)

# Subgoal predicate that always holds, so EVERY candidate is feasible and
# the only thing distinguishing candidates is the info score.
_Reached = Predicate("Reached", [_block_type], lambda s, o: True)
_PREDICATES = {_Reached}

# Subgoal predicate that never holds (block x lives in [0, 1]).
_Unreachable = Predicate("Unreachable", [_block_type],
                         lambda s, o: s.get(o[0], "x") >= 2.0)


class _FakeOptionModel:
    """Deterministic model: Move sets block.x to its parameter value."""

    last_execution_failure = None

    def __init__(self):
        self.num_calls = 0

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step, counting the call."""
        self.num_calls += 1
        nxt = state.copy()
        if len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        return nxt, 1


def _task():
    init = State({_block: np.array([0.0], dtype=np.float32)})
    return Task(init, {GroundAtom(_Reached, [_block])})


def _sketch():
    return [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_Reached, [_block])})
    ]


def _replay_pool(seed, target, budget, feasible_fn):
    """Replay the rng stream of one draw-until-target pooling pass.

    Returns ``(feasible_pool, n_draws)`` exactly as ``refine_sketch``'s
    info-seeking sampler would compute them on a fresh visit with
    ``max_samples_per_step=budget`` (the rng is consumed only by
    ``sample_params``, one draw per candidate).
    """
    rng = np.random.default_rng(seed)
    feasible, n_draws = [], 0
    while len(feasible) < target and n_draws < budget:
        x = float(sample_params(_Move, rng)[0])
        n_draws += 1
        if feasible_fn(x):
            feasible.append(x)
    return feasible, n_draws


def _refine(seed, info_scorer, n_feasible_target, max_samples_per_step=50):
    plan, success, _ = refine_sketch(
        _task(),
        _sketch(),
        _FakeOptionModel(),
        predicates=_PREDICATES,
        timeout=10.0,
        rng=np.random.default_rng(seed),
        max_samples_per_step=max_samples_per_step,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=info_scorer,
        info_n_feasible_target=n_feasible_target,
    )
    assert success
    return float(plan[0].params[0])


def test_info_seeking_picks_max_score_among_pool():
    """The scorer rewards larger x; the chosen param is the pool maximum."""
    seed, n = 7, 8
    # Replay the exact draws the seeded rng produces. The subgoal always
    # holds, so every draw is feasible and the pool is exactly the first
    # n draws (the loop stops as soon as the target is reached).
    pool, n_draws = _replay_pool(seed, n, 50, lambda x: True)
    assert n_draws == n

    chosen = _refine(seed,
                     info_scorer=lambda s, _a: s.get(_block, "x"),
                     n_feasible_target=n)
    assert chosen == max(pool)


def test_plain_path_takes_first_sample():
    """With no scorer, the first feasible sample is accepted (unchanged)."""
    seed = 7
    rng = np.random.default_rng(seed)
    first = float(sample_params(_Move, rng)[0])

    chosen = _refine(seed, info_scorer=None, n_feasible_target=1)
    assert chosen == first


def test_info_seeking_beats_first_feasible():
    """Info-seeking's pick is at least as informative as first-feasible."""
    seed, n = 7, 8
    plain = _refine(seed, info_scorer=None, n_feasible_target=1)
    info = _refine(seed,
                   info_scorer=lambda s, _a: s.get(_block, "x"),
                   n_feasible_target=n)
    # Same seed => same first draw; the pool max can only improve.
    assert info >= plain


def test_target_one_reduces_to_first_feasible():
    """n_feasible_target=1 with a scorer still returns the single sample."""
    seed = 3
    rng = np.random.default_rng(seed)
    first = float(sample_params(_Move, rng)[0])
    chosen = _refine(seed,
                     info_scorer=lambda s, _a: -s.get(_block, "x"),
                     n_feasible_target=1)
    # _info_seeking_applies requires n_feasible_target > 1, so the plain
    # single-sample path runs and returns the first draw regardless of
    # the (here inverted) scorer.
    assert chosen == first


def test_infeasible_candidates_filtered_out():
    """Only candidates whose subgoal holds enter the pool; others ignored."""
    # Subgoal holds only for x >= 0.5; scorer prefers small x. The chosen
    # param must still satisfy the subgoal (>= 0.5), i.e. the scorer can't
    # drag the pick into the infeasible region — and it must be exactly
    # the smallest x in the replayed feasible pool.
    seed, target, budget = 11, 4, 50
    reached_hi = Predicate("ReachedHi", [_block_type],
                           lambda s, o: s.get(o[0], "x") >= 0.5)
    sketch = [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(reached_hi, [_block])})
    ]
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}),
                {GroundAtom(reached_hi, [_block])})
    pool, _ = _replay_pool(seed, target, budget, lambda x: x >= 0.5)
    assert len(pool) == target  # this seed fills the pool within budget

    plan, success, _ = refine_sketch(
        task,
        sketch,
        _FakeOptionModel(),
        predicates={reached_hi},
        timeout=10.0,
        rng=np.random.default_rng(seed),
        max_samples_per_step=budget,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=lambda s, _a: -s.get(_block, "x"),  # prefers small x
        info_n_feasible_target=target,
    )
    assert success
    chosen = float(plan[0].params[0])
    assert chosen >= 0.5
    assert chosen == min(pool)


def test_draw_until_target_pools_beyond_fixed_batch():
    """Hard subgoals keep drawing until the feasible pool is full.

    Feasibility is ~10% (x >= 0.9), so a fixed batch of ``target`` draws
    would almost surely pool 0-1 feasible candidates and collapse the
    argmax to first-feasible. The draw-until-target loop keeps drawing
    well past ``target`` draws (within the step's rollout budget) until
    ``target`` feasible candidates are pooled, then picks the max-score
    one among them.
    """
    seed, target, budget = 0, 4, 200
    reached_hi = Predicate("ReachedHi", [_block_type],
                           lambda s, o: s.get(o[0], "x") >= 0.9)
    sketch = [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(reached_hi, [_block])})
    ]
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}),
                {GroundAtom(reached_hi, [_block])})
    pool, n_draws = _replay_pool(seed, target, budget, lambda x: x >= 0.9)
    assert len(pool) == target  # the pool was filled...
    assert n_draws > target  # ...and that took more draws than a fixed batch

    plan, success, _ = refine_sketch(
        task,
        sketch,
        _FakeOptionModel(),
        predicates={reached_hi},
        timeout=10.0,
        rng=np.random.default_rng(seed),
        max_samples_per_step=budget,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=lambda s, _a: s.get(_block, "x"),
        info_n_feasible_target=target,
    )
    assert success
    assert float(plan[0].params[0]) == max(pool)


def test_step_budget_caps_pooling():
    """The step budget bounds rollouts; the argmax uses the partial pool."""
    seed, target, budget = 13, 8, 10
    reached_hi = Predicate("ReachedHi", [_block_type],
                           lambda s, o: s.get(o[0], "x") >= 0.9)
    sketch = [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(reached_hi, [_block])})
    ]
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}),
                {GroundAtom(reached_hi, [_block])})
    pool, n_draws = _replay_pool(seed, target, budget, lambda x: x >= 0.9)
    assert n_draws == budget  # the budget was hit before the target...
    assert 0 < len(pool) < target  # ...leaving a partial (non-empty) pool

    model = _FakeOptionModel()
    plan, success, _ = refine_sketch(
        task,
        sketch,
        model,
        predicates={reached_hi},
        timeout=10.0,
        rng=np.random.default_rng(seed),
        max_samples_per_step=budget,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=lambda s, _a: s.get(_block, "x"),
        info_n_feasible_target=target,
    )
    assert success
    assert float(plan[0].params[0]) == max(pool)
    # Exactly budget scoring rollouts, plus the backtracking loop
    # re-executing the chosen option once for validation.
    assert model.num_calls == budget + 1


def test_budget_shared_across_attempts_fails_fast():
    """An unsatisfiable subgoal costs ~budget rollouts per node, not budget x
    attempts.

    Attempt 1 spends the whole node budget pooling (0 feasible) and
    falls back to an infeasible sample that fails validation; the
    remaining target - 1 attempts arrive with stock and budget
    exhausted, draw the 1-candidate minimum, and fail fast — then the
    step exhausts and the search backtracks.
    """
    budget, target = 10, 8
    sketch = [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_Unreachable, [_block])})
    ]
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}),
                {GroundAtom(_Unreachable, [_block])})
    model = _FakeOptionModel()
    plan, success, total_samples = refine_sketch(
        task,
        sketch,
        model,
        predicates={_Unreachable},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=budget,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=lambda s, _a: s.get(_block, "x"),
        info_n_feasible_target=target,
    )
    assert not success
    assert plan == []
    # The attempt cap equals the pool target, not budget-many attempts.
    assert total_samples == target
    # Attempt 1: budget draws + 1 execution of the fallback. Attempts
    # 2..target: 1 minimum draw + 1 execution each. Far below the old
    # nested worst case of budget * attempts rollouts.
    expected = (budget + 1) + 2 * (target - 1)
    assert model.num_calls == expected


def test_ranked_stock_replayed_across_backtracks():
    """Downstream failures pop the ranked stock — no re-pooling.

    Step 0 is info-eligible with an always-true subgoal but a target
    above the budget, so its first attempt spends the whole node budget
    pooling (every draw feasible) and banks the runner-ups. Step 1 is
    parameter-free with a never-true subgoal, so it fails immediately
    and bounces the search back to step 0. Each bounce must consume the
    next-ranked banked candidate at the cost of a single execution
    rollout — not redraw a pool — and once the stock and budget are
    gone, the remaining attempts fall back to 1-draw fillers until the
    attempt cap (= target) exhausts the step.
    """
    budget, target = 3, 8
    sketch = [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_Reached, [_block])}),
        SketchStep(option=_Noop,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_Unreachable, [_block])}),
    ]
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}),
                {GroundAtom(_Unreachable, [_block])})
    model = _FakeOptionModel()
    plan, success, total_samples = refine_sketch(
        task,
        sketch,
        model,
        predicates={_Reached, _Unreachable},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=budget,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=lambda s, _a: s.get(_block, "x"),
        info_n_feasible_target=target,
    )
    assert not success
    assert plan == []
    # Step 0 gets `target` attempts before exhausting; step 1 fails once
    # per step-0 success.
    assert total_samples == 2 * target
    # Step-0 cost: attempt 1 pools `budget` draws + 1 execution; attempts
    # 2-3 replay the two banked candidates (1 execution each, no draws);
    # attempts 4..target are 1-draw fillers (2 rollouts each). Step 1: 1
    # execution per bounce. Without stock replay, attempts 2-3 would
    # redraw and cost more.
    step0 = (budget + 1) + 2 * 1 + (target - budget) * 2
    expected = step0 + target
    assert model.num_calls == expected


def test_ranked_walk_on_goal_miss():
    """Final-goal misses walk the ranked stock best-first to success.

    The pool's candidates all satisfy the subgoal, but the task goal
    holds only for the *least* informative one. The loop's retries must
    pop candidates in descending info-score order — each a pure
    execution, no new draws — and succeed on the last-ranked one.
    """
    seed, target = 7, 4
    pool, n_draws = _replay_pool(seed, target, 50, lambda x: True)
    assert n_draws == target
    assert len(set(pool)) == target  # distinct => threshold well-defined
    ranked = sorted(pool, reverse=True)
    # Goal holds strictly below the gap between the two lowest-ranked.
    thresh = (ranked[-1] + ranked[-2]) / 2
    goal_low = Predicate("GoalLow", [_block_type],
                         lambda s, o: s.get(o[0], "x") < thresh)
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}),
                {GroundAtom(goal_low, [_block])})
    model = _FakeOptionModel()
    plan, success, total_samples = refine_sketch(
        task,
        _sketch(),
        model,
        predicates=_PREDICATES,
        timeout=10.0,
        rng=np.random.default_rng(seed),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=True,
        info_scorer=lambda s, _a: s.get(_block, "x"),
        info_n_feasible_target=target,
    )
    assert success
    # Walked best-first; only the minimum-x candidate satisfies the goal.
    assert float(plan[0].params[0]) == min(pool)
    assert total_samples == target  # one attempt per ranked candidate
    # `target` scoring rollouts (the pool) + `target` executions (the
    # walk) — the retries drew nothing new.
    assert model.num_calls == 2 * target


def test_plain_budget_semantics_unchanged():
    """Without a scorer, max_samples_per_step still means attempts.

    Anchors the budget identity for plain backtracking: an
    unsatisfiable-subgoal step burns exactly one rollout per attempt for
    max_samples_per_step attempts, then exhausts.
    """
    budget = 7
    sketch = [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_Unreachable, [_block])})
    ]
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}),
                {GroundAtom(_Unreachable, [_block])})
    model = _FakeOptionModel()
    plan, success, total_samples = refine_sketch(
        task,
        sketch,
        model,
        predicates={_Unreachable},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=budget,
        check_subgoals=True,
        check_final_goal=False,
    )
    assert not success
    assert plan == []
    assert total_samples == budget
    assert model.num_calls == budget
