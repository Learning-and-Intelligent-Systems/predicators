"""Tests for per-skill synthesized samplers in ``sketch_refinement``.

Verifies that a sampler registered under an option name in
``parameterized_samplers`` is consulted (with the step's subgoal +
objects + the option's params box) to draw that option's continuous
params during refinement — on both the plain and info-seeking paths —
and that a missing / misbehaving sampler falls back to uniform sampling
so refinement is byte-for-byte unchanged when no usable sampler is
supplied.
"""

# pylint: disable=unused-import

from types import SimpleNamespace

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils  # noqa: F401  (settles import order)
from predicators.agent_sdk.plan_execution import execute_plan_forward
from predicators.agent_sdk.sketch_parsing import parse_atoms, \
    parse_sketch_from_text, strip_subgoal_annotations
from predicators.agent_sdk.sketch_refinement import \
    refine_and_validate_report, refine_sketch, sample_params
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


# A 1-D option whose parameter becomes the post-state x of the block.
_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_true,
    terminal=_false,
)

# A zero-dim option (no continuous params) for empty-bracket parsing tests.
_Wait0 = ParameterizedOption(
    "Wait0",
    types=[_block_type],
    params_space=Box(low=np.zeros(0, dtype=np.float32),
                     high=np.zeros(0, dtype=np.float32)),
    policy=_noop_policy,
    initiable=_true,
    terminal=_false,
)

# An option that is never initiable, for forward-execution failure tests.
_NeverInit = ParameterizedOption(
    "NeverInit",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_false,
    terminal=_false,
)


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


# Subgoal uniform sampling hits only ~10% of the time (x >= 0.9), but a
# targeted sampler lands on the first draw.
_ReachedHi = Predicate("ReachedHi", [_block_type],
                       lambda s, o: s.get(o[0], "x") >= 0.9)
# Always-true subgoal so the first draw (uniform or sampled) is accepted.
_Reached = Predicate("Reached", [_block_type], lambda s, o: True)


def _task_hi():
    init = State({_block: np.array([0.0], dtype=np.float32)})
    return Task(init, {GroundAtom(_ReachedHi, [_block])})


def _sketch_hi():
    return [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_ReachedHi, [_block])})
    ]


def _easy_task_and_sketch():
    sketch = [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_Reached, [_block])})
    ]
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}),
                {GroundAtom(_Reached, [_block])})
    return task, sketch


def test_registered_sampler_is_used():
    """A targeted sampler lands the hard subgoal on the first sample."""
    calls = []

    def sampler(state, subgoal_atoms, rng, objects):
        del state, rng
        calls.append((objects, subgoal_atoms))
        return np.array([0.95], dtype=np.float32)

    model = _FakeOptionModel()
    plan, success, total = refine_sketch(
        _task_hi(),
        _sketch_hi(),
        model,
        predicates={_ReachedHi},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        parameterized_samplers={"Move": sampler})
    assert success
    assert np.isclose(float(plan[0].params[0]), 0.95)
    # Feasible on the very first attempt — none of the uniform churn.
    assert total == 1
    assert model.num_calls == 1
    # The sampler saw the right subgoal and objects.
    objs, subgoal = calls[0]
    assert [o.name for o in objs] == ["block0"]
    assert GroundAtom(_ReachedHi, [_block]) in subgoal


def test_missing_entry_falls_back_to_uniform():
    """A sampler keyed by another option leaves Move on the uniform path."""
    seed = 7
    first = float(sample_params(_Move, np.random.default_rng(seed))[0])
    task, sketch = _easy_task_and_sketch()

    def other(*_args):
        raise AssertionError("sampler for a different option was called")

    plan, success, _ = refine_sketch(
        task,
        sketch,
        _FakeOptionModel(),
        predicates={_Reached},
        timeout=10.0,
        rng=np.random.default_rng(seed),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        parameterized_samplers={"OtherOption": other})
    assert success
    # Identical to the no-sampler uniform draw.
    assert float(plan[0].params[0]) == first


def test_bad_shape_falls_back_to_uniform():
    """A wrong-shaped return is rejected; uniform sampling still succeeds."""
    task, sketch = _easy_task_and_sketch()

    def bad(*_args):
        return np.array([0.5, 0.5], dtype=np.float32)  # shape (2,) != (1,)

    plan, success, _ = refine_sketch(task,
                                     sketch,
                                     _FakeOptionModel(),
                                     predicates={_Reached},
                                     timeout=10.0,
                                     rng=np.random.default_rng(0),
                                     max_samples_per_step=50,
                                     check_subgoals=True,
                                     check_final_goal=False,
                                     parameterized_samplers={"Move": bad})
    assert success
    assert 0.0 <= float(plan[0].params[0]) <= 1.0


def test_raising_sampler_falls_back_to_uniform():
    """A sampler that raises is caught and uniform sampling proceeds."""
    task, sketch = _easy_task_and_sketch()

    def boom(*_args):
        raise ValueError("nope")

    _, success, _ = refine_sketch(task,
                                  sketch,
                                  _FakeOptionModel(),
                                  predicates={_Reached},
                                  timeout=10.0,
                                  rng=np.random.default_rng(0),
                                  max_samples_per_step=50,
                                  check_subgoals=True,
                                  check_final_goal=False,
                                  parameterized_samplers={"Move": boom})
    assert success


def test_none_samplers_unchanged():
    """parameterized_samplers=None reproduces the plain first-uniform-draw
    param."""
    seed = 7
    first = float(sample_params(_Move, np.random.default_rng(seed))[0])
    task, sketch = _easy_task_and_sketch()
    plan, success, _ = refine_sketch(task,
                                     sketch,
                                     _FakeOptionModel(),
                                     predicates={_Reached},
                                     timeout=10.0,
                                     rng=np.random.default_rng(seed),
                                     max_samples_per_step=50,
                                     check_subgoals=True,
                                     check_final_goal=False,
                                     parameterized_samplers=None)
    assert success
    assert float(plan[0].params[0]) == first


def test_sampler_used_on_info_seeking_path():
    """The info-seeking draw loop also routes through the sampler."""

    def sampler(_s, _a, rng, _o):
        # Jitter so candidates differ but all clear the x>=0.9 subgoal.
        return np.array([0.9 + 0.05 * rng.random()], dtype=np.float32)

    model = _FakeOptionModel()
    plan, success, _ = refine_sketch(
        _task_hi(),
        _sketch_hi(),
        model,
        predicates={_ReachedHi},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=lambda s, _a: s.get(_block, "x"),
        info_n_feasible_target=4,
        parameterized_samplers={"Move": sampler})
    assert success
    # Every pooled candidate came from the sampler => satisfies x >= 0.9.
    assert float(plan[0].params[0]) >= 0.9


# --------------------------------------------------------------------------- #
# LLM-proposed initial_params (tried first, with sampling fallback).
# --------------------------------------------------------------------------- #


def test_initial_params_tried_first_without_sampler():
    """LLM-proposed initial_params are used before any uniform draw."""
    step = SketchStep(option=_Move,
                      objects=[_block],
                      subgoal_atoms={GroundAtom(_ReachedHi, [_block])},
                      initial_params=np.array([0.95], dtype=np.float32))
    model = _FakeOptionModel()
    plan, success, total = refine_sketch(_task_hi(), [step],
                                         model,
                                         predicates={_ReachedHi},
                                         timeout=10.0,
                                         rng=np.random.default_rng(0),
                                         max_samples_per_step=50,
                                         check_subgoals=True,
                                         check_final_goal=False,
                                         parameterized_samplers=None)
    assert success
    # The proposal satisfied the hard subgoal on the very first attempt.
    assert np.isclose(float(plan[0].params[0]), 0.95)
    assert total == 1
    assert model.num_calls == 1


def test_initial_params_fall_back_to_uniform_on_failure():
    """A bad proposal fails the first attempt; uniform backtracking
    recovers."""
    step = SketchStep(option=_Move,
                      objects=[_block],
                      subgoal_atoms={GroundAtom(_ReachedHi, [_block])},
                      initial_params=np.array([0.0], dtype=np.float32))
    plan, success, total = refine_sketch(_task_hi(), [step],
                                         _FakeOptionModel(),
                                         predicates={_ReachedHi},
                                         timeout=10.0,
                                         rng=np.random.default_rng(0),
                                         max_samples_per_step=200,
                                         check_subgoals=True,
                                         check_final_goal=False,
                                         parameterized_samplers=None)
    assert success
    # The failed proposal was the first sample; uniform then found x >= 0.9.
    assert total > 1
    assert float(plan[0].params[0]) >= 0.9


def test_initial_params_clipped_to_box():
    """Out-of-box proposals are clipped before grounding (no ValueError)."""
    step = SketchStep(option=_Move,
                      objects=[_block],
                      subgoal_atoms={GroundAtom(_ReachedHi, [_block])},
                      initial_params=np.array([5.0], dtype=np.float32))
    plan, success, total = refine_sketch(_task_hi(), [step],
                                         _FakeOptionModel(),
                                         predicates={_ReachedHi},
                                         timeout=10.0,
                                         rng=np.random.default_rng(0),
                                         max_samples_per_step=50,
                                         check_subgoals=True,
                                         check_final_goal=False,
                                         parameterized_samplers=None)
    assert success
    # 5.0 clipped to the option's high bound (1.0), which clears x >= 0.9.
    assert np.isclose(float(plan[0].params[0]), 1.0)
    assert total == 1


def test_initial_params_seeded_and_win_on_disagreement():
    """LLM params are pooled with sampled draws; the argmax (most informative)
    is chosen.

    Here the guess is the most informative.
    """
    step = SketchStep(option=_Move,
                      objects=[_block],
                      subgoal_atoms={GroundAtom(_ReachedHi, [_block])},
                      initial_params=np.array([1.0], dtype=np.float32))
    model = _FakeOptionModel()

    # Sampled candidates clear x >= 0.9 but stay below the guess's x = 1.0.
    def sampler(_s, _a, rng, _o):
        return np.array([0.9 + 0.05 * rng.random()], dtype=np.float32)

    plan, success, _ = refine_sketch(
        _task_hi(), [step],
        model,
        predicates={_ReachedHi},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=lambda s, _a: float(s.get(_block, "x")),
        info_n_feasible_target=4,
        parameterized_samplers={"Move": sampler})
    assert success
    # The guess had the highest disagreement (x = 1.0) => argmax picked it.
    assert np.isclose(float(plan[0].params[0]), 1.0)
    # The pool was actually built (guess + draws rolled), not short-circuited
    # at the guess: a short-circuit would have rolled the option only once.
    assert model.num_calls >= 4


def test_initial_params_lose_to_more_informative_draw():
    """A feasible guess no longer short-circuits: a strictly more informative
    sampled candidate beats it in the disagreement argmax."""
    step = SketchStep(option=_Move,
                      objects=[_block],
                      subgoal_atoms={GroundAtom(_ReachedHi, [_block])},
                      initial_params=np.array([0.9], dtype=np.float32))
    plan, success, _ = refine_sketch(
        _task_hi(), [step],
        _FakeOptionModel(),
        predicates={_ReachedHi},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=lambda s, _a: float(s.get(_block, "x")),
        info_n_feasible_target=4,
        parameterized_samplers={
            "Move": lambda *_a: np.array([0.99], dtype=np.float32)
        })
    assert success
    # The seeded guess (x = 0.9) was beaten by the more informative draw
    # (0.99) — proving it is pooled, not accepted just for being first.
    assert np.isclose(float(plan[0].params[0]), 0.99)


def test_initial_params_infeasible_seed_info_seeking_recovers():
    """An infeasible guess isn't pooled; info-seeking draws still solve."""
    step = SketchStep(option=_Move,
                      objects=[_block],
                      subgoal_atoms={GroundAtom(_ReachedHi, [_block])},
                      initial_params=np.array([0.0], dtype=np.float32))
    plan, success, _ = refine_sketch(
        _task_hi(), [step],
        _FakeOptionModel(),
        predicates={_ReachedHi},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=200,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=lambda s, _a: float(s.get(_block, "x")),
        info_n_feasible_target=4,
        parameterized_samplers={
            "Move":
            lambda _s, _a, rng, _o: np.array([0.9 + 0.05 * rng.random()],
                                             dtype=np.float32)
        })
    assert success
    # The infeasible guess (x = 0) wasn't pooled; sampled candidates won.
    assert float(plan[0].params[0]) >= 0.9


def test_strip_subgoal_annotations():
    """`-> {atoms}` is removed so the params parser sees only the option."""
    out = strip_subgoal_annotations(
        "Move(block0:block)[0.7] -> {ReachedHi(block0:block)}")
    assert out == "Move(block0:block)[0.7]"
    # A line without an annotation is untouched.
    assert strip_subgoal_annotations(
        "Move(block0:block)[0.7]") == "Move(block0:block)[0.7]"


def test_parse_sketch_params_wrong_arity_drops_sketch():
    """Wrong param count is rejected by the canonical parser (empty sketch)."""
    sketch = parse_sketch_from_text(
        "Move(block0:block)[0.7, 0.8] -> {ReachedHi(block0:block)}",
        _task_hi(),
        predicates={_ReachedHi},
        options={_Move},
        types={_block_type},
        parse_continuous_params=True)
    assert not sketch


def test_parse_sketch_zero_dim_empty_brackets():
    """`[]` on a zero-param option yields an empty initial_params array."""
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}), set())
    sketch = parse_sketch_from_text("Wait0(block0:block)[]",
                                    task,
                                    predicates=set(),
                                    options={_Wait0},
                                    types={_block_type},
                                    parse_continuous_params=True)
    assert len(sketch) == 1
    assert sketch[0].initial_params is not None
    assert sketch[0].initial_params.shape == (0, )


def test_parse_sketch_strict_errors_on_bad_line():
    """``strict=True`` raises on a line the parser would otherwise silently
    drop or truncate at, so a tool never executes a different plan than the
    agent wrote."""
    with pytest.raises(ValueError, match="continuous parameter"):
        parse_sketch_from_text(
            "Move(block0:block)[0.7, 0.8] -> {ReachedHi(block0:block)}",
            _task_hi(),
            predicates={_ReachedHi},
            options={_Move},
            types={_block_type},
            parse_continuous_params=True,
            strict=True)
    with pytest.raises(ValueError, match="too many object arguments"):
        parse_sketch_from_text("Move(block0:block, block0:block)[0.7]",
                               _task_hi(),
                               predicates={_ReachedHi},
                               options={_Move},
                               types={_block_type},
                               parse_continuous_params=True,
                               strict=True)


def test_parse_sketch_strict_empty_brackets_mean_no_seed():
    """Under ``strict=True``, an explicit `[]` on a PARAMETRIZED option is the
    documented "no seed": the line parses with ``initial_params`` None so
    refinement samples the parameters (previously the count mismatch silently
    truncated the sketch at that line)."""
    sketch = parse_sketch_from_text(
        "Move(block0:block)[] -> {ReachedHi(block0:block)}",
        _task_hi(),
        predicates={_ReachedHi},
        options={_Move},
        types={_block_type},
        parse_continuous_params=True,
        strict=True)
    assert len(sketch) == 1
    assert sketch[0].initial_params is None
    assert GroundAtom(_ReachedHi, [_block]) in sketch[0].subgoal_atoms
    # A zero-param option's `[]` stays an exact (empty) param vector.
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}), set())
    sketch = parse_sketch_from_text("Wait0(block0:block)[]",
                                    task,
                                    predicates=set(),
                                    options={_Wait0},
                                    types={_block_type},
                                    parse_continuous_params=True,
                                    strict=True)
    assert len(sketch) == 1
    assert sketch[0].initial_params is not None
    assert sketch[0].initial_params.shape == (0, )


def test_parse_sketch_from_text_with_params():
    """parse_continuous_params=True populates SketchStep.initial_params."""
    sketch = parse_sketch_from_text(
        "Move(block0:block)[0.7] -> {ReachedHi(block0:block)}",
        _task_hi(),
        predicates={_ReachedHi},
        options={_Move},
        types={_block_type},
        parse_continuous_params=True)
    assert len(sketch) == 1
    assert sketch[0].initial_params is not None
    assert np.allclose(sketch[0].initial_params, [0.7])
    assert GroundAtom(_ReachedHi, [_block]) in sketch[0].subgoal_atoms


def test_parse_sketch_from_text_params_disabled_by_default():
    """Default (params off) ignores `[..]` and leaves initial_params None."""
    sketch = parse_sketch_from_text(
        "Move(block0:block)[0.7] -> {ReachedHi(block0:block)}",
        _task_hi(),
        predicates={_ReachedHi},
        options={_Move},
        types={_block_type})
    assert len(sketch) == 1
    assert sketch[0].initial_params is None
    # The option + subgoal still parse, unaffected by the trailing `[..]`.
    assert GroundAtom(_ReachedHi, [_block]) in sketch[0].subgoal_atoms


def test_parse_atoms_pos_neg():
    """parse_atoms splits positive and NOT-prefixed (negative) atoms."""
    pos, neg = parse_atoms(
        "ReachedHi(block0:block), NOT Reached(block0:block)",
        {_ReachedHi, _Reached}, [_block])
    assert pos == {GroundAtom(_ReachedHi, [_block])}
    assert neg == {GroundAtom(_Reached, [_block])}


def test_parse_atoms_unknown_skipped():
    """Atoms with an unknown predicate are skipped (not raised)."""
    pos, neg = parse_atoms("Nope(block0:block)", {_ReachedHi}, [_block])
    assert pos == set()
    assert neg == set()


def test_execute_plan_forward_success():
    """A plan that reaches the goal: success, executed_all, no failure."""
    plan = [_Move.ground([_block], np.array([0.95], dtype=np.float32))]
    seen = []
    result = execute_plan_forward(_task_hi(),
                                  plan,
                                  _FakeOptionModel(),
                                  predicates={_ReachedHi},
                                  on_step=lambda i, o: seen.append(i))
    assert result.success
    assert result.goal_reached
    assert result.executed_all
    assert result.first_failure_idx is None
    assert len(result.steps) == 1
    assert result.steps[0].num_actions == 1
    assert seen == [0]  # on_step fired once


# An option that is initiable but whose execution yields 0 actions (e.g. a
# motion-planning collision), for the continue-past-failure tests.
_Stuck = ParameterizedOption(
    "Stuck",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_true,
    terminal=_false,
)


class _StuckThenMoveModel:
    """Like _FakeOptionModel, but Stuck returns 0 actions (unchanged state)
    while leaving a post-state, so forward execution keeps going."""

    def __init__(self):
        self.num_calls = 0
        self.last_execution_failure = None

    def get_next_state_and_num_actions(self, state, option):
        """Stuck -> (unchanged state, 0 actions); Move -> sets x, 1 action."""
        self.num_calls += 1
        if option.name == "Stuck":
            self.last_execution_failure = "BiRRT collision"
            return state.copy(), 0
        nxt = state.copy()
        if len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        return nxt, 1


def test_execute_plan_forward_continues_past_zero_action_failure():
    """A 0-action (collision) step is a failure but execution continues; if a
    later step reaches the goal, goal_reached is True yet clean_to_goal is
    False — the real executor would abort at the failing step."""
    plan = [
        _Stuck.ground([_block], np.array([0.5], dtype=np.float32)),
        _Move.ground([_block], np.array([0.95], dtype=np.float32)),
    ]
    result = execute_plan_forward(_task_hi(),
                                  plan,
                                  _StuckThenMoveModel(),
                                  predicates={_ReachedHi})
    assert result.goal_reached  # goal atoms hold in the final state
    assert result.first_failure_idx == 0  # the 0-action Stuck step
    assert result.goal_step_idx == 1  # goal first holds after Move
    assert not result.clean_to_goal  # failure precedes the goal step
    assert not result.success
    assert result.steps[0].failure_reason == "BiRRT collision"
    assert result.actions_to_goal == 1  # only Move spent an action


def test_execute_plan_forward_stop_on_failure_aborts():
    """With stop_on_failure (the evaluate_option_plan path), a 0-action step
    aborts execution like the real executor: later steps don't run and the goal
    is not reached."""
    plan = [
        _Stuck.ground([_block], np.array([0.5], dtype=np.float32)),
        _Move.ground([_block], np.array([0.95], dtype=np.float32)),
    ]
    model = _StuckThenMoveModel()
    result = execute_plan_forward(_task_hi(),
                                  plan,
                                  model,
                                  predicates={_ReachedHi},
                                  stop_on_failure=True)
    assert not result.goal_reached  # aborted before the Move could run
    assert result.first_failure_idx == 0
    assert len(result.steps) == 1  # stopped after the failed step
    assert model.num_calls == 1  # Move never executed
    assert result.goal_step_idx is None
    assert not result.clean_to_goal


def test_execute_plan_forward_clean_to_goal_tracks_actions():
    """A clean plan that reaches the goal is clean_to_goal with the cumulative
    actions-to-goal recorded."""
    plan = [_Move.ground([_block], np.array([0.95], dtype=np.float32))]
    result = execute_plan_forward(_task_hi(),
                                  plan,
                                  _FakeOptionModel(),
                                  predicates={_ReachedHi})
    assert result.clean_to_goal
    assert result.goal_step_idx == 0
    assert result.actions_to_goal == 1
    assert result.total_actions == 1


def test_execute_plan_forward_goal_not_reached():
    """The step executes but doesn't reach the goal: not success, no
    failure."""
    plan = [_Move.ground([_block], np.array([0.5], dtype=np.float32))]
    result = execute_plan_forward(_task_hi(),
                                  plan,
                                  _FakeOptionModel(),
                                  predicates={_ReachedHi})
    assert not result.success
    assert not result.goal_reached
    assert result.executed_all
    assert result.first_failure_idx is None


def test_execute_plan_forward_subgoal_divergence():
    """An unmet step subgoal is recorded (but isn't an execution failure)."""
    sketch = [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_ReachedHi, [_block])})
    ]
    plan = [_Move.ground([_block], np.array([0.5], dtype=np.float32))]
    result = execute_plan_forward(_task_hi(),
                                  plan,
                                  _FakeOptionModel(),
                                  predicates={_ReachedHi},
                                  sketch=sketch)
    assert result.first_subgoal_divergence_idx == 0
    assert result.steps[0].subgoal_missing == {
        GroundAtom(_ReachedHi, [_block])
    }
    assert result.first_failure_idx is None  # divergence != execution failure


def test_execute_plan_forward_not_initiable_stops():
    """A not-initiable step fails and halts execution (no later steps run)."""
    plan = [
        _NeverInit.ground([_block], np.array([0.0], dtype=np.float32)),
        _Move.ground([_block], np.array([0.95], dtype=np.float32)),
    ]
    model = _FakeOptionModel()
    result = execute_plan_forward(_task_hi(),
                                  plan,
                                  model,
                                  predicates={_ReachedHi})
    assert result.first_failure_idx == 0
    assert not result.executed_all
    assert result.steps[0].failure_reason == "not initiable"
    assert len(result.steps) == 1  # stopped after the failed step
    assert model.num_calls == 0  # never executed (not initiable)


def test_refine_and_validate_report_returns_plan():
    """refine_and_validate_report yields (success, report, plan).

    The grounded plan is what refine_plan_sketch captures so the
    approach can return the simulator-verified answer directly.
    """
    step = SketchStep(option=_Move,
                      objects=[_block],
                      subgoal_atoms={GroundAtom(_ReachedHi, [_block])},
                      initial_params=np.array([0.95], dtype=np.float32))
    success, report, plan = refine_and_validate_report(
        _task_hi(), [step],
        _FakeOptionModel(),
        predicates={_ReachedHi},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=50,
        check_subgoals=True)
    assert success
    assert "SUCCESS" in report
    assert len(plan) == 1
    # The captured plan carries the validated continuous params.
    assert np.isclose(float(plan[0].params[0]), 0.95)


class _TrajModel(_FakeOptionModel):
    """_FakeOptionModel that also exposes per-step low-level trajectories, so
    the solved_check gate sees a non-coarse rollout."""

    last_trajectory = None

    def get_next_state_and_num_actions(self, state, option):
        nxt, n = super().get_next_state_and_num_actions(state, option)
        self.last_trajectory = SimpleNamespace(states=[state, nxt],
                                               actions=[None])
        return nxt, n


def test_refine_sketch_solved_check_rejects_during_search():
    """A rejecting solved_check fails every goal-reaching candidate as "scored
    non-solve" DURING backtracking (the search keeps sampling instead of
    accepting), and the deepest-failure near-miss records the rejection with
    the candidate's exact params."""
    task, sketch = _easy_task_and_sketch()
    seen = []

    def reject_all(states, labels, coarse):
        seen.append((len(states), list(labels), coarse))
        return False, "solved=False, reward=-0.05"

    deepest = []
    _plan, success, total = refine_sketch(task,
                                          sketch,
                                          _TrajModel(),
                                          predicates={_Reached},
                                          timeout=10.0,
                                          rng=np.random.default_rng(0),
                                          max_samples_per_step=3,
                                          check_subgoals=True,
                                          check_final_goal=True,
                                          deepest_failure_holder=deepest,
                                          solved_check=reject_all)
    assert not success
    # Every candidate passed subgoal+goal checks, reached the gate, and
    # was rejected - the search spent its full budget.
    assert total == 3
    assert len(seen) == 3
    assert deepest and deepest[0].fail_reason == (
        "scored non-solve: solved=False, reward=-0.05")
    # Non-coarse: the model exposes last_trajectory, so the gate saw
    # init + per-step states and (name, objects, params) labels.
    n_states, labels, coarse = seen[0]
    assert n_states == 2 and coarse is False
    assert labels[0][0] == "Move" and labels[0][1] == ("block0", )


def test_refine_sketch_solved_check_accepts():
    """An accepting solved_check leaves refinement untouched; a coarse stash
    (model without last_trajectory) is flagged to the callback."""
    task, sketch = _easy_task_and_sketch()
    seen_coarse = []

    def accept_all(_states, _labels, coarse):
        seen_coarse.append(coarse)
        return True, ""

    plan, success, total = refine_sketch(
        task,
        sketch,
        _FakeOptionModel(),  # no last_trajectory -> coarse
        predicates={_Reached},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=3,
        check_subgoals=True,
        check_final_goal=True,
        solved_check=accept_all)
    assert success and len(plan) == 1 and total == 1
    assert seen_coarse == [True]
