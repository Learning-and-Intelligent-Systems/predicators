"""Tests for near-miss reporting in ``sketch_refinement``.

A failed backtracking search now records the deepest rollout that
executed but failed validation - the failing step's exact params, the
missing atoms, and that rollout's post-state - and
``refine_and_validate_report`` surfaces it, so a failed
refine_plan_sketch call returns a gradient instead of only the stuck
step's name.
"""

import numpy as np
from gym.spaces import Box

from predicators.agent_sdk.plan_execution import _fmt_state_features
from predicators.agent_sdk.sketch_refinement import \
    refine_and_validate_report, refine_sketch
from predicators.agent_sdk.sketch_types import SketchStep
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)
_other = Object("other0", _block_type)


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: False,
)

_NeverInit = ParameterizedOption(
    "NeverInit",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=lambda _s, _m, _o, _p: False,
    terminal=lambda _s, _m, _o, _p: False,
)

# Impossible within the option box (x <= 1.0): every rollout executes
# but fails the subgoal check, exercising the validation-failure path.
_ReachedTwo = Predicate("ReachedTwo", [_block_type],
                        lambda s, o: s.get(o[0], "x") >= 2.0)
# Always true: an easy upstream step.
_Reached = Predicate("Reached", [_block_type], lambda s, o: True)


class _FakeOptionModel:
    """Deterministic model: Move sets block.x to its parameter value."""

    last_execution_failure = None

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step."""
        nxt = state.copy()
        if len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        return nxt, 1


def _task(goal_pred=_ReachedTwo):
    init = State({_block: np.array([0.0], dtype=np.float32)})
    return Task(init, {GroundAtom(goal_pred, [_block])})


def _step(option=_Move, subgoal_pred=_ReachedTwo):
    return SketchStep(option=option,
                      objects=[_block],
                      subgoal_atoms={GroundAtom(subgoal_pred, [_block])})


def _refine(sketch, holder, **kwargs):
    defaults = dict(predicates={_ReachedTwo, _Reached},
                    timeout=10.0,
                    rng=np.random.default_rng(0),
                    max_samples_per_step=5,
                    check_subgoals=True,
                    check_final_goal=False,
                    deepest_failure_holder=holder)
    defaults.update(kwargs)
    return refine_sketch(_task(), sketch, _FakeOptionModel(), **defaults)


def test_deepest_failure_recorded_without_truncation():
    """The holder fills on a plain (non-explorer) failed search."""
    holder = []
    _, success, _ = _refine([_step()], holder)
    assert not success
    assert len(holder) == 1
    df = holder[0]
    assert df.step_idx == 0
    assert df.fail_reason.startswith("subgoal missing")
    assert "ReachedTwo" in df.fail_reason
    assert df.option.name == "Move"
    assert 0.0 <= float(df.option.params[0]) <= 1.0
    assert df.post_state is not None


def test_deepest_failure_tracks_deepest_index_with_params_and_post_state():
    """A two-step sketch records the DEEPER failing step.

    The stashed post-state matches the failing option's own rollout.
    """
    holder = []
    sketch = [_step(subgoal_pred=_Reached), _step(subgoal_pred=_ReachedTwo)]
    _, success, _ = _refine(sketch, holder)
    assert not success
    assert holder[0].step_idx == 1
    # The fake model sets x to the option's param: the post-state is the
    # failing rollout's own, not some other attempt's.
    assert np.isclose(holder[0].post_state.get(_block, "x"),
                      float(holder[0].option.params[0]))


def test_deepest_failure_ignores_non_validation_failures():
    """Not-initiable failures never fill the holder (no rollout to report)."""
    holder = []
    _, success, _ = _refine([_step(option=_NeverInit)], holder)
    assert not success
    assert not holder


def _report(sketch, **kwargs):
    defaults = dict(predicates={_ReachedTwo, _Reached},
                    timeout=10.0,
                    rng=np.random.default_rng(0),
                    max_samples_per_step=5,
                    check_subgoals=True)
    defaults.update(kwargs)
    return refine_and_validate_report(_task(), sketch, _FakeOptionModel(),
                                      **defaults)


def test_report_shows_deepest_failure_on_sample_exhausted():
    """SAMPLE_EXHAUSTED reports the near-miss params, reason, and state."""
    success, report, _ = _report([_step()])
    assert not success
    assert "FAILURE: SAMPLE_EXHAUSTED" in report
    assert "Deepest failure: step 0 Move(block0)[" in report
    assert "subgoal missing" in report
    assert "ReachedTwo" in report
    assert "post-state: block0[x=" in report
    # Only the step's own objects are dumped.
    assert "other0" not in report


def test_report_shows_deepest_failure_on_timeout():
    """A search that times out still reports its deepest near-miss."""
    success, report, _ = _report([_step()],
                                 timeout=0.5,
                                 max_samples_per_step=10_000_000)
    assert not success
    assert "FAILURE: TIMEOUT" in report
    assert "Deepest failure: step 0 Move(block0)[" in report


def test_report_omits_deepest_failure_when_none_recorded():
    """No validation failure (e.g. nothing ever initiable) => no line."""
    success, report, _ = _report([_step(option=_NeverInit)])
    assert not success
    assert "Deepest failure" not in report


def test_truncate_on_subgoal_fail_behavior_unchanged():
    """Explorer-mode truncation still returns the deepest consistent prefix
    (inclusive of the failing step), with or without a holder attached."""
    sketch = [_step(subgoal_pred=_Reached), _step(subgoal_pred=_ReachedTwo)]
    plan_no_holder, success, _ = refine_sketch(
        _task(),
        sketch,
        _FakeOptionModel(),
        predicates={_ReachedTwo, _Reached},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=5,
        check_subgoals=True,
        check_final_goal=False,
        truncate_on_subgoal_fail=True)
    assert not success
    assert len(plan_no_holder) == 2  # prefix includes the failing step
    holder = []
    plan_with_holder, success, _ = _refine(sketch,
                                           holder,
                                           truncate_on_subgoal_fail=True)
    assert not success
    assert len(plan_with_holder) == 2
    assert holder[0].step_idx == 1


def test_fmt_state_features_object_filter():
    """The objects filter restricts the dump; default dumps everything."""
    state = State({
        _block: np.array([0.25], dtype=np.float32),
        _other: np.array([0.75], dtype=np.float32),
    })
    full = _fmt_state_features(state)
    assert "block0[x=0.2500]" in full
    assert "other0[x=0.7500]" in full
    only = _fmt_state_features(state, objects=[_block])
    assert only == "block0[x=0.2500]"
