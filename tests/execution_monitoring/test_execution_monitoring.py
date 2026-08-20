"""Tests for execution monitors."""

import numpy as np
import pytest
from gym.spaces import Box

from predicators.execution_monitoring import create_execution_monitor
from predicators.execution_monitoring.expected_atoms_monitor import \
    ExpectedAtomsExecutionMonitor
from predicators.execution_monitoring.mpc_execution_monitor import \
    MpcExecutionMonitor
from predicators.execution_monitoring.subgoal_annotations_monitor import \
    SubgoalAnnotationsExecutionMonitor, SubgoalExecutionStatus
from predicators.execution_monitoring.trivial_execution_monitor import \
    TrivialExecutionMonitor
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Type


def test_create_execution_monitor():
    """Tests for create_execution_monitor()."""
    exec_monitor = create_execution_monitor("trivial")
    assert isinstance(exec_monitor, TrivialExecutionMonitor)

    exec_monitor = create_execution_monitor("mpc")
    assert isinstance(exec_monitor, MpcExecutionMonitor)

    exec_monitor = create_execution_monitor("expected_atoms")
    assert isinstance(exec_monitor, ExpectedAtomsExecutionMonitor)

    exec_monitor = create_execution_monitor("subgoal_annotations")
    assert isinstance(exec_monitor, SubgoalAnnotationsExecutionMonitor)

    with pytest.raises(NotImplementedError) as e:
        create_execution_monitor("not a real monitor")
    assert "Unrecognized execution monitor" in str(e)


class _FakeSketchStep:
    """Duck-typed sketch step (see agent_sdk.bilevel_sketch.SketchStep)."""

    def __init__(self, option, subgoal_atoms, subgoal_neg_atoms=None):
        self.option = option
        self.subgoal_atoms = subgoal_atoms
        self.subgoal_neg_atoms = subgoal_neg_atoms


def test_subgoal_annotations_monitor():
    """Unit tests for SubgoalAnnotationsExecutionMonitor.step()."""
    block_type = Type("block", ["held"])
    block = Object("block0", block_type)
    held = Predicate("Held", [block_type],
                     lambda s, o: s.get(o[0], "held") > 0.5)
    state_held = State({block: np.array([1.0], dtype=np.float32)})
    state_free = State({block: np.array([0.0], dtype=np.float32)})

    def _make_option(terminal):
        param_opt = ParameterizedOption(
            "Pick",
            types=[block_type],
            params_space=Box(low=np.zeros(1, dtype=np.float32),
                             high=np.ones(1, dtype=np.float32)),
            policy=lambda s, m, o, p: Action(np.zeros(1, dtype=np.float32)),
            initiable=lambda s, m, o, p: True,
            terminal=lambda s, m, o, p: terminal,
        )
        return param_opt, param_opt.ground([block],
                                           np.zeros(1, dtype=np.float32))

    done_parent, done_option = _make_option(True)
    _, running_option = _make_option(False)
    held_atom = GroundAtom(held, [block])

    monitor = create_execution_monitor("subgoal_annotations")

    # No approach info (e.g. exploration): never replan.
    assert not monitor.step(state_free)

    # Info of an unexpected shape (another approach's export): ignore.
    monitor.update_approach_info([{"something": "else"}])
    assert not monitor.step(state_free)

    def _status(option, steps_initiated, pos=None, neg=None):
        step = _FakeSketchStep(done_parent, pos, neg)
        return SubgoalExecutionStatus(sketch=[step],
                                      steps_initiated=steps_initiated,
                                      current_option=option)

    # No option initiated yet (fresh policy right after a replan).
    monitor.update_approach_info([_status(None, 0, {held_atom})])
    assert not monitor.step(state_free)

    # Mid-option: the current option has not terminated.
    monitor.update_approach_info([_status(running_option, 1, {held_atom})])
    assert not monitor.step(state_free)

    # Boundary, annotation holds: no replan.
    monitor.update_approach_info([_status(done_option, 1, {held_atom})])
    assert not monitor.step(state_held)

    # Boundary, unannotated step: nothing to check.
    monitor.update_approach_info([_status(done_option, 1, None)])
    assert not monitor.step(state_free)

    # Boundary, positive atom unsatisfied: replan.
    monitor.update_approach_info([_status(done_option, 1, {held_atom})])
    assert monitor.step(state_free)

    # Boundary, negative atom violated: replan.
    monitor.update_approach_info([_status(done_option, 1, None, {held_atom})])
    assert monitor.step(state_held)
