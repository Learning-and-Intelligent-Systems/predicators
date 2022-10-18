"""Test cases for utils."""

import os
import time
from typing import Iterator, Tuple
from typing import Type as TypingType

import matplotlib.pyplot as plt
import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.envs.cover import CoverEnv, CoverMultistepOptions
from predicators.envs.pddl_env import ProceduralTasksSpannerPDDLEnv
from predicators.ground_truth_nsrts import _get_predicates_by_names, \
    get_gt_nsrts
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.settings import CFG
from predicators.structs import NSRT, Action, DefaultState, DummyOption, \
    GroundAtom, LowLevelTrajectory, ParameterizedOption, Predicate, Segment, \
    State, STRIPSOperator, Type, Variable
from predicators.utils import GoalCountHeuristic, _PyperplanHeuristicWrapper, \
    _TaskPlanningHeuristic


@pytest.mark.parametrize("max_groundings,exp_num_true,exp_num_false",
                         [(-1, 0, 0), (None, 1, 1)])
def test_count_positives_for_ops(max_groundings, exp_num_true, exp_num_false):
    """Tests for count_positives_for_ops()."""
    utils.reset_config({"segmenter": "atom_changes"})
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions = {not_on([cup_var, plate_var])}
    add_effects = {on([cup_var, plate_var])}
    delete_effects = {not_on([cup_var, plate_var])}
    strips_operator = STRIPSOperator("Pick", parameters, preconditions,
                                     add_effects, delete_effects, set())
    cup = cup_type("cup")
    plate = plate_type("plate")
    parameterized_option = utils.SingletonParameterizedOption(
        "Dummy",
        lambda s, m, o, p: Action(np.array([0.0])),
        params_space=Box(0, 1, (1, )))
    option = parameterized_option.ground([], np.array([0.0]))
    state = State({cup: [0.5], plate: [1.0]})
    action = Action(np.zeros(1, dtype=np.float32))
    action.set_option(option)
    states = [state, state]
    actions = [action]
    strips_ops = [strips_operator]
    option_specs = [(parameterized_option, [])]
    pruned_atom_data = [
        # Test empty sequence.
        (LowLevelTrajectory([state], []), [{on([cup, plate])}]),
        # Test not positive.
        (LowLevelTrajectory(states, actions), [{on([cup, plate])},
                                               set()]),
        # Test true positive.
        (LowLevelTrajectory(states, actions), [{not_on([cup, plate])},
                                               {on([cup, plate])}]),
        # Test false positive.
        (LowLevelTrajectory(states, actions), [{not_on([cup, plate])},
                                               set()]),
    ]
    segments = [
        seg for traj in pruned_atom_data for seg in segment_trajectory(traj)
    ]

    num_true, num_false, _, _ = utils.count_positives_for_ops(
        strips_ops, option_specs, segments, max_groundings=max_groundings)
    assert num_true == exp_num_true
    assert num_false == exp_num_false


def test_segment_trajectory_to_state_and_atoms_sequence():
    """Tests for segment_trajectory_to_state_sequence() and
    segment_trajectory_to_atoms_sequence()."""
    # Set up the segments.
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    state0 = State({cup: [0.5], plate: [1.0, 1.2]})
    state1 = State({cup: [0.5], plate: [1.1, 1.2]})
    state2 = State({cup: [0.8], plate: [1.5, 1.2]})
    states = [state0, state1, state2]
    action0 = Action([0.4])
    action1 = Action([0.6])
    actions = [action0, action1]
    traj1 = LowLevelTrajectory(states, actions)
    traj2 = LowLevelTrajectory(list(reversed(states)), actions)
    init_atoms = {on([cup, plate])}
    final_atoms = {not_on([cup, plate])}
    segment1 = Segment(traj1, init_atoms, final_atoms)
    segment2 = Segment(traj2, final_atoms, init_atoms)
    # Test segment_trajectory_to_state_sequence().
    state_seq = utils.segment_trajectory_to_state_sequence([segment1])
    assert state_seq == [state0, state2]
    state_seq = utils.segment_trajectory_to_state_sequence(
        [segment1, segment2])
    assert state_seq == [state0, state2, state0]
    state_seq = utils.segment_trajectory_to_state_sequence(
        [segment1, segment2, segment1, segment2])
    assert state_seq == [state0, state2, state0, state2, state0]
    with pytest.raises(AssertionError):
        # Need at least one segment in the trajectory.
        utils.segment_trajectory_to_state_sequence([])
    with pytest.raises(AssertionError):
        # Segments don't chain together correctly.
        utils.segment_trajectory_to_state_sequence([segment1, segment1])
    # Test segment_trajectory_to_atoms_sequence().
    atoms_seq = utils.segment_trajectory_to_atoms_sequence([segment1])
    assert atoms_seq == [init_atoms, final_atoms]
    atoms_seq = utils.segment_trajectory_to_atoms_sequence(
        [segment1, segment2])
    assert atoms_seq == [init_atoms, final_atoms, init_atoms]
    atoms_seq = utils.segment_trajectory_to_atoms_sequence(
        [segment1, segment2, segment1, segment2])
    assert atoms_seq == [
        init_atoms, final_atoms, init_atoms, final_atoms, init_atoms
    ]
    with pytest.raises(AssertionError):
        # Need at least one segment in the trajectory.
        utils.segment_trajectory_to_atoms_sequence([])
    with pytest.raises(AssertionError):
        # Segments don't chain together correctly.
        utils.segment_trajectory_to_atoms_sequence([segment1, segment1])


def test_num_options_in_action_sequence():
    """Tests for num_options_in_action_sequence()."""
    assert utils.num_options_in_action_sequence([]) == 0
    actions = [Action(np.array([0])) for _ in range(3)]
    with pytest.raises(AssertionError):
        # Actions must contain options for this method to be used.
        utils.num_options_in_action_sequence(actions)
    parameterized_option = ParameterizedOption("Move", [], Box(0, 1, (1, )),
                                               None, None, None)
    option1 = parameterized_option.ground([], [0.1])
    option2 = parameterized_option.ground([], [0.2])
    option3 = parameterized_option.ground([], [0.3])
    for options, expected_num in (([option1, option1,
                                    option1], 1), ([option1, option2,
                                                    option2], 2),
                                  ([option1, option2,
                                    option1], 3), ([option1, option2,
                                                    option3], 3)):
        actions = [Action(np.array([0]), options[i]) for i in range(3)]
        assert utils.num_options_in_action_sequence(actions) == expected_num


def test_entropy():
    """Tests for entropy()."""
    assert np.allclose(utils.entropy(0.0), 0.0)
    assert np.allclose(utils.entropy(1.0), 0.0)
    assert np.allclose(utils.entropy(0.5), 1.0, atol=0.001)


def test_create_state_from_dict():
    """Tests for create_state_from_dict()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    data = {cup: {"feat1": 0.3}, plate: {"feat1": 0.6}}
    with pytest.raises(KeyError):  # missing feat2
        utils.create_state_from_dict(data)
    data1 = {cup: {"feat1": 0.3}, plate: {"feat1": 0.6, "feat2": 1.3}}
    state1 = utils.create_state_from_dict(data1)
    assert np.allclose(state1[cup], np.array([0.3]))
    assert np.allclose(state1[plate], np.array([0.6, 1.3]))
    data2 = {
        cup: {
            "feat1": 0.3,
            "dummy": 1.3
        },
        plate: {
            "feat1": 0.6,
            "feat2": 1.3
        }
    }
    state2 = utils.create_state_from_dict(data2)
    assert state1.allclose(state2)
    state3 = utils.create_state_from_dict(data2, "dummy_sim_state")
    assert state3.simulator_state == "dummy_sim_state"
    with pytest.raises(NotImplementedError):  # allclose not allowed
        state2.allclose(state3)


def test_line_segment():
    """Tests for LineSegment()."""
    _, ax = plt.subplots(1, 1)
    ax.set_xlim((-5, 5))
    ax.set_ylim((-8, 8))

    seg1 = utils.LineSegment(x1=0, y1=1, x2=3, y2=7)
    assert seg1.x1 == 0
    assert seg1.y1 == 1
    assert seg1.x2 == 3
    assert seg1.y2 == 7
    seg1.plot(ax, color="red", linewidth=2)
    assert seg1.contains_point(2, 5)
    assert not seg1.contains_point(2.1, 5)
    assert not seg1.contains_point(2, 4.9)

    seg2 = utils.LineSegment(x1=2, y1=-5, x2=1, y2=6)
    seg2.plot(ax, color="blue", linewidth=2)

    seg3 = utils.LineSegment(x1=-2, y1=-3, x2=-4, y2=2)
    seg3.plot(ax, color="green", linewidth=2)

    assert utils.geom2ds_intersect(seg1, seg2)
    assert not utils.geom2ds_intersect(seg1, seg3)
    assert not utils.geom2ds_intersect(seg2, seg3)

    # Uncomment for debugging.
    # plt.savefig("/tmp/line_segment_unit_test.png")

    # Legacy tests.
    seg1 = utils.LineSegment(2, 5, 7, 6)
    seg2 = utils.LineSegment(2.5, 7.1, 7.4, 5.3)
    assert utils.geom2ds_intersect(seg1, seg2)

    seg1 = utils.LineSegment(1, 3, 5, 3)
    seg2 = utils.LineSegment(3, 7, 3, 2)
    assert utils.geom2ds_intersect(seg1, seg2)

    seg1 = utils.LineSegment(2, 5, 7, 6)
    seg2 = utils.LineSegment(2, 6, 7, 7)
    assert not utils.geom2ds_intersect(seg1, seg2)

    seg1 = utils.LineSegment(1, 1, 3, 3)
    seg2 = utils.LineSegment(2, 2, 4, 4)
    assert not utils.geom2ds_intersect(seg1, seg2)

    seg1 = utils.LineSegment(1, 1, 3, 3)
    seg2 = utils.LineSegment(1, 1, 6.7, 7.4)
    assert not utils.geom2ds_intersect(seg1, seg2)


def test_circle():
    """Tests for Circle()."""
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim((-11, 5))
    ax.set_ylim((-6, 10))

    circ1 = utils.Circle(x=0, y=1, radius=3)
    assert circ1.x == 0
    assert circ1.y == 1
    assert circ1.radius == 3
    circ1.plot(ax, color="red", alpha=0.5)

    assert circ1.contains_point(0, 1)
    assert circ1.contains_point(0.5, 1)
    assert circ1.contains_point(0, 0.5)
    assert circ1.contains_point(0.25, 1.25)
    assert not circ1.contains_point(0, 4.1)
    assert not circ1.contains_point(3.1, 0)
    assert not circ1.contains_point(0, -2.1)
    assert not circ1.contains_point(-3.1, 0)

    circ2 = utils.Circle(x=-3, y=2, radius=6)
    circ2.plot(ax, color="blue", alpha=0.5)

    circ3 = utils.Circle(x=-6, y=1, radius=1)
    circ3.plot(ax, color="green", alpha=0.5)

    assert utils.geom2ds_intersect(circ1, circ2)
    assert not utils.geom2ds_intersect(circ1, circ3)
    assert utils.geom2ds_intersect(circ2, circ3)

    # Uncomment for debugging.
    # plt.savefig("/tmp/circle_unit_test.png")


def test_triangle():
    """Tests for Triangle()."""
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim((-10.0, 10.0))
    ax.set_ylim((-10.0, 10.0))

    tri1 = utils.Triangle(5.0, 5.0, 7.5, 7.5, 5.0, 7.5)
    assert tri1.contains_point(5.5, 6)
    assert tri1.contains_point(5.9999, 6)
    assert tri1.contains_point(5.8333, 6.6667)
    assert tri1.contains_point(7.3, 7.4)
    assert not tri1.contains_point(6, 6)
    assert not tri1.contains_point(5.1, 5.1)
    assert not tri1.contains_point(5.2, 5.1)
    assert not tri1.contains_point(5.1, 7.6)
    assert not tri1.contains_point(4.9, 7.3)
    assert not tri1.contains_point(5.0, 7.5)
    assert not tri1.contains_point(7.6, 7.6)
    tri1.plot(ax, color="red", alpha=0.5)

    tri2 = utils.Triangle(-3.0, -4.0, -6.2, -5.6, -9.0, -1.7)
    tri2.plot(ax, color="blue", alpha=0.5)

    # Almost degenerate triangle.
    tri3 = utils.Triangle(0.0, 0.0, 1.0, 1.0, -1.0, -1.001)
    assert tri3.contains_point(0.0, -0.001 / 3.0)
    tri3.plot(ax, color="green", alpha=0.5)

    # Degenerate triangle (a line).
    with pytest.raises(ValueError) as e:
        utils.Triangle(0.0, 0.0, 1.0, 1.0, -1.0, -1.0)
    assert "Degenerate triangle" in str(e)

    # Uncomment for debugging.
    # plt.savefig("/tmp/triangle_unit_test.png")


def test_rectangle():
    """Tests for Rectangle()."""
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))

    rect1 = utils.Rectangle(x=-2, y=-1, width=4, height=3, theta=0)
    assert rect1.x == -2
    assert rect1.y == -1
    assert rect1.width == 4
    assert rect1.height == 3
    assert rect1.theta == 0
    rect1.plot(ax, color="red", alpha=0.5)

    assert np.allclose(rect1.center, (0, 0.5))

    circ1 = rect1.circumscribed_circle
    assert np.allclose((circ1.x, circ1.y), (0, 0.5))
    assert np.allclose(circ1.radius, 2.5)
    circ1.plot(ax,
               facecolor="none",
               edgecolor="black",
               linewidth=1,
               linestyle="dashed")

    expected_vertices = np.array([(-2, -1), (-2, 2), (2, -1), (2, 2)])
    assert np.allclose(sorted(rect1.vertices), expected_vertices)
    for (x, y) in rect1.vertices:
        v = utils.Circle(x, y, radius=0.1)
        v.plot(ax,
               facecolor="none",
               edgecolor="black",
               linewidth=1,
               linestyle="dashed")

    for seg in rect1.line_segments:
        seg.plot(ax, color="black", linewidth=1, linestyle="dashed")

    assert not rect1.contains_point(-2.1, 0)
    assert rect1.contains_point(-1.9, 0)
    assert not rect1.contains_point(0, 2.1)
    assert rect1.contains_point(0, 1.9)
    assert not rect1.contains_point(2.1, 0)
    assert rect1.contains_point(1.9, 0)
    assert not rect1.contains_point(0, -1.1)
    assert rect1.contains_point(0, -0.9)
    assert rect1.contains_point(0, 0.5)
    assert not rect1.contains_point(100, 100)

    rect2 = utils.Rectangle(x=1, y=-2, width=2, height=2, theta=0.5)
    rect2.plot(ax, color="blue", alpha=0.5)

    rect3 = utils.Rectangle(x=-1.5, y=1, width=1, height=1, theta=-0.5)
    rect3.plot(ax, color="green", alpha=0.5)

    assert utils.geom2ds_intersect(rect1, rect2)
    assert utils.geom2ds_intersect(rect1, rect3)
    assert utils.geom2ds_intersect(rect3, rect1)
    assert not utils.geom2ds_intersect(rect2, rect3)

    rect4 = utils.Rectangle(x=0.8, y=1e-5, height=0.1, width=0.07, theta=0)
    assert not rect4.contains_point(0.2, 0.05)

    rect5 = utils.Rectangle(x=-4, y=-2, height=0.25, width=2, theta=-np.pi / 4)
    rect5.plot(ax, facecolor="yellow", edgecolor="gray")
    origin = utils.Circle(x=-3.5, y=-2.3, radius=0.05)
    origin.plot(ax, color="black")
    rect6 = rect5.rotate_about_point(origin.x, origin.y, rot=np.pi / 4)
    rect6.plot(ax, facecolor="none", edgecolor="black", linestyle="dashed")

    # Uncomment for debugging.
    # plt.savefig("/tmp/rectangle_unit_test.png")


def test_line_segment_circle_intersection():
    """Tests for line_segment_intersects_circle()."""
    seg1 = utils.LineSegment(-3, 0, 0, 0)
    circ1 = utils.Circle(0, 0, 1)
    assert utils.geom2ds_intersect(seg1, circ1)
    assert utils.geom2ds_intersect(circ1, seg1)

    seg2 = utils.LineSegment(-3, 3, 4, 3)
    assert not utils.geom2ds_intersect(seg2, circ1)
    assert not utils.geom2ds_intersect(circ1, seg2)

    seg3 = utils.LineSegment(0, -2, 1, -2.5)
    assert not utils.geom2ds_intersect(seg3, circ1)
    assert not utils.geom2ds_intersect(circ1, seg3)

    seg4 = utils.LineSegment(0, -3, 0, -4)
    assert not utils.geom2ds_intersect(seg4, circ1)
    assert not utils.geom2ds_intersect(circ1, seg4)

    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))
    assert not utils.line_segment_intersects_circle(seg2, circ1, ax=ax)

    # Uncomment for debugging.
    # plt.savefig("/tmp/line_segment_circle_unit_test.png")


def test_line_segment_rectangle_intersection():
    """Tests for line_segment_intersects_rectangle()."""
    seg1 = utils.LineSegment(-3, 0, 0, 0)
    rect1 = utils.Rectangle(-1, -1, 2, 2, 0)
    assert utils.geom2ds_intersect(seg1, rect1)
    assert utils.geom2ds_intersect(rect1, seg1)

    seg2 = utils.LineSegment(-3, 3, 4, 3)
    assert not utils.geom2ds_intersect(seg2, rect1)
    assert not utils.geom2ds_intersect(rect1, seg2)

    seg3 = utils.LineSegment(0, -2, 1, -2.5)
    assert not utils.geom2ds_intersect(seg3, rect1)
    assert not utils.geom2ds_intersect(rect1, seg3)

    seg4 = utils.LineSegment(0, -3, 0, -4)
    assert not utils.geom2ds_intersect(seg4, rect1)
    assert not utils.geom2ds_intersect(rect1, seg4)


def test_rectangle_circle_intersection():
    """Tests for rectangle_intersects_circle()."""
    rect1 = utils.Rectangle(x=0, y=0, width=4, height=3, theta=0)
    circ1 = utils.Circle(x=0, y=0, radius=1)
    assert utils.geom2ds_intersect(rect1, circ1)
    assert utils.geom2ds_intersect(circ1, rect1)

    circ2 = utils.Circle(x=1, y=1, radius=0.5)
    assert utils.geom2ds_intersect(rect1, circ2)
    assert utils.geom2ds_intersect(circ2, rect1)

    rect2 = utils.Rectangle(x=1, y=1, width=1, height=1, theta=0)
    assert not utils.geom2ds_intersect(rect2, circ1)
    assert not utils.geom2ds_intersect(circ1, rect2)

    circ3 = utils.Circle(x=0, y=0, radius=100)
    assert utils.geom2ds_intersect(rect1, circ3)
    assert utils.geom2ds_intersect(circ3, rect1)
    assert utils.geom2ds_intersect(rect2, circ3)
    assert utils.geom2ds_intersect(circ3, rect2)


def test_geom2ds_intersect():
    """Tests for geom2ds_intersect()."""
    with pytest.raises(NotImplementedError):
        utils.geom2ds_intersect(None, None)


def test_get_static_preds():
    """Tests for get_static_preds()."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    static_preds = utils.get_static_preds(nsrts, env.predicates)
    assert {pred.name for pred in static_preds} == {"IsTarget", "IsBlock"}


def test_get_static_atoms():
    """Tests for get_static_atoms()."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    task = env.get_train_tasks()[0]
    objects = set(task.init)
    ground_nsrts = set()
    for nsrt in nsrts:
        ground_nsrts |= set(utils.all_ground_nsrts(nsrt, objects))
    atoms = utils.abstract(task.init, env.predicates) | task.goal
    num_blocks = sum(1 for obj in objects if obj.type.name == "block")
    num_targets = sum(1 for obj in objects if obj.type.name == "target")
    assert len(atoms) > num_blocks + num_targets
    static_atoms = utils.get_static_atoms(ground_nsrts, atoms)
    # IsBlock for every block, IsTarget for every target
    assert len(static_atoms) == num_blocks + num_targets
    # Now remove the ground NSRT for covering target0 with block0.
    nsrts_to_remove = {
        nsrt
        for nsrt in ground_nsrts if nsrt.name == "Place"
        and [obj.name for obj in nsrt.objects] == ["block0", "target0"]
    }
    assert len(nsrts_to_remove) == 1
    ground_nsrts.remove(nsrts_to_remove.pop())
    # This removal should make Covers(block0, target0) be static.
    new_static_atoms = utils.get_static_atoms(ground_nsrts, atoms)
    assert len(new_static_atoms) == len(static_atoms) + 1
    assert not static_atoms - new_static_atoms  # nothing should be deleted
    added_atom = (new_static_atoms - static_atoms).pop()
    assert added_atom.predicate.name == "Covers"
    assert [obj.name for obj in added_atom.objects] == ["block0", "target0"]


def test_run_policy():
    """Tests for run_policy()."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    policy = lambda _: Action(env.action_space.sample())
    task = env.get_task("test", 0)
    traj, metrics = utils.run_policy(policy,
                                     env,
                                     "test",
                                     0,
                                     task.goal_holds,
                                     max_num_steps=5)
    assert not task.goal_holds(traj.states[-1])
    assert len(traj.states) == 6
    assert len(traj.actions) == 5
    assert "policy_call_time" in metrics
    assert metrics["policy_call_time"] > 0.0
    traj2, _ = utils.run_policy(policy,
                                env,
                                "test",
                                0,
                                lambda s: True,
                                max_num_steps=5)
    assert not task.goal_holds(traj2.states[-1])
    assert len(traj2.states) == 1
    assert len(traj2.actions) == 0
    executed = False

    def _onestep_terminal(_):
        nonlocal executed
        terminate = executed
        executed = True
        return terminate

    traj3, _ = utils.run_policy(policy,
                                env,
                                "test",
                                0,
                                _onestep_terminal,
                                max_num_steps=5)
    assert not task.goal_holds(traj3.states[-1])
    assert len(traj3.states) == 2
    assert len(traj3.actions) == 1

    # Test exceptions_to_break_on.
    def _policy(_):
        raise ValueError("mock error")

    class _CountingMonitor(utils.Monitor):

        def __init__(self):
            self.num_observations = 0

        def observe(self, state, action):
            self.num_observations += 1

    with pytest.raises(ValueError) as e:
        utils.run_policy(_policy,
                         env,
                         "test",
                         0,
                         task.goal_holds,
                         max_num_steps=5)
    assert "mock error" in str(e)
    monitor = _CountingMonitor()
    traj4, _ = utils.run_policy(_policy,
                                env,
                                "test",
                                0,
                                task.goal_holds,
                                max_num_steps=5,
                                exceptions_to_break_on={ValueError},
                                monitor=monitor)
    assert len(traj4.states) == 1
    assert monitor.num_observations == 1

    class _MockEnv:

        @staticmethod
        def reset(train_or_test, task_idx):
            """Reset the mock environment."""
            del train_or_test, task_idx  # unused
            return DefaultState

        @staticmethod
        def step(action):
            """Step the mock environment."""
            del action  # unused
            raise utils.EnvironmentFailure("mock failure")

    mock_env = _MockEnv()
    policy = lambda _: Action(np.zeros(1, dtype=np.float32))
    monitor = _CountingMonitor()
    traj5, _ = utils.run_policy(
        policy,
        mock_env,
        "test",
        0,
        lambda s: False,
        max_num_steps=5,
        exceptions_to_break_on={utils.EnvironmentFailure},
        monitor=monitor)
    assert len(traj5.states) == 1
    assert len(traj5.actions) == 0
    assert monitor.num_observations == 1

    # Test policy call time.
    def _policy(_):
        time.sleep(0.1)
        return Action(env.action_space.sample())

    traj, metrics = utils.run_policy(_policy,
                                     env,
                                     "test",
                                     0,
                                     task.goal_holds,
                                     max_num_steps=3)
    assert metrics["policy_call_time"] >= 3 * 0.1

    # Test with monitor in case where an uncaught exception is raised.

    def _policy(_):
        raise ValueError("mock error")

    monitor = _CountingMonitor()
    try:
        utils.run_policy(_policy,
                         env,
                         "test",
                         0,
                         task.goal_holds,
                         max_num_steps=3,
                         monitor=monitor)
    except ValueError:
        pass
    assert monitor.num_observations == 1


def test_run_policy_with_simulator():
    """Tests for run_policy_with_simulator()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0, 1.2]})

    def _simulator(s, a):
        ns = s.copy()
        assert a.arr.shape == (1, )
        ns[cup][0] += a.arr.item()
        return ns

    def _policy(_):
        return Action(np.array([4]))

    traj = utils.run_policy_with_simulator(_policy,
                                           _simulator,
                                           state,
                                           lambda s: True,
                                           max_num_steps=5)
    assert len(traj.states) == 1
    assert len(traj.actions) == 0

    traj = utils.run_policy_with_simulator(_policy,
                                           _simulator,
                                           state,
                                           lambda s: False,
                                           max_num_steps=5)
    assert len(traj.states) == 6
    assert len(traj.actions) == 5

    def _terminal(s):
        return s[cup][0] > 9.9

    traj = utils.run_policy_with_simulator(_policy,
                                           _simulator,
                                           state,
                                           _terminal,
                                           max_num_steps=5)
    assert len(traj.states) == 4
    assert len(traj.actions) == 3

    # Test with monitor.
    class _NullMonitor(utils.Monitor):

        def observe(self, state, action):
            pass

    monitor = _NullMonitor()
    traj = utils.run_policy_with_simulator(_policy,
                                           _simulator,
                                           state,
                                           _terminal,
                                           max_num_steps=5,
                                           monitor=monitor)
    assert len(traj.states) == 4
    assert len(traj.actions) == 3

    # Test with monitor in case where an uncaught exception is raised.
    class _CountingMonitor(utils.Monitor):

        def __init__(self):
            self.num_observations = 0

        def observe(self, state, action):
            self.num_observations += 1

    def _policy(_):
        raise ValueError("mock error")

    monitor = _CountingMonitor()
    try:
        utils.run_policy_with_simulator(_policy,
                                        _simulator,
                                        state,
                                        _terminal,
                                        max_num_steps=5,
                                        monitor=monitor)
    except ValueError:
        pass
    assert monitor.num_observations == 1

    # Test exceptions_to_break_on.
    def _policy(_):
        raise ValueError("mock error")

    with pytest.raises(ValueError) as e:
        utils.run_policy_with_simulator(_policy,
                                        _simulator,
                                        state,
                                        _terminal,
                                        max_num_steps=5)
    assert "mock error" in str(e)
    monitor = _CountingMonitor()
    traj = utils.run_policy_with_simulator(_policy,
                                           _simulator,
                                           state,
                                           _terminal,
                                           max_num_steps=5,
                                           exceptions_to_break_on={ValueError},
                                           monitor=monitor)
    assert len(traj.states) == 1
    assert monitor.num_observations == 1

    def _simulator(state, action):
        raise utils.EnvironmentFailure("mock failure")

    _policy = lambda _: Action(np.zeros(1, dtype=np.float32))
    monitor = _CountingMonitor()
    traj = utils.run_policy_with_simulator(
        _policy,
        _simulator,
        state,
        _terminal,
        max_num_steps=5,
        exceptions_to_break_on={utils.EnvironmentFailure},
        monitor=monitor)
    assert len(traj.states) == 1
    assert len(traj.actions) == 0
    assert monitor.num_observations == 1


def test_option_plan_to_policy():
    """Tests for option_plan_to_policy()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0, 1.2]})

    def _simulator(s, a):
        ns = s.copy()
        assert a.arr.shape == (1, )
        ns[cup][0] += a.arr.item()
        return ns

    params_space = Box(0, 1, (1, ))

    def policy(_1, _2, _3, p):
        return Action(p)

    def initiable(s, _2, _3, p):
        return p > 0.25 and s[cup][0] < 1

    def terminal(s, _1, _2, _3):
        return s[cup][0] > 9.9

    parameterized_option = ParameterizedOption("Move", [], params_space,
                                               policy, initiable, terminal)
    params = [0.1]
    option = parameterized_option.ground([], params)
    plan = [option]
    policy = utils.option_plan_to_policy(plan)
    with pytest.raises(AssertionError):
        # option is not initiable from start state
        policy(state)
    params = [0.5]
    option = parameterized_option.ground([], params)
    plan = [option]
    policy = utils.option_plan_to_policy(plan)
    assert option.initiable(state)
    traj = utils.run_policy_with_simulator(option.policy,
                                           _simulator,
                                           state,
                                           option.terminal,
                                           max_num_steps=100)
    assert len(traj.actions) == len(traj.states) - 1 == 19
    for t in range(19):
        assert not option.terminal(state)
        assert state.allclose(traj.states[t])
        action = policy(state)
        assert np.allclose(action.arr, traj.actions[t].arr)
        state = _simulator(state, action)
    assert option.terminal(state)
    with pytest.raises(utils.OptionExecutionFailure):
        # Ran out of options
        policy(state)


def test_action_arrs_to_policy():
    """Tests for action_arrs_to_policy()."""
    action_arrs = [
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    ]

    state = DefaultState
    policy = utils.action_arrs_to_policy(action_arrs)
    action = policy(state)
    assert isinstance(action, Action)
    assert np.allclose(action.arr, action_arrs[0])
    action = policy(state)
    assert np.allclose(action.arr, action_arrs[1])
    action = policy(state)
    assert np.allclose(action.arr, action_arrs[2])
    with pytest.raises(IndexError):
        # Ran out of actions.
        policy(state)
    # Original list should not be modified.
    assert len(action_arrs) == 3


def test_strip_predicate():
    """Test for strip_predicate()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])

    def _classifier1(state, objects):
        cup, plate = objects
        return state[cup][0] + state[plate][0] < 2

    pred = Predicate("On", [cup_type, plate_type], _classifier1)
    cup = cup_type("cup")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup: [0.5], plate1: [1.0, 1.2], plate2: [-9.0, 1.0]})
    pred_stripped = utils.strip_predicate(pred)
    assert pred.name == pred_stripped.name
    assert pred.types == pred_stripped.types
    assert pred.holds(state, (cup, plate1))
    assert pred.holds(state, (cup, plate2))
    with pytest.raises(Exception) as e:
        pred_stripped.holds(state, (cup, plate1))
    assert "Stripped classifier should never be called" in str(e)
    with pytest.raises(Exception) as e:
        pred_stripped.holds(state, (cup, plate2))
    assert "Stripped classifier should never be called" in str(e)


def test_strip_task():
    """Test for strip_task()."""
    env = CoverEnv()
    Covers, Holding = _get_predicates_by_names("cover", ["Covers", "Holding"])
    task = env.get_train_tasks()[0]
    block0, _, _, target0, _ = list(task.init)
    # Goal is Covers(block0, target0)
    assert len(task.goal) == 1
    original_goal_atom = next(iter(task.goal))
    state = task.init.copy()
    state.set(block0, "pose", state.get(target0, "pose"))
    assert original_goal_atom.holds(state)
    # Include both Covers and Holding (don't strip them).
    stripped_task1 = utils.strip_task(task, {Covers, Holding})
    assert len(stripped_task1.goal) == 1
    new_goal_atom1 = next(iter(stripped_task1.goal))
    assert new_goal_atom1.holds(state)
    # Include Holding, but strip Covers.
    stripped_task2 = utils.strip_task(task, {Holding})
    assert len(stripped_task2.goal) == 1
    new_goal_atom2 = next(iter(stripped_task2.goal))
    with pytest.raises(Exception) as e:
        new_goal_atom2.holds(state)
    assert "Stripped classifier should never be called" in str(e)


def test_sample_subsets():
    """Tests for sample_subsets()."""
    universe = list(range(10))
    num_samples = 5
    min_set_size = 1
    max_set_size = 2
    rng = np.random.default_rng(0)
    samples = list(
        utils.sample_subsets(universe, num_samples, min_set_size, max_set_size,
                             rng))
    assert len(samples) == 5
    assert {len(s) for s in samples} == {1, 2}
    assert all(s.issubset(set(universe)) for s in samples)
    assert not list(
        utils.sample_subsets(universe, 0, min_set_size, max_set_size, rng))
    assert list(utils.sample_subsets(
        [], num_samples, 0, 0, rng)) == [set() for _ in range(num_samples)]
    with pytest.raises(AssertionError):
        next(utils.sample_subsets(universe, num_samples, min_set_size, 0, rng))
    with pytest.raises(AssertionError):
        next(
            utils.sample_subsets([], num_samples, min_set_size, max_set_size,
                                 rng))


def test_abstract():
    """Tests for abstract() and wrap_atom_predicates()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])

    def _classifier1(state, objects):
        cup, plate = objects
        return state[cup][0] + state[plate][0] < 2

    pred1 = Predicate("On", [cup_type, plate_type], _classifier1)
    wrapped_pred1 = utils.wrap_predicate(pred1, "TEST-PREFIX-")
    assert wrapped_pred1.name == "TEST-PREFIX-On"
    assert wrapped_pred1.types == pred1.types

    def _classifier2(state, objects):
        cup, _, plate = objects
        return state[cup][0] + state[plate][0] < -1

    pred2 = Predicate("Is", [cup_type, plate_type, plate_type], _classifier2)
    wrapped_pred2 = utils.wrap_predicate(pred2, "TEST-PREFIX-")
    assert wrapped_pred2.name == "TEST-PREFIX-Is"
    assert wrapped_pred2.types == pred2.types
    cup = cup_type("cup")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup: [0.5], plate1: [1.0, 1.2], plate2: [-9.0, 1.0]})
    atoms = utils.abstract(state, {pred1, pred2})
    wrapped = utils.wrap_atom_predicates(atoms, "TEST-PREFIX-G-")
    assert len(wrapped) == len(atoms)
    for atom in wrapped:
        assert atom.predicate.name.startswith("TEST-PREFIX-G-")
    lifted_atoms = {pred1([cup_type("?cup"), plate_type("?plate")])}
    wrapped = utils.wrap_atom_predicates(lifted_atoms, "TEST-PREFIX-L-")
    assert len(wrapped) == len(lifted_atoms)
    for atom in wrapped:
        assert atom.predicate.name.startswith("TEST-PREFIX-L-")
    assert len(atoms) == 4
    assert atoms == {
        pred1([cup, plate1]),
        pred1([cup, plate2]),
        pred2([cup, plate1, plate2]),
        pred2([cup, plate2, plate2])
    }
    # Wrapping a predicate should destroy its classifier.
    assert not utils.abstract(state, {wrapped_pred1, wrapped_pred2})


def test_create_new_variables():
    """Tests for create_new_variables()."""
    cup_type = Type("cup", ["feat1"])
    plate_type = Type("plate", ["feat1"])
    vs = utils.create_new_variables([cup_type, cup_type, plate_type])
    assert vs == [
        Variable("?x0", cup_type),
        Variable("?x1", cup_type),
        Variable("?x2", plate_type)
    ]
    existing_vars = {Variable("?x0", cup_type), Variable("?x5", cup_type)}
    vs = utils.create_new_variables([plate_type], existing_vars=existing_vars)
    assert vs == [Variable("?x6", plate_type)]
    existing_vars = {Variable("?x", cup_type), Variable("?xerox", cup_type)}
    vs = utils.create_new_variables([plate_type], existing_vars=existing_vars)
    assert vs == [Variable("?x0", plate_type)]
    existing_vars = {Variable("?x5", cup_type)}
    vs = utils.create_new_variables([plate_type],
                                    existing_vars=existing_vars,
                                    var_prefix="?llama")
    assert vs == [Variable("?llama0", plate_type)]


def test_unify_lifted_to_ground():
    """Tests for unify() when lifted atoms are the first argument and ground
    atoms are the second argument."""
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    var0 = cup_type("?var0")
    var1 = cup_type("?var1")
    var2 = cup_type("?var2")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: True)
    pred1 = Predicate("Pred1", [cup_type, cup_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: True)

    kb0 = frozenset({pred0([cup0])})
    q0 = frozenset({pred0([var0])})
    found, assignment = utils.unify(kb0, q0)
    assert found
    assert assignment == {cup0: var0}

    q1 = frozenset({pred0([var0]), pred0([var1])})
    found, assignment = utils.unify(kb0, q1)
    assert not found
    assert assignment == {}

    kb1 = frozenset({pred0([cup0]), pred0([cup1])})
    found, assignment = utils.unify(kb1, q0)
    assert not found  # different number of predicates/objects
    assert assignment == {}

    kb2 = frozenset({pred0([cup0]), pred2([cup2])})
    q2 = frozenset({pred0([var0]), pred2([var2])})
    found, assignment = utils.unify(kb2, q2)
    assert found
    assert assignment == {cup0: var0, cup2: var2}

    kb3 = frozenset({pred0([cup0])})
    q3 = frozenset({pred0([var0]), pred2([var2])})
    found, assignment = utils.unify(kb3, q3)
    assert not found
    assert assignment == {}

    kb4 = frozenset({pred1([cup0, cup1]), pred1([cup1, cup2])})
    q4 = frozenset({pred1([var0, var1])})
    found, assignment = utils.unify(kb4, q4)
    assert not found  # different number of predicates
    assert assignment == {}

    kb5 = frozenset({pred0([cup2]), pred1([cup0, cup1]), pred1([cup1, cup2])})
    q5 = frozenset({pred1([var0, var1]), pred0([var1]), pred0([var0])})
    found, assignment = utils.unify(kb5, q5)
    assert not found
    assert assignment == {}

    kb6 = frozenset({
        pred0([cup0]),
        pred2([cup1]),
        pred1([cup0, cup2]),
        pred1([cup2, cup1])
    })
    q6 = frozenset({pred0([var0]), pred2([var1]), pred1([var0, var1])})
    found, assignment = utils.unify(kb6, q6)
    assert not found
    assert assignment == {}

    kb7 = frozenset({pred0([cup0]), pred2([cup1])})
    q7 = frozenset({pred0([var0]), pred2([var0])})
    found, assignment = utils.unify(kb7, q7)
    assert not found  # different number of objects
    assert assignment == {}

    kb8 = frozenset({pred0([cup0]), pred2([cup0])})
    q8 = frozenset({pred0([var0]), pred2([var0])})
    found, assignment = utils.unify(kb8, q8)
    assert found
    assert assignment == {cup0: var0}

    kb9 = frozenset({pred1([cup0, cup1]), pred1([cup1, cup2]), pred2([cup0])})
    q9 = frozenset({pred1([var0, var1]), pred1([var2, var0]), pred2([var0])})
    found, assignment = utils.unify(kb9, q9)
    assert not found
    assert assignment == {}


def test_unify_other_liftedground_combinations():
    """Tests for unify() with other combinations of ground/lifted atoms."""
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup2 = cup_type("cup2")
    var0 = cup_type("?var0")
    var1 = cup_type("?var1")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: True)

    kb0 = frozenset({pred0([var0])})
    q0 = frozenset({pred0([cup0])})
    found, assignment = utils.unify(kb0, q0)
    assert found
    assert assignment == {var0: cup0}

    kb1 = frozenset({pred0([var0])})
    q1 = frozenset({pred0([var1])})
    found, assignment = utils.unify(kb1, q1)
    assert found
    assert assignment == {var0: var1}

    kb2 = frozenset({pred0([cup0])})
    q2 = frozenset({pred0([cup2])})
    found, assignment = utils.unify(kb2, q2)
    assert found
    assert assignment == {cup0: cup2}


def test_unify_preconds_effects_options():
    """Tests for unify_preconds_effects_options()."""
    # The following test checks edge cases of unification with respect to
    # the split between effects and option variables.
    # The case is basically this:
    # Add set 1: P(a, b)
    # Option 1: A(b, c)
    # Add set 2: P(w, x)
    # Option 2: A(y, z)
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    w = cup_type("?w")
    x = cup_type("?x")
    y = cup_type("?y")
    z = cup_type("?z")
    pred0 = Predicate("Pred0", [cup_type, cup_type], lambda s, o: False)
    param_option0 = ParameterizedOption("dummy0", [cup_type],
                                        Box(0.1, 1, (1, )),
                                        lambda s, m, o, p: Action(p),
                                        lambda s, m, o, p: False,
                                        lambda s, m, o, p: False)
    # Option0(cup0, cup1)
    ground_option_args = (cup0, cup1)
    # Pred0(cup1, cup2) true
    ground_add_effects = frozenset({pred0([cup1, cup2])})
    ground_delete_effects = frozenset()
    # Option0(w, x)
    lifted_option_args = (w, x)
    # Pred0(y, z) True
    lifted_add_effects = frozenset({pred0([y, z])})
    lifted_delete_effects = frozenset()
    suc, sub = utils.unify_preconds_effects_options(
        frozenset(), frozenset(), ground_add_effects, lifted_add_effects,
        ground_delete_effects, lifted_delete_effects, param_option0,
        param_option0, ground_option_args, lifted_option_args)
    assert not suc
    assert not sub
    # The following test is for an edge case where everything is identical
    # except for the name of the parameterized option. We do not want to
    # unify in this case.
    # First, a unify that should succeed.
    suc, sub = utils.unify_preconds_effects_options(frozenset(), frozenset(),
                                                    frozenset(), frozenset(),
                                                    frozenset(), frozenset(),
                                                    param_option0,
                                                    param_option0,
                                                    (cup0, cup1), (cup0, cup1))
    assert suc
    assert sub == {cup0: cup0, cup1: cup1}
    # Now, a unify that should fail because of different parameterized options.
    param_option1 = ParameterizedOption("dummy1", [cup_type],
                                        Box(0.1, 1, (1, )),
                                        lambda s, m, o, p: Action(p),
                                        lambda s, m, o, p: False,
                                        lambda s, m, o, p: False)
    suc, sub = utils.unify_preconds_effects_options(frozenset(), frozenset(),
                                                    frozenset(), frozenset(),
                                                    frozenset(), frozenset(),
                                                    param_option0,
                                                    param_option1,
                                                    (cup0, cup1), (cup0, cup1))
    assert not suc
    assert not sub


def test_get_random_object_combination():
    """Tests for get_random_object_combination()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat2"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate0 = plate_type("plate0")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    rng = np.random.default_rng(123)
    objs = utils.get_random_object_combination({cup0, cup1, cup2},
                                               [cup_type, cup_type], rng)
    assert all(obj.type == cup_type for obj in objs)
    objs = utils.get_random_object_combination(
        {cup0, cup1, cup2, plate0, plate1, plate2}, [cup_type, plate_type],
        rng)
    assert [obj.type for obj in objs] == [cup_type, plate_type]
    objs = utils.get_random_object_combination({cup0},
                                               [cup_type, cup_type, cup_type],
                                               rng)
    assert len(objs) == 3
    assert len(set(objs)) == 1
    objs = utils.get_random_object_combination({cup0}, [plate_type], rng)
    assert objs is None  # no object of type plate


def test_get_entity_combinations():
    """Tests for get_object_combinations() and get_variable_combinations()."""
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    cup_var0 = cup_type("?cup0")
    cup_var1 = cup_type("?cup1")
    cup_var2 = cup_type("?cup2")
    plate_type = Type("plate_type", ["feat1"])
    plate0 = plate_type("plate0")
    plate1 = plate_type("plate1")
    plate_var0 = plate_type("?plate0")
    plate_var1 = plate_type("?plate1")

    objects = {cup0, cup1, cup2, plate0, plate1}
    types = [cup_type, plate_type]
    assert list(utils.get_object_combinations(objects, types)) == \
        [[cup0, plate0], [cup0, plate1],
         [cup1, plate0], [cup1, plate1],
         [cup2, plate0], [cup2, plate1]]

    objects = {cup0, cup2}
    types = [cup_type, cup_type]
    assert list(utils.get_object_combinations(objects, types)) == \
        [[cup0, cup0], [cup0, cup2],
         [cup2, cup0], [cup2, cup2]]

    variables = {cup_var0, cup_var1, cup_var2, plate_var0, plate_var1}
    types = [cup_type, plate_type]
    assert list(utils.get_variable_combinations(variables, types)) == \
        [[cup_var0, plate_var0], [cup_var0, plate_var1],
         [cup_var1, plate_var0], [cup_var1, plate_var1],
         [cup_var2, plate_var0], [cup_var2, plate_var1]]

    variables = {cup_var0, cup_var2}
    types = [cup_type, cup_type]
    assert list(utils.get_variable_combinations(variables, types)) == \
        [[cup_var0, cup_var0], [cup_var0, cup_var2],
         [cup_var2, cup_var0], [cup_var2, cup_var2]]


def test_get_all_atoms_for_predicate():
    """Tests for get_all_ground_atoms_for_predicate() and
    get_all_lifted_atoms_for_predicate()."""
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    cup_var0 = cup_type("?cup0")
    cup_var1 = cup_type("?cup1")
    cup_var2 = cup_type("?cup2")
    plate_type = Type("plate_type", ["feat1"])
    plate0 = plate_type("plate0")
    plate1 = plate_type("plate1")
    plate_var0 = plate_type("?plate0")
    plate_var1 = plate_type("?plate1")

    cup_on_plate = Predicate("CupOnPlate", [cup_type, plate_type],
                             lambda s, o: True)
    cup_on_cup = Predicate("CupOnCup", [cup_type, cup_type], lambda s, o: True)

    objects = frozenset({cup0, cup1, cup2, plate0, plate1})
    pred = cup_on_plate
    assert utils.get_all_ground_atoms_for_predicate(pred, objects) == \
        {cup_on_plate([cup0, plate0]), cup_on_plate([cup0, plate1]),
         cup_on_plate([cup1, plate0]), cup_on_plate([cup1, plate1]),
         cup_on_plate([cup2, plate0]), cup_on_plate([cup2, plate1])}

    objects = frozenset({cup0, cup2})
    pred = cup_on_cup
    assert utils.get_all_ground_atoms_for_predicate(pred, objects) == \
        {cup_on_cup([cup0, cup0]), cup_on_cup([cup0, cup2]),
         cup_on_cup([cup2, cup0]), cup_on_cup([cup2, cup2])}

    variables = frozenset(
        {cup_var0, cup_var1, cup_var2, plate_var0, plate_var1})
    pred = cup_on_plate
    assert utils.get_all_lifted_atoms_for_predicate(pred, variables) == \
        {cup_on_plate([cup_var0, plate_var0]),
         cup_on_plate([cup_var0, plate_var1]),
         cup_on_plate([cup_var1, plate_var0]),
         cup_on_plate([cup_var1, plate_var1]),
         cup_on_plate([cup_var2, plate_var0]),
         cup_on_plate([cup_var2, plate_var1])}

    variables = frozenset({cup_var0, cup_var2})
    pred = cup_on_cup
    assert utils.get_all_lifted_atoms_for_predicate(pred, variables) == \
        {cup_on_cup([cup_var0, cup_var0]), cup_on_cup([cup_var0, cup_var2]),
         cup_on_cup([cup_var2, cup_var0]), cup_on_cup([cup_var2, cup_var2])}


def test_find_substitution():
    """Tests for find_substitution()."""
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    var0 = cup_type("?var0")
    var1 = cup_type("?var1")
    var2 = cup_type("?var2")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: True)
    pred1 = Predicate("Pred1", [cup_type, cup_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: True)

    kb0 = [pred0([cup0])]
    q0 = [pred0([var0])]
    found, assignment = utils.find_substitution(kb0, q0)
    assert found
    assert assignment == {var0: cup0}

    q1 = [pred0([var0]), pred0([var1])]
    found, assignment = utils.find_substitution(kb0, q1)
    assert not found
    assert assignment == {}

    q1 = [pred0([var0]), pred0([var1])]
    found, assignment = utils.find_substitution(kb0, q1, allow_redundant=True)
    assert found
    assert assignment == {var0: cup0, var1: cup0}

    kb1 = [pred0([cup0]), pred0([cup1])]
    found, assignment = utils.find_substitution(kb1, q0)
    assert found
    assert assignment == {var0: cup0}

    kb2 = [pred0([cup0]), pred2([cup2])]
    q2 = [pred0([var0]), pred2([var2])]
    found, assignment = utils.find_substitution(kb2, q2)
    assert found
    assert assignment == {var0: cup0, var2: cup2}

    kb3 = [pred0([cup0])]
    q3 = [pred0([var0]), pred2([var2])]
    found, assignment = utils.find_substitution(kb3, q3)
    assert not found
    assert assignment == {}

    kb4 = [pred1([cup0, cup1]), pred1([cup1, cup2])]
    q4 = [pred1([var0, var1])]
    found, assignment = utils.find_substitution(kb4, q4)
    assert found
    assert assignment == {var0: cup0, var1: cup1}

    kb5 = [pred0([cup2]), pred1([cup0, cup1]), pred1([cup1, cup2])]
    q5 = [pred1([var0, var1]), pred0([var1]), pred0([var0])]
    found, assignment = utils.find_substitution(kb5, q5)
    assert not found
    assert assignment == {}

    kb6 = [
        pred0([cup0]),
        pred2([cup1]),
        pred1([cup0, cup2]),
        pred1([cup2, cup1])
    ]
    q6 = [pred0([var0]), pred2([var1]), pred1([var0, var1])]
    found, assignment = utils.find_substitution(kb6, q6)
    assert not found
    assert assignment == {}

    kb7 = [pred1([cup0, cup0])]
    q7 = [pred1([var0, var0])]
    found, assignment = utils.find_substitution(kb7, q7)
    assert found
    assert assignment == {var0: cup0}

    kb8 = [pred1([cup0, cup0])]
    q8 = [pred1([var0, var1])]
    found, assignment = utils.find_substitution(kb8, q8)
    assert not found
    assert assignment == {}

    found, assignment = utils.find_substitution(kb8, q8, allow_redundant=True)
    assert found
    assert assignment == {var0: cup0, var1: cup0}

    kb9 = [pred1([cup0, cup1])]
    q9 = [pred1([var0, var0])]
    found, assignment = utils.find_substitution(kb9, q9)
    assert not found
    assert assignment == {}

    found, assignment = utils.find_substitution(kb9, q9, allow_redundant=True)
    assert not found
    assert assignment == {}

    kb10 = [pred1([cup0, cup1]), pred1([cup1, cup0])]
    q10 = [pred1([var0, var1]), pred1([var0, var2])]
    found, assignment = utils.find_substitution(kb10, q10)
    assert not found
    assert assignment == {}

    kb11 = [pred1([cup0, cup1]), pred1([cup1, cup0])]
    q11 = [pred1([var0, var1]), pred1([var1, var0])]
    found, assignment = utils.find_substitution(kb11, q11)
    assert found
    assert assignment == {var0: cup0, var1: cup1}

    plate_type = Type("plate_type", ["feat1"])
    plate0 = plate_type("plate0")
    var3 = plate_type("?var3")
    pred4 = Predicate("Pred4", [plate_type], lambda s, o: True)
    pred5 = Predicate("Pred5", [plate_type, cup_type], lambda s, o: True)

    kb12 = [pred4([plate0])]
    q12 = [pred0([var0])]
    found, assignment = utils.find_substitution(kb12, q12)
    assert not found
    assert assignment == {}

    kb13 = [pred4([plate0]), pred5([plate0, cup0])]
    q13 = [pred4([var3]), pred5([var3, var0])]
    found, assignment = utils.find_substitution(kb13, q13)
    assert found
    assert assignment == {var3: plate0, var0: cup0}


def test_SingletonParameterizedOption():
    """Tests for SingletonParameterizedOption()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state1 = State({cup: [0.0], plate: [1.0, 1.2]})
    state2 = State({cup: [1.0], plate: [1.0, 1.2]})

    def _initiable(s, _2, objs, _4):
        return s[objs[0]][0] > 0.0

    def _policy(_1, _2, _3, p):
        return Action(p)

    param_option = utils.SingletonParameterizedOption("Dummy",
                                                      _policy,
                                                      types=[cup_type],
                                                      params_space=Box(
                                                          0, 1, (1, )),
                                                      initiable=_initiable)
    assert param_option.name == "Dummy"
    assert param_option.types == [cup_type]
    assert np.allclose(param_option.params_space.low, np.array([0]))
    assert np.allclose(param_option.params_space.high, np.array([1]))
    option = param_option.ground([cup], [0.5])
    assert not option.initiable(state1)
    option = param_option.ground([cup], [0.5])
    assert option.initiable(state2)
    assert not option.terminal(state2)
    action = option.policy(state2)
    assert np.allclose(action.arr, np.array([0.5]))
    assert option.terminal(state2.copy())


def test_LinearChainParameterizedOption():
    """Tests for LinearChainParameterizedOption()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.0], plate: [1.0, 1.2]})

    # This parameterized option takes the action [-4] twice and terminates.
    def _initiable(_1, m, _3, _4):
        m["num_steps"] = 0
        return True

    def _policy(_1, m, _3, _4):
        m["num_steps"] += 1
        return Action(np.array([-4]))

    def _terminal(_1, m, _3, _4):
        return m["num_steps"] >= 2

    param_option0 = ParameterizedOption("dummy0", [cup_type],
                                        Box(0.1, 1, (1, )), _policy,
                                        _initiable, _terminal)

    # This parameterized option takes the action [2] four times and terminates.
    def _policy(_1, m, _3, _4):
        m["num_steps"] += 1
        return Action(np.array([2]))

    def _terminal(_1, m, _3, _4):
        return m["num_steps"] >= 4

    param_option1 = ParameterizedOption("dummy1", [cup_type],
                                        Box(0.1, 1, (1, )), _policy,
                                        _initiable, _terminal)

    children = [param_option0, param_option1]
    chain_param_opt = utils.LinearChainParameterizedOption("chain", children)
    assert chain_param_opt.types == [cup_type]
    assert np.allclose(chain_param_opt.params_space.low, [0.1])
    assert np.allclose(chain_param_opt.params_space.high, [1.0])
    chain_option = chain_param_opt.ground([cup], [0.5])
    assert chain_option.objects == [cup]

    assert chain_option.initiable(state)
    assert chain_option.policy(state).arr[0] == -4
    assert not chain_option.terminal(state)
    assert chain_option.policy(state).arr[0] == -4
    assert not chain_option.terminal(state)
    assert chain_option.policy(state).arr[0] == 2
    assert not chain_option.terminal(state)
    assert chain_option.policy(state).arr[0] == 2
    assert not chain_option.terminal(state)
    assert chain_option.policy(state).arr[0] == 2
    assert not chain_option.terminal(state)
    assert chain_option.policy(state).arr[0] == 2
    assert chain_option.terminal(state)

    # Cannot initialize with empty children.
    with pytest.raises(AssertionError):
        utils.LinearChainParameterizedOption("chain", [])

    # Test that AssertionError is raised when options don't chain.
    param_option2 = ParameterizedOption(
        "dummy2", [cup_type], Box(0.1, 1, (1, )),
        lambda _1, _2, _3, _4: Action(np.array([0])),
        lambda _1, _2, _3, _4: False, lambda _1, _2, _3, _4: False)

    children = [param_option0, param_option2]
    chain_param_opt = utils.LinearChainParameterizedOption("chain2", children)
    chain_option = chain_param_opt.ground([cup], [0.5])
    assert chain_option.initiable(state)
    assert chain_option.policy(state).arr[0] == -4
    assert not chain_option.terminal(state)
    assert chain_option.policy(state).arr[0] == -4
    assert not chain_option.terminal(state)
    with pytest.raises(AssertionError):
        chain_option.policy(state)


def test_nsrt_methods():
    """Tests for all_ground_nsrts(), extract_preds_and_types()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate1_var = plate_type("?plate1")
    plate2_var = plate_type("?plate1")
    parameters = [cup_var, plate1_var, plate2_var]
    preconditions = {not_on([cup_var, plate1_var])}
    add_effects = {on([cup_var, plate1_var])}
    delete_effects = {not_on([cup_var, plate1_var])}
    params_space = Box(-10, 10, (2, ))
    parameterized_option = ParameterizedOption("Pick", [cup_type],
                                               params_space,
                                               lambda s, m, o, p: 2 * p,
                                               lambda s, m, o, p: True,
                                               lambda s, m, o, p: True)
    nsrt = NSRT("PickNSRT",
                parameters,
                preconditions,
                add_effects,
                delete_effects,
                set(),
                parameterized_option, [parameters[0]],
                _sampler=None)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_nsrts = sorted(utils.all_ground_nsrts(nsrt, objects))
    assert len(ground_nsrts) == 8
    all_obj = [nsrt.objects for nsrt in ground_nsrts]
    assert [cup1, plate1, plate1] in all_obj
    assert [cup1, plate2, plate1] in all_obj
    assert [cup2, plate1, plate1] in all_obj
    assert [cup2, plate2, plate1] in all_obj
    assert [cup1, plate1, plate2] in all_obj
    assert [cup1, plate2, plate2] in all_obj
    assert [cup2, plate1, plate2] in all_obj
    assert [cup2, plate2, plate2] in all_obj
    preds, types = utils.extract_preds_and_types({nsrt})
    assert preds == {"NotOn": not_on, "On": on}
    assert types == {"plate_type": plate_type, "cup_type": cup_type}


def test_all_ground_operators():
    """Tests for all_ground_operators()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate1_var = plate_type("?plate1")
    plate2_var = plate_type("?plate2")
    parameters = [cup_var, plate1_var, plate2_var]
    preconditions = {not_on([cup_var, plate1_var])}
    add_effects = {on([cup_var, plate1_var])}
    delete_effects = {not_on([cup_var, plate1_var])}
    op = STRIPSOperator("Pick", parameters, preconditions, add_effects,
                        delete_effects, set())
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_ops = sorted(utils.all_ground_operators(op, objects))
    assert len(ground_ops) == 8
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate1, plate1] in all_obj
    assert [cup1, plate2, plate1] in all_obj
    assert [cup2, plate1, plate1] in all_obj
    assert [cup2, plate2, plate1] in all_obj
    assert [cup1, plate1, plate2] in all_obj
    assert [cup1, plate2, plate2] in all_obj
    assert [cup2, plate1, plate2] in all_obj
    assert [cup2, plate2, plate2] in all_obj
    preds, types = utils.extract_preds_and_types({op})
    assert preds == {"NotOn": not_on, "On": on}
    assert types == {"plate_type": plate_type, "cup_type": cup_type}


def test_all_ground_operators_given_partial():
    """Tests for all_ground_operators_given_partial()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate1_var = plate_type("?plate1")
    plate2_var = plate_type("?plate2")
    parameters = [cup_var, plate1_var, plate2_var]
    preconditions = {not_on([cup_var, plate1_var])}
    add_effects = {on([cup_var, plate1_var])}
    delete_effects = {not_on([cup_var, plate1_var])}
    op = STRIPSOperator("Pick", parameters, preconditions, add_effects,
                        delete_effects, set())
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    # First test empty partial sub.
    ground_ops = sorted(
        utils.all_ground_operators_given_partial(op, objects, {}))
    assert ground_ops == sorted(utils.all_ground_operators(op, objects))
    # Test with one partial sub.
    sub = {plate1_var: plate1}
    ground_ops = sorted(
        utils.all_ground_operators_given_partial(op, objects, sub))
    assert len(ground_ops) == 4
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate1, plate1] in all_obj
    assert [cup2, plate1, plate1] in all_obj
    assert [cup1, plate1, plate2] in all_obj
    assert [cup2, plate1, plate2] in all_obj
    preds, types = utils.extract_preds_and_types({op})
    assert preds == {"NotOn": not_on, "On": on}
    assert types == {"plate_type": plate_type, "cup_type": cup_type}
    # Test another single partial sub.
    sub = {plate1_var: plate2}
    ground_ops = sorted(
        utils.all_ground_operators_given_partial(op, objects, sub))
    assert len(ground_ops) == 4
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate2, plate1] in all_obj
    assert [cup2, plate2, plate1] in all_obj
    assert [cup1, plate2, plate2] in all_obj
    assert [cup2, plate2, plate2] in all_obj
    # Test multiple partial subs.
    sub = {plate1_var: plate1, plate2_var: plate2}
    ground_ops = sorted(
        utils.all_ground_operators_given_partial(op, objects, sub))
    assert len(ground_ops) == 2
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate1, plate2] in all_obj
    assert [cup2, plate1, plate2] in all_obj
    sub = {plate1_var: plate2, plate2_var: plate1, cup_var: cup1}
    ground_ops = sorted(
        utils.all_ground_operators_given_partial(op, objects, sub))
    assert len(ground_ops) == 1
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate2, plate1] in all_obj


def test_prune_ground_atom_dataset():
    """Tests for prune_ground_atom_dataset()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: False)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: False)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup1: [0.5], cup2: [0.1], plate1: [1.0], plate2: [1.2]})
    on_ground = {
        GroundAtom(on, [cup1, plate1]),
        GroundAtom(on, [cup2, plate2])
    }
    not_on_ground = {
        GroundAtom(not_on, [cup1, plate2]),
        GroundAtom(not_on, [cup2, plate1])
    }
    all_atoms = on_ground | not_on_ground
    ground_atom_dataset = [(LowLevelTrajectory([state], []), [all_atoms])]
    pruned_dataset1 = utils.prune_ground_atom_dataset(ground_atom_dataset,
                                                      {on})
    assert pruned_dataset1[0][1][0] == on_ground
    pruned_dataset2 = utils.prune_ground_atom_dataset(ground_atom_dataset,
                                                      {not_on})
    assert pruned_dataset2[0][1][0] == not_on_ground
    pruned_dataset3 = utils.prune_ground_atom_dataset(ground_atom_dataset,
                                                      {on, not_on})
    assert pruned_dataset3[0][1][0] == all_atoms
    pruned_dataset4 = utils.prune_ground_atom_dataset(ground_atom_dataset,
                                                      set())
    assert pruned_dataset4[0][1][0] == set()


def test_all_possible_ground_atoms():
    """Tests for all_possible_ground_atoms()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: False)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: False)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup1: [0.5], cup2: [0.1], plate1: [1.0], plate2: [1.2]})
    on_ground = {
        GroundAtom(on, [cup1, plate1]),
        GroundAtom(on, [cup1, plate2]),
        GroundAtom(on, [cup2, plate1]),
        GroundAtom(on, [cup2, plate2])
    }
    not_on_ground = {
        GroundAtom(not_on, [cup1, plate1]),
        GroundAtom(not_on, [cup1, plate2]),
        GroundAtom(not_on, [cup2, plate1]),
        GroundAtom(not_on, [cup2, plate2])
    }
    ground_atoms = sorted(on_ground | not_on_ground)
    assert utils.all_possible_ground_atoms(state, {on, not_on}) == ground_atoms
    assert not utils.abstract(state, {on, not_on})


def test_create_ground_atom_dataset():
    """Tests for create_ground_atom_dataset()."""
    utils.reset_config({
        "env": "test_env",
    })
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type],
                   lambda s, o: s.get(o[0], "feat1") > s.get(o[1], "feat1"))
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    states = [
        State({
            cup1: np.array([0.5]),
            cup2: np.array([0.1]),
            plate1: np.array([1.0]),
            plate2: np.array([1.2])
        }),
        State({
            cup1: np.array([1.1]),
            cup2: np.array([0.1]),
            plate1: np.array([1.0]),
            plate2: np.array([1.2])
        })
    ]
    actions = [Action(np.array([0.0]), DummyOption)]
    dataset = [LowLevelTrajectory(states, actions)]
    ground_atom_dataset = utils.create_ground_atom_dataset(dataset, {on})
    assert len(ground_atom_dataset) == 1
    assert len(ground_atom_dataset[0]) == 2
    assert len(ground_atom_dataset[0][0].states) == len(states)
    assert all(gs.allclose(s) for gs, s in \
               zip(ground_atom_dataset[0][0].states, states))
    assert len(ground_atom_dataset[0][0].actions) == len(actions)
    assert all(ga == a
               for ga, a in zip(ground_atom_dataset[0][0].actions, actions))
    assert len(ground_atom_dataset[0][1]) == len(states) == 2
    assert ground_atom_dataset[0][1][0] == set()
    assert ground_atom_dataset[0][1][1] == {GroundAtom(on, [cup1, plate1])}


def test_get_reachable_atoms():
    """Tests for get_reachable_atoms()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred1 = Predicate("Pred1", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type, plate_type], lambda s, o: True)
    pred3 = Predicate("Pred3", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    # pred3 is unreachable
    preconditions1 = {pred1([cup_var, plate_var])}
    add_effects1 = {pred2([cup_var, plate_var])}
    delete_effects1 = {}
    preconditions2 = {pred1([cup_var, plate_var])}
    add_effects2 = {}
    delete_effects2 = {pred3([cup_var, plate_var])}
    nsrt1 = NSRT("Pick",
                 parameters,
                 preconditions1,
                 add_effects1,
                 delete_effects1,
                 ignore_effects=set(),
                 option=None,
                 option_vars=[],
                 _sampler=None)
    nsrt2 = NSRT("Place",
                 parameters,
                 preconditions2,
                 add_effects2,
                 delete_effects2,
                 ignore_effects=set(),
                 option=None,
                 option_vars=[],
                 _sampler=None)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_nsrts = (set(utils.all_ground_nsrts(nsrt1, objects))
                    | set(utils.all_ground_nsrts(nsrt2, objects)))
    assert len(ground_nsrts) == 8
    atoms = {pred1([cup1, plate1]), pred1([cup1, plate2])}
    reachable_atoms = utils.get_reachable_atoms(ground_nsrts, atoms)
    assert reachable_atoms == {
        pred1([cup1, plate1]),
        pred1([cup1, plate2]),
        pred2([cup1, plate1]),
        pred2([cup1, plate2])
    }


def test_nsrt_application():
    """Tests for get_applicable_operators() and apply_operator() with a
    _GroundNSRT."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred1 = Predicate("Pred1", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type, plate_type], lambda s, o: True)
    pred3 = Predicate("Pred3", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions1 = {pred1([cup_var, plate_var])}
    add_effects1 = {pred2([cup_var, plate_var])}
    delete_effects1 = {}
    preconditions2 = {pred1([cup_var, plate_var])}
    add_effects2 = {}
    delete_effects2 = {pred3([cup_var, plate_var])}
    nsrt1 = NSRT("Pick",
                 parameters,
                 preconditions1,
                 add_effects1,
                 delete_effects1,
                 ignore_effects=set(),
                 option=None,
                 option_vars=[],
                 _sampler=None)
    nsrt2 = NSRT("Place",
                 parameters,
                 preconditions2,
                 add_effects2,
                 delete_effects2,
                 ignore_effects=set(),
                 option=None,
                 option_vars=[],
                 _sampler=None)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_nsrts = (set(utils.all_ground_nsrts(nsrt1, objects))
                    | set(utils.all_ground_nsrts(nsrt2, objects)))
    assert len(ground_nsrts) == 8
    applicable = list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup1, plate1])}))
    assert len(applicable) == 2
    all_obj = [(nsrt.name, nsrt.objects) for nsrt in applicable]
    assert ("Pick", [cup1, plate1]) in all_obj
    assert ("Place", [cup1, plate1]) in all_obj
    next_atoms = [
        utils.apply_operator(nsrt, {pred1([cup1, plate1])})
        for nsrt in applicable
    ]
    assert {pred1([cup1, plate1])} in next_atoms
    assert {pred1([cup1, plate1]), pred2([cup1, plate1])} in next_atoms
    assert list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup1, plate2])}))
    assert list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup2, plate1])}))
    assert list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup2, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred2([cup1, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred2([cup1, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred2([cup2, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred2([cup2, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred3([cup1, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred3([cup1, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred3([cup2, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred3([cup2, plate2])}))
    # Tests with ignore effects.
    ignore_effects = {pred2}
    nsrt3 = NSRT("Pick",
                 parameters,
                 preconditions1,
                 add_effects1,
                 delete_effects1,
                 ignore_effects=ignore_effects,
                 option=None,
                 option_vars=[],
                 _sampler=None)
    ground_nsrts = sorted(utils.all_ground_nsrts(nsrt3, objects))
    applicable = list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup1, plate1])}))
    assert len(applicable) == 1
    ground_nsrt = applicable[0]
    atoms = {pred1([cup1, plate1]), pred2([cup2, plate2])}
    next_atoms = utils.apply_operator(ground_nsrt, atoms)
    assert next_atoms == {pred1([cup1, plate1]), pred2([cup1, plate1])}
    # Tests when the add effects and delete effects have overlap. The add
    # effects should take precedence.
    add_effects = {pred2([cup_var, plate_var]), pred3([cup_var, plate_var])}
    delete_effects = {pred2([cup_var, plate_var])}
    nsrt4 = NSRT("Pick",
                 parameters,
                 preconditions1,
                 add_effects,
                 delete_effects,
                 ignore_effects=set(),
                 option=None,
                 option_vars=[],
                 _sampler=None)
    ground_nsrts = sorted(utils.all_ground_nsrts(nsrt4, objects))
    applicable = list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup1, plate1])}))
    assert len(applicable) == 1
    ground_nsrt = applicable[0]
    atoms = {pred1([cup1, plate1])}
    next_atoms = utils.apply_operator(ground_nsrt, atoms)
    assert next_atoms == {
        pred1([cup1, plate1]),
        pred2([cup1, plate1]),
        pred3([cup1, plate1])
    }


def test_operator_application():
    """Tests for get_applicable_operators(), apply_operator(), and
    get_successors_from_ground_ops() with a _GroundSTRIPSOperator."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred1 = Predicate("Pred1", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type, plate_type], lambda s, o: True)
    pred3 = Predicate("Pred3", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions1 = {pred1([cup_var, plate_var])}
    add_effects1 = {pred2([cup_var, plate_var])}
    delete_effects1 = {}
    preconditions2 = {pred1([cup_var, plate_var])}
    add_effects2 = {}
    delete_effects2 = {pred3([cup_var, plate_var])}
    op1 = STRIPSOperator("Pick", parameters, preconditions1, add_effects1,
                         delete_effects1, set())
    op2 = STRIPSOperator("Place", parameters, preconditions2, add_effects2,
                         delete_effects2, set())
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_ops = (set(utils.all_ground_operators(op1, objects))
                  | set(utils.all_ground_operators(op2, objects)))
    assert len(ground_ops) == 8
    applicable = list(
        utils.get_applicable_operators(ground_ops, {pred1([cup1, plate1])}))
    assert len(applicable) == 2
    all_obj = [(op.name, op.objects) for op in applicable]
    assert ("Pick", [cup1, plate1]) in all_obj
    assert ("Place", [cup1, plate1]) in all_obj
    next_atoms = [
        utils.apply_operator(op, {pred1([cup1, plate1])}) for op in applicable
    ]
    assert {pred1([cup1, plate1])} in next_atoms
    assert {pred1([cup1, plate1]), pred2([cup1, plate1])} in next_atoms
    assert list(
        utils.get_applicable_operators(ground_ops, {pred1([cup1, plate2])}))
    assert list(
        utils.get_applicable_operators(ground_ops, {pred1([cup2, plate1])}))
    assert list(
        utils.get_applicable_operators(ground_ops, {pred1([cup2, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred2([cup1, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred2([cup1, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred2([cup2, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred2([cup2, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred3([cup1, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred3([cup1, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred3([cup2, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred3([cup2, plate2])}))
    # Test for get_successors_from_ground_ops().
    # Make sure uniqueness is handled properly.
    op3 = STRIPSOperator("Pick", parameters, preconditions1, add_effects1,
                         delete_effects1, set())
    preconditions3 = {pred2([cup_var, plate_var])}
    op4 = STRIPSOperator("Place", parameters, preconditions3, add_effects2,
                         delete_effects2, set())
    op5 = STRIPSOperator("Pick2", parameters, preconditions1, add_effects1,
                         delete_effects1, set())
    ground_ops = (set(utils.all_ground_operators(op3, objects))
                  | set(utils.all_ground_operators(op4, objects))
                  | set(utils.all_ground_operators(op5, objects)))
    successors = list(
        utils.get_successors_from_ground_ops({pred1([cup1, plate1])},
                                             ground_ops))
    assert len(successors) == 1
    assert successors[0] == {pred1([cup1, plate1]), pred2([cup1, plate1])}
    successors = list(
        utils.get_successors_from_ground_ops({pred1([cup1, plate1])},
                                             ground_ops,
                                             unique=False))
    assert len(successors) == 2
    assert successors[0] == successors[1]
    assert not list(
        utils.get_successors_from_ground_ops({pred3([cup2, plate2])},
                                             ground_ops))
    # Tests with ignore effects.
    ignore_effects = {pred2}
    op3 = STRIPSOperator("Pick",
                         parameters,
                         preconditions1,
                         add_effects1,
                         delete_effects1,
                         ignore_effects=ignore_effects)
    ground_ops = sorted(utils.all_ground_operators(op3, objects))
    applicable = list(
        utils.get_applicable_operators(ground_ops, {pred1([cup1, plate1])}))
    assert len(applicable) == 1
    ground_op = applicable[0]
    atoms = {pred1([cup1, plate1]), pred2([cup2, plate2])}
    next_atoms = utils.apply_operator(ground_op, atoms)
    assert next_atoms == {pred1([cup1, plate1]), pred2([cup1, plate1])}


@pytest.mark.parametrize("heuristic_name, expected_heuristic_cls", [
    ("hadd", _PyperplanHeuristicWrapper),
    ("hmax", _PyperplanHeuristicWrapper),
    ("hff", _PyperplanHeuristicWrapper),
    ("hsa", _PyperplanHeuristicWrapper),
    ("lmcut", _PyperplanHeuristicWrapper),
    ("goal_count", GoalCountHeuristic),
])
def test_create_task_planning_heuristic(
        heuristic_name: str,
        expected_heuristic_cls: TypingType[_TaskPlanningHeuristic]):
    """Tests for create_task_planning_heuristic()."""
    heuristic = utils.create_task_planning_heuristic(heuristic_name, set(),
                                                     set(), set(), set(),
                                                     set())
    assert isinstance(heuristic, expected_heuristic_cls)


def test_create_task_planning_heuristic_raises_error_for_unknown_heuristic():
    """Test creating unknown heuristic raises a ValueError."""
    with pytest.raises(ValueError):
        utils.create_task_planning_heuristic("not a real heuristic", set(),
                                             set(), set(), set(), set())


def test_create_task_planning_heuristic_base_class():
    """Test to cover _TaskPlanningHeuristic base class."""
    base_heuristic = _TaskPlanningHeuristic("base", set(), set(), set())
    with pytest.raises(NotImplementedError):
        base_heuristic(set())


def test_goal_count_heuristic():
    """Test the goal count heuristic."""
    # Create predicate and objects
    block_type = Type("block_type", ["feat1"])
    table_type = Type("plate_type", ["feat1"])

    on = Predicate("On", [block_type, table_type], lambda s, o: False)
    block1 = block_type("block1")
    block2 = block_type("block2")
    block3 = block_type("block3")
    table1 = table_type("table1")
    table2 = table_type("table2")

    # Create goal and heuristic instance
    goal_atoms = {
        GroundAtom(on, [block1, table1]),
        GroundAtom(on, [block2, table1]),
        GroundAtom(on, [block3, table2]),
    }
    heuristic = GoalCountHeuristic("goal_count",
                                   init_atoms=set(),
                                   goal=goal_atoms,
                                   ground_ops=set())

    assert heuristic(goal_atoms) == 0
    assert heuristic({
        GroundAtom(on, [block1, table1]),
        GroundAtom(on, [block2, table1]),
        GroundAtom(on, [block3, table1]),
    }) == 1
    assert heuristic({
        GroundAtom(on, [block1, table2]),
        GroundAtom(on, [block2, table2]),
        GroundAtom(on, [block3, table2]),
    }) == 2
    assert heuristic({
        GroundAtom(on, [block1, table2]),
        GroundAtom(on, [block2, table2]),
        GroundAtom(on, [block3, table1]),
    }) == 3

    # Some edge cases
    assert heuristic(set()) == 3
    assert heuristic(
        {GroundAtom(on, [block_type("block99"),
                         table_type("table99")])}) == 3


def test_create_pddl():
    """Tests for create_pddl_domain() and create_pddl_problem()."""
    utils.reset_config({"env": "cover"})
    # All predicates and options
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    train_task = env.get_train_tasks()[0]
    state = train_task.init
    objects = list(state)
    init_atoms = utils.abstract(state, env.predicates)
    goal = train_task.goal
    domain_str = utils.create_pddl_domain(nsrts, env.predicates, env.types,
                                          "cover")
    problem_str = utils.create_pddl_problem(objects, init_atoms, goal, "cover",
                                            "cover-problem0")
    assert domain_str == """(define (domain cover)
  (:requirements :typing)
  (:types block robot target)

  (:predicates
    (Covers ?x0 - block ?x1 - target)
    (HandEmpty)
    (Holding ?x0 - block)
    (IsBlock ?x0 - block)
    (IsTarget ?x0 - target)
  )

  (:action Pick
    :parameters (?block - block)
    :precondition (and (HandEmpty)
        (IsBlock ?block))
    :effect (and (Holding ?block)
        (not (HandEmpty)))
  )

  (:action Place
    :parameters (?block - block ?target - target)
    :precondition (and (Holding ?block)
        (IsBlock ?block)
        (IsTarget ?target))
    :effect (and (Covers ?block ?target)
        (HandEmpty)
        (not (Holding ?block)))
  )
)"""

    assert problem_str == """(define (problem cover-problem0) (:domain cover)
  (:objects
    block0 - block
    block1 - block
    robby - robot
    target0 - target
    target1 - target
  )
  (:init
    (HandEmpty)
    (IsBlock block0)
    (IsBlock block1)
    (IsTarget target0)
    (IsTarget target1)
  )
  (:goal (and (Covers block0 target0)))
)
"""

    # Test spanner domain, which has hierarchical types.
    utils.reset_config({"env": "pddl_spanner_procedural_tasks"})
    # All predicates and options
    env = ProceduralTasksSpannerPDDLEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    domain_str = utils.create_pddl_domain(nsrts, env.predicates, env.types,
                                          "spanner")
    assert domain_str == """(define (domain spanner)
  (:requirements :typing)
  (:types 
    man nut spanner - locatable
    locatable location - object)

  (:predicates
    (at ?x0 - locatable ?x1 - location)
    (carrying ?x0 - man ?x1 - spanner)
    (link ?x0 - location ?x1 - location)
    (loose ?x0 - nut)
    (tightened ?x0 - nut)
    (useable ?x0 - spanner)
  )

  (:action pickup_spanner
    :parameters (?l - location ?s - spanner ?m - man)
    :precondition (and (at ?m ?l)
        (at ?s ?l))
    :effect (and (carrying ?m ?s)
        (not (at ?s ?l)))
  )

  (:action tighten_nut
    :parameters (?l - location ?s - spanner ?m - man ?n - nut)
    :precondition (and (at ?m ?l)
        (at ?n ?l)
        (carrying ?m ?s)
        (loose ?n)
        (useable ?s))
    :effect (and (tightened ?n)
        (not (loose ?n))
        (not (useable ?s)))
  )

  (:action walk
    :parameters (?start - location ?end - location ?m - man)
    :precondition (and (at ?m ?start)
        (link ?start ?end))
    :effect (and (at ?m ?end)
        (not (at ?m ?start)))
  )
)"""

    train_task = env.get_train_tasks()[0]
    state = train_task.init
    objects = list(state)
    init_atoms = utils.abstract(state, env.predicates)
    goal = train_task.goal
    problem_str = utils.create_pddl_problem(objects, init_atoms, goal,
                                            "spanner", "spanner-0")
    assert problem_str == """(define (problem spanner-0) (:domain spanner)
  (:objects
    bob - man
    gate - location
    location0 - location
    location1 - location
    location2 - location
    nut0 - nut
    shed - location
    spanner0 - spanner
    spanner1 - spanner
    spanner2 - spanner
  )
  (:init
    (at bob shed)
    (at nut0 gate)
    (at spanner0 location0)
    (at spanner1 location2)
    (at spanner2 location0)
    (link location0 location1)
    (link location1 location2)
    (link location2 gate)
    (link shed location0)
    (loose nut0)
    (useable spanner0)
    (useable spanner1)
    (useable spanner2)
  )
  (:goal (and (tightened nut0)))
)
"""


def test_VideoMonitor():
    """Tests for VideoMonitor()."""
    env = CoverMultistepOptions()
    monitor = utils.VideoMonitor(env.render)
    policy = lambda _: Action(env.action_space.sample())
    task = env.get_task("test", 0)
    traj, _ = utils.run_policy(policy,
                               env,
                               "test",
                               0,
                               task.goal_holds,
                               max_num_steps=2,
                               monitor=monitor)
    assert not task.goal_holds(traj.states[-1])
    assert len(traj.states) == 3
    assert len(traj.actions) == 2
    video = monitor.get_video()
    assert len(video) == len(traj.states)
    first_state_rendered = env.render_state(task.init, task)
    assert np.allclose(first_state_rendered, video[0])
    assert not np.allclose(first_state_rendered, video[1])


def test_SimulateVideoMonitor():
    """Tests for SimulateVideoMonitor()."""
    env = CoverMultistepOptions()
    task = env.get_task("test", 0)
    monitor = utils.SimulateVideoMonitor(task, env.render_state)
    policy = lambda _: Action(env.action_space.sample())
    traj, _ = utils.run_policy(policy,
                               env,
                               "test",
                               0,
                               task.goal_holds,
                               max_num_steps=2,
                               monitor=monitor)
    assert not task.goal_holds(traj.states[-1])
    assert len(traj.states) == 3
    assert len(traj.actions) == 2
    video = monitor.get_video()
    assert len(video) == len(traj.states)
    first_state_rendered = env.render_state(task.init, task)
    assert np.allclose(first_state_rendered, video[0])
    assert not np.allclose(first_state_rendered, video[1])


def test_save_video():
    """Tests for save_video()."""
    dirname = "_fake_tmp_video_dir"
    filename = "video.mp4"
    utils.reset_config({"video_dir": dirname})
    rng = np.random.default_rng(123)
    video = [rng.integers(255, size=(3, 3), dtype=np.uint8) for _ in range(3)]
    utils.save_video(filename, video)
    os.remove(os.path.join(dirname, filename))
    os.rmdir(dirname)


def test_get_config_path_str():
    """Tests for get_config_path_str()."""
    utils.reset_config({
        "env": "dummyenv",
        "approach": "dummyapproach",
        "seed": 321,
        "excluded_predicates": "all",
        "included_options": "Dummy1,Dummy2",
        "experiment_id": "foobar",
    })
    s = utils.get_config_path_str()
    assert s == "dummyenv__dummyapproach__321__all__Dummy1,Dummy2__foobar"
    s = utils.get_config_path_str("override_id")
    assert s == "dummyenv__dummyapproach__321__all__Dummy1,Dummy2__override_id"


def test_get_approach_save_path_str():
    """Tests for get_approach_save_path_str()."""
    dirname = "_fake_tmp_approach_dir"
    old_approach_dir = CFG.approach_dir
    utils.reset_config({
        "env": "test_env",
        "approach": "test_approach",
        "seed": 123,
        "approach_dir": dirname,
        "excluded_predicates": "test_pred1,test_pred2",
        "included_options": "Dummy1",
        "experiment_id": "baz",
    })
    save_path = utils.get_approach_save_path_str()
    assert save_path == dirname + ("/test_env__test_approach__123__"
                                   "test_pred1,test_pred2__Dummy1__baz.saved")
    utils.reset_config({
        "env": "test_env",
        "approach": "test_approach",
        "seed": 123,
        "approach_dir": dirname,
        "excluded_predicates": "",
        "included_options": "",
        "experiment_id": "",
    })
    save_path = utils.get_approach_save_path_str()
    assert save_path == dirname + "/test_env__test_approach__123______.saved"
    os.rmdir(dirname)
    utils.reset_config({"approach_dir": old_approach_dir})


def test_get_approach_load_path_str():
    """Tests for get_approach_load_path_str()."""
    dirname = "_fake_tmp_approach_dir"
    old_approach_dir = CFG.approach_dir
    utils.reset_config({
        "env": "test_env",
        "approach": "test_approach",
        "seed": 123,
        "approach_dir": dirname,
        "excluded_predicates": "test_pred1,test_pred2",
        "included_options": "Dummy1",
        "experiment_id": "baz",
        "load_experiment_id": "foo",
    })
    save_path = utils.get_approach_save_path_str()
    assert save_path == dirname + ("/test_env__test_approach__123__"
                                   "test_pred1,test_pred2__Dummy1__baz.saved")
    load_path = utils.get_approach_load_path_str()
    assert load_path == dirname + ("/test_env__test_approach__123__"
                                   "test_pred1,test_pred2__Dummy1__foo.saved")
    utils.reset_config({
        "env": "test_env",
        "approach": "test_approach",
        "seed": 123,
        "approach_dir": dirname,
        "excluded_predicates": "test_pred1,test_pred2",
        "included_options": "Dummy1",
        "experiment_id": "baz",
    })
    save_path = utils.get_approach_save_path_str()
    assert save_path == dirname + ("/test_env__test_approach__123__"
                                   "test_pred1,test_pred2__Dummy1__baz.saved")
    load_path = utils.get_approach_load_path_str()
    assert load_path == dirname + ("/test_env__test_approach__123__"
                                   "test_pred1,test_pred2__Dummy1__baz.saved")
    os.rmdir(dirname)
    utils.reset_config({"approach_dir": old_approach_dir})


def test_update_config():
    """Tests for update_config()."""
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "seed": 123,
    })
    assert CFG.env == "cover"
    assert CFG.approach == "random_actions"
    assert CFG.seed == 123
    utils.update_config({
        "env": "dummyenv",
        "approach": "dummyapproach",
        "seed": 321,
    })
    assert CFG.env == "dummyenv"
    assert CFG.approach == "dummyapproach"
    assert CFG.seed == 321
    with pytest.raises(ValueError):
        utils.update_config({"not a real setting name": 0})


def test_reset_config():
    """Tests for reset_config()."""
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "seed": 123,
    })
    assert CFG.env == "cover"
    assert CFG.approach == "random_actions"
    assert CFG.seed == 123
    utils.reset_config({
        "env": "dummyenv",
        "approach": "dummyapproach",
        "seed": 321,
    })
    assert CFG.env == "dummyenv"
    assert CFG.approach == "dummyapproach"
    assert CFG.seed == 321
    with pytest.raises(ValueError):
        utils.reset_config({"not a real setting name": 0})
    # Test that default seed gets set automatically.
    del CFG.seed
    assert "seed" not in CFG.__dict__
    with pytest.raises(AttributeError):
        _ = CFG.seed
    utils.reset_config({"env": "cover"})
    assert CFG.seed == 123


def test_run_gbfs():
    """Tests for run_gbfs()."""
    S = Tuple[int, int]  # grid (row, col)
    A = str  # up, down, left, right

    def _grid_successor_fn(state: S) -> Iterator[Tuple[A, S, float]]:
        arrival_costs = np.array([
            [1, 1, 8, 1, 1],
            [1, 8, 1, 1, 1],
            [1, 8, 1, 1, 1],
            [1, 1, 1, 8, 1],
            [1, 1, 2, 1, 1],
        ],
                                 dtype=float)

        act_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        r, c = state

        for act in sorted(act_to_delta):
            dr, dc = act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            # Check if in bounds
            if not (0 <= new_r < arrival_costs.shape[0] and \
                    0 <= new_c < arrival_costs.shape[1]):
                continue
            # Valid action
            yield (act, (new_r, new_c), arrival_costs[new_r, new_c])

    def _grid_check_goal_fn(state: S) -> bool:
        # Bottom right corner of grid
        return state == (4, 4)

    def _grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    initial_state = (0, 0)
    state_sequence, action_sequence = utils.run_gbfs(initial_state,
                                                     _grid_check_goal_fn,
                                                     _grid_successor_fn,
                                                     _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'
    ]

    # Same, but actually reaching the goal is impossible.
    state_sequence, action_sequence = utils.run_gbfs(initial_state,
                                                     lambda s: False,
                                                     _grid_successor_fn,
                                                     _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'
    ]

    # Test with an infinite branching factor.
    def _inf_grid_successor_fn(state: S) -> Iterator[Tuple[A, S, float]]:
        # Change all costs to 1.
        i = 0
        for (a, ns, _) in _grid_successor_fn(state):
            yield (a, ns, 1.)
        # Yield unnecessary and costly noops.
        # These lines should not be covered, and that's the point!
        while True:  # pragma: no cover
            action = f"noop{i}"
            yield (action, state, 100.)
            i += 1

    state_sequence, action_sequence = utils.run_gbfs(initial_state,
                                                     _grid_check_goal_fn,
                                                     _inf_grid_successor_fn,
                                                     _grid_heuristic_fn,
                                                     lazy_expansion=True)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'
    ]
    # Test limit on max evals.
    state_sequence, action_sequence = utils.run_gbfs(
        initial_state,
        _grid_check_goal_fn,
        _inf_grid_successor_fn,
        _grid_heuristic_fn,
        max_evals=2)  # note: need lazy_expansion to be False here
    assert state_sequence == [(0, 0), (1, 0)]
    assert action_sequence == ['down']

    # Test timeout.
    # We don't care about the return value. Since the goal check always
    # returns False, the fact that this test doesn't hang means that
    # the timeout is working correctly.
    utils.run_gbfs(initial_state,
                   lambda s: False,
                   _inf_grid_successor_fn,
                   _grid_heuristic_fn,
                   timeout=0.01)


def test_run_astar():
    """Tests for run_astar()."""
    S = Tuple[int, int]  # grid (row, col)
    A = str  # up, down, left, right

    def _grid_successor_fn(state: S) -> Iterator[Tuple[A, S, float]]:
        arrival_costs = np.array([
            [1, 1, 100, 1, 1],
            [1, 100, 1, 1, 1],
            [1, 100, 1, 1, 1],
            [1, 1, 1, 100, 1],
            [1, 1, 100, 1, 1],
        ],
                                 dtype=float)

        act_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        r, c = state

        for act in sorted(act_to_delta):
            dr, dc = act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            # Check if in bounds
            if not (0 <= new_r < arrival_costs.shape[0] and \
                    0 <= new_c < arrival_costs.shape[1]):
                continue
            # Valid action
            yield (act, (new_r, new_c), arrival_costs[new_r, new_c])

    def _grid_check_goal_fn(state: S) -> bool:
        # Bottom right corner of grid
        return state == (4, 4)

    def _grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    initial_state = (0, 0)
    state_sequence, action_sequence = utils.run_astar(initial_state,
                                                      _grid_check_goal_fn,
                                                      _grid_successor_fn,
                                                      _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2),
                              (2, 2), (2, 3), (2, 4), (3, 4), (4, 4)]
    assert action_sequence == [
        'down', 'down', 'down', 'right', 'right', 'up', 'right', 'right',
        'down', 'down'
    ]


def test_run_hill_climbing():
    """Tests for run_hill_climbing()."""
    S = Tuple[int, int]  # grid (row, col)
    A = str  # up, down, left, right

    def _grid_successor_fn(state: S) -> Iterator[Tuple[A, S, float]]:
        arrival_costs = np.array([
            [1, 1, 8, 1, 1],
            [1, 8, 1, 1, 1],
            [1, 8, 1, 1, 1],
            [1, 1, 1, 8, 1],
            [1, 1, 2, 1, 1],
        ],
                                 dtype=float)

        act_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        r, c = state

        for act in sorted(act_to_delta):
            dr, dc = act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            # Check if in bounds
            if not (0 <= new_r < arrival_costs.shape[0] and \
                    0 <= new_c < arrival_costs.shape[1]):
                continue
            # Valid action
            yield (act, (new_r, new_c), arrival_costs[new_r, new_c])

    def _grid_check_goal_fn(state: S) -> bool:
        # Bottom right corner of grid
        return state == (4, 4)

    def _grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    initial_state = (0, 0)
    state_sequence, action_sequence, heuristics = utils.run_hill_climbing(
        initial_state, _grid_check_goal_fn, _grid_successor_fn,
        _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        "down", "down", "down", "down", "right", "right", "right", "right"
    ]
    assert heuristics == [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]

    # Same, but actually reaching the goal is impossible.
    state_sequence, action_sequence, _ = utils.run_hill_climbing(
        initial_state, lambda s: False, _grid_successor_fn, _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        "down", "down", "down", "down", "right", "right", "right", "right"
    ]

    # Search with no successors
    def _no_successor_fn(state: S) -> Iterator[Tuple[A, S, float]]:
        if state == initial_state:
            yield "dummy_action", (2, 2), 1.0

    state_sequence, action_sequence, _ = utils.run_hill_climbing(
        initial_state, lambda s: False, _no_successor_fn, _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (2, 2)]
    assert action_sequence == ["dummy_action"]

    # Tests showing the benefit of enforced hill climbing.
    def _local_minimum_grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        if state in [(1, 0), (0, 1)]:
            return float("inf")
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    for parallelize in (False, True):
        # With enforced_depth 0, search fails.
        state_sequence, action_sequence, heuristics = utils.run_hill_climbing(
            initial_state,
            _grid_check_goal_fn,
            _grid_successor_fn,
            _local_minimum_grid_heuristic_fn,
            parallelize=parallelize)
        assert state_sequence == [(0, 0)]
        assert not action_sequence
        assert heuristics == [8.0]

        # With enforced_depth 1, search succeeds.
        state_sequence, action_sequence, heuristics = utils.run_hill_climbing(
            initial_state,
            _grid_check_goal_fn,
            _grid_successor_fn,
            _local_minimum_grid_heuristic_fn,
            enforced_depth=1,
            parallelize=parallelize)
        # Note that hill-climbing does not care about costs.
        assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
                                  (4, 1), (4, 2), (4, 3), (4, 4)]
        assert action_sequence == [
            "down", "down", "down", "down", "right", "right", "right", "right"
        ]
        assert heuristics == [
            8.0, float("inf"), 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0
        ]

    # Test early_termination_heuristic_thresh with very high value.
    initial_state = (0, 0)
    state_sequence, action_sequence, heuristics = utils.run_hill_climbing(
        initial_state,
        _grid_check_goal_fn,
        _grid_successor_fn,
        _grid_heuristic_fn,
        early_termination_heuristic_thresh=10000000)
    assert state_sequence == [(0, 0)]
    assert not action_sequence


def test_run_policy_guided_astar():
    """Tests for run_policy_guided_astar()."""
    S = Tuple[int, int]  # grid (row, col)
    A = str  # up, down, left, right

    arrival_costs = np.array([
        [1, 1, 100, 1, 1],
        [1, 100, 1, 1, 1],
        [1, 100, 1, 1, 1],
        [1, 1, 1, 100, 1],
        [1, 1, 100, 1, 1],
    ],
                             dtype=float)

    act_to_delta = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def _get_valid_actions(state: S) -> Iterator[Tuple[A, float]]:
        r, c = state
        for act in sorted(act_to_delta):
            dr, dc = act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            # Check if in bounds
            if not (0 <= new_r < arrival_costs.shape[0] and \
                    0 <= new_c < arrival_costs.shape[1]):
                continue
            yield (act, arrival_costs[new_r, new_c])

    def _get_next_state(state: S, action: A) -> S:
        r, c = state
        dr, dc = act_to_delta[action]
        return (r + dr, c + dc)

    goal = (4, 4)

    def _grid_check_goal_fn(state: S) -> bool:
        # Bottom right corner of grid
        return state == goal

    def _grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        return float(abs(state[0] - goal[0]) + abs(state[1] - goal[1]))

    def _policy(state: S) -> A:
        # Move right until we can't anymore.
        _, c = state
        if c >= arrival_costs.shape[1] - 1:
            return None
        return "right"

    initial_state = (0, 0)
    num_rollout_steps = 10

    # The policy should bias toward the path that moves all the way right, then
    # planning should move all the way down to reach the goal.
    state_sequence, action_sequence = utils.run_policy_guided_astar(
        initial_state,
        _grid_check_goal_fn,
        _get_valid_actions,
        _get_next_state,
        _grid_heuristic_fn,
        _policy,
        num_rollout_steps=num_rollout_steps,
        rollout_step_cost=0)

    assert state_sequence == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4),
                              (2, 4), (3, 4), (4, 4)]
    assert action_sequence == [
        'right', 'right', 'right', 'right', 'down', 'down', 'down', 'down'
    ]

    # With a trivial policy, should find the optimal path.
    state_sequence, action_sequence = utils.run_policy_guided_astar(
        initial_state,
        _grid_check_goal_fn,
        _get_valid_actions,
        _get_next_state,
        _grid_heuristic_fn,
        policy=lambda s: None,
        num_rollout_steps=num_rollout_steps,
        rollout_step_cost=0)

    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2),
                              (2, 2), (2, 3), (2, 4), (3, 4), (4, 4)]
    assert action_sequence == [
        'down', 'down', 'down', 'right', 'right', 'up', 'right', 'right',
        'down', 'down'
    ]

    # With a policy that outputs invalid actions, should ignore the policy
    # and find the optimal path.
    state_sequence, action_sequence = utils.run_policy_guided_astar(
        initial_state,
        _grid_check_goal_fn,
        _get_valid_actions,
        _get_next_state,
        _grid_heuristic_fn,
        policy=lambda s: "garbage",
        num_rollout_steps=num_rollout_steps,
        rollout_step_cost=0)

    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2),
                              (2, 2), (2, 3), (2, 4), (3, 4), (4, 4)]
    assert action_sequence == [
        'down', 'down', 'down', 'right', 'right', 'up', 'right', 'right',
        'down', 'down'
    ]


def test_ops_and_specs_to_dummy_nsrts():
    """Tests for ops_and_specs_to_dummy_nsrts()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions = {not_on([cup_var, plate_var])}
    add_effects = {on([cup_var, plate_var])}
    delete_effects = {not_on([cup_var, plate_var])}
    params_space = Box(-10, 10, (2, ))
    parameterized_option = ParameterizedOption("Pick", [], params_space,
                                               lambda s, m, o, p: 2 * p,
                                               lambda s, m, o, p: True,
                                               lambda s, m, o, p: True)
    strips_operator = STRIPSOperator("Pick", parameters, preconditions,
                                     add_effects, delete_effects, set())
    nsrts = utils.ops_and_specs_to_dummy_nsrts([strips_operator],
                                               [(parameterized_option, [])])
    assert len(nsrts) == 1
    nsrt = next(iter(nsrts))
    assert nsrt.parameters == parameters
    assert nsrt.preconditions == preconditions
    assert nsrt.add_effects == add_effects
    assert nsrt.delete_effects == delete_effects
    assert nsrt.option == parameterized_option
    assert not nsrt.option_vars


def test_string_to_python_object():
    """Tests for string_to_python_object()."""
    assert utils.string_to_python_object("3") == 3
    assert utils.string_to_python_object("1234") == 1234
    assert utils.string_to_python_object("3.2") == 3.2
    assert utils.string_to_python_object("test") == "test"
    assert utils.string_to_python_object("") == ""
    assert utils.string_to_python_object("True") is True
    assert utils.string_to_python_object("False") is False
    assert utils.string_to_python_object("None") is None
    assert utils.string_to_python_object("true") is True
    assert utils.string_to_python_object("false") is False
    assert utils.string_to_python_object("none") is None
    assert utils.string_to_python_object("[3.2]") == [3.2]
    assert utils.string_to_python_object("[3.2,4.3]") == [3.2, 4.3]
    assert utils.string_to_python_object("[3.2, 4.3]") == [3.2, 4.3]
    assert utils.string_to_python_object("(3.2,4.3)") == (3.2, 4.3)
    assert utils.string_to_python_object("(3.2, 4.3)") == (3.2, 4.3)
    assert utils.string_to_python_object("lambda x: x + 3")(12) == 15
    with pytest.raises(TypeError):  # invalid number of arguments
        utils.string_to_python_object("lambda: x + 3")(12)
    assert utils.string_to_python_object("lambda: 13")() == 13


def test_get_env_asset_path():
    """Tests for get_env_asset_path()."""
    path = utils.get_env_asset_path("urdf/plane.urdf")
    assert path.endswith("urdf/plane.urdf")
    assert os.path.exists(path)
    with pytest.raises(AssertionError):
        utils.get_env_asset_path("not_a_real_asset")


def test_create_video_from_partial_refinements():
    """Tests for create_video_from_partial_refinements()."""
    env = CoverEnv()
    PickPlace = list(env.options)[0]
    option = PickPlace.ground([],
                              np.zeros(PickPlace.params_space.shape,
                                       dtype=np.float32))
    partial_refinements = [([], [option])]
    utils.reset_config({"failure_video_mode": "not a real video mode"})
    with pytest.raises(NotImplementedError):
        utils.create_video_from_partial_refinements(partial_refinements,
                                                    env,
                                                    "train",
                                                    task_idx=0,
                                                    max_num_steps=10)
    utils.reset_config({"env": "cover", "failure_video_mode": "longest_only"})
    video = utils.create_video_from_partial_refinements(partial_refinements,
                                                        env,
                                                        "train",
                                                        task_idx=0,
                                                        max_num_steps=10)
    assert len(video) == 2


def test_env_failure():
    """Tests for EnvironmentFailure class."""
    cup_type = Type("cup_type", ["feat1"])
    cup = cup_type("cup")
    try:
        raise utils.EnvironmentFailure("failure123",
                                       {"offending_objects": {cup}})
    except utils.EnvironmentFailure as e:
        assert str(e) == ("EnvironmentFailure('failure123'): "
                          "{'offending_objects': {cup:cup_type}}")
        assert e.info["offending_objects"] == {cup}


def test_parse_config_excluded_predicates():
    """Tests for parse_config_excluded_predicates()."""
    # Test excluding nothing.
    utils.reset_config({
        "env": "cover",
        "excluded_predicates": "",
    })
    env = CoverEnv()
    included, excluded = utils.parse_config_excluded_predicates(env)
    assert sorted(p.name for p in included) == [
        "Covers", "HandEmpty", "Holding", "IsBlock", "IsTarget"
    ]
    assert not excluded
    # Test excluding specific predicates.
    utils.reset_config({
        "excluded_predicates": "IsBlock,HandEmpty",
    })
    included, excluded = utils.parse_config_excluded_predicates(env)
    assert sorted(p.name
                  for p in included) == ["Covers", "Holding", "IsTarget"]
    assert sorted(p.name for p in excluded) == ["HandEmpty", "IsBlock"]
    # Test excluding all (non-goal) predicates.
    utils.reset_config({
        "excluded_predicates": "all",
    })
    included, excluded = utils.parse_config_excluded_predicates(env)
    assert sorted(p.name for p in included) == ["Covers"]
    assert sorted(p.name for p in excluded) == [
        "HandEmpty", "Holding", "IsBlock", "IsTarget"
    ]
    # Can exclude goal predicates when offline_data_method is demo+ground_atoms.
    utils.reset_config({
        "offline_data_method": "demo+ground_atoms",
        "excluded_predicates": "Covers",
    })
    included, excluded = utils.parse_config_excluded_predicates(env)
    assert sorted(p.name for p in included) == [
        "HandEmpty", "Holding", "IsBlock", "IsTarget"
    ]
    assert sorted(p.name for p in excluded) == ["Covers"]
    # Cannot exclude goal predicates otherwise..
    utils.reset_config({
        "offline_data_method": "demo",
        "excluded_predicates": "Covers",
    })
    with pytest.raises(AssertionError):
        utils.parse_config_excluded_predicates(env)


def test_parse_config_included_options():
    """Tests for parse_config_included_options()."""
    # Test including nothing.
    utils.reset_config({
        "env": "cover_multistep_options",
        "included_options": "",
    })
    env = CoverMultistepOptions()
    included = utils.parse_config_included_options(env)
    assert not included
    # Test including specific options.
    utils.reset_config({
        "included_options": "Pick",
    })
    Pick, Place = sorted(env.options)
    assert Pick.name == "Pick"
    assert Place.name == "Place"
    included = utils.parse_config_included_options(env)
    assert included == {Pick}
    utils.reset_config({
        "included_options": "Place",
    })
    included = utils.parse_config_included_options(env)
    assert included == {Place}
    utils.reset_config({
        "included_options": "Pick,Place",
    })
    included = utils.parse_config_included_options(env)
    assert included == {Pick, Place}
    # Test including an unknown option.
    utils.reset_config({
        "included_options": "Pick,NotReal",
    })
    with pytest.raises(AssertionError) as e:
        utils.parse_config_included_options(env)
    assert "Unrecognized option in included_options!" in str(e)


def test_null_sampler():
    """Tests for null_sampler()."""
    assert utils.null_sampler(None, None, None, None).shape == (0, )


def test_behavior_state():
    """Tests for BehaviorState."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = utils.BehaviorState({cup: [0.5], plate: [1.0, 1.2]})
    other_state = state.copy()
    assert state.allclose(other_state)


def test_nostdout(capfd):
    """Tests for nostdout()."""

    def _hello_world():
        print("Hello world!")

    for _ in range(2):
        _hello_world()
        out, _ = capfd.readouterr()
        assert out == "Hello world!\n"
        with utils.nostdout():
            _hello_world()
        out, _ = capfd.readouterr()
        assert out == ""


def test_generate_random_string():
    """Tests for generate_random_str()."""
    rng = np.random.default_rng(123)
    assert utils.generate_random_string(0, ["a"], rng) == ""
    assert utils.generate_random_string(5, ["a"], rng) == "aaaaa"
    assert len(utils.generate_random_string(5, ["a", "b"], rng)) == 5
    with pytest.raises(AssertionError):
        utils.generate_random_string(5, ["a", "bb"], rng)
