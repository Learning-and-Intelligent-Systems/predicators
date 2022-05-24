"""Tests for low-level trajectory segmentation."""

import numpy as np
import pytest
from gym.spaces import Box

from predicators.src import utils
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_new_env
from predicators.src.nsrt_learning.segmentation import segment_trajectory
from predicators.src.structs import Action, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Type


def test_segment_trajectory():
    """Tests for segment_trajectory()."""
    utils.reset_config({"segmenter": "option_changes"})
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    pred1 = Predicate("Pred1", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    preds = {pred0, pred1, pred2}
    state0 = State({cup0: [0.4], cup1: [0.7], cup2: [0.1]})
    atoms0 = utils.abstract(state0, preds)
    state1 = State({cup0: [0.8], cup1: [0.3], cup2: [1.0]})
    atoms1 = utils.abstract(state1, preds)
    # Tests with known options.
    param_option = utils.SingletonParameterizedOption(
        "Dummy",
        lambda s, m, o, p: Action(p),
        types=[cup_type],
        params_space=Box(0.1, 1, (1, )),
    )
    option0 = param_option.ground([cup0], np.array([0.2]))
    assert option0.initiable(state0)
    action0 = option0.policy(state0)
    # The option changes, but the option spec stays the same. Want to segment.
    # Note that this is also a test for the case where the final option
    # terminates in the final state.
    option1 = param_option.ground([cup0], np.array([0.1]))
    assert option1.initiable(state0)
    action1 = option1.policy(state0)
    option2 = param_option.ground([cup1], np.array([0.1]))
    assert option2.initiable(state0)
    action2 = option2.policy(state0)
    known_option_ll_traj = LowLevelTrajectory(
        [state0.copy() for _ in range(5)],
        [action0, action1, action2, action0])
    trajectory = (known_option_ll_traj,
                  [atoms0, atoms0, atoms0, atoms0, atoms0])
    known_options_trajectory = trajectory  # used later in the test
    known_option_segments = segment_trajectory(trajectory)
    assert len(known_option_segments) == 4
    # Test case where the final option does not terminate in the final state.
    infinite_param_option = ParameterizedOption(
        "InfiniteDummy",
        types=[cup_type],
        params_space=Box(0.1, 1, (1, )),
        policy=lambda s, m, o, p: Action(p),
        initiable=lambda s, m, o, p: True,
        terminal=lambda s, m, o, p: False,
    )
    infinite_option = infinite_param_option.ground([cup0], np.array([0.2]))
    states = [state0.copy() for _ in range(5)]
    infinite_option.initiable(states[0])
    actions = [infinite_option.policy(s) for s in states[:-1]]
    trajectory = (LowLevelTrajectory(states, actions),
                  [atoms0, atoms0, atoms0, atoms0, atoms1])
    assert len(segment_trajectory(trajectory)) == 0

    # More tests for temporally extended options.
    def _initiable(s, m, o, p):
        del s, o, p  # unused
        m["steps_remaining"] = 3
        return True

    def _policy(s, m, o, p):
        del s, o  # unused
        m["steps_remaining"] -= 1
        return Action(p)

    def _terminal(s, m, o, p):
        del s, o, p  # unused
        return m["steps_remaining"] <= 0

    three_step_param_option = ParameterizedOption(
        "ThreeStepDummy",
        types=[cup_type],
        params_space=Box(0.1, 1, (1, )),
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
    )

    def _simulate(s, a):
        del a  # unused
        return s.copy()

    three_option0 = three_step_param_option.ground([cup0], np.array([0.2]))
    three_option1 = three_step_param_option.ground([cup0], np.array([0.2]))
    policy = utils.option_plan_to_policy([three_option0, three_option1])
    traj = utils.run_policy_with_simulator(
        policy,
        _simulate,
        state0,
        termination_function=lambda s: False,
        max_num_steps=6)
    atom_traj = [atoms0] * 3 + [atoms1] * 3 + [atoms0]
    trajectory = (traj, atom_traj)
    segments = segment_trajectory(trajectory)
    assert len(segments) == 2
    segment0 = segments[0]
    segment1 = segments[1]
    assert segment0.has_option()
    assert segment0.get_option() == three_option0
    assert segment0.init_atoms == atoms0
    assert segment0.final_atoms == atoms1
    assert segment1.has_option()
    assert segment1.get_option() == three_option1
    assert segment1.init_atoms == atoms1
    assert segment1.final_atoms == atoms0

    # Tests without known options.
    action0 = option0.policy(state0)
    action0.unset_option()
    action1 = option0.policy(state0)
    action1.unset_option()
    action2 = option1.policy(state0)
    action2.unset_option()
    trajectory = (LowLevelTrajectory([state0.copy() for _ in range(5)],
                                     [action0, action1, action2, action0]),
                  [atoms0, atoms0, atoms0, atoms0, atoms0])
    # Should crash, because the option_changes segmenter assumes that options
    # are known.
    with pytest.raises(AssertionError):
        segment_trajectory(trajectory)
    # Test oracle segmenter with known options. Should be the same as option
    # changes segmenter.
    utils.reset_config({"segmenter": "oracle"})
    known_option_segments = segment_trajectory(known_options_trajectory)
    assert len(known_option_segments) == 4
    # Segment with atoms changes instead.
    utils.reset_config({"segmenter": "atom_changes"})
    assert len(segment_trajectory(trajectory)) == 0
    unknown_option_ll_traj = LowLevelTrajectory(
        [state0.copy() for _ in range(5)] + [state1],
        [action0, action1, action2, action0, action1])
    trajectory = (unknown_option_ll_traj,
                  [atoms0, atoms0, atoms0, atoms0, atoms0, atoms1])
    unknown_option_segments = segment_trajectory(trajectory)
    assert len(unknown_option_segments) == 1
    segment = unknown_option_segments[0]
    assert len(segment.actions) == 5
    assert not segment.has_option()
    assert segment.init_atoms == atoms0
    assert segment.final_atoms == atoms1
    # Test oracle segmenter with unknown options. This segmenter uses the
    # ground truth NSRTs, so we need to use a real environment where those
    # are defined.
    utils.reset_config({
        "segmenter": "oracle",
        "option_learner": "oracle",
        "env": "cover_multistep_options",
        "cover_multistep_thr_percent": 0.99,
        "cover_multistep_bhr_percent": 0.99,
        "cover_initial_holding_prob": 0.0,
        "cover_num_blocks": 1,
        "cover_num_targets": 1,
        "num_train_tasks": 1,
        "offline_data_method": "demo",
    })
    env = create_new_env("cover_multistep_options", do_cache=False)
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 1
    dataset = create_dataset(env, train_tasks, known_options=set())
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
    assert len(ground_atom_dataset) == 1
    trajectory = ground_atom_dataset[0]
    ll_traj, atoms = trajectory
    assert train_tasks[0].goal.issubset(atoms[-1])
    assert len(ll_traj.actions) > 0
    assert not ll_traj.actions[0].has_option()
    segments = segment_trajectory(trajectory)
    # Should be 2 because the hyperparameters force the task to be exactly
    # one pick and one place.
    assert len(segments) == 2
    # Test unknown segmenter.
    utils.reset_config({"segmenter": "not a real segmenter"})
    with pytest.raises(NotImplementedError):
        segment_trajectory(trajectory)
    # Return for use elsewhere.
    return (known_option_ll_traj, known_option_segments,
            unknown_option_ll_traj, unknown_option_segments)


@pytest.mark.parametrize("env", [
    "stick_button", "cover_multistep_options", "doors", "coffee", "touch_point"
])
def test_contact_based_segmentation(env):
    """Tests for contact-based segmentation."""
    utils.reset_config({
        "segmenter": "contacts",
        "env": env,
        "num_train_tasks": 1,
        "offline_data_method": "demo",
        "doors_room_map_size": 2,
        "doors_min_room_exists_frac": 1.0,
        "doors_max_room_exists_frac": 1.0,
        "doors_birrt_smooth_amt": 0,
    })
    env = create_new_env(env, do_cache=False)
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 1
    dataset = create_dataset(env, train_tasks, env.options)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
    assert len(ground_atom_dataset) == 1
    trajectory = ground_atom_dataset[0]
    ll_traj, atoms = trajectory
    assert train_tasks[0].goal.issubset(atoms[-1])
    assert len(ll_traj.actions) > 0
    assert ll_traj.actions[0].has_option()
    segments = segment_trajectory(trajectory)
    # The options should be grouped together.
    for segment in segments:
        assert len(segment.actions) > 0
        segment_option = segment.get_option()
        for action in segment.actions:
            assert action.get_option() is segment_option


def test_contact_based_segmentation_failure_case():
    """Failure case tests for contact-based segmentation."""
    utils.reset_config({
        "segmenter": "contacts",
        "env": "not a real env",
    })
    with pytest.raises(NotImplementedError) as e:
        segment_trajectory(([], []))
    assert "Contact-based segmentation not implemented" in str(e)
