"""Tests for human low-level control with mobile Fetch."""

from predicators import utils
from predicators.approaches.human_low_level_control_approach import \
    HumanLowLevelControlApproach
from predicators.envs.pybullet_circuit import PyBulletCircuitEnv


def test_human_low_level_control_mobile_fetch_action_space(monkeypatch):
    """Ensure mobile fetch actions always fit the action space."""
    utils.reset_config({
        "env": "pybullet_circuit",
        "approach": "human_low_level_control",
        "pybullet_robot": "mobile_fetch",
        "pybullet_control_mode": "position",
        "use_gui": False,
    })
    env = PyBulletCircuitEnv(use_gui=False)
    state = env.reset("test", 0)

    approach = HumanLowLevelControlApproach(env.predicates, set(), env.types,
                                            env.action_space,
                                            env.get_train_tasks())
    monkeypatch.setattr(approach, "_print_instructions", lambda: None)
    monkeypatch.setattr(approach, "_setup_terminal", lambda: None)
    monkeypatch.setattr(approach, "_get_pressed_key", lambda: None)

    policy = approach.solve(env.get_test_tasks()[0], timeout=1)
    action = policy(state)
    assert env.action_space.contains(action.arr)
