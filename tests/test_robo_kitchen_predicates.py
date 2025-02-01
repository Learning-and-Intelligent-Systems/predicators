"""Tests for robo_kitchen predicates."""

import numpy as np
import pytest
from types import SimpleNamespace

import robosuite
from robosuite.controllers import load_composite_controller_config

from predicators.envs.robo_kitchen import RoboKitchenEnv
from predicators.structs import Object, State
from predicators.settings import CFG

from robocasa.environments.kitchen.single_stage.kitchen_doors import OpenSingleDoor, CloseSingleDoor


def test_open_predicate():
    """Test the Open predicate for different door angles."""
    # Set up configuration
    CFG.seed = 0
    CFG.num_test_tasks = 1
    CFG.robo_kitchen_randomize_init_state = True
    
    # Create environment
    env = RoboKitchenEnv(use_gui=False)
    
    # Create a hinge door object
    door = Object("door", env.hinge_door_type)
    
    # Test cases with different angles
    test_cases = [
        # angle, expected_open
        (-0.1, False),  # Clearly closed
        (0.0, False),   # Closed
        (0.084, False), # At threshold
        (0.085, True),  # Just above threshold
        (0.5, True),    # Clearly open
        (1.0, True),    # Fully open
    ]
    
    for angle, expected_open in test_cases:
        # Create state with the given angle
        state = State({door: np.array([angle])})
        
        # Check if Open predicate holds
        is_open = env.Open_holds(state, [door])
        
        # Assert the result matches expectation
        assert is_open == expected_open, \
            f"Door angle {angle} expected Open={expected_open} but got {is_open}"

@pytest.mark.parametrize("task_name,expected_open", [
    ("OpenSingleDoor", False),  # Should start closed
    ("CloseSingleDoor", True),  # Should start open
])
def test_door_initial_states(task_name, expected_open):
    """Test that door tasks initialize with correct states."""
    # Set up configuration
    CFG.seed = 0
    CFG.num_test_tasks = 1
    CFG.robo_kitchen_randomize_init_state = True
    
    # Create environment with the specified task
    controller_config = load_composite_controller_config(robot="PandaOmron")
    config = {
        "env_name": task_name,
        "robots": "PandaOmron",
        "controller_configs": controller_config,
        "layout_ids": 0,
        "style_ids": 0,
        "translucent_robot": True,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_camera_obs": False,
        "control_freq": 20,
    }
    raw_env = robosuite.make(**config)
    
    # Create predicators environment
    env = RoboKitchenEnv(use_gui=False)
    env._env_raw = raw_env
    
    # Reset to get initial state
    obs = raw_env.reset()
    
    # Get door angle from observation
    raw_angle = raw_env.sim.data.qpos[raw_env.hinge_qpos_addr]
    # Get normalized door state
    door_state = raw_env.door_fxtr.get_door_state(env=raw_env)
    for k, v in door_state.items():
        print(f"Door state {k}: {v}")
    door_angle = door_state['door']  # Use normalized angle
    print(f"\nTask: {task_name}")
    print(f"Raw angle: {raw_angle}")
    print(f"Normalized angle: {door_angle}")
    print(f"Door state: {door_state}")
    print(f"Hinge qpos addr: {raw_env.hinge_qpos_addr}")
    print(f"All qpos: {raw_env.sim.data.qpos}")
    
    # Create door object and state
    door = Object("door", env.hinge_door_type)
    state = State({door: np.array([door_angle])})
    
    # Check if Open predicate matches expected initial state
    is_open = env.Open_holds(state, [door])
    assert is_open == expected_open, \
        f"Task {task_name} expected initial Open={expected_open} but got {is_open} (angle={door_angle})"