"""Tests for robo_kitchen action conversion."""

import numpy as np
import pytest
import warnings

from predicators.envs.robo_kitchen import RoboKitchenEnv, MAX_CARTESIAN_DISPLACEMENT, MAX_ROTATION_DISPLACEMENT
from predicators.structs import Action
from predicators.settings import CFG

# Set up configuration
CFG.seed = 0
CFG.num_test_tasks = 1
CFG.robo_kitchen_randomize_init_state = True

def get_eef_pos(state_info):
    """Helper to get end effector position from state info."""
    return state_info["robot0_eef_pos"]

def test_action_conversion():
    """Test that predicators actions are correctly converted to robocasa actions."""
    # Create environment
    env = RoboKitchenEnv(use_gui=False)
    
    # Reset environment
    obs = env.reset("test", 0)
    assert isinstance(obs, dict), "Observation should be a dict"
    
    # Test cases with different actions and expected direction changes
    test_cases = [
        # (action_array, expected_direction)
        # Test case 1: No movement
        (np.zeros(7), np.zeros(3)),
        
        # Test case 2: Only position movement
        (np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),  # +x
        (np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # +y
        (np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),  # +z
        
        # Test case 3: Only rotation movement (position should stay relatively stable)
        (np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.zeros(3)),  # Roll
        (np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), np.zeros(3)),  # Pitch
        (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), np.zeros(3)),  # Yaw
        
        # Test case 4: Only gripper movement (position should stay stable)
        (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), np.zeros(3)),  # close gripper
        (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]), np.zeros(3)), # open gripper
        
        # Test case 5: Combined movement
        (np.array([0.5, -0.5, 0.5, 0.2, -0.2, 0.2, -1.0]), np.array([0.5, -0.5, 0.5])),
    ]
    
    for action_arr, expected_direction in test_cases:
        # Verify observation structure
        assert isinstance(obs, dict), "Observation should be a dict"
        assert "state_info" in obs, "Observation should have state_info"
        assert "obs_images" in obs, "Observation should have obs_images"
        assert isinstance(obs["state_info"], dict), "State info should be a dict"
                # Get initial position
        initial_pos = get_eef_pos(obs["state_info"])
        
        # Create and execute predicators action
        action = Action(action_arr)
        for _ in range(6):
            obs = env.step(action)
        
        # Let the movement settle with zero actions
        for _ in range(6):
            obs = env.step(Action(np.zeros(7)))
        
        # Get final position
        final_pos = get_eef_pos(obs["state_info"])
        
        # Calculate actual movement direction
        movement = final_pos - initial_pos
        
        # For non-zero expected movements, verify direction
        eps_movement = 0.03
        if not np.allclose(expected_direction, 0):
            # Normalize vectors for direction comparison
            if np.linalg.norm(movement) > eps_movement:  # Only normalize if movement is non-negligible
                movement_normalized = movement / np.linalg.norm(movement)
                expected_normalized = expected_direction / np.linalg.norm(expected_direction)
                
                # Check if movement is in roughly the right direction (dot product > 0.7)
                dot_product = np.dot(movement_normalized, expected_normalized)
                assert dot_product > 0.7, \
                    f"Movement direction mismatch. Expected {expected_direction}, got {movement}. " \
                    f"Dot product: {dot_product}"
        else:
            # For zero expected movement (rotation/gripper actions), verify position stayed relatively stable
            assert np.linalg.norm(movement) < eps_movement, \
                f"Expected no movement but got {movement}"
        


def test_action_space():
    """Test that action space is correctly defined."""
    env = RoboKitchenEnv(use_gui=False)
    
    # Check action space
    assert env.action_space.shape == (7,), \
        f"Expected 7D action space, got {env.action_space.shape}D"
    assert np.allclose(env.action_space.low, -1.0), \
        f"Expected action space low to be -1.0, got {env.action_space.low}"
    assert np.allclose(env.action_space.high, 1.0), \
        f"Expected action space high to be 1.0, got {env.action_space.high}"

def test_robo_kitchen_training_task_generation():
    """Test that we can generate training tasks for RoboKitchenEnv."""
    # Create env without GUI for testing
    env = RoboKitchenEnv(use_gui=False)
    
    # Set a small number of training tasks for testing
    CFG.num_train_tasks = 1
    
    # Generate training tasks
    train_tasks = env._generate_train_tasks()
    
    # Verify we got the expected number of tasks
    assert len(train_tasks) == CFG.num_train_tasks
    
    # Verify task structure
    task = train_tasks[0]
    assert hasattr(task, "init_obs")
    assert hasattr(task, "goal_description")
    
    # Verify observation structure
    assert "state_info" in task.init_obs
    assert "obs_images" in task.init_obs
    
    # Verify goal description matches selected task
    assert task.goal_description == env.task_selected
    
    # Verify state info contains expected keys for a robosuite env
    state_info = task.init_obs["state_info"]
    # Print state info keys for debugging
    print("\nState info keys:", list(state_info.keys()))
    expected_keys = ["base_pos", "base_rot", "eef_pos", "eef_rot", "mount_pos", "mount_rot"]
    
    # Issue warning about state info keys being different from online generated data
    warnings.warn("State info keys differ from those in online generated data", UserWarning)
    
    for key in expected_keys:
        assert key in state_info, f"Missing expected key {key} in state_info"