"""Tests for kitchen_doors.py."""

import numpy as np
import pytest
import robosuite
from robosuite.controllers import load_composite_controller_config


@pytest.mark.parametrize("task_name", ["OpenSingleDoor", "CloseSingleDoor"])
def test_door_angle_observable(task_name):
    """Test that door_angle observable returns normalized angles."""
    # Create environment
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
    env = robosuite.make(**config)
    
    # Reset environment
    obs = env.reset()
    
    # Get raw and normalized angles
    raw_angle = env.sim.data.qpos[env.hinge_qpos_addr]
    door_state = env.door_fxtr.get_door_state(env=env)
    normalized_angle = door_state['door']
    observable_angle = obs['door_angle']
    
    # The observable should return the normalized angle
    assert abs(observable_angle - normalized_angle) < 1e-6, \
        f"Task {task_name}: Observable angle {observable_angle} does not match normalized angle {normalized_angle}"
    
    # Print debug info
    print(f"\nTask: {task_name}")
    print(f"Raw angle: {raw_angle}")
    print(f"Normalized angle: {normalized_angle}")
    print(f"Observable angle: {observable_angle}")