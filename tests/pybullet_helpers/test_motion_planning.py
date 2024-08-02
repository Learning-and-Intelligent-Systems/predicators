"""Tests for PyBullet motion planning."""

import time

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import create_pybullet_block
from predicators.pybullet_helpers.camera import create_gui_connection
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.motion_planning import run_motion_planning
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot

USE_GUI = False


def test_run_motion_planning(physics_client_id):
    """Tests for run_motion_planning()."""
    ee_home_position = (1.35, 0.75, 0.75)
    ee_orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
    ee_home_pose = Pose(ee_home_position, ee_orn)
    seed = 123
    robot = create_single_arm_pybullet_robot("fetch", physics_client_id,
                                             ee_home_pose)
    robot_init_state = tuple(ee_home_position) + tuple(
        ee_orn, ) + (robot.open_fingers, )
    robot.reset_state(robot_init_state)
    joint_initial = robot.get_joints()
    # Should succeed with a path of length 2.
    joint_target = list(joint_initial)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies=set(),
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert len(path) == 2
    assert np.allclose(path[0], joint_initial)
    assert np.allclose(path[-1], joint_target)
    # Should succeed, no collisions.
    ee_target_position = np.add(ee_home_position, (0.0, 0.0, -0.05))
    ee_target = Pose(ee_target_position, ee_orn)
    joint_target = robot.inverse_kinematics(ee_target, validate=True)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies=set(),
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert np.allclose(path[0], joint_initial)
    assert np.allclose(path[-1], joint_target)
    # Should fail because the target collides with the table.
    table_pose = (1.35, 0.75, 0.0)
    table_orientation = [0., 0., 0., 1.]
    table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                          useFixedBase=True,
                          physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(table_id,
                                      table_pose,
                                      table_orientation,
                                      physicsClientId=physics_client_id)
    ee_target_position = np.add(ee_home_position, (0.0, 0.0, -0.6))
    ee_target = Pose(ee_target_position, ee_orn)
    joint_target = robot.inverse_kinematics(ee_target, validate=True)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies={table_id},
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert path is None
    # Should fail because the initial state collides with the table.
    path = run_motion_planning(robot,
                               joint_target,
                               joint_initial,
                               collision_bodies={table_id},
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert path is None
    # Should succeed, but will need to move the arm up to avoid the obstacle.
    block_pose = (1.35, 0.6, 0.5)
    block_orientation = [0., 0., 0., 1.]
    block_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.2, 0.01, 0.3),
        mass=0,  # immoveable
        friction=1,
        orientation=block_orientation,
        physics_client_id=physics_client_id)
    p.resetBasePositionAndOrientation(block_id,
                                      block_pose,
                                      block_orientation,
                                      physicsClientId=physics_client_id)
    ee_target_position = (1.35, 0.4, 0.6)
    ee_target = Pose(ee_target_position, ee_orn)
    joint_target = robot.inverse_kinematics(ee_target, validate=True)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies={table_id, block_id},
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert path is not None
    p.removeBody(block_id, physicsClientId=physics_client_id)
    # Should fail because the hyperparameters are too limited.
    utils.reset_config({
        "pybullet_birrt_num_iters": 1,
        "pybullet_birrt_num_attempts": 1,
    })
    block_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.2, 0.01, 0.3),
        mass=0,  # immoveable
        friction=1,
        orientation=block_orientation,
        physics_client_id=physics_client_id)
    p.resetBasePositionAndOrientation(block_id,
                                      block_pose,
                                      block_orientation,
                                      physicsClientId=physics_client_id)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies={table_id, block_id},
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert path is None
    p.removeBody(block_id, physicsClientId=physics_client_id)


def test_move_to_shelf():
    """Test for Panda robot moving to put a held block into a shelf.

    Notably, the robot must change its gripper orientation from top-down
    to forward-facing, so motion planning must be in position and
    orientation.

    Also notably, the held object must be collision-checked like the robot.
    """
    utils.reset_config({"pybullet_control_mode": "reset"})

    # Set up scene.
    x_lb = 1.2
    x_ub = 1.5
    y_lb = 0.4
    y_ub = 1.1
    pick_z = 0.75
    default_orn = (0.0, 0.0, 0.0, 1.0)
    table_pose = (1.35, 0.75, 0.0)
    table_orientation = (0., 0., 0., 1.)
    table_height = 0.2
    shelf_width = (x_ub - x_lb) * 0.4
    shelf_length = (y_ub - y_lb) * 0.6
    shelf_base_height = pick_z * 0.8
    shelf_ceiling_height = pick_z * 0.2
    shelf_ceiling_thickness = 0.01
    shelf_pole_girth = 0.01
    shelf_color = (0.5, 0.3, 0.05, 1.0)
    shelf_x = x_ub - shelf_width / 2
    shelf_y = y_lb + shelf_length / 2
    block_color = (1.0, 0.0, 0.0, 1.0)
    block_size = 0.05
    block_x = (x_lb + x_ub) / 2
    block_y = y_ub - block_size
    block_z = table_height + block_size / 2
    offset_z = 0.01
    obj_mass = 0.5
    obj_friction = 1.2
    robot_ee_home_orn = (0.7071, 0.7071, 0.0, 0.0)
    home_pose = Pose((block_x, block_y, block_z + offset_z), robot_ee_home_orn)

    # Target for motion planning.
    tx = shelf_x
    ty = shelf_y
    tz = table_height + shelf_base_height + block_size / 2 + offset_z
    target_orn = (0.7071, 0.0, 0.7071, 0.0)
    target_pose = Pose((tx, ty, tz), target_orn)

    if USE_GUI:  # pragma: no cover
        physics_client_id = create_gui_connection()
        # Draw the target.
        p.addUserDebugText("*",
                           target_pose.position, [1.0, 0.0, 0.0],
                           physicsClientId=physics_client_id)
    else:
        physics_client_id = p.connect(p.DIRECT)

    # Load table.
    table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                          useFixedBase=True,
                          physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(table_id,
                                      table_pose,
                                      table_orientation,
                                      physicsClientId=physics_client_id)

    # Create shelf.
    color = shelf_color
    orientation = default_orn
    base_pose = (shelf_x, shelf_y, table_height + shelf_base_height / 2)
    # Shelf base.
    # Create the collision shape.
    base_half_extents = [
        shelf_width / 2, shelf_length / 2, shelf_base_height / 2
    ]
    base_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=base_half_extents,
        physicsClientId=physics_client_id)
    # Create the visual shape.
    base_visual_id = p.createVisualShape(p.GEOM_BOX,
                                         halfExtents=base_half_extents,
                                         rgbaColor=color,
                                         physicsClientId=physics_client_id)
    # Create the ceiling.
    link_positions = []
    link_collision_shape_indices = []
    link_visual_shape_indices = []
    pose = (
        0, 0,
        shelf_base_height / 2 + shelf_ceiling_height - \
            shelf_ceiling_thickness / 2
    )
    link_positions.append(pose)
    half_extents = [
        shelf_width / 2, shelf_length / 2, shelf_ceiling_thickness / 2
    ]
    collision_id = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=half_extents,
                                          physicsClientId=physics_client_id)
    link_collision_shape_indices.append(collision_id)
    visual_id = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=half_extents,
                                    rgbaColor=color,
                                    physicsClientId=physics_client_id)
    link_visual_shape_indices.append(visual_id)
    # Create poles connecting the base to the ceiling.
    for x_sign in [-1, 1]:
        for y_sign in [-1, 1]:
            pose = (x_sign * (shelf_width - shelf_pole_girth) / 2,
                    y_sign * (shelf_length - shelf_pole_girth) / 2,
                    shelf_base_height / 2 + shelf_ceiling_height / 2)
            link_positions.append(pose)
            half_extents = [
                shelf_pole_girth / 2, shelf_pole_girth / 2,
                shelf_ceiling_height / 2
            ]
            collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=physics_client_id)
            link_collision_shape_indices.append(collision_id)
            visual_id = p.createVisualShape(p.GEOM_BOX,
                                            halfExtents=half_extents,
                                            rgbaColor=color,
                                            physicsClientId=physics_client_id)
            link_visual_shape_indices.append(visual_id)

    # Create the whole body.
    num_links = len(link_positions)
    assert len(link_collision_shape_indices) == num_links
    assert len(link_visual_shape_indices) == num_links
    link_masses = [0.1 for _ in range(num_links)]
    link_orientations = [orientation for _ in range(num_links)]
    link_intertial_frame_positions = [[0, 0, 0] for _ in range(num_links)]
    link_intertial_frame_orns = [[0, 0, 0, 1] for _ in range(num_links)]
    link_parent_indices = [0 for _ in range(num_links)]
    link_joint_types = [p.JOINT_FIXED for _ in range(num_links)]
    link_joint_axis = [[0, 0, 0] for _ in range(num_links)]
    shelf_id = p.createMultiBody(
        baseCollisionShapeIndex=base_collision_id,
        baseVisualShapeIndex=base_visual_id,
        basePosition=base_pose,
        baseOrientation=orientation,
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_shape_indices,
        linkVisualShapeIndices=link_visual_shape_indices,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=link_intertial_frame_positions,
        linkInertialFrameOrientations=link_intertial_frame_orns,
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axis,
        physicsClientId=physics_client_id)

    # Create block.
    color = block_color
    half_extents = (block_size / 2.0, block_size / 2.0, block_size / 2.0)
    block_id = create_pybullet_block(color, half_extents, obj_mass,
                                     obj_friction, default_orn,
                                     physics_client_id)
    p.resetBasePositionAndOrientation(block_id, [block_x, block_y, block_z],
                                      default_orn,
                                      physicsClientId=physics_client_id)

    # Create robot, initialized to be grasping the block.
    robot = create_single_arm_pybullet_robot("panda", physics_client_id,
                                             home_pose)
    # Close the fingers.
    joint_state = robot.get_joints()
    joint_state[robot.left_finger_joint_idx] = robot.closed_fingers
    joint_state[robot.right_finger_joint_idx] = robot.closed_fingers
    robot.set_joints(joint_state)

    # Create holding transform.
    held_obj_id = block_id
    world_to_base_link = get_link_state(
        robot.robot_id,
        robot.end_effector_id,
        physics_client_id=physics_client_id).com_pose
    base_link_to_world = np.r_[p.invertTransform(world_to_base_link[0],
                                                 world_to_base_link[1])]
    world_to_obj = np.r_[p.getBasePositionAndOrientation(
        held_obj_id, physicsClientId=physics_client_id)]
    held_obj_to_base_link = p.invertTransform(
        *p.multiplyTransforms(base_link_to_world[:3], base_link_to_world[3:],
                              world_to_obj[:3], world_to_obj[3:]))
    base_link_to_held_obj = p.invertTransform(*held_obj_to_base_link)

    def _set_state(pt: JointPositions) -> None:
        robot.set_joints(pt)
        world_to_base_link = get_link_state(
            robot.robot_id,
            robot.end_effector_id,
            physics_client_id=physics_client_id).com_pose
        world_to_held_obj = p.multiplyTransforms(world_to_base_link[0],
                                                 world_to_base_link[1],
                                                 base_link_to_held_obj[0],
                                                 base_link_to_held_obj[1])
        p.resetBasePositionAndOrientation(held_obj_id,
                                          world_to_held_obj[0],
                                          world_to_held_obj[1],
                                          physicsClientId=physics_client_id)

    # Force move to target to get the target joint positions.
    robot_state = tuple(target_pose.position) + \
        tuple(target_pose.orientation) + (robot.closed_fingers, )
    robot.reset_state(robot_state)
    target_positions = robot.get_joints()

    # Move back to start, but slightly up so that the robot is not in collision
    # with the table.
    x, y, z = home_pose.position
    z += offset_z
    robot_state = (x, y, z) + \
        tuple(home_pose.orientation) + (robot.closed_fingers, )
    robot.reset_state(robot_state)
    initial_positions = robot.get_joints()
    _set_state(initial_positions)

    collision_bodies = {shelf_id, table_id}
    plan = run_motion_planning(robot,
                               initial_positions,
                               target_positions,
                               collision_bodies,
                               held_object=held_obj_id,
                               base_link_to_held_obj=base_link_to_held_obj,
                               seed=123,
                               physics_client_id=physics_client_id)
    assert plan is not None

    # Replay the plan.
    if USE_GUI:  # pragma: no cover
        for state in plan:
            _set_state(state)
            for _ in range(100):
                p.stepSimulation(physicsClientId=physics_client_id)
                time.sleep(0.001)
