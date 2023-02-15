"""Tests for PyBullet motion planning."""

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.motion_planning import run_motion_planning
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot


def test_move_to_shelf():
    """Test for Panda robot moving to put a held block into a shelf.

    Notably, the robot must change its gripper orientation from top-down
    to forward-facing, so motion planning must be in position and
    orientation.
    """
    utils.reset_config({
        # TODO: the "digital twin" robot should always be in reset mode?
        "pybullet_control_mode": "reset"
    })

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
    shelf_y = y_ub - shelf_length / 2
    block_color = (1.0, 0.0, 0.0, 1.0)
    block_size = 0.04
    block_x = (x_lb + x_ub) / 2
    block_y = y_lb + block_size
    block_z = table_height + block_size / 2
    offset_z = 0.01
    obj_mass = 0.5
    obj_friction = 1.2
    camera_distance = 0.8
    camera_yaw = 90.0
    camera_pitch = -24
    camera_target = (1.65, 0.75, 0.42)
    robot_ee_home_orn = (0.7071, 0.7071, 0.0, 0.0)
    home_pose = Pose((block_x, block_y, block_z + offset_z), robot_ee_home_orn)
    # Target for motion planning.
    tx = shelf_x
    ty = shelf_y
    tz = table_height + shelf_base_height + block_size / 2 + offset_z
    target_orn = (0.7071, 0.0, 0.7071, 0.0)
    target_pose = Pose((tx, ty, tz), target_orn)

    physics_client_id = p.connect(p.GUI)  # TODO change to direct
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,
                               False,
                               physicsClientId=physics_client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                               False,
                               physicsClientId=physics_client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                               False,
                               physicsClientId=physics_client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                               False,
                               physicsClientId=physics_client_id)
    p.resetDebugVisualizerCamera(camera_distance,
                                 camera_yaw,
                                 camera_pitch,
                                 camera_target,
                                 physicsClientId=physics_client_id)

    # Draw the target.
    p.addUserDebugText("*",
                       target_pose.position, [1.0, 0.0, 0.0],
                       physicsClientId=physics_client_id)

    # Load table in both the main client and the copy.
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
    robot.set_motors(joint_state)

    # Create holding transform.
    held_obj_id = block_id
    base_link_to_world = np.r_[p.invertTransform(
        *p.getLinkState(robot.robot_id,
                        robot.end_effector_id,
                        physicsClientId=physics_client_id)[:2])]
    world_to_obj = np.r_[p.getBasePositionAndOrientation(
        held_obj_id, physicsClientId=physics_client_id)]
    held_obj_to_base_link = p.invertTransform(
        *p.multiplyTransforms(base_link_to_world[:3], base_link_to_world[3:],
                              world_to_obj[:3], world_to_obj[3:]))

    # Force move to target.
    robot_state = tuple(target_pose.position) + \
        tuple(target_pose.orientation) + (robot.closed_fingers, )
    robot.reset_state(robot_state)
    target_positions = robot.get_joints()

    # Move back to start.
    robot_state = tuple(home_pose.position) + \
        tuple(home_pose.orientation) + (robot.closed_fingers, )
    robot.reset_state(robot_state)

    # Apply the holding transform.
    # world_to_base_link = get_link_state(
    #     robot.robot_id,
    #     robot.end_effector_id,
    #     physics_client_id=physics_client_id).com_pose
    # base_link_to_held_obj = p.invertTransform(*held_obj_to_base_link)
    # world_to_held_obj = p.multiplyTransforms(world_to_base_link[0],
    #                                          world_to_base_link[1],
    #                                          base_link_to_held_obj[0],
    #                                          base_link_to_held_obj[1])
    # p.resetBasePositionAndOrientation(held_obj_id,
    #                                   world_to_held_obj[0],
    #                                   world_to_held_obj[1],
    #                                   physicsClientId=physics_client_id)

    collision_bodies = {shelf_id}

    initial_positions = robot.get_joints()
    plan = run_motion_planning(robot,
                               initial_positions,
                               target_positions,
                               collision_bodies,
                               seed=123,
                               physics_client_id=physics_client_id)
    assert plan is not None

    # Replay the plan.
    for state in plan:
        robot.set_joints(state)
        import time
        for _ in range(100):
            p.stepSimulation(physics_client_id)
            time.sleep(0.001)
