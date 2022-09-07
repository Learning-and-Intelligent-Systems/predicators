"""Functions that return motion plans (i.e: series of low-level motions) for
the BEHAVIOR Robot to execute."""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import scipy
from numpy.random._generator import Generator

from predicators.behavior_utils.behavior_utils import detect_robot_collision, \
    get_aabb_volume, get_closest_point_on_aabb, get_scene_body_ids, \
    reset_and_release_hand
from predicators.structs import Array

try:
    from igibson import object_states
    from igibson.envs.behavior_env import \
        BehaviorEnv  # pylint: disable=unused-import
    from igibson.external.pybullet_tools.utils import CIRCULAR_LIMITS
    from igibson.object_states.on_floor import \
        RoomFloor  # pylint: disable=unused-import
    from igibson.objects.articulated_object import URDFObject
    from igibson.utils.behavior_robot_planning_utils import \
        plan_base_motion_br, plan_hand_motion_br

except (ImportError, ModuleNotFoundError) as e:
    pass


def make_dummy_plan(
    env: "BehaviorEnv",
    obj: Union["URDFObject", "RoomFloor"],
    continuous_params: Array,
    rng: Optional[Generator] = None
) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """Function to return a defualt 'dummy' plan.

    This is useful when implementing option models/controllers where a
    plan is not actually necessary (e.g. magic open and close actions).
    Note though that doing this is technically cheating...
    """
    del env, obj, continuous_params, rng
    return ([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]])


def make_navigation_plan(
    env: "BehaviorEnv",
    obj: Union["URDFObject", "RoomFloor"],
    pos_offset: Array,
    rng: Optional[Generator] = None
) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """Function to return a series of actions to navigate to a particular
    offset from an object's position.

    If the navigation is infeasible (i.e, final pose is in collision),
    returns an indication to this effect (None). Otherwise, returns the
    plan, which is a series of (x, y, rot) for the base to move to. Also
    returns the original euler angle orientation of the robot body. Note
    that depending on CFG.behavior_option_model_rrt, this function might
    or might not run RRT to find the plan. If it does not run RRT, the
    plan will only be one step (i.e, the final pose to be navigated to).
    """
    if rng is None:
        rng = np.random.default_rng(23)

    logging.info(f"PRIMITIVE: Attempting to navigate to {obj.name} with "
                 f"params {pos_offset}")

    # test agent positions around an obj
    # try to place the agent near the object, and rotate it to the object
    valid_position = None  # ((x,y,z),(roll, pitch, yaw))
    original_orientation = p.getEulerFromQuaternion(
        env.robots[0].get_orientation())
    state = p.saveState()

    def sample_fn(env: "BehaviorEnv",
                  rng: Generator) -> Tuple[float, float, float]:
        random_point = env.scene.get_random_point(rng=rng)
        x, y = random_point[1][:2]
        theta = (
            rng.random() *
            (CIRCULAR_LIMITS[1] - CIRCULAR_LIMITS[0])) + CIRCULAR_LIMITS[0]
        return (x, y, theta)

    if not isinstance(
            obj,
            URDFObject):  # must be a URDFObject so we can get its position!
        logging.error("ERROR! Object to navigate to is not valid (not an "
                      "instance of URDFObject).")
        p.restoreState(state)
        p.removeState(state)
        logging.error(f"PRIMITIVE: navigate to {obj.name} with params "
                      f"{pos_offset} fail")
        return None

    obj_pos = obj.get_position()
    pos = [
        pos_offset[0] + obj_pos[0],
        pos_offset[1] + obj_pos[1],
        env.robots[0].initial_z_offset,
    ]
    yaw_angle = np.arctan2(pos_offset[1], pos_offset[0]) - np.pi
    orn = [0, 0, yaw_angle]
    env.robots[0].set_position_orientation(pos, p.getQuaternionFromEuler(orn))
    eye_pos = env.robots[0].parts["eye"].get_position()
    ray_test_res = p.rayTest(eye_pos, obj_pos)
    # Test to see if the robot is obstructed by some object, but make sure
    # that object is not either the robot's body or the object we want to
    # pick up!
    blocked = len(ray_test_res) > 0 and (ray_test_res[0][0] not in (
        env.robots[0].parts["body"].get_body_id(),
        obj.get_body_id(),
    ))
    if not detect_robot_collision(env.robots[0]) and not blocked:
        valid_position = (pos, orn)

    if valid_position is None:
        if blocked:
            logging.warning("WARNING: Position commanded is blocked!")
        else:
            logging.warning("WARNING: Position commanded is in collision!")
        p.restoreState(state)
        p.removeState(state)
        logging.warning(f"PRIMITIVE: navigate to {obj.name} with params "
                        f"{pos_offset} fail")
        return None

    p.restoreState(state)
    end_conf = [
        valid_position[0][0],
        valid_position[0][1],
        valid_position[1][2],
    ]
    if env.use_rrt:
        obstacles = get_scene_body_ids(env)
        if env.robots[0].parts["right_hand"].object_in_hand is not None:
            obstacles.remove(env.robots[0].parts["right_hand"].object_in_hand)
        plan = plan_base_motion_br(
            robot=env.robots[0],
            end_conf=end_conf,
            base_limits=(),
            obstacles=obstacles,
            override_sample_fn=lambda: sample_fn(env, rng),
            rng=rng,
        )
        p.restoreState(state)
    else:
        pos = env.robots[0].get_position()
        plan = [[pos[0], pos[1], original_orientation[2]], end_conf]

    if plan is None:
        p.restoreState(state)
        p.removeState(state)
        logging.info(f"PRIMITIVE: navigate to {obj.name} with params "
                     f"{pos_offset} failed; birrt failed to sample a plan!")
        return None

    p.restoreState(state)
    p.removeState(state)

    plan = [list(waypoint) for waypoint in plan]
    logging.info(f"PRIMITIVE: navigate to {obj.name} success! Plan found with "
                 f"continuous params {pos_offset}.")
    return plan, original_orientation


def make_grasp_plan(
    env: "BehaviorEnv",
    obj: Union["URDFObject", "RoomFloor"],
    grasp_offset: Array,
    rng: Optional[Generator] = None,
) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """Function to return a series of actions to grasp an object at a
    particular offset from an object's position.

    If the grasp is infeasible (i.e, final pose is in collision or agent
    is already holding a different object, etc.), returns an indication
    to this effect (None). Otherwise, returns the plan, which is (x, y,
    z, roll, pitch, yaw) waypoints for the hand to pass through. Also
    returns the original euler angle orientation of the hand. Note that
    depending on CFG.behavior_option_model_rrt, this function might or
    might not run RRT to find the plan. If it does not run RRT, the plan
    will only be one step (i.e, the pose to move the hand to to try
    grasping the object).
    """
    if rng is None:
        rng = np.random.default_rng(23)

    logging.info(f"PRIMITIVE: Attempting to grasp {obj.name} with params "
                 f"{grasp_offset}")

    obj_in_hand = env.robots[0].parts["right_hand"].object_in_hand
    # If we're holding something, fail and return None
    if obj_in_hand is not None:
        logging.info(f"PRIMITIVE: grasp {obj.name} fail, agent already has an "
                     "object in hand!")
        return None
    reset_and_release_hand(env)  # first reset the hand's internal states

    # If the object we're trying to grasp doesn't have all the attributes
    # we'll need for assistive grasping, fail and return None
    if not (isinstance(obj, URDFObject) and hasattr(obj, "states")
            and object_states.AABB in obj.states):
        logging.info(f"PRIMITIVE: grasp {obj.name} fail, no object")
        return None

    lo, hi = obj.states[object_states.AABB].get_value()
    volume = get_aabb_volume(lo, hi)

    # If the object is too big to be grasped, or bolted to its surface,
    # fail and return None
    if not (volume < 0.3 * 0.3 * 0.3 and
            not obj.main_body_is_fixed):  # say we can only grasp small objects
        logging.info(f"PRIMITIVE: grasp {obj.name} fail, too big or fixed")
        return None

    # If the object is too far away, fail and return None
    if (np.linalg.norm(
            np.array(obj.get_position()) -
            np.array(env.robots[0].get_position())) > 2):
        logging.info(f"PRIMITIVE: grasp {obj.name} fail, too far")
        return None

    # Grasping Phase 1: Compute the position and orientation of
    # the hand based on the provided continuous parameters and
    # try to create a plan to it.
    obj_pos = obj.get_position()
    x = obj_pos[0] + grasp_offset[0]
    y = obj_pos[1] + grasp_offset[1]
    z = obj_pos[2] + grasp_offset[2]
    hand_x, hand_y, hand_z = (env.robots[0].parts["right_hand"].get_position())
    minx = min(x, hand_x) - 0.5
    miny = min(y, hand_y) - 0.5
    minz = min(z, hand_z) - 0.5
    maxx = max(x, hand_x) + 0.5
    maxy = max(y, hand_y) + 0.5
    maxz = max(z, hand_z) + 0.5

    # compute the angle the hand must be in such that it can
    # grasp the object from its current offset position
    # This involves aligning the z-axis (in the world frame)
    # of the hand with the vector that goes from the hand
    # to the object. We can find the rotation matrix that
    # accomplishes this rotation by following:
    # https://math.stackexchange.com/questions/180418/
    # calculate-rotation-matrix-to-align-vector-a-to-vector
    # -b-in-3d
    hand_to_obj_vector = np.array(grasp_offset[:3])
    hand_to_obj_unit_vector = hand_to_obj_vector / \
        np.linalg.norm(
        hand_to_obj_vector
    )
    unit_z_vector = np.array([0.0, 0.0, -1.0])
    # This is because we assume the hand is originally oriented
    # so -z is coming out of the palm
    c_var = np.dot(unit_z_vector, hand_to_obj_unit_vector)
    if c_var not in [-1.0, 1.0]:
        v_var = np.cross(unit_z_vector, hand_to_obj_unit_vector)
        s_var = np.linalg.norm(v_var)
        v_x = np.array([
            [0, -v_var[2], v_var[1]],
            [v_var[2], 0, -v_var[0]],
            [-v_var[1], v_var[0], 0],
        ])
        R = (np.eye(3) + v_x + np.linalg.matrix_power(v_x, 2) * ((1 - c_var) /
                                                                 (s_var**2)))
        r = scipy.spatial.transform.Rotation.from_matrix(R)
        euler_angles = r.as_euler("xyz")
    else:
        if c_var == 1.0:
            euler_angles = np.zeros(3, dtype=float)
        else:
            euler_angles = np.array([0.0, np.pi, 0.0])

    state = p.saveState()
    end_conf = [
        x,
        y,
        z,
        euler_angles[0],
        euler_angles[1],
        euler_angles[2],
    ]
    if env.use_rrt:
        # plan a motion to the pose [x, y, z, euler_angles[0],
        # euler_angles[1], euler_angles[2]]
        plan = plan_hand_motion_br(
            robot=env.robots[0],
            obj_in_hand=None,
            end_conf=end_conf,
            hand_limits=((minx, miny, minz), (maxx, maxy, maxz)),
            obstacles=get_scene_body_ids(env,
                                         include_self=True,
                                         include_right_hand=True),
            rng=rng,
        )
        p.restoreState(state)
    else:
        pos = env.robots[0].parts["right_hand"].get_position()
        plan = [[pos[0], pos[1], pos[2]] + list(
            p.getEulerFromQuaternion(
                env.robots[0].parts["right_hand"].get_orientation())),
                end_conf]

    # NOTE: This below line is *VERY* important after the
    # pybullet state is restored. The hands keep an internal
    # track of their state, and if we don't reset this
    # state to mirror the actual pybullet state, the hand will
    # think it's elsewhere and update incorrectly accordingly
    env.robots[0].parts["right_hand"].set_position(
        env.robots[0].parts["right_hand"].get_position())
    env.robots[0].parts["left_hand"].set_position(
        env.robots[0].parts["left_hand"].get_position())

    # If RRT planning fails, fail and return None
    if plan is None:
        logging.info(f"PRIMITIVE: grasp {obj.name} fail, failed "
                     f"to find plan to continuous params {grasp_offset}")
        return None

    # Grasping Phase 2: Move along the vector from the
    # position the hand ends up in to the object and
    # then try to grasp.
    hand_pos = plan[-1][0:3]
    hand_orn = plan[-1][3:6]
    # Get the closest point on the object's bounding
    # box at which we can try to put the hand
    closest_point_on_aabb = get_closest_point_on_aabb(hand_pos, lo, hi)
    delta_pos_to_obj = [
        closest_point_on_aabb[0] - hand_pos[0],
        closest_point_on_aabb[1] - hand_pos[1],
        closest_point_on_aabb[2] - hand_pos[2],
    ]
    # we want to accomplish the motion in 25 timesteps
    # NOTE: this is an arbitrary choice
    delta_step_to_obj = [delta_pos / 25.0 for delta_pos in delta_pos_to_obj]

    # move the hand along the vector to the object until it
    # touches the object
    for _ in range(25):
        new_hand_pos = [
            hand_pos[0] + delta_step_to_obj[0],
            hand_pos[1] + delta_step_to_obj[1],
            hand_pos[2] + delta_step_to_obj[2],
        ]
        plan.append(new_hand_pos + list(hand_orn))
        hand_pos = new_hand_pos

    p.restoreState(state)
    p.removeState(state)
    original_orientation = list(
        p.getEulerFromQuaternion(
            env.robots[0].parts["right_hand"].get_orientation()))

    logging.info(f"PRIMITIVE: grasp {obj.name} success! Plan found with "
                 f"continuous params {grasp_offset}.")
    return plan, original_orientation


def make_place_plan(
    env: "BehaviorEnv",
    obj: Union["URDFObject", "RoomFloor"],
    place_rel_pos: Array,
    rng: Optional[Generator] = None,
) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """Function to return a series of actions to place an object at a
    particular offset from another object's position.

    If the placement is infeasible (i.e, final pose is in collision or
    agent is not currently holding an object, etc.), returns an
    indication to this effect (None). Otherwise, returns the plan, which
    is (x, y, z, roll, pitch, yaw) waypoints for the hand to pass
    through. Also returns the original euler angle orientation of the
    hand. Note that depending on CFG.behavior_option_model_rrt, this
    function might or might not run RRT to find the plan. If it does not
    run RRT, the plan will only be one step (i.e, the pose to move the
    hand to to try placing the object).
    """
    if rng is None:
        rng = np.random.default_rng(23)

    try:
        obj_in_hand_idx = env.robots[0].parts["right_hand"].object_in_hand
        obj_in_hand = [
            obj for obj in env.scene.get_objects()
            if obj.get_body_id() == obj_in_hand_idx
        ][0]
        logging.info(f"PRIMITIVE: attempt to place {obj_in_hand.name} ontop"
                     f"/inside {obj.name} with params {place_rel_pos}")
    except ValueError:
        logging.info("Cannot place; either no object in hand or holding "
                     "the object to be placed on-top/inside of!")
        return None

    # if the object in the agent's hand is None or not equal to the object
    # passed in as an argument to this option, fail and return None
    if not (obj_in_hand is not None and obj_in_hand != obj):
        logging.info("Cannot place; either no object in hand or holding "
                     "the object to be placed on top of!")
        return None

    # if the object is not a urdf object, fail and return None
    if not isinstance(obj, URDFObject):
        logging.info(f"PRIMITIVE: place {obj_in_hand.name} ontop "
                     f"{obj.name} fail, too far")
        return None

    state = p.saveState()
    # To check if object fits on place location
    p.restoreState(state)
    # NOTE: This below line is *VERY* important after the
    # pybullet state is restored. The hands keep an internal
    # track of their state, and if we don't reset their this
    # state to mirror the actual pybullet state, the hand will
    # think it's elsewhere and update incorrectly accordingly
    env.robots[0].parts["right_hand"].set_position(
        env.robots[0].parts["right_hand"].get_position())
    env.robots[0].parts["left_hand"].set_position(
        env.robots[0].parts["left_hand"].get_position())

    obj_in_hand_idx = env.robots[0].parts["right_hand"].object_in_hand
    obj_in_hand = [
        obj for obj in env.scene.get_objects()
        if obj.get_body_id() == obj_in_hand_idx
    ][0]
    x, y, z = np.add(place_rel_pos, obj.get_position())
    hand_x, hand_y, hand_z = env.robots[0].parts["right_hand"].get_position()

    minx = min(x, hand_x) - 1
    miny = min(y, hand_y) - 1
    minz = min(z, hand_z) - 0.5
    maxx = max(x, hand_x) + 1
    maxy = max(y, hand_y) + 1
    maxz = max(z, hand_z) + 0.5

    obstacles = get_scene_body_ids(env, include_self=False)
    obstacles.remove(env.robots[0].parts["right_hand"].object_in_hand)
    end_conf = [
        x,
        y,
        z + 0.2,
        0,
        np.pi * 7 / 6,
        0,
    ]
    if env.use_rrt:
        plan = plan_hand_motion_br(
            robot=env.robots[0],
            obj_in_hand=obj_in_hand,
            end_conf=end_conf,
            hand_limits=((minx, miny, minz), (maxx, maxy, maxz)),
            obstacles=obstacles,
            rng=rng,
        )
        p.restoreState(state)
        p.removeState(state)
    else:
        pos = env.robots[0].parts["right_hand"].get_position()
        plan = [[pos[0], pos[1], pos[2]] + list(
            p.getEulerFromQuaternion(
                env.robots[0].parts["right_hand"].get_orientation())),
                end_conf]

    # NOTE: This below line is *VERY* important after the
    # pybullet state is restored. The hands keep an internal
    # track of their state, and if we don't reset this
    # state to mirror the actual pybullet state, the hand will
    # think it's elsewhere and update incorrectly accordingly
    env.robots[0].parts["right_hand"].set_position(
        env.robots[0].parts["right_hand"].get_position())
    env.robots[0].parts["left_hand"].set_position(
        env.robots[0].parts["left_hand"].get_position())

    # If RRT planning fails, fail and return None
    if plan is None:
        logging.info(f"PRIMITIVE: placeOnTop {obj.name} fail, failed "
                     f"to find plan to continuous params {place_rel_pos}")
        return None

    original_orientation = list(
        p.getEulerFromQuaternion(
            env.robots[0].parts["right_hand"].get_orientation()))
    logging.info(f"PRIMITIVE: placeOnTop {obj.name} success! Plan found with "
                 f"continuous params {place_rel_pos}.")
    return plan, original_orientation
