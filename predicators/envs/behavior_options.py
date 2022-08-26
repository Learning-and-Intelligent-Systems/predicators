"""Hardcoded options for BehaviorEnv."""
# pylint: disable=import-error

import logging
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy
from numpy.random._generator import Generator

from predicators.settings import CFG
from predicators.structs import Array, GroundAtom, State
from predicators.utils import get_aabb_volume, get_closest_point_on_aabb

try:
    import pybullet as p
    from igibson import object_states
    from igibson.envs.behavior_env import \
        BehaviorEnv  # pylint: disable=unused-import
    from igibson.external.pybullet_tools.utils import CIRCULAR_LIMITS, \
        get_aabb, get_aabb_extent
    from igibson.object_states.on_floor import \
        RoomFloor  # pylint: disable=unused-import
    from igibson.objects.articulated_object import URDFObject
    from igibson.robots.behavior_robot import \
        BRBody  # pylint: disable=unused-import
    from igibson.robots.robot_base import \
        BaseRobot  # pylint: disable=unused-import
    from igibson.utils import sampling_utils
    from igibson.utils.behavior_robot_planning_utils import \
        plan_base_motion_br, plan_hand_motion_br
    from igibson.utils.checkpoint_utils import load_checkpoint

except (ImportError, ModuleNotFoundError) as e:
    pass

_ON_TOP_RAY_CASTING_SAMPLING_PARAMS = {
    "max_angle_with_z_axis": 0.17,
    "bimodal_stdev_fraction": 1e-6,
    "bimodal_mean_fraction": 1.0,
    "max_sampling_attempts": 50,
    "aabb_offset": 0.01,
}


def get_body_ids(
    env: "BehaviorEnv",
    include_self: bool = False,
    include_right_hand: bool = False,
) -> List[int]:
    """Function to return a list of body_ids for all objects in the scene for
    collision checking depending on whether navigation or grasping/ placing is
    being done."""
    ids = []
    for obj in env.scene.get_objects():
        if isinstance(obj, URDFObject):
            # We want to exclude the floor since we're always floating and
            # will never practically collide with it, but if we include it
            # in collision checking, we always seem to collide.
            if obj.name != "floors":
                ids.extend(obj.body_ids)

    if include_self:
        ids.append(env.robots[0].parts["left_hand"].get_body_id())
        ids.append(env.robots[0].parts["body"].get_body_id())
        ids.append(env.robots[0].parts["eye"].get_body_id())
        if not include_right_hand:
            ids.append(env.robots[0].parts["right_hand"].get_body_id())

    return ids


def detect_collision(bodyA: int, object_in_hand: Optional[int] = None) -> bool:
    """Detects collisions between bodyA in the scene (except for the object in
    the robot's hand)"""
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id in [bodyA, object_in_hand]:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision


def detect_robot_collision(robot: "BaseRobot") -> bool:
    """Function to detect whether the robot is currently colliding with any
    object in the scene."""
    object_in_hand = robot.parts["right_hand"].object_in_hand
    return (
        detect_collision(robot.parts["body"].body_id, object_in_hand)
        or detect_collision(robot.parts["left_hand"].body_id, object_in_hand)
        or detect_collision(robot.parts["right_hand"].body_id, object_in_hand))


def reset_and_release_hand(env: "BehaviorEnv") -> None:
    """Resets the state of the right hand."""
    env.robots[0].set_position_orientation(env.robots[0].get_position(),
                                           env.robots[0].get_orientation())
    for _ in range(50):
        env.robots[0].parts["right_hand"].set_close_fraction(0)
        env.robots[0].parts["right_hand"].trigger_fraction = 0
        p.stepSimulation()


def get_delta_low_level_base_action(robot_z: float,
                                    original_orientation: Tuple,
                                    old_xytheta: Array, new_xytheta: Array,
                                    action_space_shape: Tuple) -> Array:
    """Given a base movement plan that is a series of waypoints in world-frame
    position space, convert pairs of these points to a base movement action in
    velocity space.

    Note that we cannot simply subtract subsequent positions because the
    velocity action space used by BEHAVIOR is not defined in the world
    frame, but rather in the frame of the previous position.
    """
    ret_action = np.zeros(action_space_shape, dtype=np.float32)

    # First, get the old and new position and orientation in the world
    # frame as numpy arrays
    old_pos = np.array([old_xytheta[0], old_xytheta[1], robot_z])
    old_orn_quat = p.getQuaternionFromEuler(
        np.array(
            [original_orientation[0], original_orientation[1],
             old_xytheta[2]]))
    new_pos = np.array([new_xytheta[0], new_xytheta[1], robot_z])
    new_orn_quat = p.getQuaternionFromEuler(
        np.array(
            [original_orientation[0], original_orientation[1],
             new_xytheta[2]]))

    # Then, simply get the delta position and orientation by multiplying the
    # inverse of the old pose by the new pose
    inverted_old_pos, inverted_old_orn_quat = p.invertTransform(
        old_pos, old_orn_quat)
    delta_pos, delta_orn_quat = p.multiplyTransforms(inverted_old_pos,
                                                     inverted_old_orn_quat,
                                                     new_pos, new_orn_quat)

    # Finally, convert the orientation back to euler angles from a quaternion
    delta_orn = p.getEulerFromQuaternion(delta_orn_quat)

    ret_action[0:3] = np.array([delta_pos[0], delta_pos[1], delta_orn[2]])

    return ret_action


def navigate_to_param_sampler(state: State, goal: Set[GroundAtom],
                              rng: Generator,
                              objects: Sequence["URDFObject"]) -> Array:
    """Sampler for navigateTo option."""
    del goal
    from predicators.envs import \
        get_or_create_env  # pylint: disable=import-outside-toplevel
    from predicators.envs import \
        get_or_create_igibson_behavior_env  # pylint: disable=import-outside-toplevel

    # Get the current env for collision checking.
    env = get_or_create_env("behavior")
    load_checkpoint_state(state, env)
    env = get_or_create_igibson_behavior_env("behavior")

    # The navigation nsrts are designed such that the target
    # obj is always last in the params list.
    obj_to_sample_near = objects[-1]
    closeness_limit = 0.75
    nearness_limit = 0.5
    distance = nearness_limit + (
        (closeness_limit - nearness_limit) * rng.random())
    yaw = rng.random() * (2 * np.pi) - np.pi
    x = distance * np.cos(yaw)
    y = distance * np.sin(yaw)
    sampler_output = np.array([x, y])

    # The below while loop avoids sampling values that are inside
    # the bounding box of the object and therefore will
    # certainly be in collision with the object if the robot
    # tries to move there.
    logging.info("Sampling params for navigation...")
    num_samples_tried = 0
    while (check_nav_end_pose(env, obj_to_sample_near, sampler_output) is
           None):
        distance = closeness_limit * rng.random()
        yaw = rng.random() * (2 * np.pi) - np.pi
        x = distance * np.cos(yaw)
        y = distance * np.sin(yaw)
        sampler_output = np.array([x, y])
        if num_samples_tried % 50 == 0:
            logging.info(f"Number of navigation samples: {num_samples_tried}")
        num_samples_tried += 1

    assert check_nav_end_pose(env, obj_to_sample_near,
                              sampler_output) is not None
    return sampler_output


def create_navigate_policy(
    plan: List[List[float]], original_orientation: List[List[float]]
) -> Callable[[State, "BehaviorEnv"], Tuple[Array, bool]]:
    """Instantiates and returns a navigation option policy given an RRT plan,
    which is a list of 3-element lists each containing a series of (x, y, rot)
    waypoints for the robot to pass through."""

    def navigateToOptionPolicy(_state: State,
                               env: "BehaviorEnv") -> Tuple[Array, bool]:
        atol_xy = 1e-2
        atol_theta = 1e-3
        atol_vel = 1e-4

        # 1. Get current position and orientation
        current_pos = list(env.robots[0].get_position()[0:2])
        current_orn = p.getEulerFromQuaternion(
            env.robots[0].get_orientation())[2]

        expected_pos = np.array(plan[0][0:2])
        expected_orn = np.array(plan[0][2])

        # 2. if error is greater that MAX_ERROR
        if not np.allclose(current_pos, expected_pos,
                           atol=atol_xy) or not np.allclose(
                               current_orn, expected_orn, atol=atol_theta):
            # 2.a take a corrective action
            if len(plan) <= 1:
                done_bit = True
                logging.info("PRIMITIVE: navigation policy completed "
                             "execution!")
                return np.zeros(env.action_space.shape,
                                dtype=np.float32), done_bit
            low_level_action = get_delta_low_level_base_action(
                env.robots[0].get_position()[2],
                tuple(original_orientation[0:2]),
                np.array(current_pos + [current_orn]), np.array(plan[0]),
                env.action_space.shape)

            # But if the corrective action is 0, take the next action
            if np.allclose(low_level_action,
                           np.zeros((env.action_space.shape[0], 1)),
                           atol=atol_vel):
                low_level_action = get_delta_low_level_base_action(
                    env.robots[0].get_position()[2],
                    tuple(original_orientation[0:2]),
                    np.array(current_pos + [current_orn]), np.array(plan[1]),
                    env.action_space.shape)
                plan.pop(0)

            return low_level_action, False

        if (len(plan) == 1
            ):  # In this case, we're at the final position we wanted
            # to reach
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            done_bit = True
            logging.info("PRIMITIVE: navigation policy completed execution!")

        else:
            low_level_action = get_delta_low_level_base_action(
                env.robots[0].get_position()[2],
                tuple(original_orientation[0:2]), np.array(plan[0]),
                np.array(plan[1]), env.action_space.shape)
            done_bit = False

        plan.pop(0)

        # Ensure that the action is clipped to stay within the expected
        # range
        low_level_action = np.clip(low_level_action, -1.0, 1.0)
        return low_level_action, done_bit

    return navigateToOptionPolicy


def create_navigate_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        _obj_to_nav_to: "URDFObject"
) -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a navigation option model function given an RRT
    plan, which is a list of 3-element lists each containing a series of (x, y,
    rot) waypoints for the robot to pass through."""

    def navigateToOptionModel(_init_state: State, env: "BehaviorEnv") -> None:
        robot_z = env.robots[0].get_position()[2]
        target_pos = np.array([plan[-1][0], plan[-1][1], robot_z])
        robot_orn = p.getEulerFromQuaternion(env.robots[0].get_orientation())
        target_orn = p.getQuaternionFromEuler(
            np.array([robot_orn[0], robot_orn[1], plan[-1][2]]))
        env.robots[0].set_position_orientation(target_pos, target_orn)
        # this is running a zero action to step simulator so
        # the environment updates to the correct final position
        env.step(np.zeros(env.action_space.shape))

    return navigateToOptionModel


def navigate_to_obj_pos(
    env: "BehaviorEnv",
    obj: Union["URDFObject", "RoomFloor"],
    pos_offset: Array,
    rng: Optional[Generator] = None
) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """Parameterized controller for navigation.

    Runs motion planning to find a feasible trajectory to a certain x,y
    position offset from obj and selects an orientation such that the
    robot is facing the object. If the navigation is infeasible, returns
    an indication to this effect (None). Otherwise, returns the plan,
    which is a list of list of robot base poses to move to, as well as
    the original euler angle orientation of the robot body.
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

    valid_position = check_nav_end_pose(env, obj, pos_offset)

    if valid_position is None:
        p.restoreState(state)
        p.removeState(state)
        logging.warning(f"PRIMITIVE: navigate to {obj.name} with params "
                        f"{pos_offset} failed, sampler is problematic!")
        check_nav_end_pose(env, obj, pos_offset)
        return None

    p.restoreState(state)
    end_conf = [
        valid_position[0][0],
        valid_position[0][1],
        valid_position[1][2],
    ]
    if env.use_rrt:
        obstacles = get_body_ids(env)
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


def check_nav_end_pose(
        env: "BehaviorEnv", obj: Union["URDFObject", "RoomFloor"],
        pos_offset: Array) -> Optional[Tuple[List[int], List[int]]]:
    """Check that the robot can reach pos_offset from the obj without (1) being
    in collision with anything, or (2) being blocked from obj by some other
    solid object.

    If this is true, return the ((x,y,z),(roll, pitch, yaw)), else
    return None
    """
    valid_position = None
    state = p.saveState()
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

    # if blocked:
    #     logging.info(f"Params {pos_offset} blocked!")
    # elif valid_position is None:
    #     logging.info(f"Params {pos_offset} in collision!")

    # if valid_position is not None:
    #     logging.info(f"Params {pos_offset} is fine!")

    p.restoreState(state)
    p.removeState(state)

    return valid_position


# Sampler for grasp continuous params
def grasp_obj_param_sampler(state: State, goal: Set[GroundAtom],
                            rng: Generator,
                            objects: Sequence["URDFObject"]) -> Array:
    """Sampler for grasp option."""
    del state, goal, objects
    x_offset = (rng.random() * 0.4) - 0.2
    y_offset = (rng.random() * 0.4) - 0.2
    z_offset = rng.random() * 0.2
    return np.array([x_offset, y_offset, z_offset])


def get_delta_low_level_hand_action(
    body: "BRBody",
    old_pos: Union[Sequence[float], Array],
    old_orn: Union[Sequence[float], Array],
    new_pos: Union[Sequence[float], Array],
    new_orn: Union[Sequence[float], Array],
) -> Array:
    """Given a hand movement plan that is a series of waypoints for the hand in
    position space, convert pairs of these points to a hand movement action in
    velocity space.

    Note that we cannot simply subtract subsequent positions because the
    velocity action space used by BEHAVIOR is not defined in the world
    frame, but rather in the frame of the previous position.
    """
    # First, convert the supplied orientations to quaternions
    old_orn = p.getQuaternionFromEuler(old_orn)
    new_orn = p.getQuaternionFromEuler(new_orn)

    # Next, find the inverted position of the body (which we know shouldn't
    # change, since our actions move either the body or the hand, but not
    # both simultaneously)
    inverted_body_new_pos, inverted_body_new_orn = p.invertTransform(
        body.new_pos, body.new_orn)
    # Use this to compute the new pose of the hand w.r.t the body frame
    new_local_pos, new_local_orn = p.multiplyTransforms(
        inverted_body_new_pos, inverted_body_new_orn, new_pos, new_orn)

    # Next, compute the old pose of the hand w.r.t the body frame
    inverted_body_old_pos = inverted_body_new_pos
    inverted_body_old_orn = inverted_body_new_orn
    old_local_pos, old_local_orn = p.multiplyTransforms(
        inverted_body_old_pos, inverted_body_old_orn, old_pos, old_orn)

    # The delta position is simply given by the difference between these
    # positions
    delta_pos = np.array(new_local_pos) - np.array(old_local_pos)

    # Finally, compute the delta orientation
    inverted_old_local_orn_pos, inverted_old_local_orn_orn = p.invertTransform(
        [0, 0, 0], old_local_orn)
    _, delta_orn = p.multiplyTransforms(
        [0, 0, 0],
        new_local_orn,
        inverted_old_local_orn_pos,
        inverted_old_local_orn_orn,
    )

    delta_trig_frac = 0
    action = np.concatenate(
        [
            np.zeros((10), dtype=np.float32),
            np.array(delta_pos, dtype=np.float32),
            np.array(p.getEulerFromQuaternion(delta_orn), dtype=np.float32),
            np.array([delta_trig_frac], dtype=np.float32),
        ],
        axis=0,
    )

    return action


def create_grasp_policy(
    plan: List[List[float]], _original_orientation: List[List[float]]
) -> Callable[[State, "BehaviorEnv"], Tuple[Array, bool]]:
    """Instantiates and returns a navigation option policy given an RRT plan,
    which is a list of 6-element lists containing a series of (x, y, z, roll,
    pitch, yaw) waypoints for the hand to pass through."""
    # Set up two booleans to be used as 'memory', as well as
    # a 'reversed' plan to be used for our option that's
    # defined below. Note that the reversed plan makes a
    # copy of the list instead of just assigning by reference,
    # and this is critical to the functioning of our option. The reversed
    # plan is necessary because RRT just gives us a plan to move our hand
    # to the grasping location, but not to getting back.
    reversed_plan = list(reversed(plan))
    plan_executed_forwards = False
    tried_closing_gripper = False

    def graspObjectOptionPolicy(_state: State,
                                env: "BehaviorEnv") -> Tuple[Array, bool]:
        nonlocal plan
        nonlocal reversed_plan
        nonlocal plan_executed_forwards
        nonlocal tried_closing_gripper
        done_bit = False

        atol_xyz = 1e-4
        atol_theta = 0.1
        atol_vel = 5e-3

        # 1. Get current position and orientation
        current_pos, current_orn_quat = p.multiplyTransforms(
            env.robots[0].parts["right_hand"].parent.parts["body"].new_pos,
            env.robots[0].parts["right_hand"].parent.parts["body"].new_orn,
            env.robots[0].parts["right_hand"].local_pos,
            env.robots[0].parts["right_hand"].local_orn,
        )
        current_orn = p.getEulerFromQuaternion(current_orn_quat)

        if (not plan_executed_forwards and not tried_closing_gripper):
            expected_pos = np.array(plan[0][0:3])
            expected_orn = np.array(plan[0][3:])
            # 2. if error is greater that MAX_ERROR
            if not np.allclose(current_pos, expected_pos,
                               atol=atol_xyz) or not np.allclose(
                                   current_orn, expected_orn, atol=atol_theta):
                # 2.a take a corrective action
                if len(plan) <= 1:
                    done_bit = False
                    plan_executed_forwards = True
                    low_level_action = np.zeros(env.action_space.shape,
                                                dtype=np.float32)
                    return low_level_action, done_bit

                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[0][0:3]),
                    np.array(plan[0][3:]),
                ))

                # But if the corrective action is 0, take the next action
                if np.allclose(
                        low_level_action,
                        np.zeros((env.action_space.shape[0], 1)),
                        atol=atol_vel,
                ):
                    low_level_action = (get_delta_low_level_hand_action(
                        env.robots[0].parts["body"],
                        np.array(current_pos),
                        np.array(current_orn),
                        np.array(plan[1][0:3]),
                        np.array(plan[1][3:]),
                    ))
                    plan.pop(0)

                return low_level_action, False

            if len(plan) <= 1:  # In this case, we're at the final position
                low_level_action = np.zeros(env.action_space.shape,
                                            dtype=float)
                done_bit = False
                plan_executed_forwards = True
            else:
                # Step thru the plan to execute placing
                # phases 1 and 2
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    plan[0][0:3],
                    plan[0][3:],
                    plan[1][0:3],
                    plan[1][3:],
                ))
                if len(plan) == 1:
                    plan_executed_forwards = True

            plan.pop(0)
            return low_level_action, done_bit

        if (plan_executed_forwards and not tried_closing_gripper):
            # Close the gripper to see if you've gotten the
            # object
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            low_level_action[16] = 1.0
            tried_closing_gripper = True
            plan = reversed_plan
            return low_level_action, False

        expected_pos = np.array(plan[0][0:3])
        expected_orn = np.array(plan[0][3:])
        # 2. if error is greater that MAX_ERROR
        if not np.allclose(current_pos, expected_pos,
                           atol=atol_xyz) or not np.allclose(
                               current_orn, expected_orn, atol=atol_theta):
            # 2.a take a corrective action
            if len(plan) <= 1:
                done_bit = True
                logging.info("PRIMITIVE: grasp policy completed execution!")
                return np.zeros(env.action_space.shape,
                                dtype=np.float32), done_bit
            low_level_action = (get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                np.array(current_pos),
                np.array(current_orn),
                np.array(plan[0][0:3]),
                np.array(plan[0][3:]),
            ))

            # But if the corrective action is 0, take the next action
            if np.allclose(
                    low_level_action,
                    np.zeros((env.action_space.shape[0], 1)),
                    atol=atol_vel,
            ):
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[1][0:3]),
                    np.array(plan[1][3:]),
                ))
                plan.pop(0)

            return low_level_action, False

        if len(plan) == 1:  # In this case, we're at the final position
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            done_bit = True
            logging.info("PRIMITIVE: grasp policy completed execution!")

        else:
            # Grasping Phase 3: getting the hand back to
            # resting position near the robot.
            low_level_action = get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                reversed_plan[0][0:3],  # current pos
                reversed_plan[0][3:],  # current orn
                reversed_plan[1][0:3],  # next pos
                reversed_plan[1][3:],  # next orn
            )
            if len(reversed_plan) == 1:
                done_bit = True
                logging.info("PRIMITIVE: grasp policy completed execution!")

        reversed_plan.pop(0)

        # Ensure that the action is clipped to stay within the expected
        # range
        low_level_action = np.clip(low_level_action, -1.0, 1.0)
        return low_level_action, done_bit

    return graspObjectOptionPolicy


def create_grasp_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_grasp: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a grasp option model function given an RRT
    plan, which is a list of 6-element lists containing a series of (x, y, z,
    roll, pitch, yaw) waypoints for the hand to pass through."""

    # NOTE: -1 because there are 25 timesteps that we move along the vector
    # between the hand the object for until finally grasping, and we want
    # just the final orientation.
    hand_i = -1
    rh_final_grasp_postion = plan[hand_i][0:3]
    rh_final_grasp_orn = plan[hand_i][3:6]

    def graspObjectOptionModel(_state: State, env: "BehaviorEnv") -> None:
        nonlocal hand_i
        rh_orig_grasp_postion = env.robots[0].parts["right_hand"].get_position(
        )
        rh_orig_grasp_orn = env.robots[0].parts["right_hand"].get_orientation()

        # 1 Teleport Hand to Grasp offset location
        env.robots[0].parts["right_hand"].set_position_orientation(
            rh_final_grasp_postion,
            p.getQuaternionFromEuler(rh_final_grasp_orn))

        # 3. Close hand and simulate grasp
        a = np.zeros(env.action_space.shape, dtype=float)
        a[16] = 1.0
        assisted_grasp_action = np.zeros(28, dtype=float)
        assisted_grasp_action[26] = 1.0
        if isinstance(obj_to_grasp.body_id, List):
            grasp_obj_body_id = obj_to_grasp.body_id[0]
        else:
            grasp_obj_body_id = obj_to_grasp.body_id
        # 3.1 Call code that does assisted grasping
        # bypass_force_check is basically a hack we should
        # turn it off for the final system and use a real grasp
        # sampler
        if env.robots[0].parts["right_hand"].object_in_hand is None:
            env.robots[0].parts["right_hand"].trigger_fraction = 0
        env.robots[0].parts["right_hand"].handle_assisted_grasping(
            assisted_grasp_action,
            override_ag_data=(grasp_obj_body_id, -1),
            bypass_force_check=True)
        # 3.2 step the environment a few timesteps to complete grasp
        for _ in range(5):
            env.step(a)

        # 4 Move Hand to Original Location
        env.robots[0].parts["right_hand"].set_position_orientation(
            rh_orig_grasp_postion, rh_orig_grasp_orn)
        if env.robots[0].parts["right_hand"].object_in_hand is not None:
            # NOTE: This below line is necessary to update the visualizer.
            # Also, it only works for URDF objects (but if the object is
            # not a URDF object, grasping should have failed)
            obj_to_grasp.force_wakeup()
        # Step a zero-action in the environment to update the visuals of the
        # environment.
        env.step(np.zeros(env.action_space.shape))

    return graspObjectOptionModel


def grasp_obj_at_pos(
    env: "BehaviorEnv",
    obj: Union["URDFObject", "RoomFloor"],
    grasp_offset: Array,
    rng: Optional[Generator] = None,
) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """Parameterized controller for grasping.

    Runs motion planning to find a feasible trajectory to a certain
    x,y,z position offset from obj and selects an orientation such that
    the palm is facing the object. If the grasp is infeasible, returns
    an indication to this effect (None). Otherwise, returns the plan,
    which is a list of list of hand poses, as well as the original euler
    angle orientation of the hand.
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
            obstacles=get_body_ids(env,
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


def place_obj_plan(
    env: "BehaviorEnv",
    obj: Union["URDFObject", "RoomFloor"],
    original_state: int,
    place_rel_pos: Array,
    rng: Optional[Generator] = None,
) -> Optional[List[List[float]]]:
    """Function to return an RRT plan for placing an object."""
    if rng is None:
        rng = np.random.default_rng(23)
    obj_in_hand = env.scene.get_objects()[
        env.robots[0].parts["right_hand"].object_in_hand]
    x, y, z = np.add(place_rel_pos, obj.get_position())
    hand_x, hand_y, hand_z = env.robots[0].parts["right_hand"].get_position()

    minx = min(x, hand_x) - 1
    miny = min(y, hand_y) - 1
    minz = min(z, hand_z) - 0.5
    maxx = max(x, hand_x) + 1
    maxy = max(y, hand_y) + 1
    maxz = max(z, hand_z) + 0.5

    obstacles = get_body_ids(env, include_self=False)
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
        p.restoreState(original_state)
        p.removeState(original_state)
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

    return plan


def place_ontop_obj_pos_sampler(
        state: State, goal: Set[GroundAtom], rng: Generator,
        obj: Union["URDFObject", "RoomFloor"]) -> Array:
    """Sampler for placeOnTop option."""
    del state, goal
    assert rng is not None
    # objA is the object the robot is currently holding, and objB
    # is the surface that it must place onto.
    # The BEHAVIOR NSRT's are designed such that objA is the 0th
    # argument, and objB is the last.
    objA = obj[0]
    objB = obj[-1]

    params = _ON_TOP_RAY_CASTING_SAMPLING_PARAMS
    aabb = get_aabb(objA.get_body_id())
    aabb_extent = get_aabb_extent(aabb)

    random_seed_int = rng.integers(10000000)
    sampling_results = sampling_utils.sample_cuboid_on_object(
        objB,
        num_samples=1,
        cuboid_dimensions=aabb_extent,
        axis_probabilities=[0, 0, 1],
        refuse_downwards=True,
        random_seed_number=random_seed_int,
        **params,
    )

    if sampling_results[0] is None or sampling_results[0][0] is None:
        # If sampling fails, returns a random set of params
        return np.array([
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.3, 1.0)
        ])

    rnd_params = np.subtract(sampling_results[0][0], objB.get_position())
    return rnd_params


def create_place_policy(
    plan: List[List[float]], _original_orientation: List[List[float]]
) -> Callable[[State, "BehaviorEnv"], Tuple[Array, bool]]:
    """Instantiates and returns a navigation option policy given an RRT plan,
    which is a list of 6-element lists containing a series of (x, y, z, roll,
    pitch, yaw) waypoints for the hand to pass through."""

    # Note that the reversed plan code below makes a
    # copy of the list instead of just assigning by reference,
    # and this is critical to the functioning of our option. The reversed
    # plan is necessary because RRT just gives us a plan to move our hand
    # to the grasping location, but not to getting back.
    reversed_plan = list(reversed(plan))
    plan_executed_forwards = False
    tried_opening_gripper = False

    def placeOntopObjectOptionPolicy(_state: State,
                                     env: "BehaviorEnv") -> Tuple[Array, bool]:
        nonlocal plan
        nonlocal reversed_plan
        nonlocal plan_executed_forwards
        nonlocal tried_opening_gripper

        done_bit = False
        atol_xyz = 0.1
        atol_theta = 0.1
        atol_vel = 2.5

        # 1. Get current position and orientation
        current_pos, current_orn_quat = p.multiplyTransforms(
            env.robots[0].parts["right_hand"].parent.parts["body"].new_pos,
            env.robots[0].parts["right_hand"].parent.parts["body"].new_orn,
            env.robots[0].parts["right_hand"].local_pos,
            env.robots[0].parts["right_hand"].local_orn,
        )
        current_orn = p.getEulerFromQuaternion(current_orn_quat)

        if (not plan_executed_forwards and not tried_opening_gripper):
            expected_pos = np.array(plan[0][0:3])
            expected_orn = np.array(plan[0][3:])

            # 2. if error is greater that MAX_ERROR
            if not np.allclose(current_pos, expected_pos,
                               atol=atol_xyz) or not np.allclose(
                                   current_orn, expected_orn, atol=atol_theta):
                # 2.a take a corrective action
                if len(plan) <= 1:
                    done_bit = False
                    plan_executed_forwards = True
                    low_level_action = np.zeros(env.action_space.shape,
                                                dtype=np.float32)
                    return low_level_action, done_bit

                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[0][0:3]),
                    np.array(plan[0][3:]),
                ))

                # But if the corrective action is 0, take the next action
                if np.allclose(
                        low_level_action,
                        np.zeros((env.action_space.shape[0], 1)),
                        atol=atol_vel,
                ):
                    low_level_action = (get_delta_low_level_hand_action(
                        env.robots[0].parts["body"],
                        np.array(current_pos),
                        np.array(current_orn),
                        np.array(plan[1][0:3]),
                        np.array(plan[1][3:]),
                    ))
                    plan.pop(0)

                return low_level_action, False

            if len(plan) <= 1:  # In this case, we're at the final position
                low_level_action = np.zeros(env.action_space.shape,
                                            dtype=float)
                done_bit = False
                plan_executed_forwards = True

            else:
                # Step thru the plan to execute placing
                # phases 1 and 2
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    plan[0][0:3],
                    plan[0][3:],
                    plan[1][0:3],
                    plan[1][3:],
                ))
                if len(plan) == 1:
                    plan_executed_forwards = True

            plan.pop(0)
            return low_level_action, done_bit

        if (plan_executed_forwards and not tried_opening_gripper):
            # Open the gripper to see if you've released the
            # object
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            low_level_action[16] = -1.0
            tried_opening_gripper = True
            plan = reversed_plan
            return low_level_action, False

        expected_pos = np.array(plan[0][0:3])
        expected_orn = np.array(plan[0][3:])
        # 2. if error is greater that MAX_ERROR
        if not np.allclose(current_pos, expected_pos,
                           atol=atol_xyz) or not np.allclose(
                               current_orn, expected_orn, atol=atol_theta):
            # 2.a take a corrective action
            if len(plan) <= 1:
                done_bit = True
                logging.info("PRIMITIVE: place policy completed execution!")
                return np.zeros(env.action_space.shape,
                                dtype=np.float32), done_bit
            low_level_action = (get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                np.array(current_pos),
                np.array(current_orn),
                np.array(plan[0][0:3]),
                np.array(plan[0][3:]),
            ))

            # But if the corrective action is 0, take the next action
            if np.allclose(
                    low_level_action,
                    np.zeros((env.action_space.shape[0], 1)),
                    atol=atol_vel,
            ):
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[1][0:3]),
                    np.array(plan[1][3:]),
                ))
                plan.pop(0)

            return low_level_action, False

        if len(plan) == 1:  # In this case, we're at the final position
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            done_bit = True
            logging.info("PRIMITIVE: place policy completed execution!")

        else:
            # Placing Phase 3: getting the hand back to
            # resting position near the robot.
            low_level_action = get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                reversed_plan[0][0:3],  # current pos
                reversed_plan[0][3:],  # current orn
                reversed_plan[1][0:3],  # next pos
                reversed_plan[1][3:],  # next orn
            )
            if len(reversed_plan) == 1:
                done_bit = True
                logging.info("PRIMITIVE: place policy completed execution!")

        reversed_plan.pop(0)

        # Ensure that the action is clipped to stay within the expected
        # range
        low_level_action = np.clip(low_level_action, -1.0, 1.0)
        return low_level_action, done_bit

    return placeOntopObjectOptionPolicy


def create_place_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_place: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a place option model function given an RRT
    plan, which is a list of 6-element lists containing a series of (x, y, z,
    roll, pitch, yaw) waypoints for the hand to pass through."""

    def placeOntopObjectOptionModel(_init_state: State,
                                    env: "BehaviorEnv") -> None:
        released_obj_bid = env.robots[0].parts["right_hand"].object_in_hand
        rh_orig_grasp_postion = env.robots[0].parts["right_hand"].get_position(
        )
        rh_orig_grasp_orn = env.robots[0].parts["right_hand"].get_orientation()
        target_pos = plan[-1][0:3]
        target_orn = plan[-1][3:6]
        env.robots[0].parts["right_hand"].set_position_orientation(
            target_pos, p.getQuaternionFromEuler(target_orn))
        env.robots[0].parts["right_hand"].force_release_obj()
        obj_to_place.force_wakeup()
        # this is running a zero action to step simulator
        env.step(np.zeros(env.action_space.shape))
        # reset the released object to zero velocity so it doesn't
        # fly away because of residual warp speeds from teleportation!
        p.resetBaseVelocity(
            released_obj_bid,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
        )
        env.robots[0].parts["right_hand"].set_position_orientation(
            rh_orig_grasp_postion, rh_orig_grasp_orn)
        # this is running a series of zero action to step simulator
        # to let the object fall into its place
        for _ in range(15):
            env.step(np.zeros(env.action_space.shape))

    return placeOntopObjectOptionModel


def place_ontop_obj_pos(
    env: "BehaviorEnv",
    obj: Union["URDFObject", "RoomFloor"],
    place_rel_pos: Array,
    rng: Optional[Generator] = None,
) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """Parameterized controller for placeOnTop.

    Runs motion planning to find a feasible trajectory to a certain
    offset from obj and selects an orientation such that the palm is
    facing the object. If the placement is infeasible, returns an
    indication to this effect (None). Otherwise, returns the plan, which
    is a list of list of hand poses, as well as the original euler angle
    orientation of the hand.
    """
    if rng is None:
        rng = np.random.default_rng(23)

    try:
        obj_in_hand = env.scene.get_objects()[
            env.robots[0].parts["right_hand"].object_in_hand]
        logging.info(f"PRIMITIVE: attempt to place {obj_in_hand.name} ontop "
                     f"{obj.name} with params {place_rel_pos}")
    except ValueError:
        logging.info("Cannot place; either no object in hand or holding "
                     "the object to be placed on top of!")
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

    plan = place_obj_plan(env, obj, state, place_rel_pos, rng=rng)
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


def load_checkpoint_state(s: State,
                          env: BehaviorEnv,
                          reset: bool = False) -> None:
    """Sets the underlying iGibson environment to a particular saved state.

    When reset is True we will create a new BehaviorEnv and load our
    checkpoint into it. This will ensure that all the information from
    previous environment steps are reset as well.
    """
    assert s.simulator_state is not None
    # Get the new_task_num_task_instance_id associated with this state
    # from s.simulator_state.
    new_task_num_task_instance_id = (int(s.simulator_state.split("-")[0]),
                                     int(s.simulator_state.split("-")[1]))
    # If the new_task_num_task_instance_id is new, then we need to load
    # a new iGibson behavior env with our random seed saved in
    # env.new_task_num_task_instance_id_to_igibson_seed. Otherwise
    # we're already in the correct environment and can just load the
    # checkpoint. Also note that we overwrite the task.init saved checkpoint
    # so that it's compatible with the new environment!
    env.task_num = new_task_num_task_instance_id[0]
    # Since demo trajectories seeds are not saved, a seed is generated here if
    # one does not exist yet for the task num and task instance id pair.
    if not new_task_num_task_instance_id in \
        env.task_num_task_instance_id_to_igibson_seed:
        env.task_num_task_instance_id_to_igibson_seed[
            new_task_num_task_instance_id] = 0
    if (new_task_num_task_instance_id != (env.task_num, env.task_instance_id)
            and CFG.behavior_randomize_init_state) or reset:
        env.task_instance_id = new_task_num_task_instance_id[1]
        # Frame count is overwritten by set_igibson_behavior_env and needs to
        # be preserved across resets. So we save it before and set it after
        # we reset the env.
        frame_count = env.igibson_behavior_env.simulator.frame_count
        env.set_igibson_behavior_env(
            task_instance_id=new_task_num_task_instance_id[1],
            seed=env.task_num_task_instance_id_to_igibson_seed[
                new_task_num_task_instance_id])
        env.igibson_behavior_env.simulator.frame_count = frame_count
        env.set_options()
        env.current_ig_state_to_state(
        )  # overwrite the old task_init checkpoint file!
        env.igibson_behavior_env.reset()
    load_checkpoint(
        env.igibson_behavior_env.simulator,
        f"tmp_behavior_states/{CFG.behavior_scene_name}__" +
        f"{CFG.behavior_task_name}__{CFG.num_train_tasks}__" +
        f"{CFG.seed}__{env.task_num}__{env.task_instance_id}",
        int(s.simulator_state.split("-")[2]))
    np.random.seed(env.task_num_task_instance_id_to_igibson_seed[
        new_task_num_task_instance_id])
    # We step the environment to update the visuals of where the robot is!
    env.igibson_behavior_env.step(
        np.zeros(env.igibson_behavior_env.action_space.shape))
