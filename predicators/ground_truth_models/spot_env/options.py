"""Ground-truth options for Spot environments."""

import logging
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pbrspot
from bosdyn.client import math_helpers
from bosdyn.client.lease import LeaseClient
from bosdyn.client.sdk import Robot
from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.spot_env import HANDEMPTY_GRIPPER_THRESHOLD, \
    SpotRearrangementEnv, _get_sweeping_surface_for_container, \
    get_detection_id_for_object, get_robot, \
    get_robot_gripper_open_percentage, get_simulated_object, \
    get_simulated_robot
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import \
    get_grasp_pixel, get_last_detected_objects
from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext
from predicators.spot_utils.perception.spot_cameras import \
    get_last_captured_images
from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel, \
    simulated_grasp_at_pixel
from predicators.spot_utils.skills.spot_hand_move import close_gripper, \
    gaze_at_relative_pose, move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_absolute_pose, navigate_to_relative_pose, \
    simulated_navigate_to_relative_pose
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.skills.spot_sweep import sweep
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import DEFAULT_HAND_DROP_OBJECT_POSE, \
    DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE, DEFAULT_HAND_POST_DUMP_POSE, \
    DEFAULT_HAND_PRE_DUMP_LIFT_POSE, DEFAULT_HAND_PRE_DUMP_POSE, \
    get_relative_se2_from_se3, load_spot_metadata, object_to_top_down_geom
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, SpotActionExtraInfo, State, Type

###############################################################################
#            Helper functions for chaining multiple spot skills               #
###############################################################################

# Hack: options don't generally get passed rng's, but we need them primarily
# for options that do some kind of implicit sampling (e.g. sim-safe grasping).
# In the future, we probably want to pass these thru nicely.
_options_rng = np.random.default_rng(0)


def navigate_to_relative_pose_and_gaze(robot: Robot,
                                       rel_pose: math_helpers.SE2Pose,
                                       localizer: SpotLocalizer,
                                       gaze_target: math_helpers.Vec3) -> None:
    """Navigate to a pose and then gaze at a specific target afterwards."""
    # Stow first.
    stow_arm(robot)
    # First navigate to the pose.
    navigate_to_relative_pose(robot, rel_pose)
    # Get the relative gaze target based on the new robot pose.
    localizer.localize()
    robot_pose = localizer.get_last_robot_pose()
    # Transform this to the body frame.
    rel_gaze_target_body = robot_pose.inverse().transform_vec3(gaze_target)
    # Then gaze.
    gaze_at_relative_pose(robot, rel_gaze_target_body)


def simulated_navigate_to_relative_pose_and_gaze(
        sim_robot: pbrspot.spot.Spot, rel_pose: math_helpers.SE2Pose,
        gaze_target: math_helpers.Vec3) -> None:
    """Teleports the pybullet spot robot to rel_pose and then gazes at
    gaze_target via the hand."""
    del gaze_target
    simulated_navigate_to_relative_pose(sim_robot, rel_pose)
    # Somehow get the arm to gaze at the target correctly? Maybe for now we
    # can just mock this and teleport...


def _grasp_at_pixel_and_maybe_stow_or_dump(
        robot: Robot, img: RGBDImageWithContext, pixel: Tuple[int, int],
        grasp_rot: Optional[math_helpers.Quat], rot_thresh: float,
        timeout: float, retry_grasp_after_fail: bool, do_stow: bool,
        do_dump: bool) -> None:
    # Grasp.
    grasp_at_pixel(robot,
                   img,
                   pixel,
                   grasp_rot=grasp_rot,
                   rot_thresh=rot_thresh,
                   timeout=timeout,
                   retry_with_no_constraints=retry_grasp_after_fail)
    # Dump, if the grasp was successful.
    thresh = HANDEMPTY_GRIPPER_THRESHOLD
    if do_dump and get_robot_gripper_open_percentage(robot) > thresh:
        # Lift the grasped object up high enough that it doesn't collide.
        move_hand_to_relative_pose(robot, DEFAULT_HAND_PRE_DUMP_LIFT_POSE)
        # Rotate to the right.
        angle = -np.pi / 2
        navigate_to_relative_pose(robot, math_helpers.SE2Pose(0, 0, angle))
        # Move the hand to execute the dump.
        move_hand_to_relative_pose(robot, DEFAULT_HAND_PRE_DUMP_POSE)
        time.sleep(1.0)
        move_hand_to_relative_pose(robot, DEFAULT_HAND_POST_DUMP_POSE)
        # Rotate back to where we started.
        navigate_to_relative_pose(robot, math_helpers.SE2Pose(0, 0, -angle))
    # Stow.
    if do_stow:
        stow_arm(robot)


def _sim_safe_grasp_at_pixel_and_maybe_stow_or_dump(
        robot: Robot, target_obj: Object, rng: np.random.Generator,
        timeout: float, retry_grasp_after_fail: bool, do_stow: bool,
        do_dump: bool) -> None:
    """Implicitly gets the current image from the hand camera and selects a
    pixel + rotation threshold.

    This is necessary because we can't select pixels from inside
    simulation, but we want to do this in a just-in-time fashion when
    we're actually executing the skill on the robot. Basically, this
    moves all the logic from the pixel grasp sampler and the skill
    inside here.
    """
    # Select the coordinates of a pixel within the image so that
    # we grasp at that pixel!
    target_detection_id = get_detection_id_for_object(target_obj)
    rgbds = get_last_captured_images()
    _, artifacts = get_last_detected_objects()
    hand_camera = "hand_color_image"
    pixel, rot_quat = get_grasp_pixel(rgbds, artifacts, target_detection_id,
                                      hand_camera, rng)
    img = rgbds[hand_camera]
    # Use a relatively forgiving threshold for grasp constraints in general,
    # but for the ball, use a strict constraint.
    if target_obj.name == "ball":
        thresh = 0.17
    else:
        thresh = np.pi / 4

    _grasp_at_pixel_and_maybe_stow_or_dump(robot, img, pixel, rot_quat, thresh,
                                           timeout, retry_grasp_after_fail,
                                           do_stow, do_dump)


def _place_at_relative_position_and_stow(
        robot: Robot, rel_pose: math_helpers.SE3Pose) -> None:
    # NOTE: Might need to execute a move here if we are currently
    # too far away...
    # Place.
    place_at_relative_position(robot, rel_pose)
    # Now, move the arm back slightly. We do this because if we're
    # placing an object directly onto a table instead of dropping it,
    # then stowing/moving the hand immediately after might cause
    # us to knock the object off the table.
    slightly_back_and_up_pose = math_helpers.SE3Pose(
        x=rel_pose.x - 0.20,
        y=rel_pose.y,
        z=rel_pose.z + 0.1,
        rot=math_helpers.Quat.from_pitch(np.pi / 3))
    move_hand_to_relative_pose(robot, slightly_back_and_up_pose)
    # Stow.
    stow_arm(robot)


def _drop_and_stow(robot: Robot) -> None:
    # First, move the arm to a position from which the object will drop.
    move_hand_to_relative_pose(robot, DEFAULT_HAND_DROP_OBJECT_POSE)
    # Open the hand.
    open_gripper(robot)
    # Stow.
    stow_arm(robot)


def _drop_at_relative_position_and_look(
        robot: Robot, rel_pose: math_helpers.SE3Pose) -> None:
    # Place.
    place_at_relative_position(robot, rel_pose)
    # Move the hand back towards the robot so we can see whether
    # placing was successful or not.
    look_dist_to_retract = 0.35
    rel_look_down_xy = np.array([rel_pose.x, rel_pose.y, rel_pose.z])
    rel_look_down_xy_unit = rel_look_down_xy / np.linalg.norm(rel_look_down_xy)
    vec_to_move_back_xy = look_dist_to_retract * rel_look_down_xy_unit
    rel_look_pose = math_helpers.SE3Pose(rel_pose.x - vec_to_move_back_xy[0],
                                         rel_pose.y - vec_to_move_back_xy[1],
                                         rel_pose.z + 0.3,
                                         rot=math_helpers.Quat.from_pitch(
                                             np.pi / 3))
    # Look straight down.
    move_hand_to_relative_pose(robot, rel_look_pose)
    # Close the gripper after moving (to avoid accidentally regrasping the
    # object).
    close_gripper(robot)


def _move_closer_and_drop_at_relative_position_and_look(
        robot: Robot, localizer: SpotLocalizer,
        abs_pose: math_helpers.SE3Pose) -> None:
    # First, check if we're too far away in distance or angle
    # to place.
    localizer.localize()
    robot_pose = localizer.get_last_robot_pose()
    rel_pose = robot_pose.inverse() * abs_pose
    dist_to_object = np.sqrt(rel_pose.x * rel_pose.x + rel_pose.y * rel_pose.y)
    # If we're too far from the target to place directly, then move closer
    # to it first. Move an absolute distance away from the given rel_pose.
    target_distance = 0.85
    if dist_to_object > target_distance:
        rel_xy = np.array([rel_pose.x, rel_pose.y])
        unit_rel_xy = rel_xy / dist_to_object
        target_xy = rel_xy - target_distance * unit_rel_xy
        pose_to_nav_to = math_helpers.SE2Pose(target_xy[0], target_xy[1], 0.0)
        navigate_to_relative_pose(robot, pose_to_nav_to)
    # Relocalize to compute final relative pose.
    localizer.localize()
    robot_pose = localizer.get_last_robot_pose()
    rel_pose = robot_pose.inverse() * abs_pose
    _drop_at_relative_position_and_look(robot, rel_pose)


def _drag_and_release(robot: Robot, rel_pose: math_helpers.SE2Pose) -> None:
    # First navigate to the pose.
    navigate_to_relative_pose(robot, rel_pose)
    # Open the gripper.
    open_gripper(robot)
    # Move the gripper up a little bit.
    move_hand_to_relative_pose(robot, DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE)
    # Stow the arm.
    stow_arm(robot)


def _move_to_absolute_pose_and_place_push_stow(
        robot: Robot, localizer: SpotLocalizer,
        absolute_pose: math_helpers.SE2Pose,
        place_rel_pose: math_helpers.SE3Pose,
        push_rel_pose: math_helpers.SE3Pose) -> None:
    # Move to the absolute pose.
    navigate_to_absolute_pose(robot, localizer, absolute_pose)
    # Execute first place.
    move_hand_to_relative_pose(robot, place_rel_pose)
    # Push.
    move_hand_to_relative_pose(robot, push_rel_pose)
    # Necessary for correct move
    time.sleep(0.5)
    # Open the gripper.
    open_gripper(robot)
    # Move the gripper slightly up and to the right to
    # avoid collisions with the container.
    dz = 0.6
    slightly_back_and_up_pose = math_helpers.SE3Pose(x=push_rel_pose.x - 0.2,
                                                     y=push_rel_pose.y - 0.3,
                                                     z=push_rel_pose.z + dz,
                                                     rot=push_rel_pose.rot)
    move_hand_to_relative_pose(robot, slightly_back_and_up_pose)
    # Stow.
    stow_arm(robot)


def _open_and_close_gripper(robot: Robot) -> None:
    open_gripper(robot)
    close_gripper(robot)


###############################################################################
#                    Helper parameterized option policies                     #
###############################################################################


def _move_to_target_policy(name: str, distance_param_idx: int,
                           yaw_param_idx: int, robot_obj_idx: int,
                           target_obj_idx: int, do_gaze: bool, state: State,
                           memory: Dict, objects: Sequence[Object],
                           params: Array) -> Action:
    del memory  # not used

    robot, localizer, _ = get_robot()
    if not CFG.bilevel_plan_without_sim:
        sim_robot = get_simulated_robot()
    else:
        sim_robot = None

    distance = params[distance_param_idx]
    yaw = params[yaw_param_idx]

    robot_obj = objects[robot_obj_idx]
    robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

    target_obj = objects[target_obj_idx]
    target_pose = utils.get_se3_pose_from_state(state, target_obj)

    rel_pose = get_relative_se2_from_se3(robot_pose, target_pose, distance,
                                         yaw)
    target_height = state.get(target_obj, "height")
    gaze_target = math_helpers.Vec3(target_pose.x, target_pose.y,
                                    target_pose.z + target_height / 2)
    fn = navigate_to_relative_pose_and_gaze
    fn_args = (robot, rel_pose, localizer, gaze_target)

    if not CFG.bilevel_plan_without_sim:
        sim_fn: Callable = simulated_navigate_to_relative_pose_and_gaze
        sim_fn_args: Tuple = (sim_robot,
                              robot_pose.get_closest_se2_transform() *
                              rel_pose, gaze_target)
    else:
        sim_fn = lambda _: None
        sim_fn_args = ()

    if not do_gaze:
        fn = navigate_to_relative_pose  # type: ignore
        fn_args = (robot, rel_pose)  # type: ignore

        if not CFG.bilevel_plan_without_sim:
            sim_fn = simulated_navigate_to_relative_pose
            sim_fn_args = (sim_robot,
                           robot_pose.get_closest_se2_transform() * rel_pose)
        else:
            sim_fn = lambda _: None
            sim_fn_args = ()

    action_extra_info = SpotActionExtraInfo(name, objects, fn, fn_args, sim_fn,
                                            sim_fn_args)
    return utils.create_spot_env_action(action_extra_info)


def _grasp_policy(name: str,
                  target_obj_idx: int,
                  state: State,
                  memory: Dict,
                  objects: Sequence[Object],
                  params: Array,
                  do_dump: bool = False) -> Action:
    del memory  # not used

    robot, _, _ = get_robot()
    if not CFG.bilevel_plan_without_sim:
        sim_robot = get_simulated_robot()
    else:
        sim_robot = None

    assert len(params) == 6
    pixel = (int(params[0]), int(params[1]))
    target_obj = objects[target_obj_idx]
    if not CFG.bilevel_plan_without_sim:
        sim_target_obj = get_simulated_object(target_obj)
    else:
        sim_target_obj = None

    # Special case: if we're running dry, the image won't be used.
    if CFG.spot_run_dry:
        img: Optional[RGBDImageWithContext] = None
    else:
        rgbds = get_last_captured_images()
        hand_camera = "hand_color_image"
        img = rgbds[hand_camera]

    # Grasp from the top-down.
    grasp_rot = None
    if not np.all(params[2:] == 0.0):
        grasp_rot = math_helpers.Quat(params[2], params[3], params[4],
                                      params[5])
    # If the target object is reasonably large, don't try to stow!
    target_obj_volume = (state.get(target_obj, "height") *
                         state.get(target_obj, "length") *
                         state.get(target_obj, "width"))

    do_stow = not do_dump and \
              target_obj_volume < CFG.spot_grasp_stow_volume_threshold
    fn = _grasp_at_pixel_and_maybe_stow_or_dump
    sim_fn = None  # NOTE: cannot simulate using this option, so this
    # shouldn't be called anyways...

    # Use a relatively forgiving threshold for grasp constraints in general,
    # but for the ball, use a strict constraint.
    if target_obj.name == "ball":
        thresh = 0.17
    else:
        thresh = np.pi / 4

    # Retry a grasp if a failure occurs!
    retry_with_no_constraints = target_obj.name != "brush"
    action_extra_info = SpotActionExtraInfo(
        name, objects, fn, (robot, img, pixel, grasp_rot, thresh, 20.0,
                            retry_with_no_constraints, do_stow, do_dump),
        sim_fn, (sim_robot, grasp_rot, sim_target_obj))
    return utils.create_spot_env_action(action_extra_info)


def _sweep_objects_into_container_policy(name: str, robot_obj_idx: int,
                                         target_obj_idxs: Set[int],
                                         surface_obj_idx: int, state: State,
                                         memory: Dict,
                                         objects: Sequence[Object],
                                         params: Array) -> Action:
    del memory  # not used

    robot, _, _ = get_robot()

    velocity, = params
    duration = max(1 / velocity, 1e-3)

    robot_obj = objects[robot_obj_idx]
    robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

    target_obj_rel_xyzs: List[Tuple[float, float, float]] = []
    # First, compute the mean x and y positions of the object(s) to be swept
    # from atop the surface.
    for target_obj_idx in target_obj_idxs:
        target_obj = objects[target_obj_idx]
        target_pose = utils.get_se3_pose_from_state(state, target_obj)
        target_rel_pose = robot_pose.inverse() * target_pose
        rel_xyz = (target_rel_pose.x, target_rel_pose.y, target_rel_pose.z)
        target_obj_rel_xyzs.append(rel_xyz)
    mean_x, mean_y, _ = np.mean(target_obj_rel_xyzs, axis=0)
    # Next, compute the surface pose, and define hardcoded (magic number)
    # poses given the particular brush we're using for sweeping.
    surface_obj = objects[surface_obj_idx]
    surface_center_pose = utils.get_se3_pose_from_state(state, surface_obj)
    surface_height = state.get(surface_obj, "height")
    surface_width = state.get(surface_obj, "width")
    upper_left_surface_pose = math_helpers.SE3Pose(
        x=surface_center_pose.x + surface_width / 2.0 + 0.12,
        y=surface_center_pose.y - surface_height / 2.0 + 0.15,
        z=0.25,
        rot=surface_center_pose.rot)
    middle_bottom_surface_pose = math_helpers.SE3Pose(
        x=surface_center_pose.x + 0.12,
        y=surface_center_pose.y + surface_height / 2.0 - 0.10,
        z=0.25,
        rot=surface_center_pose.rot)
    upper_left_surface_rel_pose = robot_pose.inverse(
    ) * upper_left_surface_pose
    middle_bottom_surface_rel_pose = robot_pose.inverse(
    ) * middle_bottom_surface_pose
    # Now, compute the actual pose the hand should start sweeping from by
    # clamping it between the surface poses.
    start_x = np.clip(middle_bottom_surface_rel_pose.x, mean_x + 0.175,
                      upper_left_surface_rel_pose.x)
    start_y = np.clip(middle_bottom_surface_rel_pose.y, mean_y + 0.41,
                      upper_left_surface_rel_pose.y)
    # use absolute value so that we don't get messed up by noise in the
    # perception height estimate.
    start_z = 0.14
    pitch = math_helpers.Quat.from_pitch(np.pi / 2)
    yaw = math_helpers.Quat.from_yaw(np.pi / 4)
    rot = pitch * yaw
    sweep_start_pose = math_helpers.SE3Pose(x=start_x,
                                            y=start_y,
                                            z=start_z,
                                            rot=rot)
    sweep_move_dx = 0.0
    sweep_move_dy = -0.8
    sweep_move_dz = 0.0

    # Execute the sweep. Note simulation fn and args not implemented yet.
    action_extra_info = SpotActionExtraInfo(
        name, objects, sweep, (robot, sweep_start_pose, sweep_move_dx,
                               sweep_move_dy, sweep_move_dz, duration), None,
        ())
    return utils.create_spot_env_action(action_extra_info)


def _pick_and_dump_policy(name: str, robot_obj_idx: int, target_obj_idx: int,
                          state: State, memory: Dict,
                          objects: Sequence[Object], params: Array) -> Action:
    grasp_action = _grasp_policy(name,
                                 target_obj_idx,
                                 state,
                                 memory,
                                 objects,
                                 params,
                                 do_dump=True)

    # If the container starts out next to a surface while ready for sweeping,
    # put it back.
    robot = objects[robot_obj_idx]
    container = objects[target_obj_idx]
    surface = _get_sweeping_surface_for_container(container, state)
    if surface is None:
        return grasp_action
    param_dict = load_spot_metadata()["prepare_container_relative_xy"]
    prep_params = np.array(
        [param_dict["dx"], param_dict["dy"], param_dict["angle"]])
    prep_objects = [robot, container, surface, surface]
    prep_sweep_action = _prepare_container_for_sweeping_policy(
        state, memory, prep_objects, prep_params)

    # Chain the actions.
    actions = [grasp_action, prep_sweep_action]

    def _fn() -> None:
        for action in actions:
            assert isinstance(action.extra_info, (list, tuple))
            _, _, action_fn, action_fn_args, _, _ = action.extra_info
            action_fn(*action_fn_args)

    # Note simulation fn and args not implemented yet.
    action_extra_info = SpotActionExtraInfo(name, objects, _fn, tuple(), None,
                                            tuple())
    return utils.create_spot_env_action(action_extra_info)


###############################################################################
#                   Concrete parameterized option policies                    #
###############################################################################


def _move_to_hand_view_object_policy(state: State, memory: Dict,
                                     objects: Sequence[Object],
                                     params: Array) -> Action:
    name = "MoveToHandViewObject"
    distance_param_idx = 0
    yaw_param_idx = 1
    robot_obj_idx = 0
    target_obj_idx = 1
    do_gaze = True
    return _move_to_target_policy(name, distance_param_idx, yaw_param_idx,
                                  robot_obj_idx, target_obj_idx, do_gaze,
                                  state, memory, objects, params)


def _move_to_body_view_object_policy(state: State, memory: Dict,
                                     objects: Sequence[Object],
                                     params: Array) -> Action:
    name = "MoveToBodyViewObject"
    distance_param_idx = 0
    yaw_param_idx = 1
    robot_obj_idx = 0
    target_obj_idx = 1
    do_gaze = False
    return _move_to_target_policy(name, distance_param_idx, yaw_param_idx,
                                  robot_obj_idx, target_obj_idx, do_gaze,
                                  state, memory, objects, params)


def _move_to_reach_object_policy(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
    name = "MoveToReachObject"
    distance_param_idx = 0
    yaw_param_idx = 1
    robot_obj_idx = 0
    target_obj_idx = 1
    do_gaze = False
    return _move_to_target_policy(name, distance_param_idx, yaw_param_idx,
                                  robot_obj_idx, target_obj_idx, do_gaze,
                                  state, memory, objects, params)


def _pick_object_from_top_policy(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
    name = "PickObjectFromTop"
    target_obj_idx = 1
    return _grasp_policy(name, target_obj_idx, state, memory, objects, params)


def _sim_safe_pick_object_from_top_policy(state: State, memory: Dict,
                                          objects: Sequence[Object],
                                          params: Array) -> Action:
    del state, memory, params  # unused.
    name = "SimSafePickObjectFromTop"
    target_obj_idx = 1
    robot, _, _ = get_robot()
    if not CFG.bilevel_plan_without_sim:
        sim_robot = get_simulated_robot()
    else:
        sim_robot = None

    fn = _sim_safe_grasp_at_pixel_and_maybe_stow_or_dump
    fn_args = (robot, objects[target_obj_idx], _options_rng, 10.0, True, True,
               False)

    if not CFG.bilevel_plan_without_sim:
        sim_fn: Callable = simulated_grasp_at_pixel
        sim_target_obj = get_simulated_object(objects[target_obj_idx])
        sim_fn_args: Tuple = (sim_robot, sim_target_obj)
    else:
        sim_fn = lambda _: None
        sim_fn_args = ()

    action_extra_info = SpotActionExtraInfo(name, objects, fn, fn_args, sim_fn,
                                            sim_fn_args)
    return utils.create_spot_env_action(action_extra_info)


def _pick_object_to_drag_policy(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
    name = "PickObjectToDrag"
    target_obj_idx = 1
    return _grasp_policy(name, target_obj_idx, state, memory, objects, params)


def _pick_and_dump_cup_policy(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> Action:
    # Same as PickObjectFromTop; just necessary to make options 1:1 with
    # operators.
    name = "PickAndDumpCup"
    target_obj_idx = 1
    return _grasp_policy(name, target_obj_idx, state, memory, objects, params)


def _pick_and_dump_container_policy(state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> Action:
    # Pick up and dump the container, then put it back.
    name = "PickAndDumpContainer"
    robot_obj_idx = 0
    target_obj_idx = 1
    return _pick_and_dump_policy(name, robot_obj_idx, target_obj_idx, state,
                                 memory, objects, params)


def _pick_and_dump_two_container_policy(state: State, memory: Dict,
                                        objects: Sequence[Object],
                                        params: Array) -> Action:
    # Pick up and dump the container, then put it back.
    name = "PickAndDumpTwoFromContainer"
    robot_obj_idx = 0
    target_obj_idx = 1
    return _pick_and_dump_policy(name, robot_obj_idx, target_obj_idx, state,
                                 memory, objects, params)


def _place_object_on_top_policy(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
    del memory  # not used

    name = "PlaceObjectOnTop"
    robot_obj_idx = 0
    surface_obj_idx = 2

    robot, localizer, _ = get_robot()

    dx, dy, dz = params

    robot_obj = objects[robot_obj_idx]
    robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

    surface_obj = objects[surface_obj_idx]
    surface_pose = utils.get_se3_pose_from_state(state, surface_obj)

    # The dz parameter is with respect to the top of the container.
    surface_half_height = state.get(surface_obj, "height") / 2

    place_pose = math_helpers.SE3Pose(
        x=surface_pose.x + dx,
        y=surface_pose.y + dy,
        z=surface_pose.z + surface_half_height + dz,
        rot=surface_pose.rot,
    )

    # Special case: the robot is already on top of the surface (because it is
    # probably the floor). When this happens, just drop the object.
    surface_geom = object_to_top_down_geom(surface_obj, state)
    if surface_geom.contains_point(
            robot_pose.x, robot_pose.y) and surface_obj.name == "floor":
        # Note simulation fn and args not yet implemented.
        action_extra_info = SpotActionExtraInfo(name, objects, _drop_and_stow,
                                                (robot, ), None, tuple())
        return utils.create_spot_env_action(action_extra_info)

    # If we're running on the actual robot, we want to be very precise
    # about the robot's current pose when computing the relative
    # placement position.
    if not CFG.spot_run_dry:
        assert localizer is not None
        localizer.localize()
        robot_pose = localizer.get_last_robot_pose()

    place_rel_pos = robot_pose.inverse() * place_pose
    # Note simulation fn and args not yet implemented.
    action_extra_info = SpotActionExtraInfo(
        name, objects, _place_at_relative_position_and_stow,
        (robot, place_rel_pos), None, tuple())
    return utils.create_spot_env_action(action_extra_info)


def _drop_object_inside_policy(state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> Action:
    del memory  # not used

    name = "DropObjectInside"
    robot_obj_idx = 0
    container_obj_idx = 2

    robot, _, _ = get_robot()

    dx, dy, dz = params

    robot_obj = objects[robot_obj_idx]
    robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

    container_obj = objects[container_obj_idx]
    container_pose = utils.get_se3_pose_from_state(state, container_obj)
    # The dz parameter is with respect to the top of the container.
    container_half_height = state.get(container_obj, "height") / 2

    container_rel_pose = robot_pose.inverse() * container_pose
    place_z = container_rel_pose.z + container_half_height + dz
    place_rel_pos = math_helpers.Vec3(x=container_rel_pose.x + dx,
                                      y=container_rel_pose.y + dy,
                                      z=place_z)
    # Note simulation fn and args not yet implemented.
    action_extra_info = SpotActionExtraInfo(
        name, objects, _drop_at_relative_position_and_look,
        (robot, place_rel_pos), None, tuple())
    return utils.create_spot_env_action(action_extra_info)


def _drop_not_placeable_object_policy(state: State, memory: Dict,
                                      objects: Sequence[Object],
                                      params: Array) -> Action:
    del state, memory, params  # not used

    name = "DropNotPlaceableObject"
    robot, _, _ = get_robot()

    # Note simulation fn and args not yet implemented.
    action_extra_info = SpotActionExtraInfo(name, objects,
                                            _open_and_close_gripper, (robot, ),
                                            None, tuple())
    return utils.create_spot_env_action(action_extra_info)


def _move_and_drop_object_inside_policy(state: State, memory: Dict,
                                        objects: Sequence[Object],
                                        params: Array) -> Action:
    del memory  # not used

    name = "MoveAndDropObjectInside"
    robot_obj_idx = 0
    container_obj_idx = 2
    ontop_surface_obj_idx = 3

    robot, localizer, _ = get_robot()

    dx, dy, dz = params

    robot_obj = objects[robot_obj_idx]
    robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

    container_obj = objects[container_obj_idx]
    container_pose = utils.get_se3_pose_from_state(state, container_obj)

    surface_obj = objects[ontop_surface_obj_idx]

    # Special case: the robot is already on top of the surface (because it is
    # probably the floor). When this happens, just drop the object.
    surface_geom = object_to_top_down_geom(surface_obj, state)
    if surface_geom.contains_point(robot_pose.x, robot_pose.y):
        # Note simulation fn and args not yet implemented.
        action_extra_info = SpotActionExtraInfo(name, objects, _drop_and_stow,
                                                (robot, ), None, tuple())
        return utils.create_spot_env_action(action_extra_info)

    # The dz parameter is with respect to the top of the container.
    container_half_height = state.get(container_obj, "height") / 2

    place_z = container_pose.z + container_half_height + dz
    place_abs_pos = math_helpers.Vec3(x=container_pose.x + dx,
                                      y=container_pose.y + dy,
                                      z=place_z)
    # Note simulation fn and args not yet implemented.
    action_extra_info = SpotActionExtraInfo(
        name, objects, _move_closer_and_drop_at_relative_position_and_look,
        (robot, localizer, place_abs_pos), None, tuple())
    return utils.create_spot_env_action(action_extra_info)


def _drag_to_unblock_object_policy(state: State, memory: Dict,
                                   objects: Sequence[Object],
                                   params: Array) -> Action:
    del state, memory  # not used

    name = "DragToUnblockObject"
    robot, _, _ = get_robot()
    dx, dy, dyaw = params
    move_rel_pos = math_helpers.SE2Pose(dx, dy, angle=dyaw)
    # Note that simulation fn and args not yet implemented.
    action_extra_info = SpotActionExtraInfo(name, objects, _drag_and_release,
                                            (robot, move_rel_pos), None,
                                            tuple())
    return utils.create_spot_env_action(action_extra_info)


def _drag_to_block_object_policy(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
    del state, memory  # not used

    name = "DragToBlockObject"
    robot, _, _ = get_robot()
    dx, dy, dyaw = params
    move_rel_pos = math_helpers.SE2Pose(dx, dy, angle=dyaw)
    # Note that simulation fn and args not yet implemented.
    action_extra_info = SpotActionExtraInfo(name, objects, _drag_and_release,
                                            (robot, move_rel_pos), None,
                                            tuple())
    return utils.create_spot_env_action(action_extra_info)


def _sweep_into_container_policy(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
    name = "SweepIntoContainer"
    robot_obj_idx = 0
    target_obj_idxs = {2}
    surface_obj_idx = 3
    return _sweep_objects_into_container_policy(name, robot_obj_idx,
                                                target_obj_idxs,
                                                surface_obj_idx, state, memory,
                                                objects, params)


def _sweep_two_objects_into_container_policy(state: State, memory: Dict,
                                             objects: Sequence[Object],
                                             params: Array) -> Action:
    name = "SweepTwoObjectsIntoContainer"
    robot_obj_idx = 0
    target_obj_idxs = {2, 3}
    surface_obj_idx = 4
    return _sweep_objects_into_container_policy(name, robot_obj_idx,
                                                target_obj_idxs,
                                                surface_obj_idx, state, memory,
                                                objects, params)


def _prepare_container_for_sweeping_policy(state: State, memory: Dict,
                                           objects: Sequence[Object],
                                           params: Array) -> Action:
    del memory  # not used

    name = "PrepareContainerForSweeping"
    container_obj_idx = 1
    target_obj_idx = 3  # the surface

    robot, localizer, _ = get_robot()

    dx, dy, dyaw = params

    container_obj = objects[container_obj_idx]
    container_z = state.get(container_obj, "z")  # assumed fixed

    target_obj = objects[target_obj_idx]
    target_pose = utils.get_se3_pose_from_state(state, target_obj)
    absolute_move_pose = math_helpers.SE2Pose(target_pose.x + dx,
                                              target_pose.y + dy, dyaw)

    # Place in front.
    rot = math_helpers.Quat.from_pitch(np.pi / 2)
    place_rel_pose = math_helpers.SE3Pose(x=0.6,
                                          y=0.0,
                                          z=container_z - 0.15,
                                          rot=rot)

    # Push towards the target a little bit after placing.
    # Rotate the gripper a little bit to make sure the tray is aligned.
    rot = math_helpers.Quat.from_pitch(
        np.pi / 2) * math_helpers.Quat.from_roll(-np.pi / 6)
    push_rel_pose = math_helpers.SE3Pose(x=place_rel_pose.x,
                                         y=place_rel_pose.y + 0.15,
                                         z=place_rel_pose.z,
                                         rot=rot)
    # Note that simulation fn and args not yet implemented.
    action_extra_info = SpotActionExtraInfo(
        name, objects, _move_to_absolute_pose_and_place_push_stow,
        (robot, localizer, absolute_move_pose, place_rel_pose, push_rel_pose),
        None, tuple())
    return utils.create_spot_env_action(action_extra_info)


def _move_to_ready_sweep_policy(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
    name = "MoveToReadySweep"

    # Always approach from the same angle.
    yaw = np.pi / 2.0
    # Make up new params.
    distance = 0.8
    params = np.array([distance, yaw])
    distance_param_idx = 0
    yaw_param_idx = 1
    robot_obj_idx = 0
    target_obj_idx = 2
    do_gaze = False
    return _move_to_target_policy(name, distance_param_idx, yaw_param_idx,
                                  robot_obj_idx, target_obj_idx, do_gaze,
                                  state, memory, objects, params)


def _create_teleop_policy_with_name(
        name: str) -> Callable[[State, Dict, Sequence[Object], Array], Action]:

    def _teleop_policy(state: State, memory: Dict, objects: Sequence[Object],
                       params: Array) -> Action:
        nonlocal name
        del state, memory, params

        robot, _, lease_client = get_robot(use_localizer=False)

        def _teleop(robot: Robot, lease_client: LeaseClient) -> None:
            del robot  # unused.
            prompt = "Press (y) when you are done with teleop."
            while True:
                response = utils.prompt_user(prompt).strip()
                if response == "y":
                    break
                logging.info("Invalid input. Press (y) when y")
            # Take back control.
            robot, _, lease_client = get_robot(use_localizer=False)
            lease_client.take()

        fn = _teleop
        fn_args = (robot, lease_client)
        sim_fn = lambda _: None
        sim_fn_args = ()
        action_extra_info = SpotActionExtraInfo(name, objects, fn, fn_args,
                                                sim_fn, sim_fn_args)
        return utils.create_spot_env_action(action_extra_info)

    return _teleop_policy


###############################################################################
#                       Parameterized option factory                          #
###############################################################################

_OPERATOR_NAME_TO_PARAM_SPACE = {
    "MoveToReachObject": Box(-np.inf, np.inf, (2, )),  # rel dist, dyaw
    "MoveToHandViewObject": Box(-np.inf, np.inf, (2, )),  # rel dist, dyaw
    "MoveToBodyViewObject": Box(-np.inf, np.inf, (2, )),  # rel dist, dyaw
    # x, y pixel in image + quat (qw, qx, qy, qz). If quat is all 0's
    # then grasp is unconstrained
    "PickObjectFromTop": Box(-np.inf, np.inf, (6, )),
    # same as PickObjectFromTop
    "PickAndDumpCup": Box(-np.inf, np.inf, (6, )),
    # same as PickObjectFromTop
    "PickAndDumpContainer": Box(-np.inf, np.inf, (6, )),
    # same as PickObjectFromTop
    "PickAndDumpTwoFromContainer": Box(-np.inf, np.inf, (6, )),
    # same as PickObjectFromTop
    "PickObjectToDrag": Box(-np.inf, np.inf, (6, )),
    "PlaceObjectOnTop": Box(-np.inf, np.inf, (3, )),  # rel dx, dy, dz
    "DropObjectInside": Box(-np.inf, np.inf, (3, )),  # rel dx, dy, dz
    "DropObjectInsideContainerOnTop": Box(-np.inf, np.inf,
                                          (3, )),  # rel dx, dy, dz
    "DragToUnblockObject": Box(-np.inf, np.inf, (3, )),  # rel dx, dy, dyaw
    "DragToBlockObject": Box(-np.inf, np.inf, (3, )),  # rel dx, dy, dyaw
    "SweepIntoContainer": Box(-np.inf, np.inf, (1, )),  # velocity
    "SweepTwoObjectsIntoContainer": Box(-np.inf, np.inf, (1, )),  # same
    "PrepareContainerForSweeping": Box(-np.inf, np.inf, (3, )),  # dx, dy, dyaw
    "DropNotPlaceableObject": Box(0, 1, (0, )),  # empty
    "MoveToReadySweep": Box(0, 1, (0, )),  # empty
    "TeleopPick1": Box(0, 1, (0, )),  # empty
    "PlaceNextTo": Box(0, 1, (0, )),  # empty
    "TeleopPick2": Box(0, 1, (0, )),  # empty
    "Sweep": Box(0, 1, (0, )),  # empty
    "PlaceOnFloor": Box(0, 1, (0, ))  # empty
}

# NOTE: the policies MUST be unique because they output actions with extra info
# that includes the name of the operators.
_OPERATOR_NAME_TO_POLICY = {
    "MoveToReachObject": _move_to_reach_object_policy,
    "MoveToHandViewObject": _move_to_hand_view_object_policy,
    "MoveToBodyViewObject": _move_to_body_view_object_policy,
    "PickObjectFromTop": _pick_object_from_top_policy,
    "PickObjectToDrag": _pick_object_to_drag_policy,
    "PickAndDumpCup": _pick_and_dump_cup_policy,
    "PickAndDumpContainer": _pick_and_dump_container_policy,
    "PickAndDumpTwoFromContainer": _pick_and_dump_two_container_policy,
    "PlaceObjectOnTop": _place_object_on_top_policy,
    "DropObjectInside": _drop_object_inside_policy,
    "DropObjectInsideContainerOnTop": _move_and_drop_object_inside_policy,
    "DragToUnblockObject": _drag_to_unblock_object_policy,
    "DragToBlockObject": _drag_to_block_object_policy,
    "SweepIntoContainer": _sweep_into_container_policy,
    "SweepTwoObjectsIntoContainer": _sweep_two_objects_into_container_policy,
    "PrepareContainerForSweeping": _prepare_container_for_sweeping_policy,
    "DropNotPlaceableObject": _drop_not_placeable_object_policy,
    "MoveToReadySweep": _move_to_ready_sweep_policy,
    "TeleopPick1": _create_teleop_policy_with_name("TeleopPick1"),
    "PlaceNextTo": _create_teleop_policy_with_name("PlaceNextTo"),
    "TeleopPick2": _create_teleop_policy_with_name("TeleopPick2"),
    "Sweep": _create_teleop_policy_with_name("Sweep"),
    "PlaceOnFloor": _create_teleop_policy_with_name("PlaceOnFloor")
}


class _SpotParameterizedOption(utils.SingletonParameterizedOption):
    """A parameterized option for spot.

    NOTE: parameterized options MUST be singletons in order to avoid nasty
    issues with the expected atoms monitoring.

    Also note that we need to define the policies outside the class, rather
    than pass the policies into the class, to avoid pickling issues via bosdyn.
    """

    def __init__(self, operator_name: str, types: List[Type]) -> None:
        # If we're doing proper bilevel planning with a simulator, then
        # we need to make some adjustments to the params spaces
        # and policies.
        if not CFG.bilevel_plan_without_sim:
            _OPERATOR_NAME_TO_PARAM_SPACE["PickObjectFromTop"] = Box(
                0, 1, (0, ))
            _OPERATOR_NAME_TO_POLICY[
                "PickObjectFromTop"] = _sim_safe_pick_object_from_top_policy
        params_space = _OPERATOR_NAME_TO_PARAM_SPACE[operator_name]
        policy = _OPERATOR_NAME_TO_POLICY[operator_name]
        super().__init__(operator_name, policy, types, params_space)

    def __reduce__(self) -> Tuple:
        return (_SpotParameterizedOption, (self.name, self.types))


class SpotEnvsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for Spot environments."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "spot_vlm_dustpan_test_env",
            "spot_vlm_cup_table_env",
            "spot_cube_env",
            "spot_soda_floor_env",
            "spot_soda_table_env",
            "spot_soda_bucket_env",
            "spot_soda_chair_env",
            "spot_main_sweep_env",
            "spot_ball_and_cup_sticky_table_env",
            "spot_brush_shelf_env",
            "lis_spot_block_floor_env",
        }

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        # Note that these are 1:1 with the operators.
        env = get_or_create_env(env_name)
        assert isinstance(env, SpotRearrangementEnv)

        options: Set[ParameterizedOption] = set()
        for operator in env.strips_operators:
            operator_types = [p.type for p in operator.parameters]
            option = _SpotParameterizedOption(operator.name, operator_types)
            options.add(option)

        return options
