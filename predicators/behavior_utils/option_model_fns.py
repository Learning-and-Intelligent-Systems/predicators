"""Functions that consume a plan for a BEHAVIOR robot and return an option
model for that plan."""

import logging
from typing import Callable, Dict, List

import numpy as np
import pybullet as p
from numpy.random import RandomState

from predicators import utils
from predicators.behavior_utils.behavior_utils import \
    ALL_RELEVANT_OBJECT_TYPES, sample_navigation_params, \
    sample_place_inside_params, sample_place_ontop_params
from predicators.settings import CFG
from predicators.structs import Object, State, Type

try:
    from igibson import object_states
    from igibson.envs.behavior_env import \
        BehaviorEnv  # pylint: disable=unused-import
    from igibson.object_states.adjacency import VerticalAdjacency
    from igibson.object_states.on_floor import \
        RoomFloor  # pylint: disable=unused-import
    from igibson.object_states.touching import Touching
    from igibson.objects.articulated_object import \
        URDFObject  # pylint: disable=unused-import
    from igibson.robots.behavior_robot import BRBody, BREye, BRHand \
        # pylint: disable=unused-import

except (ImportError, ModuleNotFoundError) as e:
    pass

# Necessary to ensure different numbers sampled within the
# NavigateToOptionModel.
prng = RandomState(10000)


def create_navigate_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        _obj_to_nav_to: "URDFObject"
) -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a navigation option model function given a
    plan, which is a list of 3-element lists each containing a series of (x, y,
    rot) waypoints for the robot to pass through."""

    def navigateToOptionModel(_init_state: State, env: "BehaviorEnv") -> None:
        robot_z = env.robots[0].get_position()[2]
        robot_orn = p.getEulerFromQuaternion(env.robots[0].get_orientation())
        # If we're not overriding the learned samplers, then we will directly
        # use the elements of `plan`, which in turn use the outputs of the
        # learned samplers. Otherwise, we will ignore these and use our
        # oracle sampler to give us values to use.
        if not CFG.behavior_override_learned_samplers:
            desired_xpos = plan[-1][0]
            desired_ypos = plan[-1][1]
            desired_zrot = plan[-1][2]
        else:
            rng = np.random.default_rng(prng.randint(10000))
            sample_arr = sample_navigation_params(env, _obj_to_nav_to, rng)
            obj_pos = _obj_to_nav_to.get_position()
            desired_xpos = sample_arr[0] + obj_pos[0]
            desired_ypos = sample_arr[1] + obj_pos[1]
            desired_zrot = np.arctan2(sample_arr[1], sample_arr[0]) - np.pi
            logging.info(f"PRIMITIVE: Overriding sample ({plan[-1][0]}" +
                         f", {plan[-1][1]}) and attempting to " +
                         f"navigate to {_obj_to_nav_to.name} with "
                         f"params {sample_arr}")

        if CFG.simulate_nav:
            done_bit = False
            while not done_bit:
                # Get expected position and orientation from plan.
                expected_pos = np.array([plan[0][0], plan[0][1], robot_z])
                expected_orn = p.getQuaternionFromEuler(
                    np.array([robot_orn[0], robot_orn[1], plan[0][2]]))
                # In this case, we're at the final position we wanted to reach.
                if len(plan) == 1:
                    done_bit = True
                    logging.info(
                        "PRIMITIVE: navigation policy completed execution!")
                env.robots[0].set_position_orientation(expected_pos,
                                                       expected_orn)
                env.step(np.zeros(env.action_space.shape))
                plan.pop(0)
        target_pos = np.array([desired_xpos, desired_ypos, robot_z])
        target_orn = p.getQuaternionFromEuler(
            np.array([robot_orn[0], robot_orn[1], desired_zrot]))
        env.robots[0].set_position_orientation(target_pos, target_orn)
        # this is running a zero action to step simulator so
        # the environment updates to the correct final position
        env.step(np.zeros(env.action_space.shape))

    return navigateToOptionModel


def create_grasp_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_grasp: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a grasp option model function given a plan,
    which is a list of 6-element lists containing a series of (x, y, z, roll,
    pitch, yaw) waypoints for the hand to pass through."""

    # NOTE: -1 because there are 25 timesteps that we move along the vector
    # between the hand the object for until finally grasping, and we want
    # just the final orientation.
    hand_i = -1
    rh_final_grasp_postion = plan[hand_i][0:3]
    rh_final_grasp_orn = plan[hand_i][3:6]

    def graspObjectOptionModel(_state: State, env: "BehaviorEnv") -> None:
        nonlocal hand_i
        rh_orig_grasp_position = env.robots[0].parts[
            "right_hand"].get_position()
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
            rh_orig_grasp_position, rh_orig_grasp_orn)
        if env.robots[0].parts["right_hand"].object_in_hand is not None:
            # NOTE: This below line is necessary to update the visualizer.
            # Also, it only works for URDF objects (but if the object is
            # not a URDF object, grasping should have failed)
            obj_to_grasp.force_wakeup()
        # Step a zero-action in the environment to update the visuals of the
        # environment.
        env.step(np.zeros(env.action_space.shape))

    return graspObjectOptionModel


def create_place_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_place_onto: "URDFObject"
) -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns a place option model function given a plan,
    which is a list of 6-element lists containing a series of (x, y, z, roll,
    pitch, yaw) waypoints for the hand to pass through."""

    def placeOntopObjectOptionModel(_init_state: State,
                                    env: "BehaviorEnv") -> None:
        obj_in_hand_idx = env.robots[0].parts["right_hand"].object_in_hand
        obj_in_hand = [
            obj for obj in env.scene.get_objects()
            if obj.get_body_id() == obj_in_hand_idx
        ][0]
        rh_orig_grasp_position = env.robots[0].parts[
            "right_hand"].get_position()
        rh_orig_grasp_orn = env.robots[0].parts["right_hand"].get_orientation()
        # If we're not overriding the learned samplers, then we will directly
        # use the elements of `plan`, which in turn use the outputs of the
        # learned samplers. Otherwise, we will ignore these and use our
        # oracle sampler to give us values to use.
        if not CFG.behavior_override_learned_samplers:
            target_pos = plan[-1][0:3]
            target_orn = plan[-1][3:6]
        else:
            rng = np.random.default_rng(prng.randint(10000))
            sample_arr = sample_place_ontop_params(env, obj_to_place_onto, rng)
            target_pos = np.add(sample_arr, \
                obj_to_place_onto.get_position()).tolist()
            target_orn = [0, np.pi * 7 / 6, 0]
            logging.info(f"PRIMITIVE: Overriding sample ({plan[-1][0:3]}" +
                         "and attempting to " +
                         f"place ontop {obj_to_place_onto.name} with "
                         f"params {target_pos}")

        env.robots[0].parts["right_hand"].set_position_orientation(
            target_pos, p.getQuaternionFromEuler(target_orn))
        env.robots[0].parts["right_hand"].force_release_obj()
        obj_in_hand.set_position_orientation(
            target_pos, p.getQuaternionFromEuler(target_orn))
        obj_to_place_onto.force_wakeup()
        # this is running a zero action to step simulator
        env.step(np.zeros(env.action_space.shape))
        # reset the released object to zero velocity so it doesn't
        # fly away because of residual warp speeds from teleportation!
        p.resetBaseVelocity(
            obj_in_hand_idx,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
        )
        env.robots[0].parts["right_hand"].set_position_orientation(
            rh_orig_grasp_position, rh_orig_grasp_orn)
        # this is running a series of zero action to step simulator
        # to let the object fall into its place
        for _ in range(15):
            env.step(np.zeros(env.action_space.shape))
        # Check whether object is ontop of not a target object
        objs_under = set()
        for obj in env.scene.get_objects():
            if not obj_in_hand.states[Touching].get_value(obj):
                continue
            adjacency = obj.states[VerticalAdjacency].get_value()
            if obj_in_hand.get_body_id() in adjacency.positive_neighbors:
                if "floor" not in obj.category:
                    if obj != obj_to_place_onto:
                        type_name = obj.category
                        _type_name_to_type: Dict[str, Type] = {}
                        for type_name in ALL_RELEVANT_OBJECT_TYPES:
                            if type_name in _type_name_to_type:
                                continue
                            # In the future, we may need other object
                            # attributes, but for the moment, we just
                            # need position and orientation.
                            obj_type = Type(
                                type_name,
                                [
                                    "pos_x", "pos_y", "pos_z", "orn_0",
                                    "orn_1", "orn_2", "orn_3"
                                ],
                            )
                            _type_name_to_type[type_name] = obj_type
                        if isinstance(obj, (URDFObject, RoomFloor)):
                            if "board_game" in obj.name or \
                                "video_game" in obj.name:
                                obj_name = obj.name + ".n.01_1"
                            else:
                                obj_name = obj.bddl_object_scope
                        else:
                            assert isinstance(obj, (BRBody, BRHand, BREye))
                            obj_name = "agent"
                        # This checks if our obj type is in _type_name_to_type
                        # which has all relevant objects. If not, it continues
                        # and does not add the obj as an offending_object.
                        if type_name in _type_name_to_type:
                            obj_type = _type_name_to_type[type_name]
                            try:
                                objs_under.add(Object(obj_name, obj_type))
                            except:  # pylint: disable=bare-except
                                print("INVALID OBJECT TYPE:", obj_name,
                                      obj_type)
                                continue
                        else:
                            continue
        if len(objs_under) != 0:
            if not CFG.behavior_ignore_discover_failures:
                raise utils.EnvironmentFailure(
                    "collision", {"offending_objects": objs_under})

    return placeOntopObjectOptionModel


def create_open_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_open: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns an open option model given a dummy plan."""
    del plan

    def openObjectOptionModel(_init_state: State, env: "BehaviorEnv") -> None:
        logging.info(f"PRIMITIVE: Attempting to open {obj_to_open.name}")
        if np.linalg.norm(
                np.array(obj_to_open.get_position()) -
                np.array(env.robots[0].get_position())) < 2:
            if hasattr(obj_to_open,
                       "states") and object_states.Open in obj_to_open.states:
                obj_to_open.states[object_states.Open].set_value(True)
            else:
                logging.info("PRIMITIVE open failed, cannot be opened")
        else:
            logging.info("PRIMITIVE open failed, too far")
        obj_to_open.force_wakeup()
        # Step the simulator to update visuals.
        env.step(np.zeros(env.action_space.shape))

    return openObjectOptionModel


def create_close_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_close: "URDFObject") -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns an close option model given a dummy plan."""
    del plan

    def closeObjectOptionModel(_init_state: State, env: "BehaviorEnv") -> None:
        logging.info(f"PRIMITIVE: Attempting to close {obj_to_close.name}")
        if np.linalg.norm(
                np.array(obj_to_close.get_position()) -
                np.array(env.robots[0].get_position())) < 2:
            if hasattr(obj_to_close,
                       "states") and object_states.Open in obj_to_close.states:
                obj_to_close.states[object_states.Open].set_value(False)
            else:
                logging.info("PRIMITIVE close failed, cannot be opened")
        else:
            logging.info("PRIMITIVE close failed, too far")
        obj_to_close.force_wakeup()
        # Step the simulator to update visuals.
        env.step(np.zeros(env.action_space.shape))

    return closeObjectOptionModel


def create_place_inside_option_model(
        plan: List[List[float]], _original_orientation: List[List[float]],
        obj_to_place_into: "URDFObject"
) -> Callable[[State, "BehaviorEnv"], None]:
    """Instantiates and returns an placeInside option model given a dummy
    plan."""

    def placeInsideObjectOptionModel(_init_state: State,
                                     env: "BehaviorEnv") -> None:
        obj_in_hand_idx = env.robots[0].parts["right_hand"].object_in_hand
        obj_in_hand = [
            obj for obj in env.scene.get_objects()
            if obj.get_body_id() == obj_in_hand_idx
        ][0]
        rh_orig_grasp_position = env.robots[0].parts[
            "right_hand"].get_position()
        rh_orig_grasp_orn = env.robots[0].parts["right_hand"].get_orientation()
        if obj_in_hand is not None and obj_in_hand != obj_to_place_into and \
            isinstance(obj_to_place_into, URDFObject):
            logging.info(
                f"PRIMITIVE: attempt to place {obj_in_hand.name} inside "
                f"{obj_to_place_into.name}")
            if np.linalg.norm(
                    np.array(obj_to_place_into.get_position()) -
                    np.array(env.robots[0].get_position())) < 2:
                if (hasattr(obj_to_place_into, "states")
                        and object_states.Open in obj_to_place_into.states
                        and obj_to_place_into.states[object_states.Open].
                        get_value()) or (hasattr(obj_to_place_into, "states")
                                         and not object_states.Open
                                         in obj_to_place_into.states):
                    logging.info(f"PRIMITIVE: place {obj_in_hand.name} inside "
                                 f"{obj_to_place_into.name} success")

                    # If we're not overriding the learned samplers, then we
                    # will directly use the elements of `plan`, which in turn
                    # use the outputs of the learned samplers. Otherwise, we
                    # will ignore these and use our oracle sampler to give us
                    # values to use.
                    if not CFG.behavior_override_learned_samplers:
                        target_pos = plan[-1][0:3]
                        target_orn = plan[-1][3:6]
                    else:
                        rng = np.random.default_rng(prng.randint(10000))
                        place_rel_pos = sample_place_inside_params(
                            obj_to_place_into, rng)
                        target_pos_list = np.add(
                            place_rel_pos, obj_to_place_into.get_position())
                        target_pos_list[2] += 0.2
                        target_pos = target_pos_list.tolist()
                        target_orn = plan[-1][3:6]
                        logging.info(
                            f"PRIMITIVE: Overriding sample ({plan[-1][0:3]}" +
                            f", {plan[-1][3:6]}) and attempting to " +
                            f"place inside to {obj_to_place_into.name} with "
                            f"params {target_pos}")
                    env.robots[0].parts["right_hand"].force_release_obj()
                    obj_to_place_into.force_wakeup()
                    obj_in_hand.set_position_orientation(
                        target_pos, p.getQuaternionFromEuler(target_orn))
                    # this is running a zero action to step simulator
                    env.step(np.zeros(env.action_space.shape))
                    # reset the released object to zero velocity so it
                    # doesn't fly away because of residual warp speeds
                    # from teleportation!
                    p.resetBaseVelocity(
                        obj_in_hand_idx,
                        linearVelocity=[0, 0, 0],
                        angularVelocity=[0, 0, 0],
                    )
                    env.robots[0].parts["right_hand"].set_position_orientation(
                        rh_orig_grasp_position, rh_orig_grasp_orn)
                    # this is running a series of zero action to step
                    # simulator to let the object fall into its place
                    for _ in range(15):
                        env.step(np.zeros(env.action_space.shape))
                else:
                    logging.info(
                        f"PRIMITIVE: place {obj_in_hand.name} inside "
                        f"{obj_to_place_into.name} fail, need open not open")
            else:
                logging.info(f"PRIMITIVE: place {obj_in_hand.name} inside "
                             f"{obj_to_place_into.name} fail, too far")
        else:
            logging.info("PRIMITIVE: place failed with invalid obj params.")

        obj_to_place_into.force_wakeup()
        # Step the simulator to update visuals.
        env.step(np.zeros(env.action_space.shape))

    return placeInsideObjectOptionModel
