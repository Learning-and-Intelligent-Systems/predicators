"""Ground-truth options for the cover environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.cover import CoverMultistepOptions
from predicators.envs.pybullet_cover import PyBulletCoverEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class CoverGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the cover environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "cover", "cover_regrasp", "cover_handempty",
            "cover_hierarchical_types"
        }

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        PickPlace = utils.SingletonParameterizedOption("PickPlace",
                                                       _policy,
                                                       params_space=Box(
                                                           0, 1, (1, )))

        return {PickPlace}


class CoverTypedOptionsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the cover_typed_options environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cover_typed_options"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        block_type = types["block"]
        target_type = types["target"]

        def _Pick_policy(s: State, m: Dict, o: Sequence[Object],
                         p: Array) -> Action:
            del m  # unused
            # The pick parameter is a RELATIVE position, so we need to
            # add the pose of the object.
            pick_pose = s.get(o[0], "pose") + p[0]
            pick_pose = min(max(pick_pose, 0.0), 1.0)
            return Action(np.array([pick_pose], dtype=np.float32))

        Pick = utils.SingletonParameterizedOption("Pick",
                                                  _Pick_policy,
                                                  types=[block_type],
                                                  params_space=Box(
                                                      -0.1, 0.1, (1, )))

        def _Place_policy(state: State, memory: Dict,
                          objects: Sequence[Object], params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        Place = utils.SingletonParameterizedOption(
            "Place",
            _Place_policy,  # use the parent class's policy
            types=[target_type],
            params_space=Box(0, 1, (1, )))

        return {Pick, Place}


class CoverMultiStepOptionsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the cover_multistep_options environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cover_multistep_options"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        block_type = types["block"]
        target_type = types["target"]
        robot_type = types["robot"]

        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]

        # Override the original options to make them multi-step. Note that
        # the parameter spaces are designed to match what would be learned
        # by the neural option learners.

        # Pick
        def _Pick_initiable(s: State, m: Dict, o: Sequence[Object],
                            p: Array) -> bool:
            # Convert the relative parameters into absolute parameters.
            m["params"] = p
            # Get the non-static object features.
            block, robot = o
            vec = [
                s.get(block, "grasp"),
                s.get(robot, "x"),
                s.get(robot, "y"),
                s.get(robot, "grip"),
                s.get(robot, "holding"),
            ]
            m["absolute_params"] = vec + p
            return HandEmpty.holds(s, [])

        def _Pick_terminal(s: State, m: Dict, o: Sequence[Object],
                           p: Array) -> bool:
            assert np.allclose(p, m["params"])
            # Pick is done when we're holding the desired object.
            return Holding.holds(s, o)

        Pick = ParameterizedOption("Pick",
                                   types=[block_type, robot_type],
                                   params_space=Box(-np.inf, np.inf, (5, )),
                                   policy=cls._create_pick_policy(),
                                   initiable=_Pick_initiable,
                                   terminal=_Pick_terminal)

        # Place
        def _Place_initiable(s: State, m: Dict, o: Sequence[Object],
                             p: Array) -> bool:
            block, robot, _ = o
            assert block.is_instance(block_type)
            assert robot.is_instance(robot_type)
            # Convert the relative parameters into absolute parameters.
            m["params"] = p
            # Only the block and robot are changing. Get the non-static features.
            vec = [
                s.get(block, "x"),
                s.get(block, "grasp"),
                s.get(robot, "x"),
                s.get(robot, "grip"),
                s.get(robot, "holding"),
            ]
            m["absolute_params"] = vec + p
            # Place is initiable if we're holding the object.
            return Holding.holds(s, [block, robot])

        def _Place_terminal(s: State, m: Dict, o: Sequence[Object],
                            p: Array) -> bool:
            del o  # unused
            assert np.allclose(p, m["params"])
            # Place is done when the hand is empty.
            return HandEmpty.holds(s, [])

        Place = ParameterizedOption(
            "Place",
            types=[block_type, robot_type, target_type],
            params_space=Box(-np.inf, np.inf, (5, )),
            policy=cls._create_place_policy(),
            initiable=_Place_initiable,
            terminal=_Place_terminal)

        return {Pick, Place}

    @classmethod
    def _create_pick_policy(cls) -> ParameterizedPolicy:

        def policy(s: State, m: Dict, o: Sequence[Object], p: Array) -> Action:
            assert np.allclose(p, m["params"])
            del p
            absolute_params = m["absolute_params"]
            # The object is the one we want to pick.
            obj, robot = o
            x = s.get(robot, "x")
            y = s.get(robot, "y")
            by = s.get(obj, "y")
            desired_x = absolute_params[1]
            desired_y = by + 1e-3
            at_desired_x = abs(desired_x - x) < 1e-5

            lb, ub = CFG.cover_multistep_action_limits
            # If we're above the object, move down and turn on the gripper.
            if at_desired_x:
                delta_y = np.clip(desired_y - y, lb, ub)
                return Action(np.array([0., delta_y, 1.0], dtype=np.float32))
            # If we're not above the object, but we're at a safe height,
            # then move left/right.
            if y >= CoverMultistepOptions.initial_robot_y:
                delta_x = np.clip(desired_x - x, lb, ub)
                return Action(np.array([delta_x, 0., 1.0], dtype=np.float32))
            # If we're not above the object, and we're not at a safe height,
            # then move up.
            delta_y = np.clip(CoverMultistepOptions.initial_robot_y + 1e-2 - y,
                              lb, ub)
            return Action(np.array([0., delta_y, 1.0], dtype=np.float32))

        return policy

    @classmethod
    def _create_place_policy(cls) -> ParameterizedPolicy:

        def policy(s: State, m: Dict, o: Sequence[Object], p: Array) -> Action:
            assert np.allclose(p, m["params"])
            del p
            absolute_params = m["absolute_params"]
            # The object is the one we want to place at.
            obj, robot, _ = o
            x = s.get(robot, "x")
            y = s.get(robot, "y")
            bh = s.get(obj, "height")
            desired_x = absolute_params[2]
            desired_y = bh + 1e-3

            at_desired_x = abs(desired_x - x) < 1e-5

            lb, ub = CFG.cover_multistep_action_limits
            # If we're already above the object, move down and turn off the magnet.
            if at_desired_x:
                delta_y = np.clip(desired_y - y, lb, ub)
                return Action(np.array([0., delta_y, -1.0], dtype=np.float32))
            # If we're not above the object, but we're at a safe height,
            # then move left/right.
            if y >= CoverMultistepOptions.initial_robot_y:
                delta_x = np.clip(desired_x - x, lb, ub)
                return Action(np.array([delta_x, 0., 1.0], dtype=np.float32))
            # If we're not above the object, and we're not at a safe height,
            # then move up.
            delta_y = np.clip(CoverMultistepOptions.initial_robot_y + 1e-2 - y,
                              lb, ub)
            return Action(np.array([0., delta_y, 1.0], dtype=np.float32))

        return policy
