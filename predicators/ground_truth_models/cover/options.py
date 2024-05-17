"""Ground-truth options for the cover environment."""

from typing import ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.cover import CoverMultistepOptions
from predicators.envs.pybullet_cover import PyBulletCoverEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
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


class BumpyCoverGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the bumpy cover environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"bumpy_cover"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        block_type = types["block"]
        target_type = types["target"]

        Pick = utils.SingletonParameterizedOption("Pick",
                                                  _policy,
                                                  types=[block_type],
                                                  params_space=Box(
                                                      0, 1, (1, )))

        Place = utils.SingletonParameterizedOption(
            "Place",
            _policy,
            types=[block_type, target_type],
            params_space=Box(0, 1, (1, )))

        return {Pick, Place}


class RegionalBumpyCoverGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the regional bumpy cover environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"regional_bumpy_cover"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        block_type = types["block"]
        target_type = types["target"]

        options: Set[ParameterizedOption] = set()

        PickFromSmooth = utils.SingletonParameterizedOption("PickFromSmooth",
                                                            _policy,
                                                            types=[block_type],
                                                            params_space=Box(
                                                                0, 1, (1, )))
        options.add(PickFromSmooth)

        PickFromBumpy = utils.SingletonParameterizedOption("PickFromBumpy",
                                                           _policy,
                                                           types=[block_type],
                                                           params_space=Box(
                                                               0, 1, (1, )))
        options.add(PickFromBumpy)

        PickFromTarget = utils.SingletonParameterizedOption(
            "PickFromTarget",
            _policy,
            types=[block_type, target_type],
            params_space=Box(0, 1, (1, )))
        options.add(PickFromTarget)

        PlaceOnTarget = utils.SingletonParameterizedOption(
            "PlaceOnTarget",
            _policy,
            types=[block_type, target_type],
            params_space=Box(0, 1, (1, )))
        options.add(PlaceOnTarget)

        PlaceOnBumpy = utils.SingletonParameterizedOption("PlaceOnBumpy",
                                                          _policy,
                                                          types=[block_type],
                                                          params_space=Box(
                                                              0, 1, (1, )))
        options.add(PlaceOnBumpy)

        if CFG.regional_bumpy_cover_include_impossible_nsrt:

            def _impossible_policy(state: State, memory: Dict,
                                   objects: Sequence[Object],
                                   params: Array) -> Action:
                del memory, objects, params  # unused
                # Find a place to click that is effectively a no-op.
                obj_regions: List[Tuple[float, float]] = []
                objs = state.get_objects(block_type) + \
                    state.get_objects(target_type)
                for obj in objs:
                    pose = state.get(obj, "pose")
                    width = state.get(obj, "width")
                    obj_regions.append((pose - width, pose + width))
                for x in np.linspace(0, 1, 100):
                    if not any(lb <= x <= ub for lb, ub in obj_regions):
                        return Action(np.array([x], dtype=np.float32))
                raise utils.OptionExecutionFailure(
                    "No noop possible.")  # pragma: no cover

            ImpossiblePickPlace = utils.SingletonParameterizedOption(
                "ImpossiblePickPlace",
                _impossible_policy,
                types=[block_type, target_type])
            options.add(ImpossiblePickPlace)

        return options


class CoverTypedOptionsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the cover_typed_options environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cover_typed_options", "cover_place_hard"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        block_type = types["block"]
        target_type = types["target"]

        Holding = predicates["Holding"]
        HandEmpty = predicates["HandEmpty"]

        def _Pick_policy(s: State, m: Dict, o: Sequence[Object],
                         p: Array) -> Action:
            del m  # unused
            # The pick parameter is a RELATIVE position, so we need to
            # add the pose of the object.
            if CFG.env == "cover_typed_options":
                pick_pose = s.get(o[0], "pose") + p[0]
                pick_pose = min(max(pick_pose, 0.0), 1.0)
                return Action(np.array([pick_pose], dtype=np.float32))
            return Action(p)

        lb, ub = (0.0, 1.0)
        if CFG.env == "cover_typed_options":
            lb, ub = (-0.1, 0.1)

        def _Pick_terminal(s: State, m: Dict, o: Sequence[Object],
                           p: Array) -> bool:
            # Pick is done when we're holding the desired object.
            return Holding.holds(s, o)

        Pick = utils.SingletonParameterizedOption("Pick",
                                                _Pick_policy,
                                                types=[block_type],
                                                params_space=Box(
                                                      lb, ub, (1, )),
                                                # terminal=_Pick_terminal,
                                                )

        def _Place_policy(state: State, memory: Dict,
                          objects: Sequence[Object], params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        def _Place_terminal(s: State, m: Dict, o: Sequence[Object],
                            p: Array) -> bool:
            # del o  # unused
            # Place is done when the hand is empty.
            return HandEmpty.holds(s, [])

        place_types = [block_type, target_type]
        if CFG.env == "cover_typed_options":
            place_types = [target_type]
        Place = utils.SingletonParameterizedOption(
            "Place",
            _Place_policy,  # use the parent class's policy
            types=place_types,
            params_space=Box(0, 1, (1, )),
            # terminal=_Place_terminal,
            )

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
            # Only the block and robot are changing. Get the changing features.
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
            # If we're already above the object, move down and turn off
            # the magnet.
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


class PyBulletCoverGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the pybullet_cover environment."""

    # Robot parameters.
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_cover"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        _, pybullet_robot, _ = \
            PyBulletCoverEnv.initialize_pybullet(using_gui=False)

        # Note: this isn't exactly correct because the first argument should be
        # the current finger joint value, which we don't have in the State `s`.
        # This could lead to slippage or bad grasps, but we haven't seen this
        # in practice, so we'll leave it as is instead of changing the State.
        HandEmpty = predicates["HandEmpty"]
        toggle_fingers_func = lambda s, _1, _2: (
            (pybullet_robot.open_fingers, pybullet_robot.closed_fingers)
            if HandEmpty.holds(s, []) else
            (pybullet_robot.closed_fingers, pybullet_robot.open_fingers))

        PickPlace = utils.LinearChainParameterizedOption(
            "PickPlace",
            [
                # Move to far above the location we will pick/place at.
                cls._create_cover_move_option(
                    name="MoveEndEffectorToPrePose",
                    pybullet_robot=pybullet_robot,
                    target_z=PyBulletCoverEnv.workspace_z,
                    predicates=predicates,
                    types=types),
                # Move down to pick/place.
                cls._create_cover_move_option(
                    name="MoveEndEffectorToPose",
                    pybullet_robot=pybullet_robot,
                    target_z=PyBulletCoverEnv.pickplace_z,
                    predicates=predicates,
                    types=types),
                # Toggle fingers.
                create_change_fingers_option(
                    pybullet_robot, "ToggleFingers", [], Box(
                        0, 1, (1, )), toggle_fingers_func,
                    CFG.pybullet_max_vel_norm, PyBulletCoverEnv.grasp_tol),
                # Move back up.
                cls._create_cover_move_option(
                    name="MoveEndEffectorBackUp",
                    pybullet_robot=pybullet_robot,
                    target_z=PyBulletCoverEnv.workspace_z,
                    predicates=predicates,
                    types=types)
            ])

        return {PickPlace}

    @classmethod
    def _create_cover_move_option(
            cls, name: str, pybullet_robot: SingleArmPyBulletRobot,
            target_z: float, predicates: Dict[str, Predicate],
            types: Dict[str, Type]) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose in Cover."""
        option_types: Sequence[Type] = []
        params_space = Box(0, 1, (1, ))
        HandEmpty = predicates["HandEmpty"]
        robot_type = types["robot"]
        home_orn = PyBulletCoverEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not objects
            robot, = state.get_objects(robot_type)
            hand = state.get(robot, "hand")
            # De-normalize hand feature to actual table coordinates.
            current_y = PyBulletCoverEnv.y_lb + (PyBulletCoverEnv.y_ub -
                                                 PyBulletCoverEnv.y_lb) * hand
            current_position = (state.get(robot, "pose_x"), current_y,
                                state.get(robot, "pose_z"))
            current_pose = Pose(current_position, home_orn)
            y_norm, = params
            # De-normalize parameter to actual table coordinates.
            target_y = PyBulletCoverEnv.y_lb + (PyBulletCoverEnv.y_ub -
                                                PyBulletCoverEnv.y_lb) * y_norm
            target_position = (PyBulletCoverEnv.workspace_x, target_y,
                               target_z)
            target_pose = Pose(target_position, home_orn)
            if HandEmpty.holds(state, []):
                finger_status = "open"
            else:
                finger_status = "closed"
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot, name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)
