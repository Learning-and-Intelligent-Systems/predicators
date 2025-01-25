"""Ground-truth options for the coffee environment."""

from functools import lru_cache
from typing import ClassVar, Dict, Sequence, Set
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.envs.pybullet_grow import PyBulletGrowEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.coffee.options import \
    PyBulletCoffeeGroundTruthOptionFactory
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


@lru_cache
def _get_pybullet_robot() -> SingleArmPyBulletRobot:
    _, pybullet_robot, _ = \
        PyBulletCoffeeEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletGrowGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletGrowEnv]] = PyBulletGrowEnv
    pick_policy_tol: ClassVar[float] = 1e-3
    pour_policy_tol: ClassVar[float] = 1e-3
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_grow"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        cup_type = types["cup"]
        # Predicates
        Holding = predicates["Holding"]
        Grown = predicates["Grown"]

        # PickJug
        def _PickJug_terminal(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            del memory, params  # unused
            robot, jug = objects
            holds = Holding.holds(state, [robot, jug])
            return holds

        PickJug = ParameterizedOption(
            name="PickJug",
            types=[robot_type, jug_type],
            params_space=Box(0, 1, (0, )),
            policy=PyBulletCoffeeGroundTruthOptionFactory.  # pylint: disable=protected-access
            _create_pick_jug_policy(),
            # policy=cls._create_pick_jug_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_PickJug_terminal)

        # Pour
        def _Pour_terminal(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
            del memory, params
            _, _, cup = objects
            return Grown.holds(state, [cup])

        Pour = ParameterizedOption(
            "Pour",
            [robot_type, jug_type, cup_type],
            params_space=Box(0, 1, (0, )),
            policy=PyBulletCoffeeGroundTruthOptionFactory.  # pylint: disable=protected-access
            _create_pour_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Pour_terminal)

        # Place
        def _Place_terminal(state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
            del memory, params
            robot, jug = objects
            return not Holding.holds(state, [robot, jug])

        Place = ParameterizedOption("Place", [robot_type, jug_type],
                                    params_space=Box(0, 1, (2, )),
                                    policy=cls._crete_place_policy(),
                                    initiable=lambda s, m, o, p: True,
                                    terminal=_Place_terminal)

        return {PickJug, Pour, Place}

    @classmethod
    def _crete_place_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory
            robot, jug = objects

            # Get the current robot position.
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            tilt = state.get(robot, "tilt")
            wrist = state.get(robot, "wrist")
            robot_pos = (x, y, z)

            # Get the difference between the jug location and the target.
            # Use the jug position as the origin.
            jx = state.get(jug, "x")
            jy = state.get(jug, "y")
            jz = state.get(jug, "z")
            # jz = cls.env_cls.z_lb + cls.env_cls.jug_height()
            current_jug_pos = (jx, jy, jz)
            x_norm, y_norm = params
            target_jug_pos = (cls.env_cls.x_lb +
                              (cls.env_cls.x_ub - cls.env_cls.x_lb) * x_norm,
                              cls.env_cls.y_lb +
                              (cls.env_cls.y_ub - cls.env_cls.y_lb) * y_norm,
                              cls.env_cls.z_lb + cls.env_cls.jug_height / 2)

            dtilt = cls.env_cls.robot_init_tilt - tilt
            dwrist = cls.env_cls.robot_init_wrist - wrist
            dx, dy, dz = np.subtract(target_jug_pos, current_jug_pos)

            # Get the target robot position.
            target_robot_pos = (x + dx, y + dy, z + dz)
            # If close enough, place.
            sq_dist_to_place = np.sum(
                np.subtract(robot_pos, target_robot_pos)**2)
            if sq_dist_to_place < cls.env_cls.place_jug_tol:
                return PyBulletCoffeeGroundTruthOptionFactory._get_place_action(  # pylint: disable=protected-access
                    state)

            # only move down if it has arrived at target x, y
            if abs(dx) < 0.01 and abs(dy) < 0.01:
                return PyBulletCoffeeGroundTruthOptionFactory._get_move_action(  # pylint: disable=protected-access
                    state,
                    target_robot_pos,
                    robot_pos,
                    finger_status="closed",
                    dtilt=dtilt,
                    dwrist=dwrist,
                )

            target_robot_pos = (x + dx, y + dy, z)
            return PyBulletCoffeeGroundTruthOptionFactory._get_move_action(  # pylint: disable=protected-access
                state,
                target_robot_pos,
                robot_pos,
                finger_status="closed",
                dtilt=dtilt,
                dwrist=dwrist,
            )

        return policy
