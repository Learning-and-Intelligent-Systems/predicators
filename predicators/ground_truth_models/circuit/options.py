"""Ground-truth options for the coffee environment."""

import logging
from functools import lru_cache
from typing import ClassVar, Dict, Sequence, Set
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.envs.pybullet_circuit import PyBulletCircuitEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.coffee.options import \
    PyBulletCoffeeGroundTruthOptionFactory
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type
from predicators import utils


@lru_cache
def _get_pybullet_robot() -> SingleArmPyBulletRobot:
    _, pybullet_robot, _ = \
        PyBulletCoffeeEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletCircuitGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletCircuitEnv]] = PyBulletCircuitEnv
    pick_policy_tol: ClassVar[float] = 1e-3
    place_policy_tol: ClassVar[float] = 1e-4

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_circuit"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        # Types
        robot_type = types["robot"]
        wire_type = types["wire"]
        light_type = types["light"]
        battery_type = types["battery"]

        # Predicates
        Holding = predicates["Holding"]
        ConnectedToLight = predicates["ConnectedToLight"]
        ConnectedToBattery = predicates["ConnectedToBattery"]

        options = set()
        # PickWire
        def _PickWire_terminal(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            del memory, params
            robot, wire = objects
            holds = Holding.holds(state, [robot, wire])
            return holds

        PickWire = ParameterizedOption(
            "PickWire",
            types=[robot_type, wire_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_pick_wire_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_PickWire_terminal)
        
        def _RestoreForPickWire_terminal(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            del memory, params
            robot, wire = objects
            robot_pos = (state.get(robot, "x"), state.get(robot, "y"),
                            state.get(robot, "z"))
            wx = state.get(wire, "x")
            wy = state.get(wire, "y")
            target_pos = (wx, wy, cls.env_cls.robot_init_z) 
            return bool(np.allclose(robot_pos, target_pos, atol=1e-1)) 

        RestoreForPickWire = ParameterizedOption(
            "RestoreForPickWire",
            types=[robot_type, wire_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_move_to_above_position_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_RestoreForPickWire_terminal)
        PickWire = utils.LinearChainParameterizedOption(
            "PickWire", [
                # RestoreForPickWire, 
                PickWire])
        options.add(PickWire)

        # Connect
        def _Connect_terminal(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            del memory, params
            robot, wire, light, battery = objects
            # connected_to_light = ConnectedToLight.holds(state, [wire, light])
            # connected_to_battery = ConnectedToBattery.holds(state, [wire, 
            #                                                         battery])
            # return connected_to_light and connected_to_battery
            return not Holding.holds(state, [robot, wire])

        Connect = ParameterizedOption(
            "Connect",
            types=[robot_type, wire_type, light_type, battery_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_connect_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_Connect_terminal)

        def _RestoreForConnect_terminal(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            del memory, params
            robot, wire, light, battery = objects
            robot_pos = (state.get(robot, "x"), state.get(robot, "y"),
                            state.get(robot, "z"))
            robot_init_pos = (robot_pos[0], 
                              robot_pos[1],
                              cls.env_cls.robot_init_z)
            return bool(np.allclose(robot_pos, robot_init_pos, atol=1e-1)) 

        RestoreForConnect = ParameterizedOption(
            "RestRestoreForConnecoreForPickWire",
            types=[robot_type, wire_type, light_type, battery_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_move_to_above_position_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_RestoreForConnect_terminal)
        
        Connect = utils.LinearChainParameterizedOption(
            "Connect", [Connect, 
                        RestoreForConnect
                        ])
        options.add(Connect)

        return options
    @classmethod
    def _create_move_to_above_position_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            # This policy moves the robot to the initial position
            del memory, params
            robot, wire = objects[:2]
            robot_pos = (state.get(robot, "x"), 
                         state.get(robot, "y"), 
                         state.get(robot, "z"))
            wrot = state.get(wire, "rot")
            rrot = state.get(robot, "wrist")
            dwrist = wrot - rrot
            target_pos = (robot_pos[0], robot_pos[1],
                          cls.env_cls.robot_init_z)
            return PyBulletCoffeeGroundTruthOptionFactory._get_move_action(state,
                                        target_pos,
                                        robot_pos,
                                        finger_status="open",
                                        dwrist=dwrist)

        return policy 
    @classmethod
    def _create_pick_wire_policy(cls) -> ParameterizedPolicy:
        def pick_wire_policy(state: State, memory: Dict,
                             objects: Sequence[Object],
                             params: Array) -> Action:
            """Pick wire by 1) Rotate, 2) Pick up
            """
            del memory, params
            robot, wire = objects
            wx = state.get(wire, "x")
            wy = state.get(wire, "y")
            wz = state.get(wire, "z")
            wr = state.get(wire, "rot")
            wpos = (wx, wy, wz)
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            rz = state.get(robot, "z")
            rr = state.get(robot, "wrist")
            rpos = (rx, ry, rz)
            dwrist = wr - rr
            dwrist = np.clip(dwrist, -cls.env_cls.max_angular_vel,
                        cls.env_cls.max_angular_vel)

            way_point = (wpos[0], wpos[1], rz) #cls.env_cls.robot_init_z)
            sq_dist_to_way_point = np.sum((np.array(rpos) - 
                                           np.array(way_point))**2)
            if sq_dist_to_way_point > cls.pick_policy_tol:
                return PyBulletCoffeeGroundTruthOptionFactory._get_move_action(
                    state,
                    way_point,
                    rpos,
                    finger_status="open",
                    dwrist=dwrist)

            sq_dist = np.sum((np.array(wpos) - np.array(rpos))**2)
            if sq_dist < cls.pick_policy_tol:
                return PyBulletCoffeeGroundTruthOptionFactory._get_pick_action(
                    state)
            else:
                return PyBulletCoffeeGroundTruthOptionFactory._get_move_action(
                                            state,
                                            wpos,
                                            rpos,
                                            finger_status="open",
                                            dwrist=dwrist)
        return pick_wire_policy

    @classmethod
    def _create_connect_policy(cls) -> ParameterizedPolicy:
        def connect_policy(state: State, memory: Dict,
                           objects: Sequence[Object],
                           params: Array) -> Action:
            """Connect wire to light and battery by 1) Rotate, 2) Connect
            """
            del memory, params
            robot, wire, light, battery = objects
            wx = state.get(wire, "x")
            wy = state.get(wire, "y")
            wz = state.get(wire, "z")
            wr = state.get(wire, "rot")
            target_rot = 0
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            rz = state.get(robot, "z")
            rr = state.get(robot, "wrist")
            cur_pos = (rx, ry, rz)
            # cur_pos = (wx, wy, wz)
            dwrist = target_rot - rr

            lx = state.get(light, "x")
            ly = state.get(light, "y")
            lz = state.get(light, "z")
            bx = state.get(battery, "x")

            at_top = 1 if (wy > ly) else -1
            target_x = (lx + bx) / 2
            target_y = ly + at_top * (cls.env_cls.bulb_snap_length / 2 + 
                                      cls.env_cls.snap_width / 2 - 0.01)
            target_pos = (target_x, target_y, rz)
            # logging.debug(f"current pos: {cur_pos}")
            # logging.debug(f"target pos: {target_pos}")
            # logging.debug(f"current wrist: {rr}")
            # logging.debug(f"target wrist: {wr}")

            sq_dist = np.sum((np.array(cur_pos) - np.array(target_pos))**2)
            if sq_dist < cls.place_policy_tol:
                # logging.debug("Place")
                return PyBulletCoffeeGroundTruthOptionFactory.\
                        _get_place_action(state)

            return PyBulletCoffeeGroundTruthOptionFactory._get_move_action(
                state,
                target_pos,
                cur_pos,
                finger_status="closed",
                dwrist=dwrist)
        return connect_policy