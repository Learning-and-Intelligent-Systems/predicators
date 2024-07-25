"""Ground-truth options for the doors environment."""

from typing import Dict, Iterator, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.doors import DoorKnobsEnv, DoorsEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, \
    ParameterizedInitiable, ParameterizedOption, ParameterizedPolicy, \
    Predicate, State, Type
from predicators.utils import Rectangle, SingletonParameterizedOption, \
    StateWithCache


class DoorsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the doors environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"doors"}

    @staticmethod
    def _MoveToDoor_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
        del state, objects, params  # unused
        assert memory["action_plan"], "Motion plan did not reach its goal"
        return memory["action_plan"].pop(0)

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        door_type = types["door"]

        InDoorway = predicates["InDoorway"]
        TouchingDoor = predicates["TouchingDoor"]
        DoorIsOpen = predicates["DoorIsOpen"]
        InRoom = predicates["InRoom"]

        # MoveToDoor

        def _MoveToDoor_terminal(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> bool:
            del memory, params  # unused
            # Terminate as soon as we are in the doorway.
            robot, door = objects
            return InDoorway.holds(state, [robot, door])

        MoveToDoor = ParameterizedOption(
            "MoveToDoor",
            types=[robot_type, door_type],
            # No parameters; the option always moves to the doorway center.
            params_space=Box(0, 1, (0, )),
            # The policy is a motion planner.
            policy=cls._MoveToDoor_policy,
            # Only initiable when the robot is in a room for the doory.
            initiable=cls._create_move_to_door_initiable(predicates),
            terminal=_MoveToDoor_terminal)

        # OpenDoor
        def _OpenDoor_policy(state: State, memory: Dict,
                             objects: Sequence[Object],
                             params: Array) -> Action:
            del memory  # unused
            door, _ = objects
            delta_rot, _ = params
            current_rot = state.get(door, "rot")
            target = current_rot + delta_rot
            return Action(np.array([0.0, 0.0, target], dtype=np.float32))

        def _OpenDoor_initiable(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> bool:
            del memory, params  # unused
            # Can only open the door if touching it.
            door, robot = objects
            return TouchingDoor.holds(state, [robot, door])

        def _OpenDoor_terminal(state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> bool:
            del memory, params  # unused
            # Terminate when the door is open.
            door, _ = objects
            return DoorIsOpen.holds(state, [door])

        OpenDoor = ParameterizedOption(
            "OpenDoor",
            types=[door_type, robot_type],
            # Even though this option does not need to be parameterized, we
            # make it so, because we want to match the parameter space of the
            # option that will get learned during option learning. This is
            # useful for when we want to use sampler_learner = "oracle" too.
            params_space=Box(-np.inf, np.inf, (2, )),
            policy=_OpenDoor_policy,
            # Only initiable when the robot is in the doorway.
            initiable=_OpenDoor_initiable,
            terminal=_OpenDoor_terminal)

        # MoveThroughDoor
        def _MoveThroughDoor_terminal(state: State, memory: Dict,
                                      objects: Sequence[Object],
                                      params: Array) -> bool:
            del params  # unused
            robot, door = objects
            target_room = memory["target_room"]
            # Sanity check: we should never leave the doorway.
            assert InDoorway.holds(state, [robot, door])
            # Terminate as soon as we enter the other room.
            return InRoom.holds(state, [robot, target_room])

        MoveThroughDoor = ParameterizedOption(
            "MoveThroughDoor",
            types=[robot_type, door_type],
            # No parameters; the option always moves straight through.
            params_space=Box(0, 1, (0, )),
            # The policy just moves in a straight line. No motion planning
            # required, because there are no obstacles in the doorway.
            policy=cls._create_move_through_door_policy(action_space),
            # Only initiable when the robot is in the doorway.
            initiable=cls._create_move_through_door_initiable(predicates),
            terminal=_MoveThroughDoor_terminal)

        return {MoveToDoor, OpenDoor, MoveThroughDoor}

    @classmethod
    def _create_move_to_door_initiable(
            cls, predicates: Dict[str, Predicate]) -> ParameterizedInitiable:

        InRoom = predicates["InRoom"]

        def initiable(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del params  # unused
            robot, door = objects
            # The robot must be in one of the rooms for the door.
            for r in DoorsEnv.door_to_rooms(door, state):
                if InRoom.holds(state, [robot, r]):
                    room = r
                    break
            else:
                return False
            # Make a plan and store it in memory for use in the policy. Note
            # that policies are assumed to be deterministic, but RRT is
            # stochastic. We enforce determinism by using a constant seed.
            rng = np.random.default_rng(CFG.seed)

            room_rect = DoorsEnv.object_to_geom(room, state)

            def _sample_fn(_: Array) -> Array:
                # Sample a point in the room that is far enough away from the
                # wall (to save on collision checking).
                assert isinstance(room_rect, Rectangle)
                x_lb = room_rect.x + DoorsEnv.robot_radius
                x_ub = room_rect.x + DoorsEnv.room_size - DoorsEnv.robot_radius
                y_lb = room_rect.y + DoorsEnv.robot_radius
                y_ub = room_rect.y + DoorsEnv.room_size - DoorsEnv.robot_radius
                x = rng.uniform(x_lb, x_ub)
                y = rng.uniform(y_lb, y_ub)
                return np.array([x, y], dtype=np.float32)

            def _extend_fn(pt1: Array, pt2: Array) -> Iterator[Array]:
                # Make sure that we obey the bounds on actions.
                distance = np.linalg.norm(pt2 - pt1)
                num = int(distance / DoorsEnv.action_magnitude) + 1
                for i in range(1, num + 1):
                    yield pt1 * (1 - i / num) + pt2 * i / num

            def _collision_fn(pt: Array) -> bool:
                # Make a hypothetical state for the robot at this point and
                # check if there would be collisions.
                x, y = pt
                s = state.copy()
                s.set(robot, "x", x)
                s.set(robot, "y", y)
                return DoorsEnv.state_has_collision(s)

            def _distance_fn(from_pt: Array, to_pt: Array) -> float:
                return np.sum(np.subtract(from_pt, to_pt)**2)

            birrt = utils.BiRRT(_sample_fn,
                                _extend_fn,
                                _collision_fn,
                                _distance_fn,
                                rng,
                                num_attempts=CFG.doors_birrt_num_attempts,
                                num_iters=CFG.doors_birrt_num_iters,
                                smooth_amt=CFG.doors_birrt_smooth_amt)

            # Set up the initial and target inputs for the motion planner.
            robot_x = state.get(robot, "x")
            robot_y = state.get(robot, "y")
            target_x, target_y = cls._get_position_in_doorway(
                room, door, state)
            initial_state = np.array([robot_x, robot_y])
            target_state = np.array([target_x, target_y])
            # Run planning.
            position_plan = birrt.query(initial_state, target_state)
            # In very rare cases, motion planning fails (it is stochastic after
            # all). In this case, determine the option to be not initiable.
            if position_plan is None:  # pragma: no cover
                return False
            # The position plan is used for the termination check, and for debug
            # drawing in the rendering.
            memory["position_plan"] = position_plan
            # Convert the plan from position space to action space.
            deltas = np.subtract(position_plan[1:], position_plan[:-1])
            action_plan = [
                Action(np.array([dx, dy, 0.0], dtype=np.float32))
                for (dx, dy) in deltas
            ]
            memory["action_plan"] = action_plan
            return True

        return initiable

    @classmethod
    def _create_move_through_door_policy(
            cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del params  # unused
            robot, _ = objects
            desired_x, desired_y = memory["target"]
            robot_x = state.get(robot, "x")
            robot_y = state.get(robot, "y")
            delta = np.subtract([desired_x, desired_y], [robot_x, robot_y])
            delta_norm = np.linalg.norm(delta)
            if delta_norm > DoorsEnv.action_magnitude:
                delta = DoorsEnv.action_magnitude * delta / delta_norm
            dx, dy = delta
            action = Action(np.array([dx, dy, 0.0], dtype=np.float32))
            assert action_space.contains(action.arr)
            return action

        return policy

    @classmethod
    def _create_move_through_door_initiable(
            cls, predicates: Dict[str, Predicate]) -> ParameterizedInitiable:

        InDoorway = predicates["InDoorway"]
        DoorIsOpen = predicates["DoorIsOpen"]
        InRoom = predicates["InRoom"]

        def initiable(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del params  # unused
            robot, door = objects
            # The robot must be in the doorway.
            if not InDoorway.holds(state, [robot, door]):
                return False
            # The door must be open.
            if not DoorIsOpen.holds(state, [door]):
                return False
            # The option is initiable. Memorize the target -- otherwise, we
            # wouldn't know which side of the door to move toward.
            room1, room2 = DoorsEnv.door_to_rooms(door, state)
            if InRoom.holds(state, [robot, room1]):
                end_room = room2
            else:
                assert InRoom.holds(state, [robot, room2])
                end_room = room1
            memory["target"] = cls._get_position_in_doorway(
                end_room, door, state)
            memory["target_room"] = end_room
            return True

        return initiable

    @classmethod
    def _get_position_in_doorway(cls, room: Object, door: Object,
                                 state: State) -> Tuple[float, float]:
        assert isinstance(state, StateWithCache)
        position_cache = state.cache["position_in_doorway"]
        if (room, door) not in position_cache:
            # Find the two vertices of the doorway that are in the room.
            doorway_geom = DoorsEnv.door_to_doorway_geom(door, state)
            room_geom = DoorsEnv.object_to_geom(room, state)
            vertices_in_room = []
            for (x, y) in doorway_geom.vertices:
                if room_geom.contains_point(x, y):
                    vertices_in_room.append((x, y))
            assert len(vertices_in_room) == 2
            (x0, y0), (x1, y1) = vertices_in_room
            tx = (x0 + x1) / 2
            ty = (y0 + y1) / 2
            position_cache[(room, door)] = (tx, ty)
        return position_cache[(room, door)]


class DoorknobsGroundTruthOptionFactory(DoorsGroundTruthOptionFactory):
    """Ground-truth options for the doorknobs environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"doorknobs"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        door_type = types["door"]

        InDoorway = predicates["InDoorway"]
        InRoom = predicates["InRoom"]

        # MoveToDoor

        def _MoveToDoor_terminal(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> bool:
            del memory, params  # unused
            # Terminate as soon as we are in the doorway.
            robot, door = objects
            return InDoorway.holds(state, [robot, door])

        MoveToDoor = ParameterizedOption(
            "MoveToDoor",
            types=[robot_type, door_type],
            # No parameters; the option always moves to the doorway center.
            params_space=Box(0, 1, (0, )),
            # The policy is a motion planner.
            policy=cls._MoveToDoor_policy,
            # Only initiable when the robot is in a room for the doory.
            initiable=cls._create_move_to_door_initiable(predicates),
            terminal=_MoveToDoor_terminal)

        # OpenDoor
        def _OpenDoor_policy(_: State, memory: Dict, __: Sequence[Object],
                             params: Array) -> Action:
            del memory  # unused
            delta_rot = params
            return Action(np.array([0.0, 0.0, delta_rot], dtype=np.float32))

        OpenDoor = SingletonParameterizedOption(
            "OpenDoor",
            types=[robot_type],
            # Even though this option does not need to be parameterized, we
            # make it so, because we want to match the parameter space of the
            # option that will get learned during option learning. This is
            # useful for when we want to use sampler_learner = "oracle" too.
            params_space=Box(-np.inf, np.inf, (1, )),
            policy=_OpenDoor_policy,
        )

        # MoveThroughDoor
        def _MoveThroughDoor_terminal(state: State, memory: Dict,
                                      objects: Sequence[Object],
                                      params: Array) -> bool:
            del params  # unused
            robot, door = objects
            target_room = memory["target_room"]
            # Sanity check: we should never leave the doorway.
            assert InDoorway.holds(state, [robot, door])
            # Terminate as soon as we enter the other room.
            return InRoom.holds(state, [
                robot, target_room
            ]) or memory["starting_state"].pretty_str() == state.pretty_str()

        MoveThroughDoor = ParameterizedOption(
            "MoveThroughDoor",
            types=[robot_type, door_type],
            # No parameters; the option always moves straight through.
            params_space=Box(0, 1, (0, )),
            # The policy just moves in a straight line. No motion planning
            # required, because there are no obstacles in the doorway.
            policy=cls._create_move_through_door_policy(action_space),
            # Only initiable when the robot is in the doorway.
            initiable=cls._create_move_through_door_initiable(predicates),
            terminal=_MoveThroughDoor_terminal)

        return {MoveToDoor, OpenDoor, MoveThroughDoor}

    @classmethod
    def _create_move_to_door_initiable(
            cls, predicates: Dict[str, Predicate]) -> ParameterizedInitiable:

        InRoom = predicates["InRoom"]

        def initiable(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del params  # unused
            robot, door = objects
            # The robot must be in one of the rooms for the door.
            for ro in DoorsEnv.door_to_rooms(door, state):
                if InRoom.holds(state, [robot, ro]):
                    room = ro
                    break
            else:
                return False
            # Make a plan and store it in memory for use in the policy. Note
            # that policies are assumed to be deterministic, but RRT is
            # stochastic. We enforce determinism by using a constant seed.
            rng = np.random.default_rng(CFG.seed)

            room_rect = DoorsEnv.object_to_geom(room, state)

            def _sample_fn(_: Array) -> Array:
                # Sample a point in the room that is far enough away from the
                # wall (to save on collision checking).
                assert isinstance(room_rect, Rectangle)  # pragma: no cover
                x_lb = room_rect.x + DoorsEnv.robot_radius  # pragma: no cover
                x_ub = room_rect.x + DoorsEnv.room_size - DoorsEnv.robot_radius  # pragma: no cover
                y_lb = room_rect.y + DoorsEnv.robot_radius  # pragma: no cover
                y_ub = room_rect.y + DoorsEnv.room_size - DoorsEnv.robot_radius  # pragma: no cover
                x = rng.uniform(x_lb, x_ub)  # pragma: no cover
                y = rng.uniform(y_lb, y_ub)  # pragma: no cover
                return np.array([x, y], dtype=np.float32)  # pragma: no cover

            def _extend_fn(pt1: Array, pt2: Array) -> Iterator[Array]:
                # Make sure that we obey the bounds on actions.
                distance = np.linalg.norm(pt2 - pt1)
                num = int(distance / DoorsEnv.action_magnitude) + 1
                for i in range(1, num + 1):
                    yield pt1 * (1 - i / num) + pt2 * i / num

            def _collision_fn(pt: Array) -> bool:
                return DoorKnobsEnv.state_has_collision(state, pt)

            def _distance_fn(from_pt: Array, to_pt: Array) -> float:
                return np.sum(np.subtract(from_pt, to_pt)**2)  # pragma: no cover

            birrt = utils.BiRRT(_sample_fn,
                                _extend_fn,
                                _collision_fn,
                                _distance_fn,
                                rng,
                                num_attempts=CFG.doors_birrt_num_attempts,
                                num_iters=CFG.doors_birrt_num_iters,
                                smooth_amt=CFG.doors_birrt_smooth_amt)

            # Set up the initial and target inputs for the motion planner.
            robot_x = state.get(robot, "x")
            robot_y = state.get(robot, "y")
            target_x, target_y = cls._get_position_in_doorway(
                room, door, state)
            initial_state = np.array([robot_x, robot_y])
            target_state = np.array([target_x, target_y])
            # Run planning.
            position_plan = birrt.query(initial_state, target_state)
            # In very rare cases, motion planning fails (it is stochastic after
            # all). In this case, determine the option to be not initiable.
            if position_plan is None:  # pragma: no cover
                return False
            # The position plan is used for the termination check, and for debug
            # drawing in the rendering.
            memory["position_plan"] = position_plan
            # Convert the plan from position space to action space.
            deltas = np.subtract(position_plan[1:], position_plan[:-1])
            action_plan = [
                Action(np.array([dx, dy, 0.0], dtype=np.float32))
                for (dx, dy) in deltas
            ]
            memory["action_plan"] = action_plan
            return True

        return initiable

    @classmethod
    def _create_move_through_door_initiable(
            cls, predicates: Dict[str, Predicate]) -> ParameterizedInitiable:

        InDoorway = predicates["InDoorway"]
        InRoom = predicates["InRoom"]

        def initiable(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del params  # unused
            robot, door = objects
            # The robot must be in the doorway.
            if not InDoorway.holds(state, [robot, door]):
                return False
            # The door must be open.
            # The option is initiable. Memorize the target -- otherwise, we
            # wouldn't know which side of the door to move toward.
            room1, room2 = DoorsEnv.door_to_rooms(door, state)
            if InRoom.holds(state, [robot, room1]):
                end_room = room2
            else:
                assert InRoom.holds(state, [robot, room2])
                end_room = room1
            memory["target"] = cls._get_position_in_doorway(
                end_room, door, state)
            memory["target_room"] = end_room
            memory["starting_state"] = state
            return True

        return initiable
