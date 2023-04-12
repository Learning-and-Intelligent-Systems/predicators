"""Ground-truth options for the narrow passage environment."""

import time
from typing import Dict, Iterator, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, \
    ParameterizedInitiable, ParameterizedOption, Predicate, State, Type


class NarrowPassageGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the narrow passage environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"narrow_passage"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        door_type = types["door"]
        target_type = types["target"]

        DoorIsOpen = predicates["DoorIsOpen"]
        TouchedGoal = predicates["TouchedGoal"]

        # MoveToTarget
        def _MoveToTarget_policy(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
            del state, objects, params  # unused
            assert memory["action_plan"], "Motion plan did not reach its goal"
            return memory["action_plan"].pop(0)

        def _MoveToTarget_terminal(state: State, memory: Dict,
                                   objects: Sequence[Object],
                                   params: Array) -> bool:
            del memory, params  # unused
            return TouchedGoal.holds(state, objects)

        MoveToTarget = ParameterizedOption(
            "MoveToTarget",
            types=[robot_type, target_type],
            params_space=Box(0, 1, (1, )),
            policy=_MoveToTarget_policy,
            initiable=cls._create_move_to_target_initiable(),
            terminal=_MoveToTarget_terminal,
        )

        # MoveAndOpenDoor
        def _MoveAndOpenDoor_policy(state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> Action:
            del state, objects, params  # unused
            assert memory["action_plan"], "Motion plan did not reach its goal"
            return memory["action_plan"].pop(0)

        def _MoveAndOpenDoor_terminal(state: State, memory: Dict,
                                      objects: Sequence[Object],
                                      params: Array) -> bool:
            del memory, params  # unused
            return DoorIsOpen.holds(state, objects[1:])

        MoveAndOpenDoor = ParameterizedOption(
            "MoveAndOpenDoor",
            types=[robot_type, door_type],
            params_space=Box(0, 1, (1, )),
            policy=_MoveAndOpenDoor_policy,
            initiable=cls._create_move_and_open_door_initiable(predicates),
            terminal=_MoveAndOpenDoor_terminal,
        )

        return {MoveToTarget, MoveAndOpenDoor}

    @classmethod
    def _create_move_to_target_initiable(cls) -> ParameterizedInitiable:

        def initiable(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            robot, target = objects
            # Set up the target input for the motion planner.
            target_x = state.get(target, "x")
            target_y = state.get(target, "y")
            success = cls._run_birrt(state, memory, params, robot,
                                     np.array([target_x, target_y]))
            return success

        return initiable

    @classmethod
    def _create_move_and_open_door_initiable(
            cls, predicates: Dict[str, Predicate]) -> ParameterizedInitiable:

        DoorIsOpen = predicates["DoorIsOpen"]

        def initiable(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            robot, door = objects
            # If door is already open, this is not initiable
            if DoorIsOpen.holds(state, [door]):
                return False
            # If robot is already within range of the door, just open the door
            if NarrowPassageEnv.robot_near_door(state):
                memory["action_plan"] = [
                    Action(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                ]
                return True
            # Select target point slightly above door
            door_center_x = state.get(door, "x")
            door_target_y = (
                NarrowPassageEnv.y_ub - NarrowPassageEnv.y_lb
            ) / 2 + NarrowPassageEnv.y_lb + \
                NarrowPassageEnv.door_sensor_radius - \
                NarrowPassageEnv.robot_radius
            success = cls._run_birrt(state, memory, params, robot,
                                     np.array([door_center_x, door_target_y]))
            if not success:
                # Failed to find motion plan, so option is not initiable
                return False
            # Append open door action to memory action plan
            memory["action_plan"].append(
                Action(np.array([0.0, 0.0, 1.0], dtype=np.float32)))
            # Opening the door takes a little bit of time to plan, artificially
            time.sleep(CFG.narrow_passage_open_door_refine_penalty)
            return True

        return initiable

    @classmethod
    def _run_birrt(cls, state: State, memory: Dict, params: Array,
                   robot: Object, target_position: Array) -> bool:
        """Runs BiRRT to motion plan from start to target positions, and store
        the position and action plans in memory if successful.

        Returns true if successful, else false
        """
        # The seed is determined by the parameter passed into the option.
        # This is a hack for bilevel planning from giving up if motion planning
        # fails on the first attempt. We make the params array non-empty so it
        # is resampled, and this sets the BiRRT rng.
        rng = np.random.default_rng(int(params[0] * 1e4))

        def _sample_fn(_: Array) -> Array:
            # Sample a point in the environment
            x = rng.uniform(NarrowPassageEnv.x_lb, NarrowPassageEnv.x_ub)
            y = rng.uniform(NarrowPassageEnv.y_lb, NarrowPassageEnv.y_ub)
            return np.array([x, y], dtype=np.float32)

        def _extend_fn(pt1: Array, pt2: Array) -> Iterator[Array]:
            # Make sure that we obey the bounds on actions.
            distance = np.linalg.norm(pt2 - pt1)
            num = int(distance / NarrowPassageEnv.action_magnitude) + 1
            for i in range(1, num + 1):
                yield pt1 * (1 - i / num) + pt2 * i / num

        def _collision_fn(pt: Array) -> bool:
            # Make a hypothetical state for the robot at this point and check
            # if there would be collisions.
            x, y = pt
            s = state.copy()
            s.set(robot, "x", x)
            s.set(robot, "y", y)
            return NarrowPassageEnv.state_has_collision(s)

        def _distance_fn(from_pt: Array, to_pt: Array) -> float:
            return np.sum(np.subtract(from_pt, to_pt)**2)

        birrt = utils.BiRRT(_sample_fn,
                            _extend_fn,
                            _collision_fn,
                            _distance_fn,
                            rng,
                            num_attempts=CFG.narrow_passage_birrt_num_attempts,
                            num_iters=CFG.narrow_passage_birrt_num_iters,
                            smooth_amt=CFG.narrow_passage_birrt_smooth_amt)
        # Run planning.
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        start_position = np.array([robot_x, robot_y])
        position_plan = birrt.query(start_position, target_position)
        # If motion planning fails, determine the option to be not initiable.
        if position_plan is None:
            return False
        # The position plan is used for the termination check, and possibly
        # can be used for debug drawing in the rendering in the future.
        memory["position_plan"] = position_plan
        # Convert the plan from position space to action space.
        deltas = np.subtract(position_plan[1:], position_plan[:-1])
        action_plan = [
            Action(np.array([dx, dy, 0.0], dtype=np.float32))
            for (dx, dy) in deltas
        ]
        memory["action_plan"] = action_plan
        return True
