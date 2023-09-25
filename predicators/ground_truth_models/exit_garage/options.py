"""Ground-truth options for the exit garage environment."""

import time
from typing import Callable, Dict, Iterator, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.exit_garage import ExitGarageEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type


class ExitGarageGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the exit garage environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"exit_garage"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        car_type = types["car"]
        robot_type = types["robot"]
        obstacle_type = types["obstacle"]
        storage_type = types["storage"]

        CarHasExited = predicates["CarHasExited"]
        ObstacleCleared = predicates["ObstacleCleared"]
        ObstacleNotCleared = predicates["ObstacleNotCleared"]

        def _motion_plan_policy(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
            del state, objects, params  # unused
            assert memory["action_plan"], "Motion plan did not reach its goal"
            return memory["action_plan"].pop(0)

        # DriveCarToExit
        def _DriveCarToExit_terminal(state: State, memory: Dict,
                                     objects: Sequence[Object],
                                     params: Array) -> bool:
            del memory, params  # unused
            return CarHasExited.holds(state, objects)

        def _DriveCarToExit_initiable(state: State, memory: Dict,
                                      objects: Sequence[Object],
                                      params: Array) -> bool:
            car, = objects

            # Set up the RRT goal function
            def _goal_fn(pt: Array) -> bool:
                # Create a hypothetical state for the car at this point and
                # check whether CarHasExited is True
                x, y, theta = pt
                s = state.copy()
                s.set(car, "x", x)
                s.set(car, "y", y)
                s.set(car, "theta", theta)
                return CarHasExited.holds(state, objects)

            # Set up the target input for the motion planner.
            target_x = 0.95
            target_y = 0.4 - ExitGarageEnv.exit_width / 2
            target_theta = 0
            if CFG.exit_garage_motion_planning_ignore_obstacles:
                start_pos_list = [
                    state.get(car, "x"),
                    state.get(car, "y"),
                ]
                start_position = np.array(start_pos_list)
                memory["action_plan"] = []
                memory["position_plan"] = []
                cls._plan_direct(memory, params, start_position,
                                 np.array([target_x, target_y]), 0, 1)
                return True
            success = cls._run_rrt(state,
                                   memory,
                                   params,
                                   car,
                                   np.array([target_x, target_y,
                                             target_theta]),
                                   goal_fn=_goal_fn)
            return success

        DriveCarToExit = ParameterizedOption(
            "DriveCarToExit",
            types=[car_type],
            params_space=Box(0, 1, (1, )),
            policy=_motion_plan_policy,
            initiable=_DriveCarToExit_initiable,
            terminal=_DriveCarToExit_terminal,
        )

        # ClearObstacle
        def _ClearObstacle_terminal(state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> bool:
            del memory, params  # unused
            _, obstacle = objects
            return ObstacleCleared.holds(state, [obstacle])

        def _ClearObstacle_initiable(state: State, memory: Dict,
                                     objects: Sequence[Object],
                                     params: Array) -> bool:
            robot, obstacle = objects
            if not ObstacleNotCleared.holds(state, [obstacle]):
                return False  # obstacle already cleared

            memory["action_plan"] = []
            memory["position_plan"] = []
            start_pos_list = [
                state.get(robot, "x"),
                state.get(robot, "y"),
            ]
            start_position = np.array(start_pos_list)

            # Straight-line plan to pickup obstacle
            pickup_target_x = state.get(obstacle, "x")
            pickup_target_y = state.get(obstacle, "y")
            pickup_position = np.array([pickup_target_x, pickup_target_y])
            cls._plan_direct(memory, params, start_position, pickup_position,
                             2, 3)
            # Append pickup action to memory plans
            memory["action_plan"].append(
                Action(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)))

            # Straight-line plan to place obstacle
            storage, = state.get_objects(storage_type)
            num_stored = state.get(storage, "num_stored")
            # Set up the target input for the motion planner.
            target_x = (0.01 + ExitGarageEnv.obstacle_radius * 2) * num_stored
            target_x += ExitGarageEnv.obstacle_radius
            target_y = (ExitGarageEnv.y_ub -
                        ExitGarageEnv.storage_area_height / 2)
            cls._plan_direct(memory, params, pickup_position,
                             np.array([target_x, target_y]), 2, 3)
            # Append place action to memory action plan
            memory["action_plan"].append(
                Action(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)))

            # Moving an obstacle takes a bit of time to plan, artificially
            time.sleep(CFG.exit_garage_clear_refine_penalty)

            return True

        ClearObstacle = ParameterizedOption(
            "ClearObstacle",
            types=[robot_type, obstacle_type],
            params_space=Box(0, 1, (1, )),
            policy=_motion_plan_policy,
            initiable=_ClearObstacle_initiable,
            terminal=_ClearObstacle_terminal,
        )

        return {DriveCarToExit, ClearObstacle}

    @classmethod
    def _run_rrt(cls, state: State, memory: Dict, params: Array,
                 move_obj: Object, target_position: Array,
                 goal_fn: Callable[[Array], bool]) -> bool:
        """Runs RRT to motion plan from start car configuration to the target,
        or when the car reaches the exit. Store the position and action plans
        in memory if successful.

        Returns True if successful, else False.
        """
        rng = np.random.default_rng(int(params[0] * 1e4))

        def _sample_fn(_: Array) -> Array:
            # Sample a point in the environment
            sample = [
                rng.uniform(ExitGarageEnv.x_lb, ExitGarageEnv.x_ub),  # x
                rng.uniform(
                    ExitGarageEnv.y_lb, ExitGarageEnv.y_ub -
                    ExitGarageEnv.storage_area_height),  # y
                rng.uniform(-np.pi, np.pi),  # theta
            ]
            return np.array(sample, dtype=np.float32)

        def _distance_fn(from_pt: Array, to_pt: Array) -> float:
            distance = np.sum(np.subtract(from_pt[:2], to_pt[:2])**2)
            angle_dist = (from_pt[2] - to_pt[2] + np.pi) % (2 * np.pi) - np.pi
            # We need to scale the weight of the angle for the distance down
            # because it should matter but not as much as the position diff
            scaled_angle_dist = angle_dist / (2 * np.pi)
            distance += scaled_angle_dist**2
            return distance

        def _sample_control_result(pt: Array) -> Array:
            x, y, theta = pt
            vel = rng.uniform(-ExitGarageEnv.car_max_absolute_vel,
                              ExitGarageEnv.car_max_absolute_vel)
            omega = rng.uniform(-ExitGarageEnv.car_steering_omega_limit,
                                ExitGarageEnv.car_steering_omega_limit)
            new_x = x + np.cos(theta) * vel
            new_y = y + np.sin(theta) * vel
            new_theta = (theta + omega + np.pi) % (2 * np.pi) - np.pi
            return np.array([new_x, new_y, new_theta], dtype=np.float32)

        def _extend_fn(pt1: Array, pt2: Array) -> Iterator[Array]:
            # Make sure that we obey the bounds on actions.
            # For non-holonomic, we need to sample motion controls
            # and choose the one that produces the closest result
            current_pt = pt1
            distance_thres = CFG.exit_garage_rrt_extend_fn_threshold
            while _distance_fn(current_pt, pt2) > distance_thres:
                current_pt = min(
                    (_sample_control_result(current_pt)
                     for _ in range(CFG.exit_garage_rrt_num_control_samples)),
                    key=lambda pt: _distance_fn(pt, pt2))
                yield current_pt

        def _collision_fn(pt: Array) -> bool:
            # Check for collision of car in non-holonomic case
            x, y, theta = pt
            # Make a hypothetical state for the car at this point and check
            # if there would be collisions.
            s = state.copy()
            s.set(move_obj, "x", x)
            s.set(move_obj, "y", y)
            s.set(move_obj, "theta", theta)
            collision = ExitGarageEnv.get_car_collision_object(s) is not None
            return collision or ExitGarageEnv.coords_out_of_bounds(x, y)

        rrt = utils.RRT(
            _sample_fn,
            _extend_fn,
            _collision_fn,
            _distance_fn,
            rng,
            num_attempts=CFG.exit_garage_rrt_num_attempts,
            num_iters=CFG.exit_garage_rrt_num_iters,
            # No smoothing because of non-holonomic movement
            smooth_amt=0)
        # Run planning.
        start_pos_list = [
            state.get(move_obj, "x"),
            state.get(move_obj, "y"),
            state.get(move_obj, "theta"),
        ]
        start_position = np.array(start_pos_list)
        position_plan = rrt.query_to_goal_fn(
            start_position,
            lambda: target_position,
            goal_fn,
            sample_goal_eps=CFG.exit_garage_rrt_sample_goal_eps,
        )
        # If motion planning fails, determine the option to be not initiable.
        if position_plan is None:
            return False
        # The position plan is used for the termination check, and possibly
        # can be used for debug drawing in the rendering in the future.
        memory["position_plan"] = position_plan
        # Convert the plan from position space to action space.
        action_plan = []
        for i in range(len(position_plan) - 1):
            x1, y1, theta1 = position_plan[i]
            x2, y2, theta2 = position_plan[i + 1]
            vel = ((x2 - x1) / np.cos(theta1) if np.cos(theta1) != 0 else
                   (y2 - y1) / np.sin(theta1))
            assert abs(vel) < ExitGarageEnv.car_max_absolute_vel
            omega = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
            assert abs(omega) < ExitGarageEnv.car_steering_omega_limit
            action_plan.append(
                Action(np.array([vel, omega, 0.0, 0.0, 0.0],
                                dtype=np.float32)))
        memory["action_plan"] = action_plan
        return True

    @classmethod
    def _plan_direct(cls, memory: Dict, params: Array, start_position: Array,
                     target_position: Array, x_action_idx: int,
                     y_action_idx: int) -> None:
        """Set position and action plans for a straight line from the starting
        position to the target position.

        Returns True.
        """
        del params  # unused

        def _extend_fn(pt1: Array, pt2: Array) -> Iterator[Array]:
            # Make sure that we obey the bounds on actions.
            distance = np.linalg.norm(pt2 - pt1)
            num = int(distance / ExitGarageEnv.robot_action_magnitude) + 1
            for i in range(1, num + 1):
                yield pt1 * (1 - i / num) + pt2 * i / num

        # Run planning.
        extender = _extend_fn(start_position, target_position)
        position_plan = [start_position] + list(extender)
        # The position plan is used for the termination check, and possibly
        # can be used for debug drawing in the rendering in the future.
        memory["position_plan"].extend(position_plan)
        # Convert the plan from position space to action space.
        deltas = np.subtract(position_plan[1:], position_plan[:-1])

        def _create_action(dx: float, dy: float) -> Action:
            arr = np.zeros(5, dtype=np.float32)
            arr[x_action_idx] = dx
            arr[y_action_idx] = dy
            return Action(arr)

        action_plan = [_create_action(dx, dy) for (dx, dy) in deltas]
        memory["action_plan"].extend(action_plan)
