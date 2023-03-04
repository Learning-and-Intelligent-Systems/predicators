"""Ground-truth options for the touch point environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import CozmoAction, Array, Object, ParameterizedOption, \
    Predicate, State, Type

import cozmo
from cozmo.util import distance_mm
import asyncio

def look_at_obj(robot, object_id, is_charger=False):
    # look around and try to find a cube
    look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    seen_obj = None
    try:
        while seen_obj is None or seen_obj.object_id != object_id:
            if is_charger:
                seen_obj = robot.world.wait_for_observed_charger(timeout=30)
            else:
                seen_obj = robot.world.wait_for_observed_light_cube(timeout=30)
    except asyncio.TimeoutError:
        print("Didn't find a cube")
    finally:
        # whether we find it or not, we want to stop the behavior
        look_around.stop()
    return seen_obj

class CozmoGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the touch point environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cozmo"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        cube_type = types["cube"]
        NextTo = predicates["NextTo"]

        def _MoveTo_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> CozmoAction:
            # Move in the direction of the target.
            del memory, params  # unused
            assert len(objects) == 2
            def cozmo_program(robot: cozmo.robot.Robot):
                obj = look_at_obj(robot, 0)
                current_action = robot.go_to_object(obj, distance_mm(65.0))
                current_action.wait_for_completed()
            return CozmoAction(np.array([0]), _run=cozmo_program)

        def _MoveTo_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            return NextTo.holds(state, objects)

        MoveTo = ParameterizedOption("MoveTo",
                                     types=[robot_type, cube_type],
                                     params_space=Box(0, 1, (0, )),
                                     policy=_MoveTo_policy,
                                     initiable=lambda s, m, o, p: True,
                                     terminal=_MoveTo_terminal)

        return {MoveTo}
