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
import time

def look_at_obj(robot, object_id, is_charger=False):
    # look around and try to find a cube
    look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    seen_obj = None
    if is_charger:
        seen_obj = robot.world.wait_for_observed_charger(timeout=30)
    else:
        cubes = robot.world.wait_until_observe_num_objects(num=3, object_type=cozmo.objects.LightCube, timeout=60)
        for cube in cubes:
            if cube.object_id == object_id:
                seen_obj = cube
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
        # Predicates
        Reachable = predicates["Reachable"]
        NextTo = predicates["NextTo"]
        Touched = predicates["Touched"]
        IsRed = predicates["IsRed"]
        IsBlue = predicates["IsBlue"]
        IsGreen = predicates["IsGreen"]
        OnTop = predicates["OnTop"]
        Under = predicates["Under"]

        # MoveTo Option
        def _MoveTo_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> CozmoAction:
            # Move in the direction of the target.
            del memory, params  # unused
            assert len(objects) == 2
            object_id = state.get(objects[1], "id")
            def cozmo_program(robot: cozmo.robot.Robot):
                print("Running MoveTo")
                object_id = state.get(objects[1], "id")
                # obj = look_at_obj(robot, object_id)
                obj = robot.world.get_light_cube(object_id) 
                current_action = robot.go_to_object(obj, distance_mm(65.0))
                current_action.wait_for_completed()
            return CozmoAction(np.array([0, 0, object_id]), _run=cozmo_program)

        def _MoveTo_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            return Reachable.holds(state, objects)

        MoveTo = ParameterizedOption("MoveTo",
                                     types=[robot_type, cube_type],
                                     params_space=Box(0, 1, (0, )),
                                     policy=_MoveTo_policy,
                                     initiable=lambda s, m, o, p: True,
                                     terminal=_MoveTo_terminal)

        # Touch Option
        def _Touch_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> CozmoAction:
            # Move in the direction of the target.
            del memory, params  # unused
            assert len(objects) == 1
            object_id = state.get(objects[0], "id")
            def cozmo_program(robot: cozmo.robot.Robot):
                print("Running Touch")
                obj = robot.world.get_light_cube(object_id) 
                current_action = robot.go_to_object(obj, distance_mm(40.0))
                current_action.wait_for_completed()
                robot.play_anim_trigger(cozmo.anim.Triggers.OnSpeedtapTap).wait_for_completed()
            return CozmoAction(np.array([1, object_id, 0]), _run=cozmo_program)

        def _Touch_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            return Touched.holds(state, objects)

        Touch = ParameterizedOption("Touch",
                                     types=[cube_type],
                                     params_space=Box(0, 1, (0, )),
                                     policy=_Touch_policy,
                                     initiable=lambda s, m, o, p: True,
                                     terminal=_Touch_terminal)
        
        # Paint Option
        def _Paint_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> CozmoAction:
            # Move in the direction of the target.
            del memory  # unused
            assert len(objects) == 2
            assert len(params) == 1
            object_id = state.get(objects[1], "id")
            def cozmo_program(robot: cozmo.robot.Robot):
                print("Running Paint")
                obj = robot.world.get_light_cube(object_id)
                if int(params[0]) == 1:
                    block_color = cozmo.lights.red_light
                    cozmo_color = cozmo.lights.red_light
                elif int(params[0]) == 2:
                    block_color = cozmo.lights.blue_light
                    cozmo_color = cozmo.lights.blue_light
                elif int(params[0]) == 3:
                    block_color = cozmo.lights.green_light
                    cozmo_color = cozmo.lights.green_light
                robot.set_all_backpack_lights(block_color)
                time.sleep(2)
                robot.play_anim_trigger(cozmo.anim.Triggers.OnSpeedtapTap).wait_for_completed()
                obj.set_lights(cozmo_color)
                robot.set_center_backpack_lights(cozmo.lights.white_light)
                time.sleep(3)
            return CozmoAction(np.array([2, int(params[0]), object_id]), _run=cozmo_program)

        def _Paint_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del memory # unused
            return state.get(objects[1], "color") == int(params[0])

        Paint = ParameterizedOption("Paint",
                                     types=[robot_type, cube_type],
                                     params_space=Box(low=0, high=3, shape=(1,), dtype=np.float32),
                                     policy=_Paint_policy,
                                     initiable=lambda s, m, o, p: True,
                                     terminal=_Paint_terminal)

        # PlaceOntop Option
        def _PlaceOntop_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> CozmoAction:
            # Move in the direction of the target.
            del memory, params  # unused
            assert len(objects) == 2
            object_id1 = state.get(objects[0], "id")
            object_id2 = state.get(objects[1], "id")
            def cozmo_program(robot: cozmo.robot.Robot):
                print("Running PlaceOntop")
                # Lookaround until Cozmo knows where the 2 cubes are
                obj1 = robot.world.get_light_cube(object_id1) 
                obj2 = robot.world.get_light_cube(object_id2) 
                # Try and pickup the 1st cube
                current_action = robot.pickup_object(obj1, num_retries=3)
                current_action.wait_for_completed()
                if current_action.has_failed:
                    code, reason = current_action.failure_reason
                    result = current_action.result
                    print("Pickup Cube failed: code=%s reason='%s' result=%s" % (code, reason, result))
                    return

                # Now try to place that cube on the 2nd one
                current_action = robot.place_on_object(obj2, num_retries=3)
                current_action.wait_for_completed()
                if current_action.has_failed:
                    code, reason = current_action.failure_reason
                    result = current_action.result
                    print("Place On Cube failed: code=%s reason='%s' result=%s" % (code, reason, result))
                    return

                print("Cozmo successfully stacked 2 blocks!")
            return CozmoAction(np.array([3, object_id1, object_id2]), _run=cozmo_program)

        def _PlaceOntop_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            return OnTop.holds(state, objects)

        PlaceOntop = ParameterizedOption("PlaceOntop",
                                     types=[cube_type, cube_type],
                                     params_space=Box(0, 1, (0, )),
                                     policy=_PlaceOntop_policy,
                                     initiable=lambda s, m, o, p: True,
                                     terminal=_PlaceOntop_terminal)

        return {MoveTo, Touch, Paint, PlaceOntop}
