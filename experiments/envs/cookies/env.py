from typing import List, Optional, Sequence, Set

import numpy as np
from predicators.envs.base_env import BaseEnv
from predicators.structs import Action, EnvironmentTask, Object, Predicate, State, Type
import gym
import matplotlib

__all__ = ['Cookies']

class Cookies(BaseEnv):
    """Base environment."""

    # Settings
    coveredin_thresh: float = 0.5
    held_thresh: float = 0.5
    nextto_dist_thresh: float = 1

    # Variables for parametric types and predicates
    toppings: List[str] = [
        "Sprinkles", "Frosting", "Sugar", "ChocolateChips", "Strawberries",
        "Blueberries", "Nuts", "Honey", "Cinnamon", "Coconut",
    ]
    amount_format = "amount{topping}"
    covered_in_format = "CoveredIn{topping}"

    # Types
    _object_type = Type("obj", ["x", "y"])
    _robot_type = Type("robot", ["x", "y", "fingers"], _object_type)
    _donut_type = Type("donut", ["x", "y", "grasp"] + list(map(amount_format.format, toppings), _object_type))
    _shelf_type = Type("shelf", ["x", "y"], _object_type)
    _box_type = Type("box", ["x", "y"], _object_type)
    _topper_type = Type("topper", ["x", "y"], _object_type)

    # Predicates
    ## CoveredIn Predicates
    class CoveredIn_holds:
        def __init__(self, topping: str):
            global amount_format
            self._amount = amount_format.format(topping)

        def __call__(self, state: State, objects: Sequence[Object]) -> bool:
            donut, = objects
            return state.get(donut, self._amount) >= Cookies.coveredin_thresh

    _CoveredInPreds = {}
    for topping in toppings: # Comprehension does not see class scope
        _CoveredInPreds[covered_in_format.format(topping)] = Predicate(
            f"CoveredIn{topping}", _donut_type, CoveredIn_holds(topping)
        )

    ## NextTo Predicate
    @staticmethod
    def _objects_close(state: State, obj1: Object, obj2: Object, atol: float) -> bool:
        return np.linalg.norm([
            state.get(obj1, "x") - state.get(obj2, "x"),
            state.get(obj1, "y") - state.get(obj2, "y"),
        ]) < atol

    @staticmethod
    def _NextTo_holds(state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        return Cookies._objects_close(state, obj1, obj2, Cookies.nextto_dist_thresh)

    _NextTo = Predicate("NextTo", [_object_type, _object_type], _NextTo_holds)

    ## Held Predicate
    @staticmethod
    def _Held_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, donut = objects
        held = state.get(donut, "held") >= Cookies.held_thresh
        assert not held or Cookies._objects_close(state, robot, donut, 1e-2)
        return held

    _Held = Predicate("held", [_robot_type, _donut_type], _Held_holds)

    # Common objects
    _robot = Object("robot", _robot_type)
    _topper = Object("topper", _topper_type)

    @classmethod
    def get_name(cls) -> str:
        return "cookies"

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Override me!")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        raise NotImplementedError("Override me!")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        raise NotImplementedError("Override me!")

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Held, self._NextTo} | set(self._CoveredInPreds.values())

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._NextTo} | set(self._CoveredInPreds.values())

    @property
    def types(self) -> Set[Type]:
        {self._object_type, self._robot_type, self._donut_type, self._box_type, self._shelf_type}

    @property
    def action_space(self) -> gym.spaces.Box:
        raise NotImplementedError("Override me!")

    def render_state_plt(
        self,
        state: State,
        task: EnvironmentTask,
        action: Optional[Action] = None,
        caption: Optional[str] = None
    ) -> matplotlib.figure.Figure:
        raise NotImplementedError("Matplotlib rendering not implemented!")