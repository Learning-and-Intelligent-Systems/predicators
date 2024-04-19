"""A bunch of environments to help test VLM-based Predicate Invention."""

from typing import List, Optional, Sequence, Set

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class VLMPredicateEnv(BaseEnv):

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._object_type = Type("object", [])
        self._goal_object_type = Type("goal_object", ["goal_true"],
                                      self._object_type)

        # Predicates
        self._DummyGoal = Predicate("DummyGoal", [self._goal_object_type],
                                    self._Dummy_Goal_holds)

    def simulate(self, state: State, action: Action) -> State:
        raise ValueError("simulate shouldn't be getting called!")

    @property
    def types(self) -> Set[Type]:
        return {self._object_type, self._goal_object_type}

    def _Dummy_Goal_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "goal_true") > 0.5

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return set([self._DummyGoal])

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return set([self._DummyGoal])

    @property
    def action_space(self) -> Box:
        return Box(low=0.0, high=1.0, shape=(0, ), dtype=np.float32)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise ValueError("shouldn't be trying to render env at any point!")

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        del num, rng  # unused
        return []


class AppleCoringEnv(VLMPredicateEnv):

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Env-specific types.
        self._apple_type = Type("apple", [], self._object_type)
        self._slicing_tool_type = Type("slicing_tool", [], self._object_type)
        self._plate_type = Type("plate", [], self._object_type)
        self._hand_type = Type("hand", [], self._object_type)

    @classmethod
    def get_name(cls) -> str:
        return "apple_coring"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._apple_type, self._slicing_tool_type, self._plate_type,
            self._hand_type
        }

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        dummy_goal_obj = Object("dummy_goal_obj", self._goal_object_type)
        apple_obj = Object("apple", self._apple_type)
        plate_obj = Object("plate", self._plate_type)
        slicing_tool_obj = Object("slicing_tool", self._slicing_tool_type)
        hand_obj = Object("hand", self._hand_type)
        init_state = State({
            dummy_goal_obj: [0.0],
            apple_obj: [],
            plate_obj: [],
            slicing_tool_obj: [],
            hand_obj: []
        })
        return [
            EnvironmentTask(
                init_state,
                set([GroundAtom(self._DummyGoal, [dummy_goal_obj])]))
            for _ in range(num)
        ]


class IceTeaMakingEnv(VLMPredicateEnv):

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Env-specific types.
        self._teabag_type = Type("teabag", [], self._object_type)
        # self._ice_type = Type("ice", [], self._object_type)
        self._spoon_type = Type("spoon", [], self._object_type)
        self._cup_type = Type("cup", [], self._object_type)
        self._plate_type = Type("plate", [], self._object_type)
        self._hand_type = Type("hand", [], self._object_type)

    @classmethod
    def get_name(cls) -> str:
        return "iced_tea_making"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._teabag_type, self._spoon_type, self._cup_type,
            self._plate_type, self._hand_type
        }

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        dummy_goal_obj = Object("dummy_goal_obj", self._goal_object_type)
        teabag_obj = Object("teabag", self._teabag_type)
        # ice_obj = Object("ice", self._ice_type)
        spoon_obj = Object("spoon", self._spoon_type)
        cup_obj = Object("cup", self._cup_type)
        plate_obj = Object("plate", self._plate_type)
        hand_obj = Object("hand", self._hand_type)
        init_state = State({
            dummy_goal_obj: [0.0],
            teabag_obj: [],
            plate_obj: [],
            spoon_obj: [],
            cup_obj: [],
            hand_obj: []
        })
        return [
            EnvironmentTask(
                init_state,
                set([GroundAtom(self._DummyGoal, [dummy_goal_obj])]))
            for _ in range(num)
        ]
