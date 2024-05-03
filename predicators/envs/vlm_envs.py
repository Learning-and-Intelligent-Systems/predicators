"""A bunch of environments useful for testing VLM-based Predicate Invention.

Will likely be updated and potentially split into separate files in the
future.
"""

import abc
from typing import List, Optional, Sequence, Set

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type

DUMMY_GOAL_OBJ_NAME = "dummy_goal_obj"  # used in VLM parsing as well.


class VLMPredicateEnv(BaseEnv):
    """Environments that use VLM Predicates.

    Note that no simulate function or ground truth model is implemented
    for these yet. These are forthcoming.
    """

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
        raise ValueError("Simulate shouldn't be getting called!")

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
        return {self._DummyGoal}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._DummyGoal}

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

    def _get_tasks(
        self, num: int, rng: np.random.Generator
    ) -> List[EnvironmentTask]:  # pragma: no cover.
        del num, rng
        raise NotImplementedError("Override!")

    @property
    @abc.abstractmethod
    def vlm_debug_atom_strs(self) -> Set[str]:
        """Return a set of atom strings that should be sufficient for a VLM to
        label demonstrations consistently to learn good operators."""
        raise NotImplementedError(
            "VLM debug atom strings not implemented for this environment.")


class IceTeaMakingEnv(VLMPredicateEnv):
    """A (simplified) version of a tea-making task that's closer to pick-and-
    place than real tea-making."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Env-specific types.
        self._teabag_type = Type("teabag", [], self._object_type)
        self._spoon_type = Type("spoon", [], self._object_type)
        self._cup_type = Type("cup", [], self._object_type)
        self._plate_type = Type("plate", [], self._object_type)
        self._hand_type = Type("hand", [], self._object_type)

    @classmethod
    def get_name(cls) -> str:
        return "ice_tea_making"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._teabag_type, self._spoon_type, self._cup_type,
            self._plate_type, self._hand_type
        }

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng  # unused.
        dummy_goal_obj = Object(DUMMY_GOAL_OBJ_NAME, self._goal_object_type)
        teabag_obj = Object("teabag", self._teabag_type)
        spoon_obj = Object("spoon", self._spoon_type)
        cup_obj = Object("cup", self._cup_type)
        plate_obj = Object("plate", self._plate_type)
        hand_obj = Object("hand", self._hand_type)
        init_state = State({
            dummy_goal_obj: np.array([0.0]),
            teabag_obj: np.array([]),
            plate_obj: np.array([]),
            spoon_obj: np.array([]),
            cup_obj: np.array([]),
            hand_obj: np.array([])
        })
        return [
            EnvironmentTask(
                init_state,
                set([GroundAtom(self._DummyGoal, [dummy_goal_obj])]))
            for _ in range(num)
        ]

    @property
    def vlm_debug_atom_strs(self) -> Set[str]:
        """A 'debug grammar' set of predicates that should be sufficient for
        completing the task; useful for comparing different methods of VLM
        truth-value labelling given the same set of atom proposals to label."""
        return set([
            "hand_grasping_spoon(hand, spoon)",
            "hand_grasping_teabag(hand, teabag)", "spoon_in_cup(spoon, cup)",
            "spoon_on_plate(spoon, plate)", "teabag_in_cup(teabag, cup)",
            "teabag_on_plate(teabag, plate)"
        ])

class BurgerMakingEnv(VLMPredicateEnv):
    """A simple 2D gridworld environment for burger making."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Env-specific types
        self._robot_type = Type("robot", [], self._object_type)
        self._cutting_board_type = Type("cutting_board", [], self._object_type)
        self._grill_type = Type("grill", [], self._object_type)
        self._tomato_type = Type("tomato", [], self._object_type)
        self._cheese_type = Type("cheese", [], self._object_type)
        self._patty_type = Type("patty", [], self._object_type)
        self._bottom_bun_type = Type("bottom_bun", [], self._object_type)
        self._top_bun_type = Type("top_bun", [], self._object_type)

    @classmethod
    def get_name(cls) -> str:
        return "burger_making"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._robot_type, self._cutting_board_type, self._grill_type, self._tomato_type,
            self._cheese_type, self._patty_type, self._bottom_bun_type, self._top_bun_type}

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng  # unused
        dummy_goal_obj = Object(DUMMY_GOAL_OBJ_NAME, self._goal_object_type)
        robot_obj = Object("robot", self._robot_type)
        cutting_board_obj = Object("cutting_board", self._cutting_board_type)
        grill_obj = Object("grill", self._grill_type)
        tomato_obj = Object("tomato", self._tomato_type)
        cheese_obj = Object("cheese", self._cheese_type)
        patty_obj = Object("patty", self._patty_type)
        bottom_bun_obj = Object("bottom_bun", self._bottom_bun_type)
        top_bun_obj = Object("top_bun", self._top_bun_type)

        init_state = State({
            dummy_goal_obj: np.array([0.0]),
            robot_obj: np.array([]),
            cutting_board_obj: np.array([]),
            grill_obj: np.array([]),
            tomato_obj: np.array([]),
            cheese_obj: np.array([]),
            patty_obj: np.array([]),
            bottom_bun_obj: np.array([]),
            top_bun_obj: np.array([])
        })
        return [
            EnvironmentTask(
                init_state,
                set([GroundAtom(self._DummyGoal, [dummy_goal_obj])]))
            for _ in range(num)
        ]