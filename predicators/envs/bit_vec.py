"""Toy environment for testing non-lifted operator learning."""

from typing import Dict, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class BitVectorEnv(BaseEnv):
    """An environment with a single global object that has N binary features.

    Actions flip the binary features.

    There are two given options: one for flipping the leftmost true feature
    and one for flipping the leftmost false feature.

    There are two possible goals: all features true, or all features false.

    The given predicates specify the number of true or false features that are
    in the state, but abstract away the positions of those features. So there
    are 2N predicates but 2^N possible low-level states.
    """

    def __init__(self) -> None:
        super().__init__()
        # Get N, the dimension of the bit vector.
        self._N = CFG.bit_vec_env_num_slots
        # Types
        self._world_type = Type("world", [f"f{n}" for n in range(self._N)])
        # Predicates
        self._n_to_gt_pred = {
            n: self._create_greater_than_predicate(n)
            for n in range(self._N)
        }
        self._n_to_lt_pred = {
            n: self._create_less_than_predicate(n)
            for n in range(1, self._N + 1)
        }
        # Options
        self._ToggleTrue = utils.SingletonParameterizedOption(
            "ToggleTrue",
            policy=self._ToggleTrue_policy,
            initiable=self._ToggleTrue_initiable)
        self._ToggleFalse = utils.SingletonParameterizedOption(
            "ToggleFalse",
            policy=self._ToggleFalse_policy,
            initiable=self._ToggleFalse_initiable)
        # Static objects (always exist no matter the settings).
        self._world = Object("world", self._world_type)

    @classmethod
    def get_name(cls) -> str:
        return "bit_vec"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        slot_val, = action.arr
        slot = int(np.round(slot_val))
        feat = f"f{slot}"
        next_state = state.copy()
        old_val = state.get(self._world, feat)
        assert old_val in [0.0, 1.0]
        new_val = not bool(old_val)
        next_state.set(self._world, feat, new_val)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        lt_preds = set(self._n_to_lt_pred.values())
        gt_preds = set(self._n_to_gt_pred.values())
        return lt_preds | gt_preds

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._n_to_gt_pred[self._N - 1], self._n_to_lt_pred[1]}

    @property
    def types(self) -> Set[Type]:
        return {self._world_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._ToggleFalse, self._ToggleTrue}

    @property
    def action_space(self) -> Box:
        # An angle in radians.
        return Box(0.0, self._N - 1, (1, ))

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(self._N, 2))
        bit_vec = self._state_to_bit_vec(state)
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
        ax.imshow(np.reshape(bit_vec, (1, self._N)),
                  norm=normalize,
                  cmap="binary_r")
        ax.set_xticks([])
        ax.set_yticks([])
        title = f"Goal: {task.goal}"
        if caption is not None:
            title += f";\n{caption}"
        plt.suptitle(title, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        # The initial state features are randomly selected.
        tasks: List[Task] = []
        while len(tasks) < num:
            # Randomly pick one of the possible goals.
            goal_pred = rng.choice(sorted(self.goal_predicates))
            goal_atom = GroundAtom(goal_pred, [])
            goal = {goal_atom}
            feat_vals = rng.choice([True, False], size=self._N)
            feat_dict = dict(zip(self._world_type.feature_names, feat_vals))
            state = utils.create_state_from_dict({self._world: feat_dict})
            # Make sure goal is not satisfied.
            if not goal_atom.holds(state):
                tasks.append(Task(state, goal))
        return tasks

    def _ToggleTrue_policy(self, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
        del memory, params, objects  # unused
        bit_vec = np.array(self._state_to_bit_vec(state))
        left_most_false = np.argwhere(~bit_vec)[0, 0]
        val = left_most_false
        return Action(np.array([val], dtype=np.float32))

    def _ToggleTrue_initiable(self, state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> Action:
        del memory, params, objects  # unused
        bit_vec = np.array(self._state_to_bit_vec(state))
        # Something is False.
        return not bit_vec.all()

    def _ToggleFalse_policy(self, state: State, memory: Dict,
                            objects: Sequence[Object],
                            params: Array) -> Action:
        del memory, params, objects  # unused
        bit_vec = np.array(self._state_to_bit_vec(state))
        left_most_true = np.argwhere(bit_vec)[0, 0]
        val = left_most_true
        return Action(np.array([val], dtype=np.float32))

    def _ToggleFalse_initiable(self, state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> Action:
        del memory, params, objects  # unused
        bit_vec = np.array(self._state_to_bit_vec(state))
        # Something is True.
        return bit_vec.any()

    def _create_greater_than_predicate(self, n: int) -> Predicate:

        def _holds(state: State, objects: Sequence[Object]) -> bool:
            del objects  # unused
            bit_vec = self._state_to_bit_vec(state)
            return sum(bit_vec) > n

        return Predicate(f"GreaterThan{n}True", [], _holds)

    def _create_less_than_predicate(self, n: int) -> Predicate:

        def _holds(state: State, objects: Sequence[Object]) -> bool:
            del objects  # unused
            bit_vec = self._state_to_bit_vec(state)
            return sum(bit_vec) < n

        return Predicate(f"LessThan{n}True", [], _holds)

    def _state_to_bit_vec(self, state: State) -> List[bool]:
        return [
            bool(state.get(self._world, feat))
            for feat in self._world_type.feature_names
        ]
