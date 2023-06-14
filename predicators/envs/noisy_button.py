"""A button is somewhere on a 1D axis. The goal is to press the button. The
initial position of the button is unknown. A special "find" action locates the
button. If the button is pressed on its edges, the press is unsuccessful and
the button position is moved to an unknown position. The real position of the
button is stored in the simulator state.

The purpose of this environment is to test the development of a meta-
approach that always performs "find" when the button position is
unknown.
"""
from __future__ import annotations

from typing import ClassVar, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class _NoisyButtonState(State):

    @property
    def real_position(self) -> float:
        """Expose the real button position."""
        assert isinstance(self.simulator_state, float)
        return self.simulator_state

    def copy(self) -> _NoisyButtonState:
        state = super().copy()
        return _NoisyButtonState(state.data, state.simulator_state)

    def allclose(self, other: State) -> bool:
        return State(self.data).allclose(State(other.data))


class NoisyButtonEnv(BaseEnv):
    """Toy environment for testing meta-approaches."""

    _successful_press_thresh: ClassVar[float] = 1e-2
    _press_thresh: ClassVar[float] = 2 * _successful_press_thresh

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._button_type = Type("button",
                                 ["position", "position_known", "pressed"])
        # Predicates
        self._Pressed = Predicate("Pressed", [self._button_type],
                                  self._Pressed_holds)
        # Static objects (always exist no matter the settings).
        self._button = Object("button", self._button_type)

    @classmethod
    def get_name(cls) -> str:
        return "noisy_button"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Pressed}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Pressed}

    @property
    def types(self) -> Set[Type]:
        return {self._button_type}

    @property
    def action_space(self) -> Box:
        # A point on the 1D axis and a bit indicating a "find".
        return Box(0, 1, (2, ))

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        assert isinstance(state, _NoisyButtonState)
        position_act, find_act = action.arr
        next_state = state.copy()
        # Execute find action.
        if find_act > 0.5:
            next_state.set(self._button, "position", state.real_position)
            next_state.set(self._button, "position_known", 1.0)
            return next_state
        # Execute poke action.
        # If the poke is close enough to the real button center, succeed.
        pos = state.real_position
        dist = abs(pos - position_act)
        if dist < self._successful_press_thresh:
            next_state.set(self._button, "pressed", 1.0)
            return next_state
        # If the poke is too far, do nothing.
        if dist > self._press_thresh:
            return next_state
        # Otherwise, the poke "moved" the button out of view. To conform to the
        # assumption that simulate is deterministic, we will displace the
        # object by putting it exactly where the press occurred, but it will be
        # unknown to the agent.
        next_state.set(self._button, "position", -1.0)  # unknown
        next_state.set(self._button, "position_known", 0.0)
        next_state.simulator_state = float(position_act)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        # There is only one goal in this environment.
        goal_atom = GroundAtom(self._Pressed, [self._button])
        goal = {goal_atom}
        # The initial position of the button varies.
        tasks: List[EnvironmentTask] = []
        while len(tasks) < num:
            state = utils.create_state_from_dict({
                self._button: {
                    "position": -1.0,  # unknown
                    "position_known": 0.0,
                    "pressed": 0.0,
                },
            })
            # Sample the real position.
            pos = rng.uniform(0.0, 1.0)
            button_state = _NoisyButtonState(state.data, simulator_state=pos)
            tasks.append(EnvironmentTask(button_state, goal))
        return tasks

    def _Pressed_holds(self, state: State, objects: Sequence[Object]) -> bool:
        button, = objects
        return state.get(button, "pressed") > 0.5

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        assert isinstance(state, _NoisyButtonState)

        fig, _ = plt.subplots(1, 1)
        # Draw main line.
        plt.plot([-0.2, 1.2], [0.0, 0.0], color="black")
        # Draw last action.
        if action is not None and action.arr[1] < 0.5:
            pos_act = action.arr[0]
            plt.scatter(pos_act, 0.0, color="red", s=250, alpha=0.5)
        else:
            plt.scatter(0.5, 0.1, color="red", s=250, alpha=0.5)
        # Draw the real button position.
        alpha = 1.0 if state.get(self._button, "position_known") > 0.5 else 0.2
        plt.scatter(state.real_position, 0.0, color="blue", s=250, alpha=alpha)
        # Finish the plot.
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.25, 0.5)
        plt.yticks([])
        if caption is not None:
            plt.suptitle(caption, wrap=True)
        plt.tight_layout()
        return fig
