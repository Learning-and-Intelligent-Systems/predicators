"""Base class for an environment."""

import abc
from typing import Callable, List, Optional, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.structs import Action, DefaultState, DefaultTask, \
    ParameterizedOption, Predicate, State, Task, Type, Video


class BaseEnv(abc.ABC):
    """Base environment."""

    def __init__(self) -> None:
        self._current_state = DefaultState  # set in reset
        self._current_task = DefaultTask  # set in reset
        self._set_seed(CFG.seed)
        # These are generated lazily when get_train_tasks or get_test_tasks is
        # called. This is necessary because environment attributes are often
        # initialized in __init__ in subclasses, and super().__init__ needs
        # to be called in those subclasses first, to set the env seed.
        self._train_tasks: List[Task] = []
        self._test_tasks: List[Task] = []

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this environment, used as the argument to
        `--env`."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def simulate(self, state: State, action: Action) -> State:
        """Get the next state, given a state and an action.

        Note that this action is a low-level action (i.e., its array
        representation is a member of self.action_space), NOT an option.

        This function is primarily used in the default option model, and
        for implementing the default self.step(action). It is not meant to
        be part of the "final system", where the environment is the real world.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _generate_train_tasks(self) -> List[Task]:
        """Create an ordered list of tasks for training."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _generate_test_tasks(self) -> List[Task]:
        """Create an ordered list of tasks for testing / evaluation."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def predicates(self) -> Set[Predicate]:
        """Get the set of predicates that are given with this environment."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def goal_predicates(self) -> Set[Predicate]:
        """Get the subset of self.predicates that are used in goals."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def types(self) -> Set[Type]:
        """Get the set of types that are given with this environment."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def options(self) -> Set[ParameterizedOption]:
        """Get the set of parameterized options that are given with this
        environment."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def action_space(self) -> Box:
        """Get the action space of this environment."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        """Render a state and action into a Matplotlib figure.

        Like simulate, this function is not meant to be part of the
        "final system", where the environment is the real world. It is
        just for convenience, e.g., in test coverage.

        For environments which don't use Matplotlib for rendering, this
        function should be overriden to simply crash.

        NOTE: Users of this method must remember to call `plt.close()`,
        because this method returns an active figure object!
        """
        raise NotImplementedError("Matplotlib rendering not implemented!")

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        """Render a state and action into a list of images.

        Like simulate, this function is not meant to be part of the
        "final system", where the environment is the real world. It is
        just for convenience, e.g., in test coverage.

        By default, calls render_state_plt, but subclasses may override,
        e.g. if they do not use Matplotlib for rendering, and thus do not
        define a render_state_plt() function.
        """
        fig = self.render_state_plt(state, task, action, caption)
        img = utils.fig2data(fig)
        plt.close()
        return [img]

    def render_plt(self,
                   action: Optional[Action] = None,
                   caption: Optional[str] = None) -> matplotlib.figure.Figure:
        """Render the current state and action into a Matplotlib figure.

        By default, calls render_state_plt, but subclasses may override.

        NOTE: Users of this method must remember to call `plt.close()`,
        because this method returns an active figure object!
        """
        return self.render_state_plt(self._current_state, self._current_task,
                                     action, caption)

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:
        """Render the current state and action into a list of images.

        By default, calls render_state, but subclasses may override.
        """
        return self.render_state(self._current_state, self._current_task,
                                 action, caption)

    def get_train_tasks(self) -> List[Task]:
        """Return the ordered list of tasks for training."""
        if not self._train_tasks:
            self._train_tasks = self._generate_train_tasks()
        return self._train_tasks

    def get_test_tasks(self) -> List[Task]:
        """Return the ordered list of tasks for testing / evaluation."""
        if not self._test_tasks:
            self._test_tasks = self._generate_test_tasks()
        return self._test_tasks

    def get_task(self, train_or_test: str, task_idx: int) -> Task:
        """Return the train or test task at the given index."""
        if train_or_test == "train":
            tasks = self.get_train_tasks()
        elif train_or_test == "test":
            tasks = self.get_test_tasks()
        else:
            raise ValueError(f"get_task called with invalid train_or_test: "
                             f"{train_or_test}.")
        return tasks[task_idx]

    def _set_seed(self, seed: int) -> None:
        """Reset seed and rngs."""
        self._seed = seed
        # The train/test rng should be used when generating
        # train/test tasks respectively.
        self._train_rng = np.random.default_rng(self._seed)
        self._test_rng = np.random.default_rng(self._seed +
                                               CFG.test_env_seed_offset)

    def reset(self, train_or_test: str, task_idx: int) -> State:
        """Resets the current state to the train or test task initial state."""
        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_state = self._current_task.init
        # Copy to prevent external changes to the environment's state.
        return self._current_state.copy()

    def step(self, action: Action) -> State:
        """Apply the action, and update and return the current state.

        Note that this action is a low-level action (i.e., action.arr
        is a member of self.action_space), NOT an option.

        By default, this function just calls self.simulate. However,
        environments that maintain a more complicated internal state,
        or that don't implement simulate(), may override this method.
        """
        self._current_state = self.simulate(self._current_state, action)
        # Copy to prevent external changes to the environment's state.
        return self._current_state.copy()

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:
        """The optional environment-specific method that is used for generating
        demonstrations from a human, with a GUI.

        Returns a function that maps state and Matplotlib event to an
        action in this environment; before returning this function, it's
        recommended to log some instructions about the controls.
        """
        raise NotImplementedError("This environment did not implement an "
                                  "interface for human demonstrations!")
