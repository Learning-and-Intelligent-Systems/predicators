"""Toy environment for isolating and testing the issue of multiple instances of
a predicate being in the effects of options.

Here, the move option can turn on any number of NextTo predicates.
"""

from typing import ClassVar, Dict, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class RepeatedNextToEnv(BaseEnv):
    """RepeatedNextToEnv environment definition.

    Simple 1D problem.
    """
    env_lb: ClassVar[float] = 0.0
    env_ub: ClassVar[float] = 100.0
    grasped_thresh: ClassVar[float] = 0.5
    nextto_thresh: ClassVar[float] = 0.5

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._robot_type = Type("robot", ["x"])
        self._dot_type = Type("dot", ["x", "grasped"])
        # Predicates
        self._NextTo = Predicate("NextTo", [self._robot_type, self._dot_type],
                                 self._NextTo_holds)
        self._NextToNothing = Predicate("NextToNothing", [self._robot_type],
                                        self._NextToNothing_holds)
        self._Grasped = Predicate("Grasped",
                                  [self._robot_type, self._dot_type],
                                  self._Grasped_holds)
        # Options
        self._Move = utils.SingletonParameterizedOption(
            "Move",
            self._Move_policy,
            types=[self._robot_type, self._dot_type],
            params_space=Box(-1, 1, (1, )))
        self._Grasp = utils.SingletonParameterizedOption(
            "Grasp",
            policy=self._Grasp_policy,
            types=[self._robot_type, self._dot_type])
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "repeated_nextto"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        move_or_grasp, norm_robot_x, norm_dot_x = action.arr
        next_state = state.copy()
        if move_or_grasp < 0.5:
            # Handle move action.
            robot_x = norm_robot_x * (self.env_ub - self.env_lb) + self.env_lb
            next_state.set(self._robot, "x", robot_x)
        else:
            # Handle grasp action.
            robot_x = state.get(self._robot, "x")
            desired_x = norm_dot_x * (self.env_ub - self.env_lb) + self.env_lb
            dots = state.get_objects(self._dot_type)
            dot_to_grasp = min(
                dots, key=lambda dot: abs(state.get(dot, "x") - desired_x))
            dot_to_grasp_x = state.get(dot_to_grasp, "x")
            if abs(dot_to_grasp_x - desired_x) > 1e-4:
                # There is no dot near the desired_x action argument.
                return next_state
            if abs(robot_x - dot_to_grasp_x) > self.nextto_thresh:
                # Robot must be next to dot in order to grasp it.
                return next_state
            next_state.set(dot_to_grasp, "grasped", 1.0)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._NextTo, self._NextToNothing, self._Grasped}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Grasped}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._dot_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Move, self._Grasp}

    @property
    def action_space(self) -> Box:
        # First dimension is move (less than 0.5) or grasp (greater than 0.5).
        # Second dimension is normalized new robot x (if first dim is move).
        # Third dimension is normalized location of dot to grasp (if second
        # dim is grasp). Normalization is [self.env_lb, self.env_ub] -> [0, 1].
        return Box(0, 1, (3, ))

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1)
        robot_x = state.get(self._robot, "x")
        for dot in state.get_objects(self._dot_type):
            dot_x = state.get(dot, "x")
            if state.get(dot, "grasped") > self.grasped_thresh:
                color = "green"
            elif abs(robot_x - dot_x) < self.nextto_thresh:
                color = "orange"
            else:
                color = "red"
            plt.scatter(x=dot_x, y=0, color=color)
        plt.scatter(x=robot_x, y=0.2)
        ax.set_xlim(self.env_lb - 1, self.env_ub + 1)
        ax.set_ylim(-0.1, 0.25)
        title = ("red = not next to, orange = next to, green = grasped, "
                 "blue = robot")
        if caption is not None:
            title += f";\n{caption}"
        plt.suptitle(title, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        tasks = []
        dots = []
        for i in range(CFG.repeated_nextto_num_dots):
            dots.append(Object(f"dot{i}", self._dot_type))
        goal1 = {GroundAtom(self._Grasped, [self._robot, dots[0]])}
        goal2 = {
            GroundAtom(self._Grasped, [self._robot, dots[0]]),
            GroundAtom(self._Grasped, [self._robot, dots[1]]),
        }
        goal3 = {
            GroundAtom(self._Grasped, [self._robot, dots[0]]),
            GroundAtom(self._Grasped, [self._robot, dots[1]]),
            GroundAtom(self._Grasped, [self._robot, dots[2]]),
        }
        goals = [goal1, goal2, goal3]
        for i in range(num):
            data: Dict[Object, Array] = {}
            for dot in dots:
                dot_x = rng.uniform(self.env_lb, self.env_ub)
                data[dot] = np.array([dot_x, 0.0])
            robot_x = rng.uniform(self.env_lb, self.env_ub)
            data[self._robot] = np.array([robot_x])
            tasks.append(Task(State(data), goals[i % len(goals)]))
        return tasks

    def _Move_policy(self, state: State, memory: Dict,
                     objects: Sequence[Object], params: Array) -> Action:
        del memory  # unused
        _, dot = objects
        dot_x = state.get(dot, "x")
        delta, = params
        robot_x = max(min(self.env_ub, dot_x + delta), self.env_lb)
        norm_robot_x = (robot_x - self.env_lb) / (self.env_ub - self.env_lb)
        return Action(np.array([0, norm_robot_x, 0], dtype=np.float32))

    def _Grasp_policy(self, state: State, memory: Dict,
                      objects: Sequence[Object], params: Array) -> Action:
        del memory, params  # unused
        _, dot = objects
        dot_x = state.get(dot, "x")
        norm_dot_x = (dot_x - self.env_lb) / (self.env_ub - self.env_lb)
        return Action(np.array([1, 0, norm_dot_x], dtype=np.float32))

    def _NextTo_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, dot = objects
        return (state.get(dot, "grasped") < self.grasped_thresh
                and abs(state.get(robot, "x") - state.get(dot, "x")) <
                self.nextto_thresh)

    def _NextToNothing_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        robot, = objects
        for obj in state:
            if obj.type == self._dot_type and \
               self._NextTo_holds(state, [robot, obj]):
                return False
        return True

    def _Grasped_holds(self, state: State, objects: Sequence[Object]) -> bool:
        _, dot = objects
        return state.get(dot, "grasped") > self.grasped_thresh


class RepeatedNextToSingleOptionEnv(RepeatedNextToEnv):
    """A variation on RepeatedNextToEnv with a single parameterized option."""

    def __init__(self) -> None:
        super().__init__()
        # Options
        del self._Move
        del self._Grasp

        # The first parameter dimension modulates whether the action will be a
        # move (negative value) or a grasp (nonnegative value). The second
        # dimension is the same as that for self._Move in the parent class.
        self._MoveGrasp = utils.SingletonParameterizedOption(
            "MoveGrasp",
            policy=self._MoveGrasp_policy,
            types=[self._robot_type, self._dot_type],
            params_space=Box(-1, 1, (2, )))

    @classmethod
    def get_name(cls) -> str:
        return "repeated_nextto_single_option"

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._MoveGrasp}

    def _MoveGrasp_policy(self, state: State, memory: Dict,
                          objects: Sequence[Object], params: Array) -> Action:
        if params[0] < 0:
            return self._Move_policy(state, memory, objects, params[1:])
        return self._Grasp_policy(state, memory, objects, params[1:])


class RepeatedNextToAmbiguousEnv(RepeatedNextToEnv):
    """A variation on RepeatedNextToEnv with ambiguous demonstrations that can
    lead to the backchaining algorithm learning complex move operators."""

    @classmethod
    def get_name(cls) -> str:
        return "repeated_nextto_ambiguous"

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        assert self.env_ub - self.env_lb > self.nextto_thresh
        tasks = []
        dots = []
        assert CFG.repeated_nextto_num_dots >= 3
        for i in range(CFG.repeated_nextto_num_dots):
            dots.append(Object(f"dot{i}", self._dot_type))
        goal1 = {GroundAtom(self._Grasped, [self._robot, dots[0]])}
        goal2 = {
            GroundAtom(self._Grasped, [self._robot, dots[0]]),
            GroundAtom(self._Grasped, [self._robot, dots[1]]),
        }
        goal3 = {
            GroundAtom(self._Grasped, [self._robot, dots[0]]),
            GroundAtom(self._Grasped, [self._robot, dots[1]]),
            GroundAtom(self._Grasped, [self._robot, dots[2]]),
        }
        goals = [goal3, goal2, goal1]
        for i in range(num):
            data: Dict[Object, Array] = {}
            for dot in dots:
                dot_x = rng.uniform(self.env_ub - 0.5, self.env_ub)
                data[dot] = np.array([dot_x, 0.0])
            robot_x = self.env_lb
            data[self._robot] = np.array([robot_x])
            tasks.append(Task(State(data), goals[i % len(goals)]))
        return tasks
