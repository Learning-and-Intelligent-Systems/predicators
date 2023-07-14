from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, \
    Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib import patches

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class SamplerViz2Env(BaseEnv):
    """SamplerViz2 domain"""

    container_x : ClassVar[float] = 0
    container_y : ClassVar[float] = 0
    container_width_a : ClassVar[float] = 1
    container_height_a : ClassVar[float] = 1
    container_width_b : ClassVar[float] = 0.5
    container_height_b : ClassVar[float] = 3
    block_a_width : ClassVar[float] = 0.7
    block_a_height : ClassVar[float] = 0.3
    block_b_width : ClassVar[float] = 0.3
    block_b_height : ClassVar[float] = 2.31
    _container_color: ClassVar[List[float]] = [0.89, 0.82, 0.68]
    _block_a_color: ClassVar[str] = "#377eb8"
    _block_b_color: ClassVar[str] = "#ff7f00"
    # _block_a_color: ClassVar[List[float]] = [1, 0.41, 0.71]
    # _block_b_color: ClassVar[List[float]] = [0.31, 0.78, 0.47]



    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._block_type = Type("block", ["pose_x", "pose_y", "width", "height", "yaw"])
        self._container_type = Type("container", ["pose_x", "pose_y", "width_a", "height_a", "width_b", "height_b"])

        # Predicates
        self._InContainer = Predicate("InContainer", [self._block_type, self._container_type], self._InContainer_holds)

        # Options
        lo = [self.container_x, self.container_y, -np.pi]
        hi = [self.container_x + self.container_width_a, self.container_y + self.container_height_b, np.pi]
        self._PlaceBlock = utils.SingletonParameterizedOption(
            # variables: [block to place, container]
            # params: [x, y, yaw]
            "PlaceBlock",
            self._PlaceBlock_policy,
            types=[self._block_type, self._container_type],
            params_space=Box(np.array(lo, dtype=np.float32),
                             np.array(hi, dtype=np.float32)))

        # Static objects
        self._container = Object("container", self._container_type)
        self._block_a = Object("block_a", self._block_type)
        self._block_b = Object("block_b", self._block_type)

    @classmethod
    def get_name(cls) -> str:
        return "sampler_viz2"

    def simulate(self, state: State, action: Action) -> State:
        x, y, yaw, a_or_b = action.arr
        next_state = state.copy()

        if a_or_b == 0:
            block = self._block_a
        else:
            block = self._block_b

        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "yaw", yaw)
        # if self.check_collision(next_state):
        #     return state.copy()
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InContainer}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InContainer}

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._container_type}


    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._PlaceBlock}

    @property
    def action_space(self) -> Box:
        lowers = np.array([self.container_x, self.container_y, -np.pi + 1e-4, 0], dtype=np.float32)
        # uppers = lowers + np.array([self.container_width_a, self.container_height_b, np.pi - 1e-4, 1], dtype=np.float32)
        uppers = lowers + np.array([self.container_width_a, self.container_height_b, 2 * np.pi - 1e-4, 1], dtype=np.float32)
        return Box(lowers, uppers)

    def render_overlying_states_plt(self, states):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        container = self._container
        block_a = self._block_a
        block_b = self._block_b
        # Draw container
        rect_container_a = utils.Rectangle(self.container_x, self.container_y, self.container_width_a, self.container_height_a, 0)
        rect_container_b = utils.Rectangle(self.container_x, self.container_y, self.container_width_b, self.container_height_b, 0)
        rect_container_a.plot(ax, facecolor=self._container_color)
        rect_container_b.plot(ax, facecolor=self._container_color)

        for state in states:


            # Draw blocks
            block_a_x = state.get(self._block_a, "pose_x")
            block_a_y = state.get(self._block_a, "pose_y")
            block_a_yaw = state.get(self._block_a, "yaw")
            rect_block_a = utils.Rectangle(block_a_x, block_a_y, self.block_a_width, self.block_a_height, block_a_yaw)
            rect_block_a.plot(ax, facecolor=self._block_a_color, alpha=0.1)

            # block_b_x = state.get(self._block_b, "pose_x")
            # block_b_y = state.get(self._block_b, "pose_y")
            # block_b_yaw = state.get(self._block_b, "yaw")
            # rect_block_b = utils.Rectangle(block_b_x, block_b_y, self.block_b_width, self.block_b_height, block_b_yaw)
            # rect_block_b.plot(ax, facecolor=self._block_b_color, alpha=0.1)

            min_ = min(self.container_x, self.container_y)
            max_ = max(self.container_x + self.container_width_a, self.container_y + self.container_height_b)
            ax.set_xlim(min_, max_)
            ax.set_ylim(min_, max_)
            # ax.set_xlim(self.container_x, self.container_x + self.container_width_a)
            # ax.set_ylim(self.container_y, self.container_y + self.container_height_b)
            # plt.suptitle(caption, fontsize=12, wrap=True)
        plt.axis("off")
        plt.tight_layout()

        return fig



    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        container = self._container
        block_a = self._block_a
        block_b = self._block_b

        # Draw container
        rect_container_a = utils.Rectangle(self.container_x, self.container_y, self.container_width_a, self.container_height_a, 0)
        rect_container_b = utils.Rectangle(self.container_x, self.container_y, self.container_width_b, self.container_height_b, 0)
        rect_container_a.plot(ax, facecolor=self._container_color)
        rect_container_b.plot(ax, facecolor=self._container_color)

        # Draw blocks
        block_a_x = state.get(self._block_a, "pose_x")
        block_a_y = state.get(self._block_a, "pose_y")
        block_a_yaw = state.get(self._block_a, "yaw")
        rect_block_a = utils.Rectangle(block_a_x, block_a_y, self.block_a_width, self.block_a_height, block_a_yaw)
        rect_block_a.plot(ax, facecolor=self._block_a_color)

        block_b_x = state.get(self._block_b, "pose_x")
        block_b_y = state.get(self._block_b, "pose_y")
        block_b_yaw = state.get(self._block_b, "yaw")
        rect_block_b = utils.Rectangle(block_b_x, block_b_y, self.block_b_width, self.block_b_height, block_b_yaw)
        rect_block_b.plot(ax, facecolor=self._block_b_color)

        min_ = min(self.container_x, self.container_y)
        max_ = max(self.container_x + self.container_width_a, self.container_y + self.container_height_b)
        ax.set_xlim(min_, max_)
        ax.set_ylim(min_, max_)
        # plt.suptitle(caption, fontsize=12, wrap=True)
        plt.axis("off")
        plt.tight_layout()
        return fig

    def _get_tasks(self, num_tasks: int) -> List[Task]:
        tasks = []
        data = {}
        data[self._container] = np.array([self.container_x, self.container_y, self.container_width_a, self.container_height_a, self.container_width_b, self.container_height_b])
        data[self._block_a] = np.array([1, 1, self.block_a_width, self.block_a_height, 0])
        data[self._block_b] = np.array([1, 1, self.block_b_width, self.block_b_height, 0])

        if CFG.sampler_viz_singlestep_goal:
            goal = {GroundAtom(self._InContainer, [self._block_a, self._container])}
        else:
            goal = {GroundAtom(self._InContainer, [self._block_a, self._container]),
                    GroundAtom(self._InContainer, [self._block_b, self._container])}
        state = State(data)
        tasks = [Task(state, goal) for _ in range(num_tasks)]
        return tasks

    def check_collision(self, state: State) -> bool:
        block_a_x = state.get(self._block_a, "pose_x")
        block_a_y = state.get(self._block_a, "pose_y")
        block_a_yaw = state.get(self._block_a, "yaw")
        rect_block_a = utils.Rectangle(block_a_x, block_a_y, self.block_a_width, self.block_a_height, block_a_yaw)

        block_b_x = state.get(self._block_b, "pose_x")
        block_b_y = state.get(self._block_b, "pose_y")
        block_b_yaw = state.get(self._block_b, "yaw")
        rect_block_b = utils.Rectangle(block_b_x, block_b_y, self.block_b_width, self.block_b_height, block_b_yaw)

        return utils.rectangles_intersect(rect_block_a, rect_block_b)

    def _PlaceBlock_policy(self, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
        del memory
        block, container = objects
        x, y, yaw = params
        if block.name == self._block_a.name:
            a_or_b = 0
        else:
            a_or_b = 1
        arr = np.array([x, y, yaw, a_or_b], dtype=np.float32)
        arr = np.clip(arr, self.action_space.low, self.action_space.high)
        return Action(arr)

    def _InContainer_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        block, container = objects

        block_x = state.get(block, "pose_x")
        block_y = state.get(block, "pose_y")
        block_w = state.get(block, "width")
        block_h = state.get(block, "height")
        block_yaw = state.get(block, "yaw")
        rect_block = utils.Rectangle(block_x, block_y, block_w, block_h, block_yaw)

        rect_container_a = utils.Rectangle(self.container_x, self.container_y, self.container_width_a, self.container_height_a, 0)
        rect_container_b = utils.Rectangle(self.container_x, self.container_y, self.container_width_b, self.container_height_b, 0)

        return ((rect_container_a.contains_point(*(rect_block.vertices[0])) and 
                 rect_container_a.contains_point(*(rect_block.vertices[1])) and 
                 rect_container_a.contains_point(*(rect_block.vertices[2])) and 
                 rect_container_a.contains_point(*(rect_block.vertices[3]))) or (
                 rect_container_b.contains_point(*(rect_block.vertices[0])) and 
                 rect_container_b.contains_point(*(rect_block.vertices[1])) and 
                 rect_container_b.contains_point(*(rect_block.vertices[2])) and 
                 rect_container_b.contains_point(*(rect_block.vertices[3]))))