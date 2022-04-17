"""An Atari Breakout environment."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from gym.wrappers import FrameStack
from numpy.typing import NDArray

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Image, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class BreakoutEnv(BaseEnv):
    """An Atari Breakout environment."""

    brick_height: ClassVar[int] = 6
    brick_width: ClassVar[int] = 8
    brick_top_row: ClassVar[int] = 57
    brick_left_col: ClassVar[int] = 8
    brick_num_rows: ClassVar[int] = 6
    brick_num_cols: ClassVar[int] = 18
    paddle_height: ClassVar[int] = 4
    paddle_width: ClassVar[int] = 16
    paddle_row: ClassVar[int] = 189
    ball_width: ClassVar[int] = 2
    ball_height: ClassVar[int] = 4
    side_wall_width: ClassVar[int] = 8
    top_panel_height: ClassVar[int] = 32

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._paddle_type = Type("paddle", ["c"])
        self._ball_type = Type("ball", ["r", "c", "dr", "dc"])
        self._brick_type = Type("brick", ["r", "c", "alive"])
        # Predicates
        self._BrickAlive = Predicate("BrickAlive", [self._brick_type],
                                     self._BrickAlive_holds)
        self._BrickDead = Predicate("BrickDead", [self._brick_type],
                                    self._BrickDead_holds)
        # Options
        # TODO
        # Static objects (always exist no matter the settings).
        self._paddle = Object("paddle", self._paddle_type)
        self._ball = Object("ball", self._ball_type)
        # Gym environment.
        self._gym_env = FrameStack(gym.make("BreakoutNoFrameskip-v0"), 2)

    @classmethod
    def get_name(cls) -> str:
        return "breakout"

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not supported for Gym envs.")

    def reset(self, train_or_test: str, task_idx: int) -> State:
        if train_or_test == "train":
            seed_offset = 0
        else:
            assert train_or_test == "test"
            seed_offset = CFG.test_env_seed_offset
        seed = task_idx + seed_offset
        self._current_obs = self._reset_initial_state_from_seed(seed)
        return self._observation_to_state(self._current_obs)

    def step(self, action: Action) -> State:
        # Actions are [0, 1, 2, 3] = ['NOOP', 'FIRE', 'RIGHT', 'LEFT'].
        continuous_action, = action.arr
        if continuous_action <= -0.5:
            gym_action = 3  # left
        elif continuous_action >= 0.5:
            gym_action = 2  # right
        else:
            assert -0.5 < continuous_action < 0.5
            gym_action = 0  # noop
        self._current_obs, _, _, _ = self._gym_env.step(gym_action)
        return self._observation_to_state(self._current_obs)

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, seed_offset=0)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               seed_offset=CFG.test_env_seed_offset)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._BrickAlive, self._BrickDead}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._BrickDead}

    @property
    def types(self) -> Set[Type]:
        return {self._paddle_type, self._ball_type, self._brick_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return set()

    @property
    def action_space(self) -> Box:
        # Move the paddle left or right. Magnitudes don't matter.
        return Box(-1, 1, (1, ), dtype=np.float32)

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        raise NotImplementedError("Render state not supported for Gym envs.")

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> List[Image]:
        assert caption is None
        del action  # unused

        assert self._current_obs.shape[0] == 2
        most_recent_obs = self._current_obs[1]

        # For debugging perception.
        if CFG.breakout_debug_render:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.set_xlim((0, most_recent_obs.shape[1]))
            ax.set_ylim((most_recent_obs.shape[0], 0))
            ax.imshow(most_recent_obs, alpha=0.5)

            state = self._observation_to_state(self._current_obs)
            for brick in state.get_objects(self._brick_type):
                if not self._BrickAlive_holds(state, [brick]):
                    continue
                r = state.get(brick, "r")
                c = state.get(brick, "c")
                rect = patches.Rectangle((c - 0.5, r - 0.5),
                                         self.brick_width,
                                         self.brick_height,
                                         linewidth=1,
                                         edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)
            r = self.paddle_row
            c = state.get(self._paddle, "c")
            rect = patches.Rectangle((c - 0.5, r - 0.5),
                                     self.paddle_width,
                                     self.paddle_height,
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            r = state.get(self._ball, "r")
            c = state.get(self._ball, "c")
            rect = patches.Rectangle((c - 0.5, r - 0.5),
                                     self.ball_width,
                                     self.ball_height,
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

            plt.tight_layout()
            img = utils.fig2data(fig)
            plt.close()
        else:
            img = most_recent_obs

        return [img]

    def _get_tasks(self, num: int, seed_offset: int) -> List[Task]:
        tasks = []
        for i in range(num):
            seed = i + seed_offset
            obs = self._reset_initial_state_from_seed(seed)
            init_state = self._observation_to_state(obs)
            bricks = init_state.get_objects(self._brick_type)
            goal = {GroundAtom(self._BrickDead, [b]) for b in bricks}
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int) -> NDArray[np.uint8]:
        self._gym_env.seed(seed)
        self._gym_env.reset()
        # Firing starts the game. Occasionally, we need to fire multiple
        # times to get it started (as detected by the ball appearing).
        while True:
            obs, _, _, _ = self._gym_env.step(1)
            init_state = self._observation_to_state(obs)
            # The ball has appeared.
            if init_state.get(self._ball, "r") >= 0:
                break
        return obs

    def _observation_to_state(self, obs: NDArray[np.uint8]) -> State:
        """Extract a State from a self._gym_env observation."""

        # Expecting two frames stacked together.
        assert len(obs.shape) == 4
        assert obs.shape[0] == 2

        state_dict = {}
        all_crop_bounds = []

        # Use the current frame to detect the bricks and paddle.
        frame = obs[1]

        # Start with the bricks.
        for brick_row in range(self.brick_num_rows):
            r = self.brick_top_row + self.brick_height * brick_row
            for brick_col in range(self.brick_num_cols):
                c = self.brick_left_col + self.brick_width * brick_col
                crop = frame[r:r + self.brick_height, c:c + self.brick_width]
                all_crop_bounds.append(
                    (r, r + self.brick_height, c, c + self.brick_width))
                alive = np.any(crop)
                name = f"brick{brick_row}-{brick_col}"
                brick = Object(name, self._brick_type)
                state_dict[brick] = {"r": r, "c": c, "alive": alive}

        # Add the paddle.
        left_pad = self.side_wall_width
        detection_line = frame[self.paddle_row, left_pad:].max(axis=-1)
        # The logical and here is to handle the case where the ball is in the
        # same row as the paddle.
        shift = self.ball_width + 1
        shifted_line = np.zeros_like(detection_line)
        shifted_line[:-shift] = detection_line[shift:]
        offset_c = np.argwhere(detection_line & shifted_line)[0].item()
        c = left_pad + offset_c
        all_crop_bounds.append(
            (self.paddle_row, self.paddle_row + self.paddle_height, c,
             c + self.paddle_width))
        state_dict[self._paddle] = {"c": c}

        # Add the ball.
        r0, c0 = self._frame_to_ball_position(obs[0], all_crop_bounds)
        r1, c1 = self._frame_to_ball_position(obs[1], all_crop_bounds)
        # Special case: we lost the ball.
        if r1 == -1:
            dr = 0
            dc = 0
        else:
            dr = r1 - r0
            dc = c1 - c0
        state_dict[self._ball] = {"r": r1, "c": c1, "dr": dr, "dc": dc}

        return utils.create_state_from_dict(state_dict)

    def _frame_to_ball_position(
        self, frame: NDArray[np.uint8],
        all_crop_bounds: Sequence[Tuple[int, int, int,
                                        int]]) -> Tuple[int, int]:
        ablated_frame = frame.copy()
        for (sr, er, sc, ec) in all_crop_bounds:
            ablated_frame[sr:er, sc:ec] = 0
        # Remove the walls.
        ablated_frame[:, :self.side_wall_width] = 0
        ablated_frame[:, -self.side_wall_width:] = 0
        ablated_frame[:self.top_panel_height] = 0
        # The ball should now be the only remaining colorful thing.
        colorful_idxs = np.argwhere(ablated_frame.max(-1))
        if not len(colorful_idxs):
            # We lost the ball!
            return -1, -1
        r, c = tuple(colorful_idxs[0])
        return r, c

    @staticmethod
    def _BrickAlive_holds(state: State, objects: Sequence[Object]) -> bool:
        brick, = objects
        return state.get(brick, "alive") > 0.5

    def _BrickDead_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        return not self._BrickAlive_holds(state, objects)
