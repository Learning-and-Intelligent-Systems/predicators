"""Four rooms environment based on the original options paper."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Image, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class FourRoomsEnv(BaseEnv):
    """An environment where a 2D robot must navigate between rooms that are
    connected by hallways to reach a target room.

    The robot is rectangular. The size of that rectangle varies between
    tasks in a way that sometimes requires the robot to rotate before
    moving through hallways. This setup is inspired by the situation where
    a person (or dog) is carrying a long stick and is trying to move through
    a narrow passage.

    The action space is 1D, which rotates the robot. At each time step, the
    rotation is applied first, and then the robot moves forward by a constant
    displacement.

    For consistency with plt.Rectangle, all rectangles are oriented at their
    bottom left corner.
    """
    room_size: ClassVar[float] = 1.0
    hallway_length: ClassVar[float] = 0.2
    wall_width: ClassVar[float] = 0.01
    robot_min_length: ClassVar[float] = 0.1
    robot_max_length: ClassVar[float] = 0.3
    robot_width: ClassVar[float] = 0.05
    action_magnitude: ClassVar[float] = 0.05

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._robot_type = Type("robot", ["x", "y", "rot", "length"])
        self._room_type = Type("room", ["x", "y", "hall_top", "hall_bottom", "hall_left", "hall_right"])
        # Predicates
        self._InRoom = Predicate("InRoom",
                                  [self._robot_type, self._room_type],
                                  self._InRoom_holds)
        # Options
        self._Move = ParameterizedOption(
            "Move",
            types=[self._robot_type, self._room_type, self._room_type],
            # The parameter is the angle of rotation with which the robot
            # will pass through the hallway between the rooms.
            params_space=Box(-np.pi, np.pi, (1, )),
            policy=self._Move_policy,
            initiable=self._Move_initiable,
            terminal=self._Move_terminal)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "four_rooms"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        drot, = action.arr
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        rot = state.get(self._robot, "rot")
        new_rot = rot + drot
        new_x = x + np.cos(new_rot) * self.action_magnitude
        new_y = y + np.sin(new_rot) * self.action_magnitude
        next_state = state.copy()
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)
        next_state.set(self._robot, "rot", new_rot)
        # Check for collisions.
        if self._state_has_collision(next_state):
            return state.copy()
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InRoom}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InRoom}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._room_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Move}

    @property
    def action_space(self) -> Box:
        # An angle in radians. Constrained to prevent very large movements.
        return Box(-np.pi / 10, np.pi / 10, (1, ))

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        background_color = "white"
        # Draw robot.
        robot_color = "blue"
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        rot = state.get(self._robot, "rot")
        angle = rot * 180 / np.pi
        l = state.get(self._robot, "length")
        w = self.robot_width
        rect = patches.Rectangle((x, y), l, w, angle, color=robot_color)
        ax.add_patch(rect)
        if action:
            arrow_width = self.robot_width / 10
            drot, = action.arr
            dx = np.cos(rot + drot) * self.action_magnitude
            dy = np.sin(rot + drot) * self.action_magnitude
            ax.arrow(x, y, dx, dy, width=arrow_width, color="black")

        # Draw rooms.
        wall_color = "black"
        for room in state.get_objects(self._room_type):
            # Draw walls.
            x = state.get(room, "x")
            y = state.get(room, "y")
            room_size = self.room_size
            wall_width = self.wall_width
            has_top_hall = state.get(room, "hall_top")
            has_bottom_hall = state.get(room, "hall_bottom")
            has_left_hall = state.get(room, "hall_left")
            has_right_hall = state.get(room, "hall_right")
            # (center x, center y, length, width, has_hall)
            wall_params = [
                # Top wall.
                (x, y + room_size, room_size, wall_width, has_top_hall),
                # Bottom wall.
                (x, y, room_size, wall_width, has_bottom_hall),
                # Left wall.
                (x, y, wall_width, room_size, has_left_hall),
                # Right wall.
                (x + room_size, y, wall_width, room_size, has_right_hall),
            ]
            for (x, y, l, w, has_hall) in wall_params:
                # Draw wall.
                rect = patches.Rectangle((x, y), l, w, color=wall_color)
                ax.add_patch(rect)
                # Draw hallway.
                if has_hall:
                    # Determine if vertical or horizontal.
                    if abs(l) > abs(w):
                        hall_l = self.hallway_length
                        hall_w = 5 * w
                        center_x = x + l / 2
                        hall_x = center_x - self.hallway_length / 2
                        hall_y = y - w
                    else:
                        hall_l = 4 * l
                        hall_w = self.hallway_length
                        center_y = y + w / 2
                        hall_y = center_y - self.hallway_length / 2
                        hall_x = x - l
                    rect = patches.Rectangle((hall_x, hall_y), hall_l, hall_w, color=background_color)
                    ax.add_patch(rect)

            

        x_lb, x_ub, y_lb, y_ub = self._get_world_boundaries()
        pad = 1.1 * self.wall_width
        ax.set_xlim(x_lb - pad, x_ub + pad)
        ax.set_ylim(y_lb - pad, y_ub + pad)

        assert caption is None
        # plt.axis("off")
        plt.tight_layout()
        img = utils.fig2data(fig)
        plt.close()
        return [img]

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        tasks: List[Task] = []
        # Create the static parts of the initial state.
        top_left_room = Object("top_left_room", self._room_type)
        top_right_room = Object("top_right_room", self._room_type)
        bottom_left_room = Object("bottom_left_room", self._room_type)
        bottom_right_room = Object("bottom_right_room", self._room_type)
        init_state = utils.create_state_from_dict({
            self._robot: {
                "x": np.inf,  # will get overriden
                "y": np.inf,  # will get overriden
                "rot": np.inf,  # will get overriden
                "length": np.inf,  # will get overriden
            },
            top_left_room: {
                "x": -self.room_size,
                "y": 0,
                "hall_top": 0,
                "hall_bottom": 1,
                "hall_left": 0,
                "hall_right": 1,
            },
            top_right_room: {
                "x": 0,
                "y": 0,
                "hall_top": 0,
                "hall_bottom": 1,
                "hall_left": 1,
                "hall_right": 0,
            },
            bottom_left_room: {
                "x": -self.room_size,
                "y": -self.room_size,
                "hall_top": 1,
                "hall_bottom": 0,
                "hall_left": 0,
                "hall_right": 1,
            },
            bottom_right_room: {
                "x": 0,
                "y": -self.room_size,
                "hall_top": 1,
                "hall_bottom": 0,
                "hall_left": 1,
                "hall_right": 0,
            },
        })
        x_lb, x_ub, y_lb, y_ub = self._get_world_boundaries()
        rot_lb = -np.pi
        rot_ub = np.pi
        length_lb = self.robot_min_length
        length_ub = self.robot_max_length
        rooms = sorted(init_state.get_objects(self._room_type))
        while len(tasks) < num:
            # Sample an initial state.
            state = init_state.copy()
            x = rng.uniform(x_lb, x_ub)
            y = rng.uniform(y_lb, y_ub)
            rot = rng.uniform(rot_lb, rot_ub)
            length = rng.uniform(length_lb, length_ub)
            state.set(self._robot, "x", x)
            state.set(self._robot, "y", y)
            state.set(self._robot, "rot", rot)
            state.set(self._robot, "length", length)
            # Make sure the state is collision-free.
            if self._state_has_collision(state):
                continue
            # Sample a goal.
            goal_room_idx = rng.choice(len(rooms))
            goal_room = rooms[goal_room_idx]
            goal_atom = GroundAtom(self._InRoom, [self._robot, goal_room])
            goal = {goal_atom}
            # Make sure goal is not satisfied.
            if not goal_atom.holds(state):
                tasks.append(Task(state, goal))
        return tasks

    @staticmethod
    def _Move_policy(state: State, memory: Dict, objects: Sequence[Object],
                     params: Array) -> Action:
        del memory  # unused
        import ipdb; ipdb.set_trace()
        return Action(np.array([rot], dtype=np.float32))

    def _Move_initiable(self, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        robot, start_room, _ = objects
        return self._InRoom_holds(state, [robot, start_room])

    def _Move_terminal(self, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        robot, _, target_room = objects
        return self._InRoom_holds(state, [robot, target_room])

    def _InRoom_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, room = objects
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        room_x = state.get(room, "x")
        room_y = state.get(room, "y")
        return room_x < robot_x < room_x + self.room_size and \
               room_y < robot_y < room_y + self.room_size

    def _state_has_collision(self, state: State) -> bool:
        # TODO
        return False

    def _get_world_boundaries(self) -> Tuple[float, float, float, float]:
        x_lb = -self.room_size
        x_ub = self.room_size
        y_lb = -self.room_size
        y_ub = self.room_size
        return x_lb, x_ub, y_lb, y_ub
