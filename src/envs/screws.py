"""Toy screw picking environment."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class ScrewsEnv(BaseEnv):
    """Toy screw picking env.

    In this environment, the agent controls a magnetic gripper that can
    be used to pick up screws of different colors/sizes. The challenge
    is for the agent to learn actions that don't precisely model all the
    irrelevant screws that might be picked up when picking a desired
    screw.
    """
    gripper_width: ClassVar[float] = 0.5
    _receptacle_width: ClassVar[float] = 0.6
    _receptacle_height: ClassVar[float] = 0.2
    _screw_width: ClassVar[float] = 0.05
    _screw_height: ClassVar[float] = 0.05
    _magnetic_field_dist: ClassVar[float] = 0.5
    _num_screw_clusters: ClassVar[int] = 3
    # Reachable zone boundaries.
    rz_x_lb = -5.0
    rz_x_ub = 5.0
    rz_y_lb = 0.0
    rz_y_ub = 5.0

    def __init__(self) -> None:
        super().__init__()
        self.num_screws_train = CFG.screws_num_screws_train
        self.num_screws_test = CFG.screws_num_screws_test
        self._rz_width = self.rz_x_ub - self.rz_x_lb
        self._rz_height = self.rz_y_ub - self.rz_y_lb

        # Types
        self._screw_type = Type("screw", ["pose_x", "pose_y", "held"])
        self._gripper_type = Type("gripper", ["pose_x", "pose_y"])
        self._receptacle_type = Type("receptacle", ["pose_x", "pose_y"])
        # Predicates.
        self._GripperCanPickScrew = Predicate(
            "GripperCanPickScrew", [self._gripper_type, self._screw_type],
            self._GripperCanPickScrew_holds)
        self._AboveReceptacle = Predicate(
            "AboveReceptacle", [self._gripper_type, self._receptacle_type],
            self._AboveReceptacle_holds)
        self._HoldingScrew = Predicate("HoldingScrew",
                                       [self._gripper_type, self._screw_type],
                                       self._HoldingScrew_holds)
        self._ScrewInReceptacle = Predicate(
            "ScrewInReceptacle", [self._screw_type, self._receptacle_type],
            self._ScrewInReceptacle_holds)

        # Options.
        self._MoveToScrew = utils.SingletonParameterizedOption(
            # variables: [robot, screw to pick up].
            # params: [].
            "MoveToScrew",
            self._MoveToScrew_policy,
            types=[self._gripper_type, self._screw_type])
        self._MoveToReceptacle = utils.SingletonParameterizedOption(
            # variables: [robot, receptacle].
            # params: [].
            "MoveToReceptacle",
            self._MoveToReceptacle_policy,
            types=[
                self._gripper_type, self._receptacle_type, self._screw_type
            ])
        self._MagnetizeGripper = utils.SingletonParameterizedOption(
            # variables: [robot].
            # params: [].
            "MagnetizeGripper",
            self._MagnetizeGripper_policy,
            types=[self._gripper_type])
        self._DemagnetizeGripper = utils.SingletonParameterizedOption(
            # variables: [robot].
            # params: [].
            "DemagnetizeGripper",
            self._DemagnetizeGripper_policy,
            types=[self._gripper_type])

        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._gripper_type)
        self._receptacle = Object("receptacle", self._receptacle_type)

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               possible_num_screws=self.num_screws_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               possible_num_screws=self.num_screws_test,
                               rng=self._test_rng)

    @classmethod
    def get_name(cls) -> str:
        return "screws"

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1)
        plt.xlim([
            self.rz_x_lb - 2 * self._screw_width,
            self.rz_x_ub + 2 * self._screw_width
        ])
        plt.ylim([self.rz_y_lb, self.rz_y_ub])

        # Draw receptacle.
        receptacle_x = state.get(self._receptacle, "pose_x")
        receptacle_y = state.get(self._receptacle, "pose_y")
        plt.plot([
            receptacle_x - self._receptacle_width / 2.0,
            receptacle_x + self._receptacle_width / 2.0
        ], [receptacle_y, receptacle_y],
                 color="green")

        # Draw the gripper.
        gripper_x = state.get(self._robot, "pose_x")
        gripper_y = state.get(self._robot, "pose_y")
        plt.plot([(gripper_x - self.gripper_width / 2.0) + self._screw_width,
                  (gripper_x + self.gripper_width / 2.0) + self._screw_width],
                 [gripper_y, gripper_y],
                 color="black")
        plt.plot([(gripper_x - self.gripper_width / 2.0) + self._screw_width,
                  (gripper_x - self.gripper_width / 2.0) + self._screw_width],
                 [gripper_y - self.gripper_width / 4.0, gripper_y],
                 color="black")
        plt.plot([(gripper_x + self.gripper_width / 2.0) + self._screw_width,
                  (gripper_x + self.gripper_width / 2.0) + self._screw_width],
                 [gripper_y - self.gripper_width / 4.0, gripper_y],
                 color="black")

        # Find which object is the goal screw.
        assert len(task.goal) == 1
        goal_pred = list(task.goal)[0]
        goal_screw = goal_pred.objects[0]

        # Draw screws.
        for screw in state.get_objects(self._screw_type):
            screw_x = state.get(screw, "pose_x")
            screw_y = state.get(screw, "pose_y")
            if screw == goal_screw:
                color = "gold"
            else:
                color = "grey"
            rect = plt.Rectangle((screw_x - self._screw_width / 2.0,
                                  screw_y - self._screw_height / 2.0),
                                 self._screw_width,
                                 self._screw_height,
                                 edgecolor=color,
                                 facecolor=color)
            ax.add_patch(rect)

        return fig

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._GripperCanPickScrew, self._AboveReceptacle,
            self._HoldingScrew, self._ScrewInReceptacle
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._ScrewInReceptacle}

    @property
    def types(self) -> Set[Type]:
        return {self._screw_type, self._gripper_type, self._receptacle_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {
            self._MoveToScrew, self._MoveToReceptacle, self._MagnetizeGripper,
            self._DemagnetizeGripper
        }

    @property
    def action_space(self) -> Box:
        # dimensions: [dx, dy, magnetized vs. unmagnetized].
        lowers = np.array([-self._rz_width, -self._rz_height, 0.0],
                          dtype=np.float32)
        uppers = np.array([self._rz_width, self._rz_height, 1.0],
                          dtype=np.float32)
        return Box(lowers, uppers)

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        dx, dy, magnetization = action.arr
        state_after_move = self._transition_move(state, dx, dy)
        if magnetization > 0.5:
            return self._transition_magnetize(state_after_move)
        return self._transition_demagnetize(state_after_move)

    def _get_tasks(self, num_tasks: int, possible_num_screws: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []

        for _ in range(num_tasks):
            # Sample center locations for each of the screw clusters.
            screw_cluster_xcenters = []
            for _ in range(self._num_screw_clusters):
                screw_cluster_xcenters.append(
                    rng.uniform(self.rz_x_lb,
                                self.rz_x_ub - self._receptacle_width))
            screw_name_to_pos: Dict[str, Tuple[float, float]] = {}
            num_screws = rng.choice(possible_num_screws)
            for si in range(num_screws):
                existing_xys = set(screw_name_to_pos.values())
                while True:
                    # Sample a random initial position for the screw
                    # such that it is on the ground, and near one of the
                    # screw clusters.
                    y_pos = self.rz_y_lb + self._screw_height / 2.0
                    screw_cluster_xcenter = rng.choice(screw_cluster_xcenters)
                    x_pos = rng.normal(screw_cluster_xcenter)
                    # if this position doesn't make this screw in collision
                    # with another screw, then put the screw here.
                    if self._surface_xy_is_valid(x_pos, y_pos, existing_xys):
                        screw_name_to_pos[f"screw_{si}"] = (x_pos, y_pos)
                        break
            tasks.append(self._get_single_task(screw_name_to_pos, rng))
        return tasks

    def _get_single_task(self, screw_name_to_pos: Dict[str, Tuple[float,
                                                                  float]],
                         rng: np.random.Generator) -> Task:
        state_dict = {}
        goal_atoms: Set[GroundAtom] = set()
        # Select a random screw that needs to be in the receptacle as the goal
        # GroundAtom.
        goal_screw_name = rng.choice(sorted(screw_name_to_pos))

        # Create screw objects.
        for screw_name, (x_pos, y_pos) in screw_name_to_pos.items():
            screw_obj = Object(screw_name, self._screw_type)
            state_dict[screw_obj] = {
                "pose_x": x_pos,
                "pose_y": y_pos,
                "held": 0.0,
            }

            # If the screw is the goal screw, then add it being in the
            # receptacle to the goal atoms.
            if screw_name == goal_screw_name:
                goal_atoms.add(
                    GroundAtom(self._ScrewInReceptacle,
                               [screw_obj, self._receptacle]))

        # Create receptacle object such that it is attached to the
        # right wall and halfway between the ceiling and floor.
        state_dict[self._receptacle] = {
            "pose_x": self.rz_x_ub - (self._receptacle_width / 2.0),
            "pose_y": (self.rz_y_ub - self.rz_y_lb) / 2.0,
        }

        # Create gripper object such that it is at the middle of
        # the screen.
        state_dict[self._robot] = {
            "pose_x": (self.rz_x_lb + self.rz_x_ub) / 2.0,
            "pose_y": (self.rz_y_lb + self.rz_y_ub) / 2.0
        }

        init_state = utils.create_state_from_dict(state_dict)
        assert len(goal_atoms) == 1
        task = Task(init_state, goal_atoms)
        return task

    def _surface_xy_is_valid(self, x: float, y: float,
                             existing_xys: Set[Tuple[float, float]]) -> bool:
        # First, check that there isn't another screw at this location.
        for (existing_x, existing_y) in existing_xys:
            if (abs(existing_x - x) < self._screw_width
                    and abs(existing_y - y) < self._screw_height):
                return False
        # Next, check that the location is valid.
        y_is_valid = (y == self.rz_y_lb + self._screw_height / 2.0)
        x_is_valid = (self.rz_x_lb + self._screw_width / 2.0) <= x <= (
            self.rz_x_ub - (self._receptacle_width + self._screw_width / 2.0))
        return x_is_valid and y_is_valid

    def _transition_move(self, state: State, dx: float, dy: float) -> State:
        next_state = state.copy()
        rx = state.get(self._robot, "pose_x")
        ry = state.get(self._robot, "pose_y")
        new_rx = rx + dx
        new_rx_min = new_rx - self.gripper_width / 2.0
        new_rx_max = new_rx + self.gripper_width / 2.0
        new_ry = ry + dy

        # If the new position is outside the boundaries, move till
        # the robot collides with the boundary.
        if new_rx_min < self.rz_x_lb:
            new_rx = self.rz_x_lb + (self.gripper_width / 2.0)
        elif new_rx_max > self.rz_x_ub:
            new_rx = self.rz_x_ub - (self.gripper_width / 2.0)
        if new_ry < self.rz_y_lb:
            new_ry = self.rz_y_lb
        elif new_ry > self.rz_y_ub:
            new_ry = self.rz_y_ub
        next_state.set(self._robot, "pose_x", new_rx)
        next_state.set(self._robot, "pose_y", new_ry)

        # Recompute dx and dy in case it's changed.
        dx = new_rx - rx
        dy = new_ry - ry

        # Find out which (if any) screws are currently being held by
        # the gripper and update their positions to follow the gripper.
        all_screws = state.get_objects(self._screw_type)
        for screw in all_screws:
            if self._HoldingScrew_holds(state, [self._robot, screw]):
                screw_x = state.get(screw, "pose_x")
                screw_y = state.get(screw, "pose_y")
                next_state.set(screw, "pose_x", screw_x + dx)
                next_state.set(screw, "pose_y", screw_y + dy)

        return next_state

    def _transition_magnetize(self, state: State) -> State:
        next_state = state.copy()
        ry = state.get(self._robot, "pose_y")
        all_screws = state.get_objects(self._screw_type)
        for screw in all_screws:
            if not self._HoldingScrew_holds(state, [self._robot, screw]):
                # Check if the gripper is in the right position to
                # pick up this screw, and if so, move the screw to
                # the gripper.
                if self._GripperCanPickScrew_holds(state,
                                                   [self._robot, screw]):
                    next_state.set(screw, "pose_y",
                                   ry - (self._screw_height / 2.0))
                    next_state.set(screw, "held", 1.0)

        return next_state

    def _transition_demagnetize(self, state: State) -> State:
        next_state = state.copy()
        all_screws = state.get_objects(self._screw_type)
        for screw in all_screws:
            if self._HoldingScrew_holds(state, [self._robot, screw]):
                if self._AboveReceptacle_holds(
                        state, [self._robot, self._receptacle]):
                    receptacle_y = state.get(self._receptacle, "pose_y")
                    next_state.set(screw, "pose_y",
                                   receptacle_y + (self._screw_height / 2.0))
                else:
                    next_state.set(screw, "pose_y",
                                   self.rz_y_lb + (self._screw_height / 2.0))
                next_state.set(screw, "held", 0.0)

        return next_state

    def _GripperCanPickScrew_holds(self, state: State,
                                   objects: Sequence[Object]) -> bool:
        gripper, screw = objects
        gripper_y = state.get(gripper, "pose_y")
        screw_y = state.get(screw, "pose_y")
        screw_max_y = screw_y + self._screw_height / 2.0

        gripper_center_pose = state.get(gripper, "pose_x")
        gripper_min_x = gripper_center_pose - self.gripper_width / 2.0
        gripper_max_x = gripper_center_pose + self.gripper_width / 2.0
        screw_x = state.get(screw, "pose_x")
        screw_min_x = screw_x - (self._screw_width / 2.0)
        screw_max_x = screw_x + (self._screw_width / 2.0)

        return screw_max_y <= gripper_y and \
            gripper_y - screw_max_y < self._magnetic_field_dist and \
            gripper_min_x < screw_max_x < gripper_max_x and \
            gripper_min_x < screw_min_x < gripper_max_x

    def _AboveReceptacle_holds(self, state: State,
                               objects: Sequence[Object]) -> bool:
        gripper, receptacle = objects
        gripper_y = state.get(gripper, "pose_y")
        receptacle_y = state.get(receptacle, "pose_y")

        gripper_center_pose = state.get(gripper, "pose_x")
        gripper_min_x = gripper_center_pose - self.gripper_width / 2.0
        gripper_max_x = gripper_center_pose + self.gripper_width / 2.0
        receptacle_x = state.get(receptacle, "pose_x")
        receptacle_min_x = receptacle_x - self._receptacle_width / 2.0
        receptacle_max_x = receptacle_x + self._receptacle_width / 2.0

        return gripper_y > receptacle_y and \
            receptacle_min_x < gripper_min_x and \
            receptacle_max_x > gripper_max_x

    @staticmethod
    def _HoldingScrew_holds(state: State, objects: Sequence[Object]) -> bool:
        _, screw = objects
        return state.get(screw, "held") > 0.5

    def _ScrewInReceptacle_holds(self, state: State,
                                 objects: Sequence[Object]) -> bool:
        screw, receptacle = objects

        screw_center_pose = state.get(screw, "pose_x")
        screw_min_x = screw_center_pose - self._screw_width / 2.0
        screw_max_x = screw_center_pose + self._screw_width / 2.0
        receptacle_x = state.get(receptacle, "pose_x")
        receptacle_max_x = receptacle_x + self._receptacle_width / 2.0
        receptacle_min_x = receptacle_x - self._receptacle_width / 2.0

        screw_held = state.get(screw, "held")
        return screw_held <= 0.5 and screw_min_x > receptacle_min_x \
            and screw_max_x < receptacle_max_x

    def _MoveToScrew_policy(self, state: State, memory: Dict,
                            objects: Sequence[Object],
                            params: Array) -> Action:
        """Policy to return an action that moves to a position above a
        particular screw such that the screw can be grasped."""
        del memory, params  # unused.
        _, screw = objects
        screw_x = state.get(screw, "pose_x")
        screw_y = state.get(screw, "pose_y")

        target_x = screw_x
        target_y = screw_y + (self._screw_height /
                              2.0) + (self._magnetic_field_dist / 2.0)

        current_x = state.get(self._robot, "pose_x")
        current_y = state.get(self._robot, "pose_y")

        return Action(
            np.array([target_x - current_x, target_y - current_y, 0.0],
                     dtype=np.float32))

    def _MoveToReceptacle_policy(self, state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
        """Policy to return an action that moves to a position above the
        receptacle."""
        del memory, params  # unused.
        _, receptacle, _ = objects
        receptacle_x = state.get(receptacle, "pose_x")
        receptacle_y = state.get(receptacle, "pose_y")

        target_x = receptacle_x
        target_y = receptacle_y + (self._magnetic_field_dist)

        current_x = state.get(self._robot, "pose_x")
        current_y = state.get(self._robot, "pose_y")

        return Action(
            np.array([target_x - current_x, target_y - current_y, 1.0],
                     dtype=np.float32))

    @staticmethod
    def _MagnetizeGripper_policy(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
        """Policy to return an action that magnetizes the gripper at its
        current position."""
        del state, memory, objects, params  # unused.
        return Action(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    @staticmethod
    def _DemagnetizeGripper_policy(state: State, memory: Dict,
                                   objects: Sequence[Object],
                                   params: Array) -> Action:
        """Policy to return an action that demagnetizes the gripper at its
        current position."""
        del state, memory, objects, params  # unused.
        return Action(np.array([0.0, 0.0, 0.0], dtype=np.float32))
