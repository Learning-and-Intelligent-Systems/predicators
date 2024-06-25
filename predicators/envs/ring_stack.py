"""Ring stacking domain.

This environment is used to test more advanced sampling methods,
specifically for grasping. The environment consists of stacking
rings on a pole which has a radius that is barely less than the
inner radius of the rings.
"""

import json
import logging
from pathlib import Path
from typing import ClassVar, Collection, Dict, List, Optional, Sequence, Set, \
    Tuple, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib import patches

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    Object, Predicate, State, Type


class RingStackEnv(BaseEnv):
    """Ring stacking domain."""

    # Parameters that aren't important enough to need to clog up settings.py
    table_height: ClassVar[float] = 0.2
    # The table x bounds are (1.1, 1.6),
    x_lb: ClassVar[float] = 1.2
    x_ub: ClassVar[float] = 1.4
    # The table y bounds are (0.3, 1.2)
    y_lb: ClassVar[float] = 0.5
    y_ub: ClassVar[float] = 1.0

    pick_z: ClassVar[float] = 0.5  # Q: Maybe picking height for blocks?

    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = pick_z

    # Q: Error tolerances maybe for sampling?
    held_tol: ClassVar[float] = 0.5
    pick_tol: ClassVar[float] = 0.0001
    on_tol: ClassVar[float] = 0.01
    around_tol: ClassVar[float] = 0.1

    collision_padding: ClassVar[float] = 2.0  # Q: Variable to explore

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types #TODO
        self._ring_type = Type("ring", [
            "pose_x", "pose_y", "pose_z", "held"  # Q: Maybe add orientations in the future
        ])

        # pose taken from center of base of pole
        self._pole_type = Type("pole", [
            "pose_x", "pose_y", "pose_z"
        ])

        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "pose_z", "orn_x", "orn_y", "orn_z", "orn_w", "fingers"])
        # Predicates
        self._On = Predicate("On", [self._ring_type, self._ring_type],
                             self._On_holds)
        self._OnTable = Predicate("OnTable", [self._ring_type],
                                  self._OnTable_holds)
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding", [self._ring_type],
                                  self._Holding_holds)

        self._Around = Predicate("Around", [self._ring_type, self._pole_type],
                                 self._Around_holds)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)

        # Hyperparameters  Q: TODO
        self._ring_size = CFG.ring_size  # should be larger than ring_radius
        self._ring_height = CFG.ring_height
        self._num_rings_train = 100
        self._num_rings_test = 50

        self._pole_base_height = CFG.pole_base_height
        self._pole_height = CFG.pole_height

        # Hyperparameters from CFG. # TODO
        # self._block_size = CFG.blocks_block_size
        # self._num_blocks_train = CFG.blocks_num_blocks_train
        # self._num_blocks_test = CFG.blocks_num_blocks_test

    @classmethod
    def get_name(cls) -> str:
        return "ring_stack"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        x, y, z, fingers = action.arr

        # Infer which transition function to follow
        if fingers < 0.5:
            logging.info("transition pick")
            return self._transition_pick(state, x, y, z)
        if z < self.table_height + self._ring_height:
            logging.info("transition put on table")
            return self._transition_putontable(state, x, y, z)
        logging.info("transition around pole")
        return self._transition_around_pole(state, x, y, z)

    def _transition_pick(self, state: State, x: float, y: float,
                         z: float) -> State:
        next_state = state.copy()
        # Can only pick if fingers are open
        if not self._GripperOpen_holds(state, [self._robot]):
            logging.info("no gripper open")
            return next_state
        ring = self._get_rings_at_xyz(state, x, y, z)
        if ring is None:  # no ring at this pose
            logging.info("no ring at pose")
            return next_state

        # Execute pick
        next_state.set(ring, "pose_x", x)
        next_state.set(ring, "pose_y", y)
        next_state.set(ring, "pose_z", self.pick_z)
        next_state.set(ring, "held", 1.0)
        next_state.set(self._robot, "fingers", 0.0)  # close fingers

        return next_state

    def _transition_putontable(self, state: State, x: float, y: float,
                               z: float) -> State:
        next_state = state.copy()
        # Can only putontable if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            logging.info("no gripper open")
            return next_state
        ring = self._get_held_ring(state)
        assert ring is not None
        # Check that table surface is clear at this pose
        poses = [[
            state.get(r, "pose_x"),
            state.get(r, "pose_y"),
            state.get(r, "pose_z")
        ] for r in state if r.is_instance(self._ring_type)]
        existing_xys = {(float(p[0]), float(p[1])) for p in poses}
        if not self._table_xy_is_clear(x, y, existing_xys):
            return next_state
        # Execute putontable
        next_state.set(ring, "pose_x", x)
        next_state.set(ring, "pose_y", y)
        next_state.set(ring, "pose_z", z)
        next_state.set(ring, "held", 0.0)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers

        return next_state

    def _transition_around_pole(self, state: State, x: float, y: float,
                                z: float) -> State:
        next_state = state.copy()
        # Can only put around pole if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            logging.info("no gripper open")
            return next_state

        # check ring exists
        ring = self._get_held_ring(state)
        assert ring is not None

        # check pole exists
        pole = self._get_pole(state)
        assert pole is not None

        # Execute put around pole by snapping into place
        cur_x = state.get(pole, "pose_x")
        cur_y = state.get(pole, "pose_y")
        cur_z = state.get(pole, "pose_z")
        next_state.set(ring, "pose_x", cur_x)
        next_state.set(ring, "pose_y", cur_y)
        next_state.set(ring, "pose_z", cur_z + self._pole_base_height)
        next_state.set(ring, "held", 0.0)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers

        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnTable, self._GripperOpen, self._Holding, self._Around
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Around}

    @property
    def types(self) -> Set[Type]:
        return {self._ring_type, self._pole_type, self._robot_type}

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, fingers]
        lowers = np.array([self.x_lb, self.y_lb, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 10.0, 1.0], dtype=np.float32)
        return Box(lowers, uppers)

    def _get_tasks(self, num_tasks: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            logging.info("SAMPLING TASK!")
            while True:  # repeat until goal is not satisfied
                init_state, ring, pole = self._sample_state(rng)
                goal = self._sample_goal([ring, pole])
                if not all(goal_atom.holds(init_state) for goal_atom in goal):
                    break
            tasks.append(EnvironmentTask(init_state, goal))
        return tasks

    def _sample_state(self, rng: np.random.Generator) -> State:
        data: Dict[Object, Array] = {}
        existing_xys = set()
        # Create pole state
        pole_x, pole_y = self._sample_initial_xy(rng, existing_xys)
        logging.info(f"Pole_x: {pole_x}, Pole_y: {pole_y}")
        existing_xys.add((pole_x, pole_y))
        pole_z = self.table_height + self._pole_base_height * 0.5
        pole = Object(f"pole", self._pole_type)
        data[pole] = np.array([pole_x, pole_y, pole_z])

        ring_x, ring_y, = self._sample_initial_xy(rng, existing_xys)
        existing_xys.add((ring_x, ring_y))
        ring_z = self.table_height + self._ring_height * 0.5
        ring = Object(f"ring", self._ring_type)
        data[ring] = np.array([ring_x, ring_y, ring_z, 0.0])
        logging.info(f"ring_x,y,z: {[ring_x, ring_y, ring_z, 0.0]}")
        # [pose_x, pose_y, pose_z, fingers]
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = 1.0  # fingers start out open
        data[self._robot] = np.array([rx, ry, rz, 0, 0, 0, 1, rf], dtype=np.float32)
        return State(data), ring, pole

    def _sample_goal(self, objects: Sequence[Object], ) -> Set[GroundAtom]:
        ring, pole, = objects
        # Create goal from piles
        goal_atoms = set()
        goal_atoms.add(GroundAtom(self._Around, [ring, pole]))
        goal_atoms.add(GroundAtom(self._GripperOpen, [self._robot]))
        return goal_atoms

    def _sample_initial_xy(
            self, rng: np.random.Generator,
            existing_xys: Set[Tuple[float, float]]) -> Tuple[float, float]:
        while True:
            x = rng.uniform(self.x_lb, self.x_ub)
            y = rng.uniform(self.y_lb, self.y_ub)
            if self._table_xy_is_clear(x, y, existing_xys):
                return (x, y)

    def _table_xy_is_clear(self, x: float, y: float,
                           existing_xys: Set[Tuple[float, float]]) -> bool:
        if all(
                abs(x - other_x) > self.collision_padding * self._ring_size
                for other_x, _ in existing_xys):
            return True
        if all(
                abs(y - other_y) > self.collision_padding * self._ring_size
                for _, other_y in existing_xys):
            return True
        return False

    def _ring_is_clear(self, ring: Object, state: State) -> bool:
        return self._Clear_holds(state, [ring])

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        ring1, ring2 = objects
        if state.get(ring1, "held") >= self.held_tol or \
                state.get(ring2, "held") >= self.held_tol:
            return False
        x1 = state.get(ring1, "pose_x")
        y1 = state.get(ring1, "pose_y")
        z1 = state.get(ring1, "pose_z")
        x2 = state.get(ring2, "pose_x")
        y2 = state.get(ring2, "pose_y")
        z2 = state.get(ring2, "pose_z")
        return np.allclose([x1, y1, z1], [x2, y2, z2 + self._ring_height],
                           atol=self.on_tol)

    # Q: Might need some modifying
    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        ring, = objects
        z = state.get(ring, "pose_z")
        desired_z = self.table_height + self._ring_height * 0.5

        return (state.get(ring, "held") < self.held_tol) and \
            (
                    desired_z - self.on_tol - self._pole_base_height * 0.5 < z < desired_z + self.on_tol + self._pole_base_height * 0.5)

    @staticmethod
    def _GripperOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        rf = state.get(robot, "fingers")
        assert rf in (0.0, 1.0)
        return rf == 1.0

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        ring, = objects
        return self._get_held_ring(state) == ring

    def _Around_holds(self, state: State, objects: Sequence[Object]):
        ring, pole, = objects
        pole_x = state.get(pole, "pose_x")
        pole_y = state.get(pole, "pose_y")
        pole_z = state.get(pole, "pose_z")

        ring_x = state.get(ring, "pose_x")
        ring_y = state.get(ring, "pose_y")
        ring_z = state.get(ring, "pose_z")

        point_in_circle = (pole_x - ring_x) ** 2 + (pole_y - ring_y) ** 2 <= self._ring_size - self._ring_height
        correct_height = pole_z < ring_z <= pole_z + self._pole_height - self._ring_height
        logging.info("Checking around holds")
        logging.info(f'pole in ring?: {point_in_circle}')
        logging.info(f'ring: {[ring_x, ring_y]}, pole: {[pole_x, pole_y]}')

        return ((pole_x - ring_x) ** 2 + (
                pole_y - ring_y) ** 2 <= self._ring_size - self._ring_height) and correct_height

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        if self._Holding_holds(state, objects):
            return False
        ring, = objects
        for other_ring in state:
            if other_ring.type != self._ring_type:
                continue
            if self._On_holds(state, [other_ring, ring]):
                return False
        return True

    def _get_held_ring(self, state: State) -> Optional[Object]:
        for ring in state:
            if not ring.is_instance(self._ring_type):
                continue
            if state.get(ring, "held") >= self.held_tol:
                return ring
        return None

    def _get_pole(self, state: State) -> Optional[Object]:
        for pole in state:
            if pole.is_instance(self._pole_type):
                return pole
        return None

    def _get_rings_at_xyz(self, state: State, x: float, y: float,
                          z: float) -> Optional[Object]:
        close_rings = []
        for ring in state:
            if not ring.is_instance(self._ring_type):
                continue
            ring_pose = np.array([
                state.get(ring, "pose_x"),
                state.get(ring, "pose_y"),
                state.get(ring, "pose_z")
            ])
            if np.allclose([x, y, z], ring_pose, atol=self.pick_tol):
                dist = np.linalg.norm(np.array([x, y, z]) - ring_pose)
                close_rings.append((ring, float(dist)))
        if not close_rings:
            return None

        return min(close_rings, key=lambda x: x[1])[0]  # min distance

    # Q: TODO Might remove
    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        raise NotImplementedError

        # with open(json_file, "r", encoding="utf-8") as f:
        #     task_spec = json.load(f)
        # # Create the initial state from the task spec.
        # # One day, we can make the block size a feature of the blocks, but
        # # for now, we'll just make sure that the block size in the real env
        # # matches what we expect in sim.
        # assert np.isclose(task_spec["block_size"], self._block_size)
        # state_dict: Dict[Object, Dict[str, float]] = {}
        # id_to_obj: Dict[str, Object] = {}  # used in the goal construction
        # for block_id, block_spec in task_spec["blocks"].items():
        #     block = Object(block_id, self._block_type)
        #     id_to_obj[block_id] = block
        #     x, y, z = block_spec["position"]
        #     # Make sure that the block is in bounds.
        #     if not (self.x_lb <= x <= self.x_ub and \
        #             self.y_lb <= y <= self.y_ub and \
        #             self.table_height <= z):
        #         logging.warning("Block out of bounds in initial state!")
        #     r, g, b = block_spec["color"]
        #     state_dict[block] = {
        #         "pose_x": x,
        #         "pose_y": y,
        #         "pose_z": z,
        #         "held": 0,
        #         "color_r": r,
        #         "color_b": b,
        #         "color_g": g,
        #     }
        # # Add the robot at a constant initial position.
        # rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        # rf = 1.0  # fingers start out open
        # state_dict[self._robot] = {
        #     "pose_x": rx,
        #     "pose_y": ry,
        #     "pose_z": rz,
        #     "fingers": rf,
        # }
        # init_state = utils.create_state_from_dict(state_dict)
        # # Create the goal from the task spec.
        # if "goal" in task_spec:
        #     goal = self._parse_goal_from_json(task_spec["goal"], id_to_obj)
        # elif "language_goal" in task_spec:
        #     goal = self._parse_language_goal_from_json(
        #         task_spec["language_goal"], id_to_obj)
        # else:
        #     raise ValueError("JSON task spec must include 'goal'.")
        # env_task = EnvironmentTask(init_state, goal)
        # assert not env_task.task.goal_holds(init_state)
        # return env_task

    def _get_language_goal_prompt_prefix(self, object_names: Collection[str]) -> str:
        raise NotImplementedError

    def get_event_to_action_fn(self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:
        raise NotImplementedError

    def render_state_plt(self, state: State, task: EnvironmentTask, action: Optional[Action] = None,
                         caption: Optional[str] = None) -> matplotlib.figure.Figure:
        r = self._ring_size * 0.5  # block radius

        width_ratio = max(
            1. / 5,
            min(
                5.,  # prevent from being too extreme
                (self.y_ub - self.y_lb) / (self.x_ub - self.x_lb)))
        fig, (xz_ax, yz_ax) = plt.subplots(
            1,
            2,
            figsize=(20, 8),
            gridspec_kw={'width_ratios': [1, width_ratio]})
        xz_ax.set_xlabel("x", fontsize=24)
        xz_ax.set_ylabel("z", fontsize=24)
        xz_ax.set_xlim((self.x_lb - 2 * r, self.x_ub + 2 * r))
        xz_ax.set_ylim((self.table_height, r * 16 + 0.1))
        yz_ax.set_xlabel("y", fontsize=24)
        yz_ax.set_ylabel("z", fontsize=24)
        yz_ax.set_xlim((self.y_lb - 2 * r, self.y_ub + 2 * r))
        yz_ax.set_ylim((self.table_height, r * 16 + 0.1))

        rings = [o for o in state if o.is_instance(self._ring_type)]
        held = "None"
        for ring in sorted(rings):
            x = state.get(ring, "pose_x")
            y = state.get(ring, "pose_y")
            z = state.get(ring, "pose_z")
            # RGB values are between 0 and 1.
            color = (1, 0, 0)
            if state.get(ring, "held") > self.held_tol:
                assert held == "None"
                held = f"{ring.name}"

            # xz axis
            xz_rect = patches.Rectangle((x - r, z - r),
                                        2 * r,
                                        2 * r,
                                        zorder=-y,
                                        linewidth=1,
                                        edgecolor='black',
                                        facecolor=color)
            xz_ax.add_patch(xz_rect)

            # yz axis
            yz_rect = patches.Rectangle((y - r, z - r),
                                        2 * r,
                                        2 * r,
                                        zorder=-x,
                                        linewidth=1,
                                        edgecolor='black',
                                        facecolor=color)
            yz_ax.add_patch(yz_rect)

        title = f"Held: {held}"
        if caption is not None:
            title += f"; {caption}"
        plt.suptitle(title, fontsize=24, wrap=True)
        plt.tight_layout()
        return fig
