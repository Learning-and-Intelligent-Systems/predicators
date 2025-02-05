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


def get_ring_outer_radius(major_radius, minor_radius):
    return major_radius + minor_radius


def get_ring_inner_radius(major_radius, minor_radius):
    return major_radius - minor_radius


class RingStackEnv(BaseEnv):
    """Ring stacking domain."""

    # Parameters that aren't important enough to need to clog up settings.py
    table_height: ClassVar[float] = 0.2
    # The table x bounds are (1.1, 1.6),
    x_lb: ClassVar[float] = 1.175
    x_ub: ClassVar[float] = 1.5
    # The table y bounds are (0.3, 1.2)
    y_lb: ClassVar[float] = 0.375
    y_ub: ClassVar[float] = 1.125

    pick_z: ClassVar[float] = 0.5  # Q: Maybe picking height for blocks?

    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = 0.7
    finger_length = 0.05

    # Q: Error tolerances maybe for sampling?
    held_tol: ClassVar[float] = 0.5
    pick_tol: ClassVar[float] = 0.0001
    on_tol: ClassVar[float] = 0.01
    around_tol: ClassVar[float] = 0.1
    small_ring_radius: ClassVar[float] = 0.037

    collision_padding: ClassVar[float] = 2.0  # Q: Variable to explore

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types #TODO
        self._ring_type = Type("ring", [
            "pose_x", "pose_y", "pose_z", "id", "major_radius", "minor_radius", "held"
            # Q: Maybe add orientations in the future
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
        self._num_rings_train = 100
        self._num_rings_test = 50

        self._pole_base_height = CFG.pole_base_height
        self._pole_height = CFG.pole_height
        self._pole_radius = CFG.pole_radius
        self._pole_base_radius = CFG.pole_base_radius
        self._max_rings = CFG.ring_stack_max_num_rings
        self._id_to_geometry = {}
        # Hyperparameters from CFG. # TODO
        # self._block_size = CFG.blocks_block_size
        # self._num_blocks_train = CFG.blocks_num_blocks_train
        # self._num_blocks_test = CFG.blocks_num_blocks_test

    @classmethod
    def get_name(cls) -> str:
        return "ring_stack"

    @classmethod
    def find_intersection_with_circle(cls, circle_center, radius, start_point, direction_vector):
        # Unpack variables
        x_c, y_c = circle_center
        x_0, y_0 = start_point
        v_x, v_y = direction_vector

        # Normalize the direction vector
        direction_norm = np.linalg.norm(direction_vector)
        v_x, v_y = v_x / direction_norm, v_y / direction_norm

        # Translate start point to circle's frame
        delta_x = x_0 - x_c
        delta_y = y_0 - y_c

        # Quadratic coefficients
        B = 2 * (delta_x * v_x + delta_y * v_y)
        C = delta_x ** 2 + delta_y ** 2 - radius ** 2

        # Solve the quadratic equation for t
        discriminant = B ** 2 - 4 * C
        if discriminant < 0:
            return None  # No intersection

        # Find the smaller positive t
        t1 = (-B + np.sqrt(discriminant)) / 2
        t2 = (-B - np.sqrt(discriminant)) / 2
        t = min(t1, t2) if min(t1, t2) > 0 else max(t1, t2)

        if t < 0:
            return None  # Intersection is behind the start point

        # Calculate intersection point
        intersection_x = x_0 + t * v_x
        intersection_y = y_0 + t * v_y

        return np.array([intersection_x, intersection_y])

    def simulate(self, state: State, action: Action) -> State:
        logging.info(action.arr[:4])
        assert self.action_space.contains(action.arr[:4])
        if len(action.arr) == 8:
            x, y, z, fingers, finger1_x, finger1_y, finger2_x, finger2_y = action.arr
        else:
            x, y, z, fingers = action.arr

        # Infer which transition function to follow
        if fingers < 0.5:
            logging.info("transition pick")
            return self._transition_pick(state, x, y, z, [finger1_x, finger1_y, finger2_x, finger2_y])

        if z < self.table_height + CFG.ring_max_tubular_radius + self._pole_base_height:
            logging.info("transition put on table")
            return self._transition_putontable(state, x, y, z)

        logging.info("transition around pole")
        return self._transition_around_pole(state, x, y, z)

    def _transition_pick(self, state: State, x: float, y: float,
                         z: float, finger_positions: list[float]) -> State:
        next_state = state.copy()
        # Can only pick if fingers are open
        if not self._GripperOpen_holds(state, [self._robot]):
            logging.info("no gripper open")
            return next_state
        ring = self._get_rings_at_xyz(state, x, y, z)
        if ring is None:  # no ring at this pose
            logging.info("no ring at pose")
            return next_state

        pole = None
        for obj in state:
            if obj.is_instance(self._pole_type):
                pole = obj

        pole_x, pole_y = state.get(pole, "pose_x"), state.get(pole, "pose_y")

        ring_outer_radius = get_ring_outer_radius(
            state.get(ring, "major_radius"),
            state.get(ring, "minor_radius"))

        ring_inner_radius = get_ring_inner_radius(
            state.get(ring, "major_radius"),
            state.get(ring, "minor_radius"))

        gripper_location_in_ring = ((x - state.get(ring, "pose_x")) ** 2 + (
                y - state.get(ring, "pose_y")) ** 2) ** 0.5

        finger1_location_in_pole_base = ((finger_positions[0] - pole_x) ** 2 + (
                finger_positions[1] - pole_y) ** 2) ** 0.5

        finger2_location_in_pole_base = ((finger_positions[2] - pole_x) ** 2 + (
                finger_positions[3] - pole_y) ** 2) ** 0.5

        if finger1_location_in_pole_base <= self._pole_base_radius + 0.015 \
            or finger2_location_in_pole_base <= self._pole_base_radius + 0.015:
            return next_state


        finger1_location_in_ring = ((finger_positions[0] - state.get(ring, "pose_x")) ** 2 + (
                finger_positions[1] - state.get(ring, "pose_y")) ** 2) ** 0.5

        finger2_location_in_ring = ((finger_positions[2] - state.get(ring, "pose_x")) ** 2 + (
                finger_positions[3] - state.get(ring, "pose_y")) ** 2) ** 0.5


        finger_ring_pick_buffer = 0.002 if ring_inner_radius < self.small_ring_radius - 0.005 else 0.01


        finger1_in_ring = finger1_location_in_ring < ring_inner_radius - finger_ring_pick_buffer
        finger2_in_ring = finger2_location_in_ring < ring_inner_radius - finger_ring_pick_buffer


        finger_to_finger_vector = np.array([finger_positions[2]-finger_positions[0],finger_positions[3]-finger_positions[1]])

        position_on_ring_outer_edge = self.find_intersection_with_circle((state.get(ring, "pose_x"),state.get(ring, "pose_y")),
                                                                   ring_outer_radius,
                                                                   (finger_positions[0],finger_positions[1]),
                                                                   finger_to_finger_vector)

        position_on_ring_inner_edge = self.find_intersection_with_circle(
            (state.get(ring, "pose_x"), state.get(ring, "pose_y")),
            ring_inner_radius,
            (finger_positions[0], finger_positions[1]),
            finger_to_finger_vector)

        logging.info(f"finger positions: {finger_positions}")
        logging.info(f"ring outer positions: {position_on_ring_outer_edge}")
        logging.info(f"ring inner positions: {position_on_ring_inner_edge}")

        try:
            distance_finger_ring_inner_2 = ((finger_positions[2] - position_on_ring_inner_edge[0])**2 + (finger_positions[3] - position_on_ring_inner_edge[1])**2)**0.5

            distance_finger_ring_outer_2 = ((finger_positions[2] - position_on_ring_outer_edge[0]) ** 2 + (
                        finger_positions[3] - position_on_ring_outer_edge[1]) ** 2) ** 0.5

            distance_finger_ring_inner_1 = ((finger_positions[0] - position_on_ring_inner_edge[0]) ** 2 + (
                        finger_positions[1] - position_on_ring_inner_edge[1]) ** 2) ** 0.5
            distance_finger_ring_outer_1 = ((finger_positions[0] - position_on_ring_outer_edge[0]) ** 2 + (
                    finger_positions[1] - position_on_ring_outer_edge[1]) ** 2) ** 0.5

            distance_finger_ring_1 = min(distance_finger_ring_inner_1, distance_finger_ring_outer_1)
            distance_finger_ring_2 = min(distance_finger_ring_inner_2, distance_finger_ring_outer_2)
        except:
            logging.info("Error in distance calculations, grasp unlikely. Skipping.")
            return next_state

        if distance_finger_ring_1 <= distance_finger_ring_2:
            position_ring_edge = position_on_ring_inner_edge

        else:
            position_ring_edge = position_on_ring_inner_edge

        position_ring_edge_xy_diff = np.array((x-position_ring_edge[0], y-position_ring_edge[1])) * 0.5
        new_ring_x = position_ring_edge_xy_diff[0] + state.get(ring, "pose_x")
        new_ring_y = position_ring_edge_xy_diff[1] + state.get(ring, "pose_y")

        logging.info(f"small ring?: {ring_outer_radius <= self.small_ring_radius}")
        logging.info(f"ring_outer_radius: {ring_outer_radius}")
        logging.info(f"finger1_in_ring: {finger1_in_ring}")
        logging.info(f"finger2_in_ring: {finger2_in_ring}")
        logging.info(f"position_on_ring_edge: {position_ring_edge}")

        logging.info(f"CHECKING RING OUTER RADIUS: {ring_outer_radius}")
        logging.info(f"CHECKING RING INNER RADIUS: {ring_inner_radius}")

        logging.info(f'finger1_location_in_ring: {finger1_location_in_ring}')
        logging.info(f'finger2_location_in_ring: {finger2_location_in_ring}')

        logging.info(f'old_ring_x: {state.get(ring, "pose_x")}')
        logging.info(f'old_ring_y: {state.get(ring, "pose_y")}')
        logging.info(f"new_ring_x: {new_ring_x}")
        logging.info(f"new_ring_y: {new_ring_y}")
        logging.info(f"gripper_location: {x,y}")

        valid_grip = ((ring_outer_radius < self.small_ring_radius and gripper_location_in_ring < 0.005 and
                       finger1_location_in_ring > ring_outer_radius + 0.002 and
                       finger2_location_in_ring > ring_outer_radius + 0.002)
                      or (ring_outer_radius >= self.small_ring_radius and ((finger1_in_ring and not finger2_in_ring)  or (not finger1_in_ring and finger2_in_ring))))


        for other_ring in state:
            if not other_ring.is_instance(self._ring_type) or other_ring == ring:
                continue

            other_ring_outer_radius = get_ring_outer_radius(
                state.get(other_ring, "major_radius"),
                state.get(other_ring, "minor_radius"))

            other_ring_inner_radius = get_ring_inner_radius(
                state.get(other_ring, "major_radius"),
                state.get(other_ring, "minor_radius"))

            finger1_location_in_other_ring = ((finger_positions[0] - state.get(other_ring, "pose_x")) ** 2 + (
                    finger_positions[1] - state.get(other_ring, "pose_y")) ** 2) ** 0.5

            finger2_location_in_other_ring = ((finger_positions[2] - state.get(other_ring, "pose_x")) ** 2 + (
                    finger_positions[3] - state.get(other_ring, "pose_y")) ** 2) ** 0.5

            finger1_in_other_ring = finger1_location_in_other_ring < ring_outer_radius + 0.015
            finger2_in_other_ring = finger2_location_in_other_ring < ring_outer_radius + 0.015

            if finger1_in_other_ring or finger2_in_other_ring:
                valid_grip = False
                logging.info("Overlapping rings in grasp")
                break


        if not valid_grip:
            logging.info("invalid grasp")
            return next_state
        else:
            logging.info("valid grasp found")

        if ring_outer_radius < self.small_ring_radius:
            new_ring_x, new_ring_y = (x,y)

        # Execute pick
        next_state.set(ring, "pose_x", new_ring_x)
        next_state.set(ring, "pose_y", new_ring_y)
        next_state.set(ring, "pose_z", self.pick_z)
        next_state.set(ring, "held", 1.0)
        next_state.set(self._robot, "fingers", 0.0)  # close fingers
        next_state.set(self._robot, "pose_x", x)
        next_state.set(self._robot, "pose_y", y)
        next_state.set(self._robot, "pose_z", self.pick_z)

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

        rings  = [(
            state.get(r, "pose_x"),
            state.get(r, "pose_y"),
            get_ring_outer_radius(state.get(r, "major_radius"),state.get(r, "minor_radius")),
        ) for r in state if r.is_instance(self._ring_type) and r != ring]

        ring_circle = (state.get(ring, "pose_x"),
                       state.get(ring, "pose_y"),
                       get_ring_outer_radius(state.get(ring, "major_radius"),
                                             state.get(ring, "minor_radius")))

        if not self._table_xy_is_clear(ring_circle, rings, padding=0.015):
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

        snap_z = 0

        # check ring exists
        ring = self._get_held_ring(state)
        assert ring is not None

        # check pole exists
        pole = self._get_pole(state)
        assert pole is not None

        # Execute put around pole by snapping into place
        pole_x = state.get(pole, "pose_x")
        pole_y = state.get(pole, "pose_y")
        pole_z = state.get(pole, "pose_z")

        current_r_pose = np.array([state.get(self._robot, "pose_x"),
                                   state.get(self._robot, "pose_y"),
                                   state.get(self._robot, "pose_z")])

        new_r_pose = np.array([x, y, z])

        xy_diff_vector = new_r_pose[:2] - current_r_pose[:2]

        current_ring_pose = np.array([state.get(ring, "pose_x"),
                                      state.get(ring, "pose_y"),
                                      state.get(ring, "pose_z")])

        # Get ring's new pose given gripper's new pose
        new_ring_pose = np.r_[current_ring_pose[:2] + xy_diff_vector, new_r_pose[2]].astype(np.float32)


        # check if a ring is already around pole
        otherring = self._get_highest_ring_below(state, pole_x, pole_y, pole_z + self._pole_height)
        if otherring is not None:
            otherring_z = state.get(otherring, "pose_z")
            snap_z = otherring_z + state.get(otherring, "minor_radius") * CFG.ring_height_modifier + \
                     state.get(ring, "minor_radius") * CFG.ring_height_modifier

            logging.info("transition stack from transition around pole")
        else:
            snap_z = pole_z + self._pole_base_height

        # if larger ring, else small ring
        # Was 0.021
        finger_account = 0.021 if get_ring_outer_radius(
            state.get(ring, "major_radius"),
            state.get(ring, "minor_radius")) >= self.small_ring_radius else 0.002

        # if larger ring with small inner radius
        if get_ring_inner_radius(state.get(ring, "major_radius"),
            state.get(ring, "minor_radius")) <= self.small_ring_radius - 0.005 and get_ring_outer_radius(
            state.get(ring, "major_radius"),
            state.get(ring, "minor_radius")) >= self.small_ring_radius:
            finger_account = 0.015

        logging.info(f"finger account: {finger_account}")

        # Check if pole is inside the ring
        if ((pole_x - new_ring_pose[0]) ** 2 + (
                pole_y - new_ring_pose[1]) ** 2) ** 0.5 + self._pole_radius >= get_ring_inner_radius(
            state.get(ring, "major_radius"),
            state.get(ring, "minor_radius")) - finger_account:
            logging.info(f"pole outside ring")
            return next_state

        else:
            logging.info(f"Pole inside ring!")


        next_state.set(ring, "pose_x", new_ring_pose[0])
        next_state.set(ring, "pose_y", new_ring_pose[1])
        next_state.set(ring, "pose_z", snap_z)
        next_state.set(ring, "held", 0.0)
        next_state.set(self._robot, "fingers", 1.0)  # open
        next_state.set(self._robot, "pose_x", new_r_pose[0])
        next_state.set(self._robot, "pose_y", new_r_pose[1])
        next_state.set(self._robot, "pose_z", self.pick_z)


        logging.info(f"NEW RING POSE: {new_ring_pose}")
        logging.info(f"xydiff: {xy_diff_vector}")
        logging.info(f"OLD ROBOT POSE: {current_r_pose}")
        logging.info(f"NEW ROBOT POSE: {new_r_pose}")

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
        return {self._Around, self._OnTable}

    @property
    def types(self) -> Set[Type]:
        return {self._ring_type, self._pole_type, self._robot_type}

    @classmethod
    def get_action_space(cls) -> Box:
        lowers = np.array([cls.x_lb, cls.y_lb, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([cls.x_ub, cls.y_ub, 10.0, 1.0], dtype=np.float32)
        return Box(lowers, uppers)

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
                init_state, rings, pole = self._sample_state(rng)
                goal = self._sample_goal([rings, pole], init_state, rng)
                logging.info(f"task with goal: {goal} created")
                if not all(goal_atom.holds(init_state) for goal_atom in goal):
                    break

            tasks.append(EnvironmentTask(init_state, goal))
        return tasks

    def _sample_state(self, rng: np.random.Generator) -> State:
        data: Dict[Object, Array] = {}
        existing_circles = set()
        # Create pole state
        pole_x, pole_y = self._sample_initial_xy(rng, self._pole_base_radius, existing_circles, modifier=0.1)
        logging.info(f"Pole_x: {pole_x}, Pole_y: {pole_y}")
        existing_circles.add((pole_x, pole_y, self._pole_base_radius))
        pole_z = self.table_height + self._pole_base_height * 0.5
        pole = Object(f"pole", self._pole_type)
        data[pole] = np.array([pole_x, pole_y, pole_z])

        num_rings = int(rng.uniform(0, self._max_rings)) + 1
        rings = []
        for i in range(num_rings):
            task_ring_idx = int(rng.uniform(0, CFG.ring_dataset_size))
            major_radius, minor_radius = self.retrieve_geometry_data_from_obj(
                utils.get_env_asset_path(f"rings/ring_{task_ring_idx}.obj"))
            self._id_to_geometry[task_ring_idx] = (major_radius, minor_radius)

            radius = get_ring_outer_radius(major_radius, minor_radius)

            ring_x, ring_y, = self._sample_initial_xy(rng, radius, existing_circles,modifier=0.03)
            existing_circles.add((ring_x, ring_y, radius))
            ring_z = self.table_height + minor_radius * CFG.ring_height_modifier
            ring = Object(f"ring{i}", self._ring_type)
            data[ring] = np.array([ring_x, ring_y, ring_z, task_ring_idx, major_radius, minor_radius, 0.0])
            logging.info(f"{ring}_x,y,z: {[ring_x, ring_y, ring_z, task_ring_idx, major_radius, minor_radius, 0.0]}")
            rings.append(ring)

        # [pose_x, pose_y, pose_z, fingers]
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = 1.0  # fingers start out open
        data[self._robot] = np.array([rx, ry, rz, 0.7071, 0.7071, 0, 0, rf], dtype=np.float32)
        return State(data), rings, pole

    def _sample_goal(self, objects: Sequence[Object], init_state: State, rng: np.random.Generator) -> Set[GroundAtom]:
        rings, pole, = objects
        rings_around_pole = set()

        # choose which rings should be around pole, minimum 1
        first_ring_around_pole = rings[int(rng.uniform(0, len(rings)))]
        rings_around_pole.add(first_ring_around_pole)

        for ring in rings:
            if ring in rings_around_pole:
                continue

            # coin flip for other rings are around pole
            if rng.uniform(0, 1) < 0.5:
                rings_around_pole.add(ring)

        rings_around_pole = list(rings_around_pole)
        rings_stack_order = []

        while len(rings_around_pole) > 0:
            i = int(rng.uniform(0, len(rings_around_pole)))
            logging.info(i)
            rings_stack_order.append(rings_around_pole.pop(i))

        goal_atoms = set()

        i = 0
        while i < len(rings_stack_order):
            ring = rings_stack_order[i]
            if i == 0:
                goal_atoms.add(GroundAtom(self._Around, [ring, pole]))
                # goal_atoms.add(GroundAtom(self._OnTable, [ring]))
            else:
                # bottom_ring_outer_radius = get_ring_outer_radius(
                #     init_state.get(rings_stack_order[i - 1], "major_radius"),
                #     init_state.get(rings_stack_order[i - 1], "minor_radius"))
                #
                # bottom_ring_inner_radius = get_ring_inner_radius(
                #     init_state.get(rings_stack_order[i - 1], "major_radius"),
                #     init_state.get(rings_stack_order[i - 1], "minor_radius"))
                #
                # ring_inner_radius = get_ring_inner_radius(init_state.get(ring, "major_radius"),
                #                                           init_state.get(ring, "minor_radius"))
                #
                # ring_outer_radius = get_ring_outer_radius(init_state.get(ring, "major_radius"),
                #                                           init_state.get(ring, "minor_radius"))
                goal_atoms.add(GroundAtom(self._Around, [ring, pole]))
                # if ring_inner_radius > bottom_ring_outer_radius:
                #     goal_atoms.add(GroundAtom(self._Around, [ring, pole]))
                #     rings_stack_order.pop(i)
                #     i -= 1
                # elif ring_outer_radius < bottom_ring_inner_radius:
                #     goal_atoms.add(GroundAtom(self._Around, [ring, pole]))
                # else:
                #     goal_atoms.add(GroundAtom(self._On, [ring, rings_stack_order[i - 1]]))
                #     goal_atoms.add(GroundAtom(self._Around, [ring, pole]))
            i += 1

        # goal_atoms.add(GroundAtom(self._GripperOpen, [self._robot]))
        return goal_atoms

    def _sample_initial_xy(
            self, rng: np.random.Generator,
            circle_radius: float,
            existing_circles: Set[Tuple[float, float, float]],
            modifier: float= 1.0) -> Tuple[float, float]:
        while True:
            x = rng.uniform(self.x_lb+modifier, self.x_ub-modifier)
            y = rng.uniform(self.y_lb+modifier, self.y_ub-modifier)
            logging.info(f"sampling with {x},{y}")
            if self._table_xy_is_clear((x,y, circle_radius), existing_circles):
                return (x, y)

    def _table_xy_is_clear(self, circle: Tuple[float, float, float],
                           existing_circles: Set[Tuple[float, float, float]],
                           padding: float = 0.015) -> bool:
        '''
        Circle: [x, y, radius]
        '''

        for other_circle in existing_circles:
            d = ((circle[0]-other_circle[0])**2 + (circle[1]-other_circle[1])**2)**0.5
            if d <= abs(circle[2] - (other_circle[2]+padding)) or d <= (circle[2] + (other_circle[2]+padding)):
                return False

        return True

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        ring1, ring2 = objects
        if state.get(ring1, "held") >= self.held_tol or \
                state.get(ring2, "held") >= self.held_tol:
            return False

        if ring1 is ring2:
            return False

        x1 = state.get(ring1, "pose_x")
        y1 = state.get(ring1, "pose_y")
        z1 = state.get(ring1, "pose_z")
        x2 = state.get(ring2, "pose_x")
        y2 = state.get(ring2, "pose_y")
        z2 = state.get(ring2, "pose_z")

        On = np.allclose([x1, y1], [x2, y2],
                         atol=0.05) and np.allclose([z1], [z2 + CFG.ring_height_modifier * (
                state.get(ring2, "minor_radius") + state.get(ring1, "minor_radius"))], atol=0.035)

        logging.info(f"On holds: {On}")
        if not On:
            logging.info(([x1, y1, z1], [x2, y2, z2]))
            logging.info(([x1, y1, z1], [x2, y2, z2 + CFG.ring_height_modifier *
                                         (state.get(ring2, "minor_radius") + state.get(ring1, "minor_radius"))]))
        return On

    # Q: Might need some modifying
    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        ring, = objects
        z = state.get(ring, "pose_z")
        desired_z = self.table_height + state.get(ring, "minor_radius") * CFG.ring_height_modifier
        holds = (state.get(ring, "held") < self.held_tol) and \
                (
                        desired_z - self.on_tol < z < desired_z + self.on_tol + self._pole_base_height)
        if not holds:
            logging.info(f'on table false. desired_z: {desired_z}, z:{z}')
        else:
            logging.info(f'{ring} on table holds!')

        return holds

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

        pole_in_ring = ((pole_x - ring_x) ** 2 + (
                pole_y - ring_y) ** 2) ** 0.5 + self._pole_radius < get_ring_outer_radius(
            state.get(ring, "major_radius"),
            state.get(ring, "minor_radius"))

        correct_height = self.table_height + self._pole_base_height < ring_z <= pole_z + self._pole_height * 0.85

        logging.info(f'Around holds: {pole_in_ring and correct_height}')
        return pole_in_ring and correct_height

    def _get_highest_ring_below(self, state: State, x: float, y: float,
                                z: float) -> Optional[Object]:
        rings_here = []
        for ring in state:
            if not ring.is_instance(self._ring_type):
                continue
            ring_pose = np.array(
                [state.get(ring, "pose_x"),
                 state.get(ring, "pose_y")])
            ring_z = state.get(ring, "pose_z")
            logging.info(f'checking highest ring below x,y,z: {([x, y, z], [ring_pose, ring_z])}')
            if np.allclose([x, y], ring_pose, atol=0.02) and \
                    ring_z < z - self.pick_tol:
                rings_here.append((ring, ring_z))
        if not rings_here:
            return None
        return max(rings_here, key=lambda x: x[1])[0]  # highest z

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

            rx = state.get(ring, "pose_x"),
            ry = state.get(ring, "pose_y"),
            rz = state.get(ring, "pose_z")

            dist = ((x - rx) ** 2 + (y - ry) ** 2) ** 0.5

            if dist <= get_ring_outer_radius(state.get(ring, "major_radius"), state.get(ring, "minor_radius")):
                logging.info("CORRECT XY!")
                if not np.allclose([z], [rz], atol=state.get(ring, "minor_radius")):
                    logging.info(f"INCORRECT Z: {z}, {rz}")
                    logging.info(f'diff: {abs(z - rz)}, ring:height: {state.get(ring, "minor_radius")}')

            if dist <= get_ring_outer_radius(state.get(ring, "major_radius"), state.get(ring, "minor_radius")) and \
                    np.allclose([z], [rz], atol=self.pick_tol):
                close_rings.append((ring, float(dist)))

        if not close_rings:
            return None

        logging.info("RING AT XYZ found!")
        return min(close_rings, key=lambda x: x[1])[0]  # min distance

    def retrieve_geometry_data_from_obj(self, file_path):
        # Open the file and read the first line
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()

        # Check if the first line is a comment
        if first_line.startswith('#'):
            # Remove the '#' and split the comment by comma
            comment = first_line[1:].strip()
            values = comment.split(',')

            if len(values) == 2:
                major_radius = values[0].strip()
                minor_radius = values[1].strip()
                return float(major_radius), float(minor_radius)
            else:
                raise ValueError("Comment does not contain exactly two comma-separated values")
        else:
            raise ValueError("The first line is not a comment")

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
