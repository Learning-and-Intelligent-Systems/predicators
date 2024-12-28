"""python predicators/main.py --approach oracle --env pybullet_grow --seed 1 \

--num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    Object, Predicate, State, Type


class PyBulletGrowEnv(PyBulletEnv):
    """A simple PyBullet environment involving two cups (red and blue) and two
    jugs (red and blue). Each jug contains infinite liquid. Pouring from a jug
    of matching color into a cup increases the 'growth' in that cup.

    We want the 'growth' of both cups to exceed some threshold as a goal.
    from PyBullet Coffee domain.
    x: cup <-> jug,
    y: robot <-> machine
    z: up <-> down
    """

    # Workspace bounds. You can adjust these as you wish.
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = 0.2
    z_ub: ClassVar[float] = 0.75
    init_padding = 0.05

    # robot config
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2
    tilt_lb: ClassVar[float] = robot_init_tilt
    tilt_ub: ClassVar[float] = tilt_lb - np.pi / 4

    # jug configs
    jug_height: ClassVar[float] = 0.12
    red_jug_x = 0.66
    red_jug_y = 1.32
    blue_jug_x = 1
    blue_jug_y = 1.38

    # Target height for the plants
    growth_height: ClassVar[float] = 0.3

    # Hard-coded finger states for open/close
    open_fingers: ClassVar[float] = 0.4
    closed_fingers: ClassVar[float] = 0.1
    # Tolerance for "close enough" to pick up or hold
    grasp_tol: ClassVar[float] = 1e-2
    place_jug_tol: ClassVar[float] = 1e-3
    # How much we pour if we tilt near the max angle
    pour_rate: ClassVar[float] = 0.1

    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -38  # lower
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Define Types:
        self._robot_type = Type("robot",
                                ["x", "y", "z", "fingers", "tilt", "wrist"])
        self._cup_type = Type("cup", ["x", "y", "z", "growth", "color"])
        self._jug_type = Type("jug",
                              ["x", "y", "z", "is_held", "rot", "color"])

        # Define Objects; temp (we will create them in tasks).
        self._robot = Object("robot", self._robot_type)
        self._red_cup = Object("red_cup", self._cup_type)
        self._blue_cup = Object("blue_cup", self._cup_type)
        self._red_jug = Object("red_jug", self._jug_type)
        self._blue_jug = Object("blue_jug", self._jug_type)

        # Define Predicates
        self._Grown = Predicate("Grown", [self._cup_type], self._Grown_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._jug_type],
                                  self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._OnTable = Predicate("OnTable", [self._jug_type],
                                  self._OnTable_holds)
        self._SameColor = Predicate("SameColor",
                                    [self._cup_type, self._jug_type],
                                    self._SameColor_holds)

        # TODO: These two can also be simplified
        self._cup_to_liquid_id: Dict[Object, Optional[int]] = {}

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_grow"

    @property
    def predicates(self) -> Set[Predicate]:
        # We only define Grown in this simple example, but you could
        # define Holding, JugPickedUp, etc.
        return {
            self._Grown, self._Holding, self._HandEmpty, self._OnTable,
            self._SameColor
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Grown}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._cup_type, self._jug_type}

    # -------------------------------------------------------------------------
    # Environment Setup

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Create the PyBullet environment and the robot.

        Subclasses typically add additional environment assets here.
        """
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # You can add a table or floor plane, etc.
        # For example, let's add a table below the workspace:
        table_id = create_object(asset_path="urdf/table.urdf",
                                    position=(0.75, 1.35, 0.0),
                                    orientation=p.getQuaternionFromEuler(
                                        [0., 0., np.pi / 2]),
                                    scale=1.0
                                    use_fixed_base=True,
                                    physics_client_id=physics_client_id,
                                )
        bodies["table_id"] = table_id

        # TODO: Create the pots and jugs here
        num_cups = 2
        num_jugs = 2
        cup_ids = []
        for i in range(num_cups):
            cup_id = create_object(asset_path="urdf/pot-pixel.urdf",
                                    color=(1, 0.3, 0.3, 1) if i == 0 else \
                                        (0.3, 0.3, 1, 1),
                                    physics_client_id=physics_client_id)
            cup_ids.append(cup_id)
        bodies["cup_ids"] = cup_ids

        jug_ids = []
        for i in range(num_jugs):
            jug_id = create_object(asset_path="urdf/jug-pixel.urdf",
                                    orientation=p.getQuaternionFromEuler(
                                        [0.0, 0.0, -np.pi / 2]),
                                    color=(1, 0, 0, 1) if i == 0 else (0, 0, 1, 1),
                                    physics_client_id=physics_client_id)
            jug_ids.append(jug_id)
        bodies["jug_ids"] = jug_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to PyBullet IDs for environment assets."""
        # Update self._obj_id_to_obj here; these are used for labeling objects
        # in rendering. Alternatively, we could just keep a set of objects,
        # and store IDs in object instances themselves.
        self._red_cup.id = pybullet_bodies["cup_ids"][0]
        self._blue_cup.id = pybullet_bodies["cup_ids"][1]
        self._red_jug.id = pybullet_bodies["jug_ids"][0]
        self._blue_jug.id = pybullet_bodies["jug_ids"][1]

    # -------------------------------------------------------------------------
    # State Management: Get, (Re)Set, Update
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return IDs of jugs (since we can only hold jugs)."""
        assert self._red_jug.id is not None and self._blue_jug.id is not None
        return [self._red_jug.id, self._blue_jug.id]

    def _get_state(self) -> State:
        """Create a State object from the current PyBullet simulation.

        We read off robot state, plus the cups/jugs positions, plus any
        relevant features (like 'growth').
        """
        state_dict: Dict[Object, Dict[str, float]] = {}

        # 1) Robot
        rx, ry, rz, qx, qy, qz, qw, rf = self._pybullet_robot.get_state()
        # Convert orientation back into tilt/wrist
        _, tilt, wrist = p.getEulerFromQuaternion([qx, qy, qz, qw])
        # We store the robot's features
        state_dict[self._robot] = {
            "x": rx,
            "y": ry,
            "z": rz,
            "fingers": self._fingers_joint_to_state(rf),
            "tilt": tilt,
            "wrist": wrist
        }

        # 2) For each cup and jug, we would have loaded them in _reset_state.
        for body_id, obj in self._obj_id_to_obj.items():
            if obj.type == self._cup_type:
                (cx, cy, cz), _ = p.getBasePositionAndOrientation(
                    body_id, physicsClientId=self._physics_client_id)

                # No liquid object is created if the current liquid is 0.
                if self._cup_to_liquid_id.get(obj, None) is not None:
                    liquid_id = self._cup_to_liquid_id[obj]
                    liquid_height = p.getVisualShapeData(
                        liquid_id,
                        physicsClientId=self._physics_client_id,
                    )[0][3][2]  # Get the height of the cuboidal plant
                    current_growth = liquid_height
                else:
                    current_growth = 0.0

                color_val = 1.0 if "red" in obj.name else 2.0  # arbitrary code
                state_dict[obj] = {
                    "x": cx,
                    "y": cy,
                    "z": cz,
                    "growth": current_growth,
                    "color": color_val
                }

            elif obj.type == self._jug_type:
                (jx, jy, jz), orn = p.getBasePositionAndOrientation(
                    body_id, physicsClientId=self._physics_client_id)
                # is_held we track from constraints
                is_held = 1.0 if (body_id == self._held_obj_id) else 0.0
                rot = utils.wrap_angle(
                    p.getEulerFromQuaternion(orn)[2] + np.pi / 2)
                color_val = 1.0 if "red" in obj.name else 2.0
                state_dict[obj] = {
                    "x": jx,
                    "y": jy,
                    "z": jz,
                    "is_held": is_held,
                    "rot": rot,
                    "color": color_val
                }

        # Convert the dictionary to a State.
        state = utils.create_state_from_dict(state_dict)
        joint_positions = self._pybullet_robot.get_joints()
        pyb_state = utils.PyBulletState(
            state.data, simulator_state={"joint_positions": joint_positions})
        return pyb_state

    def _reset_state(self, state: State) -> None:
        """Called whenever we do reset() or simulate() on a new state that
        differs from the environment's current state.

        We must reflect that state in PyBullet.
        """
        super()._reset_state(state)  # Clears constraints, resets robot

        # new reset cups and jugs
        for cup_obj in [self._red_cup, self._blue_cup]:
            cx = state.get(cup_obj, "x")
            cy = state.get(cup_obj, "y")
            cz = state.get(cup_obj, "z")
            update_object(cup_obj.id,
                          position=(cx, cy, cz),
                          physics_client_id=self._physics_client_id)

        # Create liquid in cups.
        for liquid_id in self._cup_to_liquid_id.values():
            if liquid_id is not None:
                p.removeBody(liquid_id,
                             physicsClientId=self._physics_client_id)
        self._cup_to_liquid_id.clear()

        for cup in state.get_objects(self._cup_type):
            liquid_id = self._create_pybullet_liquid_for_cup(cup, state)
            self._cup_to_liquid_id[cup] = liquid_id

        # Re-create jugs in PyBullet
        for jug_obj in [self._red_jug, self._blue_jug]:
            jx = state.get(jug_obj, "x")
            jy = state.get(jug_obj, "y")
            jz = state.get(jug_obj, "z")
            update_object(jug_obj.id,
                          position=(jx, jy, jz),
                          orientation=p.getQuaternionFromEuler(
                              [0.0, 0.0, -np.pi / 2]),
                          physics_client_id=self._physics_client_id)

        # Check if either jug is held => forcibly attach constraints.
        # (Though in our tasks we often start is_held=0.)
        # for jug_body_id, jug_obj in zip(self._jug_ids,
        #                                 [self._red_jug, self._blue_jug]):
        #     if state.get(jug_obj, "is_held") > 0.5:
        #         # Create a constraint with the robot's gripper
        #         self._create_grasp_constraint_for_object(jug_body_id)

        # After re-adding objects, compare with the new state
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.warning("Could not reconstruct state exactly!")

    # -------------------------------------------------------------------------
    # Custom pouring logic: if we see the robot has a jug at max tilt over a cup
    # of matching color, we increase cup growth.

    def step(self, action: Action, render_obs: bool = False) -> State:
        """We let the parent class handle the robot stepping & constraints.

        Then, we post-process: if the robot is tilting a jug over a
        matching-color cup, we increase the cup's growth in, 
        and update the environment accordingly.
        """
        # breakpoint()
        next_state = super().step(action, render_obs=render_obs)

        # If a jug is in the robot's hand, and tilt is large,
        # check if over a cup
        if self._held_obj_id is not None:
            # Which jug is being held?
            jug_obj = self._obj_id_to_obj[self._held_obj_id]
            # Check tilt
            tilt = next_state.get(self._robot, "tilt")
            if abs(tilt - np.pi / 4) < 0.1:
                # We assume that means "pouring"
                # Find if there's a cup below that is color-matched
                _ = next_state.get(jug_obj,
                                   "color")  # 1.0 for red, 2.0 for blue
                # We can do a simple proximity check in XY
                jug_x = next_state.get(jug_obj, "x")
                jug_y = next_state.get(jug_obj, "y")
                for _, cup_obj in [(cid, obj)
                                   for cid, obj in self._obj_id_to_obj.items()
                                   if obj.type == self._cup_type]:
                    cx = next_state.get(cup_obj, "x")
                    cy = next_state.get(cup_obj, "y")
                    dist = np.hypot(jug_x - cx, jug_y - cy)
                    logging.debug(f"Dist to cup {cup_obj.name}: {dist}")
                    if dist < 0.13:  # "over" the cup
                        # cup_color = next_state.get(cup_obj, "color")
                        # if abs(cup_color - jug_color) < 0.1:
                        # Color match => increase growth
                        current_growth = next_state.get(cup_obj, "growth")
                        new_growth = min(1.0, current_growth + self.pour_rate)
                        old_liquid_id = self._cup_to_liquid_id[cup_obj]
                        if old_liquid_id is not None:
                            p.removeBody(
                                old_liquid_id,
                                physicsClientId=self._physics_client_id)
                        next_state.set(cup_obj, "growth", new_growth)
                        self._cup_to_liquid_id[cup_obj] =\
                            self._create_pybullet_liquid_for_cup(cup_obj,
                                                                 next_state)

        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    # -------------------------------------------------------------------------
    # Predicates

    @staticmethod
    def _Grown_holds(state: State, objects: Sequence[Object]) -> bool:
        """A cup is "grown" if 'growth' > growth_height."""
        cup, = objects
        growth = state.get(cup, "growth")
        return growth > PyBulletGrowEnv.growth_height

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, jug = objects
        return state.get(jug, "is_held") > 0.5

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.2

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        jug, = objects
        if self._Holding_holds(state, [self._robot, jug]):
            return False
        return True

    def _SameColor_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        cup, jug = objects
        return state.get(cup, "color") == state.get(jug, "color")

    # -------------------------------------------------------------------------
    # Task Generation
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int,
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng
        tasks = []
        for _ in range(num_tasks):
            # Initialize random positions for cups & jugs
            # red_cup_x = rng.uniform(self.x_lb, self.x_ub)
            # red_cup_y = rng.uniform(self.y_lb, self.y_ub)
            # blue_cup_x = rng.uniform(self.x_lb, self.x_ub)
            # blue_cup_y = rng.uniform(self.y_lb, self.y_ub)
            # red_jug_x = rng.uniform(self.x_lb, self.x_ub)
            # red_jug_y = rng.uniform(self.y_lb, self.y_ub)
            # blue_jug_x = rng.uniform(self.x_lb, self.x_ub)
            # blue_jug_y = rng.uniform(self.y_lb, self.y_ub)
            red_cup_x = 0.75
            red_cup_y = 1.44
            blue_cup_x = 0.5
            blue_cup_y = 1.3

            # Robot at center
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist
            }

            # Cup initial
            red_cup_dict = {
                "x": red_cup_x,
                "y": red_cup_y,
                "z": self.z_lb + self.jug_height / 2,
                "growth": 0.0,  # empty plant
                "color": 1.0  # red
            }
            blue_cup_dict = {
                "x": blue_cup_x,
                "y": blue_cup_y,
                "z": self.z_lb + self.jug_height / 2,
                "growth": 0.0,
                "color": 2.0  # blue
            }

            # Jug initial
            red_jug_dict = {
                "x": self.red_jug_x,
                "y": self.red_jug_y,
                "z": self.z_lb + self.jug_height / 2,
                "is_held": 0.0,  # not in hand
                "rot": 0.0,
                "color": 1.0
            }
            blue_jug_dict = {
                "x": self.blue_jug_x,
                "y": self.blue_jug_y,
                "z": self.z_lb + self.jug_height / 2,
                "is_held": 0.0,
                "rot": 0.0,
                "color": 2.0
            }

            init_dict = {
                self._robot: robot_dict,
                self._red_cup: red_cup_dict,
                self._blue_cup: blue_cup_dict,
                self._red_jug: red_jug_dict,
                self._blue_jug: blue_jug_dict,
            }

            init_state = utils.create_state_from_dict(init_dict)

            # The goal is for both cups to be "Grown"
            # i.e. growth > growth_height
            goal_atoms = {
                GroundAtom(self._Grown, [self._red_cup]),
                GroundAtom(self._Grown, [self._blue_cup]),
                GroundAtom(self._OnTable, [self._red_jug]),
                GroundAtom(self._OnTable, [self._blue_jug]),
            }
            tasks.append(EnvironmentTask(init_state, goal_atoms))
        return self._add_pybullet_state_to_tasks(tasks)
    
    # -------------------------------------------------------------------------
    # Helpers
    def _create_pybullet_liquid_for_cup(self, cup: Object,
                                        state: State) -> Optional[int]:
        current_liquid = state.get(cup, "growth")
        liquid_height = current_liquid
        liquid_half_extents = [0.03, 0.03, liquid_height / 2]
        if current_liquid == 0:
            return None
        cx = state.get(cup, "x")
        cy = state.get(cup, "y")
        cz = self.z_lb + current_liquid / 2

        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=liquid_half_extents,
            physicsClientId=self._physics_client_id)

        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=liquid_half_extents,
            rgbaColor=(0.35, 1, 0.3, 1.0),
            physicsClientId=self._physics_client_id)

        pose = (cx, cy, cz)
        orientation = self._default_orn
        return p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=collision_id,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=pose,
                                 baseOrientation=orientation,
                                 physicsClientId=self._physics_client_id)
