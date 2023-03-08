"""A 2D continuous satellites domain loosely inspired by the IPC domain of the
same name.

There are some number of satellites, each carrying an instrument. The possible
instruments are: (1) a camera, (2) an infrared sensor, (3) a Geiger counter.
Additionally, each satellite may be able to shoot Chemical X and/or Chemical
Y. The satellites have a viewing cone within which they can see everything
that is not occluded. The goal is for specific satellites to take readings
of specific objects with calibrated instruments.

The interesting challenges in this domain come from 2 things. (1) Because
there are multiple satellites and also random objects floating around,
moving to a particular target will not guarantee that the satellite can
actually see this target. It may be that the target is occluded, and this will
not be modeled at the high level. (2) Some coordination amongst the satellites
may be necessary for certain readings. In particular, to get a camera reading
of a particular object, the object must first be shot with Chemical X; to get
an infrared reading, the object must first be shot with Chemical Y. Geiger
readings can be taken without any sort of chemical reaction.
"""

from typing import ClassVar, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class SatellitesEnv(BaseEnv):
    """A 2D continuous satellites domain loosely inspired by the IPC domain of
    the same name."""
    radius: ClassVar[float] = 0.02
    init_padding: ClassVar[float] = 0.05
    fov_angle: ClassVar[float] = np.pi / 4
    fov_dist: ClassVar[float] = 0.3
    id_tol: ClassVar[float] = 1e-3
    location_tol: ClassVar[float] = 1e-3

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        ## `instrument` can be camera (0.0 - 0.33), infrared (0.33 - 0.66), or
        ## Geiger (0.66 - 1.0). `calibration_obj_id` is the ID of the object
        ## that this satellite can be calibrated against. `read_obj_id` is the
        ## ID of the object that this satellite has read (used its instrument
        ## on), if any. The `read_obj_id` for every satellite is always -1 (a
        ## dummy value) in the initial state of every task. The task goals
        ## require setting each `read_obj_id` to particular values.
        self._sat_type = Type("satellite", [
            "x", "y", "theta", "instrument", "calibration_obj_id",
            "is_calibrated", "read_obj_id", "shoots_chem_x", "shoots_chem_y"
        ])
        self._obj_type = Type("object",
                              ["id", "x", "y", "has_chem_x", "has_chem_y"])

        # Predicates
        self._Sees = Predicate("Sees", [self._sat_type, self._obj_type],
                               self._Sees_holds)
        self._CalibrationTarget = Predicate("CalibrationTarget",
                                            [self._sat_type, self._obj_type],
                                            self._CalibrationTarget_holds)
        self._IsCalibrated = Predicate("IsCalibrated", [self._sat_type],
                                       self._IsCalibrated_holds)
        self._HasCamera = Predicate("HasCamera", [self._sat_type],
                                    self._HasCamera_holds)
        self._HasInfrared = Predicate("HasInfrared", [self._sat_type],
                                      self._HasInfrared_holds)
        self._HasGeiger = Predicate("HasGeiger", [self._sat_type],
                                    self._HasGeiger_holds)
        self._ShootsChemX = Predicate("ShootsChemX", [self._sat_type],
                                      self._ShootsChemX_holds)
        self._ShootsChemY = Predicate("ShootsChemY", [self._sat_type],
                                      self._ShootsChemY_holds)
        self._HasChemX = Predicate("HasChemX", [self._obj_type],
                                   self._HasChemX_holds)
        self._HasChemY = Predicate("HasChemY", [self._obj_type],
                                   self._HasChemY_holds)
        self._CameraReadingTaken = Predicate("CameraReadingTaken",
                                             [self._sat_type, self._obj_type],
                                             self._CameraReadingTaken_holds)
        self._InfraredReadingTaken = Predicate(
            "InfraredReadingTaken", [self._sat_type, self._obj_type],
            self._InfraredReadingTaken_holds)
        self._GeigerReadingTaken = Predicate("GeigerReadingTaken",
                                             [self._sat_type, self._obj_type],
                                             self._GeigerReadingTaken_holds)

    @classmethod
    def get_name(cls) -> str:
        return "satellites"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        # Note: target_sat_x and target_sat_y are only used if we're not
        # doing calibration, shooting Chemical X or Y, or using an instrument.
        cur_sat_x, cur_sat_y, obj_x, obj_y, target_sat_x, target_sat_y, \
            calibrate, shoot_chem_x, shoot_chem_y, use_instrument = action.arr
        next_state = state.copy()
        sat = self._xy_to_entity(state, cur_sat_x, cur_sat_y)
        obj = self._xy_to_entity(state, obj_x, obj_y)
        if sat is None or obj is None:
            # Invalid first 4 dimensions of action.
            return next_state
        if calibrate > 0.5:
            # Handle calibration.
            if not self._Sees_holds(state, [sat, obj]):
                # Cannot calibrate if the satellite cannot see the object.
                return next_state
            if not self._CalibrationTarget_holds(state, [sat, obj]):
                # Cannot calibrate against the wrong object.
                return next_state
            next_state.set(sat, "is_calibrated", 1.0)
        elif shoot_chem_x > 0.5:
            # Handle shooting Chemical X.
            if not self._Sees_holds(state, [sat, obj]):
                # Cannot shoot if the satellite cannot see the object.
                return next_state
            if not self._ShootsChemX_holds(state, [sat]):
                # Cannot shoot if the satellite doesn't have this chemical.
                return next_state
            next_state.set(obj, "has_chem_x", 1.0)
        elif shoot_chem_y > 0.5:
            # Handle shooting Chemical Y.
            if not self._Sees_holds(state, [sat, obj]):
                # Cannot shoot if the satellite cannot see the object.
                return next_state
            if not self._ShootsChemY_holds(state, [sat]):
                # Cannot shoot if the satellite doesn't have this chemical.
                return next_state
            next_state.set(obj, "has_chem_y", 1.0)
        elif use_instrument > 0.5:
            # Handle using the instrument on this satellite.
            if not self._Sees_holds(state, [sat, obj]):
                # Cannot take a reading if the satellite cannot see the object.
                return next_state
            if not self._IsCalibrated_holds(state, [sat]):
                # Cannot take a reading if the satellite is not calibrated.
                return next_state
            if self._HasCamera_holds(state, [sat]) and \
               not self._HasChemX_holds(state, [obj]):
                # Cannot take a camera reading without Chemical X.
                return next_state
            if self._HasInfrared_holds(state, [sat]) and \
               not self._HasChemY_holds(state, [obj]):
                # Cannot take an infrared reading without Chemical Y.
                return next_state
            next_state.set(sat, "read_obj_id", state.get(obj, "id"))
        else:
            # Handle moving.
            cur_circles = self._get_all_circles(state)
            proposed_circle = utils.Circle(target_sat_x, target_sat_y,
                                           self.radius)
            if any(circ.intersects(proposed_circle) for circ in cur_circles):
                # Cannot move to a location that is in collision.
                return next_state
            next_state.set(sat, "x", target_sat_x)
            next_state.set(sat, "y", target_sat_y)
            theta = np.arctan2(obj_y - target_sat_y, obj_x - target_sat_x)
            next_state.set(sat, "theta", theta)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_sat_lst=CFG.satellites_num_sat_train,
                               num_obj_lst=CFG.satellites_num_obj_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               num_sat_lst=CFG.satellites_num_sat_test,
                               num_obj_lst=CFG.satellites_num_obj_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Sees, self._CalibrationTarget, self._IsCalibrated,
            self._HasCamera, self._HasInfrared, self._HasGeiger,
            self._ShootsChemX, self._ShootsChemY, self._HasChemX,
            self._HasChemY, self._CameraReadingTaken,
            self._InfraredReadingTaken, self._GeigerReadingTaken
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {
            self._CameraReadingTaken, self._InfraredReadingTaken,
            self._GeigerReadingTaken
        }

    @property
    def types(self) -> Set[Type]:
        return {self._sat_type, self._obj_type}

    @property
    def options(self) -> Set[ParameterizedOption]:  # pragma: no cover
        raise NotImplementedError(
            "This base class method will be deprecated soon!")

    @property
    def action_space(self) -> Box:
        # [cur sat x, cur sat y, obj x, obj y, target sat x, target sat y,
        # calibrate, shoot Chemical X, shoot Chemical Y, use instrument]
        return Box(low=0.0, high=1.0, shape=(10, ), dtype=np.float32)

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        figsize = (1, 1)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.suptitle(caption, wrap=True)
        # Draw the satellites and FOV triangles.
        for sat in state.get_objects(self._sat_type):
            if state.get(sat, "read_obj_id") != -1:
                color = "green"
            elif self._IsCalibrated_holds(state, [sat]):
                color = "blue"
            else:
                color = "red"
            x = state.get(sat, "x")
            y = state.get(sat, "y")
            circ = utils.Circle(x, y, self.radius)
            circ.plot(ax,
                      facecolor=color,
                      edgecolor="black",
                      alpha=0.75,
                      linewidth=0.1)
            tri = self._get_fov_geom(state, sat)
            tri.plot(ax, color="purple", alpha=0.25, linewidth=0)
        # Draw the objects.
        for obj in state.get_objects(self._obj_type):
            color = "purple"
            x = state.get(obj, "x")
            y = state.get(obj, "y")
            circ = utils.Circle(x, y, self.radius)
            circ.plot(ax, color="black")
            text_x = x - self.radius
            text_y = y - self.radius
            if state.get(obj, "has_chem_x") > 0.5 and \
               state.get(obj, "has_chem_y") > 0.5:
                ax.text(text_x, text_y, "X/Y", fontsize=1.5, color="white")
            elif state.get(obj, "has_chem_x") > 0.5:
                ax.text(text_x, text_y, "X", fontsize=2, color="white")
            elif state.get(obj, "has_chem_y") > 0.5:
                ax.text(text_x, text_y, "Y", fontsize=2, color="white")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int, num_sat_lst: List[int],
                   num_obj_lst: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        radius = self.radius + self.init_padding
        for _ in range(num):
            state_dict = {}
            num_sat = num_sat_lst[rng.choice(len(num_sat_lst))]
            num_obj = num_obj_lst[rng.choice(len(num_obj_lst))]
            sats = [Object(f"sat{i}", self._sat_type) for i in range(num_sat)]
            # Sample initial positions for satellites, making sure to keep
            # them far enough apart from one another.
            collision_geoms: Set[utils.Circle] = set()
            some_sat_shoots_chem_x = False
            some_sat_shoots_chem_y = False
            for sat in sats:
                # Assuming that the dimensions are forgiving enough that
                # infinite loops are impossible.
                while True:
                    x = rng.uniform()
                    y = rng.uniform()
                    geom = utils.Circle(x, y, radius)
                    # Keep only if no intersections with existing objects.
                    if not any(geom.intersects(g) for g in collision_geoms):
                        break
                collision_geoms.add(geom)
                theta = rng.uniform(-np.pi, np.pi)
                instrument = rng.uniform()
                calibration_obj_id = rng.choice(num_obj)
                shoots_chem_x = rng.choice([0.0, 1.0])
                if shoots_chem_x > 0.5:
                    some_sat_shoots_chem_x = True
                shoots_chem_y = rng.choice([0.0, 1.0])
                if shoots_chem_y > 0.5:
                    some_sat_shoots_chem_y = True
                state_dict[sat] = {
                    "x": x,
                    "y": y,
                    "theta": theta,
                    "instrument": instrument,
                    "calibration_obj_id": calibration_obj_id,
                    "is_calibrated": 0.0,
                    "read_obj_id": -1.0,  # dummy, different from all obj IDs
                    "shoots_chem_x": shoots_chem_x,
                    "shoots_chem_y": shoots_chem_y
                }
            # Ensure that at least one satellite shoots Chemical X.
            if not some_sat_shoots_chem_x:
                sat = sats[rng.choice(len(sats))]
                state_dict[sat]["shoots_chem_x"] = 1.0
            # Ensure that at least one satellite shoots Chemical Y.
            if not some_sat_shoots_chem_y:
                sat = sats[rng.choice(len(sats))]
                state_dict[sat]["shoots_chem_y"] = 1.0
            objs = [Object(f"obj{i}", self._obj_type) for i in range(num_obj)]
            # Sample initial positions for objects, making sure to keep
            # them far enough apart from one another and from satellites.
            for i, obj in enumerate(objs):
                # Assuming that the dimensions are forgiving enough that
                # infinite loops are impossible.
                while True:
                    x = rng.uniform()
                    y = rng.uniform()
                    geom = utils.Circle(x, y, radius)
                    # Keep only if no intersections with existing objects.
                    if not any(geom.intersects(g) for g in collision_geoms):
                        break
                collision_geoms.add(geom)
                state_dict[obj] = {
                    "id": i,
                    "x": x,
                    "y": y,
                    "has_chem_x": 0.0,
                    "has_chem_y": 0.0
                }
            init_state = utils.create_state_from_dict(state_dict)
            goal = set()
            for sat in sats:
                # For each satellite, choose an object for it to read, and
                # add a goal atom based on the satellite's instrument.
                goal_obj_for_sat = objs[rng.choice(len(objs))]
                if self._HasCamera_holds(init_state, [sat]):
                    goal_pred = self._CameraReadingTaken
                elif self._HasInfrared_holds(init_state, [sat]):
                    goal_pred = self._InfraredReadingTaken
                elif self._HasGeiger_holds(init_state, [sat]):
                    goal_pred = self._GeigerReadingTaken
                goal.add(GroundAtom(goal_pred, [sat, goal_obj_for_sat]))
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    def _Sees_holds(self, state: State, objects: Sequence[Object]) -> bool:
        sat, obj = objects
        triangle = self._get_fov_geom(state, sat)
        sat_x = state.get(sat, "x")
        sat_y = state.get(sat, "y")
        obj_x = state.get(obj, "x")
        obj_y = state.get(obj, "y")
        # Note: we require only that the center of the object
        # is in the view cone, ignoring the object's radius.
        if not triangle.contains_point(obj_x, obj_y):
            return False
        # Now check if the line of sight is occluded by another entity.
        dist_denom = np.sqrt((sat_x - obj_x)**2 + (sat_y - obj_y)**2)
        for ent in state:
            if ent in (sat, obj):
                continue
            # Compute the projection distance of this entity onto the line
            # segment connecting `sat` and `obj`.
            ent_x = state.get(ent, "x")
            ent_y = state.get(ent, "y")
            dist = abs((obj_x - sat_x) * (sat_y - ent_y) - (sat_x - ent_x) *
                       (obj_y - sat_y)) / dist_denom
            if dist < self.radius * 2:
                return False
        return True

    def _CalibrationTarget_holds(self, state: State,
                                 objects: Sequence[Object]) -> bool:
        sat, obj = objects
        return abs(
            state.get(sat, "calibration_obj_id") -
            state.get(obj, "id")) < self.id_tol

    @staticmethod
    def _IsCalibrated_holds(state: State, objects: Sequence[Object]) -> bool:
        sat, = objects
        return state.get(sat, "is_calibrated") > 0.5

    @staticmethod
    def _HasCamera_holds(state: State, objects: Sequence[Object]) -> bool:
        sat, = objects
        return 0.0 < state.get(sat, "instrument") < 0.33

    @staticmethod
    def _HasInfrared_holds(state: State, objects: Sequence[Object]) -> bool:
        sat, = objects
        return 0.33 < state.get(sat, "instrument") < 0.66

    @staticmethod
    def _HasGeiger_holds(state: State, objects: Sequence[Object]) -> bool:
        sat, = objects
        return 0.66 < state.get(sat, "instrument") < 1.0

    @staticmethod
    def _ShootsChemX_holds(state: State, objects: Sequence[Object]) -> bool:
        sat, = objects
        return state.get(sat, "shoots_chem_x") > 0.5

    @staticmethod
    def _ShootsChemY_holds(state: State, objects: Sequence[Object]) -> bool:
        sat, = objects
        return state.get(sat, "shoots_chem_y") > 0.5

    @staticmethod
    def _HasChemX_holds(state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "has_chem_x") > 0.5

    @staticmethod
    def _HasChemY_holds(state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "has_chem_y") > 0.5

    def _CameraReadingTaken_holds(self, state: State,
                                  objects: Sequence[Object]) -> bool:
        sat, obj = objects
        return self._HasCamera_holds(state, [sat]) and \
            abs(state.get(sat, "read_obj_id") -
                state.get(obj, "id")) < self.id_tol

    def _InfraredReadingTaken_holds(self, state: State,
                                    objects: Sequence[Object]) -> bool:
        sat, obj = objects
        return self._HasInfrared_holds(state, [sat]) and \
            abs(state.get(sat, "read_obj_id") -
                state.get(obj, "id")) < self.id_tol

    def _GeigerReadingTaken_holds(self, state: State,
                                  objects: Sequence[Object]) -> bool:
        sat, obj = objects
        return self._HasGeiger_holds(state, [sat]) and \
            abs(state.get(sat, "read_obj_id") -
                state.get(obj, "id")) < self.id_tol

    def _get_all_circles(self, state: State) -> Set[utils.Circle]:
        """Get all entities in the state as utils.Circle objects."""
        circles = set()
        for ent in state:
            x = state.get(ent, "x")
            y = state.get(ent, "y")
            circles.add(utils.Circle(x, y, self.radius))
        return circles

    def _get_fov_geom(self, state: State, sat: Object) -> utils.Triangle:
        """Get the FOV of the given satellite as a utils.Triangle."""
        x1 = state.get(sat, "x")
        y1 = state.get(sat, "y")
        theta_mid = state.get(sat, "theta")
        theta_low = theta_mid - self.fov_angle / 2.0
        x2 = x1 + self.fov_dist * np.cos(theta_low)
        y2 = y1 + self.fov_dist * np.sin(theta_low)
        theta_high = theta_mid + self.fov_angle / 2.0
        x3 = x1 + self.fov_dist * np.cos(theta_high)
        y3 = y1 + self.fov_dist * np.sin(theta_high)
        return utils.Triangle(x1, y1, x2, y2, x3, y3)

    def _xy_to_entity(self, state: State, x: float,
                      y: float) -> Optional[Object]:
        """Given x/y coordinates, return the entity (satellite or object) at
        those coordinates."""
        for ent in state:
            if abs(state.get(ent, "x") - x) < self.location_tol and \
               abs(state.get(ent, "y") - y) < self.location_tol:
                return ent
        return None


class SatellitesSimpleEnv(SatellitesEnv):
    """A simple version of the SatellitesEnv that only ever has 1 object."""

    @classmethod
    def get_name(cls) -> str:
        return "satellites_simple"

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_sat_lst=CFG.satellites_num_sat_train,
                               num_obj_lst=[1],
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               num_sat_lst=CFG.satellites_num_sat_test,
                               num_obj_lst=[1],
                               rng=self._test_rng)
