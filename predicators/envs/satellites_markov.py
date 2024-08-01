"""A 2D continuous satellites domain loosely inspired by the IPC domain of the
same name.

This is a Markov version of Satellites Domain. Here:
(1) Current low-level states are enough for making current decison, as the 
satellites have to first MoveAway(). Before it can MoveTo(). 
In this way "Sees" predicate won't be "remebered".
(2) The objects are initially placed in a way that one satellite can't see
two objects at the same time. This is to make sure that high-level plan/states
are strictly a deterministic function of low-level states.
"""

from typing import ClassVar, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.satellites import SatellitesEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class SatellitesMarkovEnv(SatellitesEnv):
    """A 2D continuous satellites domain loosely inspired by the IPC domain of
    the same name."""
    radius: ClassVar[float] = 0.02
    init_padding: ClassVar[float] = 0.02
    fov_angle: ClassVar[float] = np.pi / 4
    fov_dist: ClassVar[float] = 0.16 # this is the hypotenuse of isosceles triangle
    id_tol: ClassVar[float] = 1e-3
    location_tol: ClassVar[float] = 1e-3

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # ViewClear predicate is added
        self._ViewClear = Predicate("ViewClear", [self._sat_type],
                                    self._ViewClear_holds)

    @classmethod
    def get_name(cls) -> str:
        return "satellites-markov"

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
            if "view_clear" in self._sat_type.feature_names:
                all_objects = list(next_state.get_objects(self._obj_type))
                view_clear = True
                for obj in all_objects:
                    pair = [sat, obj]
                    if self._Sees_holds(next_state, pair):
                        view_clear = False
                        break
                if view_clear:
                    next_state.set(sat, "view_clear", 1.0)
                else:
                    next_state.set(sat, "view_clear", 0.0)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_sat_lst=CFG.satellites_markov_num_sat_train,
                               num_obj_lst=CFG.satellites_markov_num_obj_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               num_sat_lst=CFG.satellites_markov_num_sat_test,
                               num_obj_lst=CFG.satellites_markov_num_obj_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Sees, self._CalibrationTarget, self._IsCalibrated,
            self._HasCamera, self._HasInfrared, self._HasGeiger,
            self._ShootsChemX, self._ShootsChemY, self._ViewClear,
            self._HasChemX, self._HasChemY, self._CameraReadingTaken,
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
    def action_space(self) -> Box:
        # [cur sat x, cur sat y, obj x, obj y, target sat x, target sat y,
        # calibrate, shoot Chemical X, shoot Chemical Y, use instrument]
        return Box(low=0.0, high=1.0, shape=(10, ), dtype=np.float32)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None,
            save_path: Optional[str] = None) -> matplotlib.figure.Figure:
        figsize = (8, 8)  # Increase the figure size for better visibility
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.suptitle(caption, wrap=True)
        
        legend_info = []  # Collect legend information here
        
        # Draw the satellites and FOV triangles.
        for idx, sat in enumerate(state.get_objects(self._sat_type)):
            if state.get(sat, "read_obj_id") != -1:
                color = "green"
            elif self._IsCalibrated_holds(state, [sat]):
                color = "blue"
            else:
                color = "red"
            
            x = state.get(sat, "x")
            y = state.get(sat, "y")
            circ = utils.Circle(x, y, self.radius)
            circ.plot(ax, facecolor=color, edgecolor="black", alpha=0.75, 
                      linewidth=0.1)
            tri = self._get_fov_geom(state, sat)
            tri.plot(ax, color="purple", alpha=0.25, linewidth=0)
            
            # Add satellite ID as text
            sat_id = int(idx)
            ax.text(x + self.radius, y + self.radius, sat_id, fontsize=8, 
                    color="black", verticalalignment='top')

            # Collect instrument and capability information for legend
            calibration_obj_id = int(state.get(sat, "calibration_obj_id"))
            instrument = state.get(sat, "instrument")
            has_camera = "Cam" if instrument <= 0.33 else ""
            has_infrared = "Infr" if 0.33 < instrument <= 0.66 else ""
            has_geiger = "Geig" if instrument > 0.66 else ""
            shoots_chem_x = "ChemX" if state.get(sat, "shoots_chem_x") > 0.5 \
                                else ""
            shoots_chem_y = "ChemY" if state.get(sat, "shoots_chem_y") > 0.5 \
                                else ""
            
            capabilities = ", ".join(filter(None, [has_camera, has_infrared, \
                                                   has_geiger, shoots_chem_x, \
                                                    shoots_chem_y]))
            legend_info.append(f"Sat {sat_id}: Cali={calibration_obj_id}, \
                               {capabilities}")
        
        # Draw the objects.
        for obj in state.get_objects(self._obj_type):
            x = state.get(obj, "x")
            y = state.get(obj, "y")
            circ = utils.Circle(x, y, self.radius)
            if self._HasChemX_holds(state, [obj]) and \
                self._HasChemY_holds(state, [obj]):
                circ.plot(ax, facecolor='none', edgecolor="black", hatch="++")
                legend_info.append(\
                    f"Object {int(state.get(obj, 'id'))}: HasX+HasY")
            elif self._HasChemX_holds(state, [obj]):
                circ.plot(ax, facecolor='none', edgecolor="black", hatch="|||")
                legend_info.append(f"Object {int(state.get(obj, 'id'))}: HasX")
            elif self._HasChemY_holds(state, [obj]):
                circ.plot(ax, facecolor='none', edgecolor="black", hatch="---")
                legend_info.append(f"Object {int(state.get(obj, 'id'))}: HasY")
            else:
                circ.plot(ax, color="black", hatch="")
            
            # Add object ID as text
            obj_id = int(state.get(obj, "id"))
            ax.text(x + self.radius, y + self.radius, obj_id, fontsize=8, color="black", \
                    verticalalignment='top')

            # legend_info.append(f"Object {obj_id}")

        if task is not None:
            # Draw the goal objects.
            for goal in list(task.goal):
                goal_ori = goal._str
                goal_shortened = self.shorten_goal(goal_ori)
                legend_info.append(goal_shortened)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
        # Adding x and y axis labels and grid for better visibility
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True)
        
        # Adding legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', \
                       markersize=5, label='Objs'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', \
                       markersize=5, label='Sats-unC-unR'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', \
                       markersize=5, label='Sats-C-unR'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', \
                       markersize=5, label='Sats-C-R')
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0, 1.1), \
                  fontsize=8)
        
        # Adding legend information text
        legend_text = "\n".join(legend_info)
        plt.figtext(0.25, 0.7, legend_text, horizontalalignment='left', fontsize=8, wrap=True)
        
        plt.tight_layout(rect=[0, 0, 0.7, 0.8])  # Adjust layout to make room for legend text
        plt.close()
        
        if save_path:
            fig.savefig(save_path, format='png', dpi=600)
        
        return fig

    def shorten_goal(self, goal: str) -> str:
        goal_shortened = goal.replace("CameraReadingTaken", "Cam")
        goal_shortened = goal_shortened.replace("InfraredReadingTaken", "Infr")
        goal_shortened = goal_shortened.replace("GeigerReadingTaken", "Geig")
        goal_shortened = goal_shortened.replace(":satellite", "")
        goal_shortened = goal_shortened.replace(":object", "")
        return goal_shortened
    
    def _get_tasks(self, num: int, num_sat_lst: List[int],
                   num_obj_lst: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        # note: fov dist is hypotenuse of isosceles triangle, not height
        radius = max(self.radius + self.init_padding, self.fov_dist + self.init_padding)
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
                    # satellites can be near each other, but not overlapping
                    geom = utils.Circle(x, y, self.radius + self.init_padding)
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
                if "view_clear" in self._sat_type.feature_names:
                    state_dict[sat] = {
                        "x": x,
                        "y": y,
                        "theta": theta,
                        "instrument": instrument,
                        "calibration_obj_id": calibration_obj_id,
                        "is_calibrated": 0.0,
                        "read_obj_id": -1.0,  # dummy, different from all obj IDs
                        "shoots_chem_x": shoots_chem_x,
                        "shoots_chem_y": shoots_chem_y,
                        "view_clear": 1.0
                    }
                else:
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
                    # objects must be far enough from satellites and other objects
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
            task = EnvironmentTask(init_state, goal)
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

            # Check if the entity is between the satellite and the object
            dot_product = (ent_x - sat_x) * (obj_x - sat_x) + (ent_y - sat_y) \
                * (obj_y - sat_y)
            if dot_product < 0: # before start
                continue

            if dot_product > dist_denom**2: # after end
                continue

            # Compute perpendicular distance from entity to line
            dist = abs((obj_x - sat_x) * (sat_y - ent_y) - (sat_x - ent_x) * \
                       (obj_y - sat_y)) / dist_denom
            if dist < self.radius * 2:
                return False
        return True
    
    def _ViewClear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        sat, = objects
        all_objects = list(state.get_objects(self._obj_type))
        view_clear = True
        for obj in all_objects:
            pair = [sat, obj]
            if self._Sees_holds(state, pair):
                view_clear = False
                break
        return view_clear


class SatellitesSimpleEnv(SatellitesMarkovEnv):
    """A simple version of the SatellitesEnv that only has 1 object.
       The satellites also have a feature indicating whether its view is clear.
       "Sees" does not consider occlusion.
       This allows us to learn all the predicates with the assumption that the 
       predicates are a function of only their argument's states.
    """

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

        # Add attr
        self._sat_type = Type("satellite", [
            "x", "y", "theta", "instrument", "calibration_obj_id",
            "is_calibrated", "read_obj_id", "shoots_chem_x", "shoots_chem_y",
            "view_clear"
        ])
        # Override Sees and ViewClear Predicates
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
        self._ViewClear = Predicate("ViewClear", [self._sat_type],
                                    self._ViewClear_holds)
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
        return "satellites-markov_simple"

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_sat_lst=CFG.satellites_num_sat_train,
                               num_obj_lst=[1],
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               num_sat_lst=CFG.satellites_num_sat_test,
                               num_obj_lst=[1],
                               rng=self._test_rng)
    
    def _Sees_holds(self, state: State, objects: Sequence[Object]) -> bool:
        sat, obj = objects
        triangle = self._get_fov_geom(state, sat)
        obj_x = state.get(obj, "x")
        obj_y = state.get(obj, "y")
        # Note: we require only that the center of the object
        # is in the view cone, ignoring the object's radius.
        if not triangle.contains_point(obj_x, obj_y):
            return False
        return True
    
    def _ViewClear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        sat, = objects
        return state.get(sat, "view_clear") == 1

class SatellitesMediumEnv(SatellitesMarkovEnv):
    """A medium version of the SatellitesEnv that only ever has 1 object.
       "Sees" does not consider occlusion.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

    @classmethod
    def get_name(cls) -> str:
        return "satellites-markov_medium"

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_sat_lst=CFG.satellites_num_sat_train,
                               num_obj_lst=[1],
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               num_sat_lst=CFG.satellites_num_sat_test,
                               num_obj_lst=[1],
                               rng=self._test_rng)
    
    def _Sees_holds(self, state: State, objects: Sequence[Object]) -> bool:
        sat, obj = objects
        triangle = self._get_fov_geom(state, sat)
        obj_x = state.get(obj, "x")
        obj_y = state.get(obj, "y")
        # Note: we require only that the center of the object
        # is in the view cone, ignoring the object's radius.
        if not triangle.contains_point(obj_x, obj_y):
            return False
        return True
