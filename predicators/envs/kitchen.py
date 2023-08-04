"""A Kitchen environment wrapping kitchen from https://github.com/google-
research/relay-policy-learning."""
import copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

import matplotlib
import numpy as np
from gym.spaces import Box

try:
    import gymnasium as mujoco_kitchen_gym
    from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, \
        get_site_xmat, get_site_xpos
    from gymnasium_robotics.utils.rotations import mat2quat
    _MJKITCHEN_IMPORTED = True
except (ImportError, RuntimeError):
    _MJKITCHEN_IMPORTED = False
from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Image, Object, \
    Observation, Predicate, State, Type, Video

_TRACKED_SITES = [
    "hinge_site1", "hinge_site2", "kettle_site", "microhandle_site",
    "knob1_site", "knob2_site", "knob3_site", "knob4_site", "light_site",
    "slide_site", "EEF"
]

_TRACKED_SITE_TO_JOINT = {
    "knob1_site": "knob_Joint_1",
    "knob2_site": "knob_Joint_2",
    "knob3_site": "knob_Joint_3",
    "knob4_site": "knob_Joint_4"
}

_TRACKED_BODIES = ["Burner 1", "Burner 2", "Burner 3", "Burner 4"]


class KitchenEnv(BaseEnv):
    """Kitchen environment wrapping dm_control Kitchen."""

    gripper_type = Type("gripper", ["x", "y", "z", "qw", "qx", "qy", "qz"])
    object_type = Type("object", ["x", "y", "z"])
    on_off_type = Type("on_off", ["x", "y", "z", "angle"], parent=object_type)
    hinge_door_type = Type("hinge_door", ["x", "y", "z", "angle"],
                           parent=on_off_type)
    knob_type = Type("knob", ["x", "y", "z", "angle"], parent=on_off_type)
    switch_type = Type("switch", ["x", "y", "z", "angle"], parent=on_off_type)
    surface_type = Type("surface", ["x", "y", "z"], parent=object_type)
    kettle_type = Type("kettle", ["x", "y", "z"], parent=object_type)

    obj_name_to_type = {
        "gripper": gripper_type,
        "hinge2": hinge_door_type,
        "kettle": kettle_type,
        "microhandle": hinge_door_type,
        "knob1": knob_type,
        "knob2": knob_type,
        "knob3": knob_type,
        "knob4": knob_type,
        "light": switch_type,
        "slide": hinge_door_type,
        "hinge1": hinge_door_type,
        "burner1": surface_type,
        "burner2": surface_type,
        "burner3": surface_type,
        "burner4": surface_type,
    }

    at_pre_turn_atol = 0.05  # tolerance for AtPreTurnOn/Off
    ontop_atol = 0.15  # tolerance for OnTop
    on_angle_thresh = -0.4  # dial is On if less than this threshold
    light_on_thresh = -0.39  # light is On if less than this threshold
    microhandle_open_thresh = -0.68
    at_pre_pushontop_yz_atol = 0.1  # tolerance for AtPrePushOnTop
    at_pre_pullontop_yz_atol = 0.04  # tolerance for AtPrePullOnTop
    at_pre_pushontop_x_atol = 1.0  # other tolerance for AtPrePushOnTop

    obj_name_to_pre_push_dpos = {
        ("kettle", "on"): (-0.05, -0.3, -0.12),
        ("kettle", "off"): (0.0, -0.05, 0.06),
        ("knob4", "on"): (-0.1, -0.15, 0.05),
        ("knob4", "off"): (0.05, -0.12, -0.05),
        ("light", "on"): (0.1, -0.05, -0.05),
        ("light", "off"): (-0.1, -0.05, -0.05),
        ("microhandle", "on"): (0.12, 0.03, 0.17),
        ("microhandle", "off"): (0.0, -0.1, 0.2),
        ("hinge1", "on"): (0.1, -0.1, 0.0),
        ("hinge1", "off"): (-0.3, 0.1, 0.0),
        ("hinge2", "on"): (0.1, -0.15, 0.0),
        ("hinge2", "off"): (-0.1, -0.1, 0.0),
    }

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        assert _MJKITCHEN_IMPORTED, "Failed to import kitchen gym env. \
Install from https://github.com/SiddarGu/Gymnasium-Robotics.git"

        if use_gui:
            assert not CFG.make_test_videos or CFG.make_failure_videos, \
                "Turn off --use_gui to make videos in kitchen env"

        self._pred_name_to_pred = self.create_predicates()

        render_mode = "human" if self._using_gui else "rgb_array"
        self._gym_env = mujoco_kitchen_gym.make("FrankaKitchen-v1",
                                                render_mode=render_mode)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, train_or_test="train")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, train_or_test="test")

    @classmethod
    def get_name(cls) -> str:
        return "kitchen"

    def get_observation(self) -> Observation:
        return self._copy_observation(self._current_observation)

    def get_object_centric_state_info(self) -> Dict[str, Any]:
        """Parse State into Object Centric State."""
        mujoco_model = self._gym_env.model  # type: ignore
        mujoco_data = self._gym_env.data  # type: ignore
        mujoco_model_names = self._gym_env.robot_env.model_names  # type: ignore
        state_info = {}
        for site in _TRACKED_SITES:
            state_info[site] = get_site_xpos(mujoco_model, mujoco_data,
                                             site).copy()
            # Include rotation for gripper.
            if site == "EEF":
                xmat = get_site_xmat(mujoco_model, mujoco_data, site).copy()
                quat = mat2quat(xmat)
                state_info[site] = np.concatenate([state_info[site], quat])
        for joint in _TRACKED_SITE_TO_JOINT.values():
            state_info[joint] = get_joint_qpos(mujoco_model, mujoco_data,
                                               joint).copy()
        for body in _TRACKED_BODIES:
            body_id = mujoco_model_names.body_name2id[body]
            state_info[body] = mujoco_data.xpos[body_id].copy()
        return state_info

    @classmethod
    def get_pre_push_delta_pos(cls, obj: Object,
                               on_or_off: str) -> Tuple[float, float, float]:
        """Get dx, dy, dz offset for pushing."""
        try:
            dx, dy, dz = cls.obj_name_to_pre_push_dpos[(obj.name, on_or_off)]
        except KeyError:
            dx, dy, dz = (0.0, 0.0, 0.0)
        return (dx, dy, dz)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise NotImplementedError("A gym environment cannot render "
                                  "arbitrary states.")

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:
        assert caption is None
        arr: Image = self._gym_env.render()  # type: ignore
        return [arr]

    @property
    def predicates(self) -> Set[Predicate]:
        return set(self._pred_name_to_pred.values())

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {
            self._pred_name_to_pred["OnTop"],
            self._pred_name_to_pred["TurnedOn"],
            self._pred_name_to_pred["Open"]
        }

    @classmethod
    def create_predicates(cls) -> Dict[str, Predicate]:
        """Exposed for perceiver."""
        preds = {
            Predicate("AtPreTurnOff", [cls.gripper_type, cls.on_off_type],
                      cls._AtPreTurnOff_holds),
            Predicate("AtPreTurnOn", [cls.gripper_type, cls.on_off_type],
                      cls._AtPreTurnOn_holds),
            Predicate("AtPrePushOnTop", [cls.gripper_type, cls.kettle_type],
                      cls._AtPrePushOnTop_holds),
            Predicate("AtPrePullKettle", [cls.gripper_type, cls.kettle_type],
                      cls._AtPrePullKettle_holds),
            Predicate("OnTop", [cls.kettle_type, cls.surface_type],
                      cls._OnTop_holds),
            Predicate("NotOnTop", [cls.kettle_type, cls.surface_type],
                      cls._NotOnTop_holds),
            Predicate("TurnedOn", [cls.on_off_type], cls.On_holds),
            Predicate("TurnedOff", [cls.on_off_type], cls.Off_holds),
            Predicate("Open", [cls.on_off_type], cls.Open_holds),
            Predicate("Closed", [cls.on_off_type], cls.Closed_holds),
        }
        return {p.name: p for p in preds}

    @property
    def types(self) -> Set[Type]:
        return {
            self.gripper_type, self.object_type, self.on_off_type,
            self.knob_type, self.kettle_type, self.switch_type,
            self.hinge_door_type, self.surface_type
        }

    @property
    def action_space(self) -> Box:
        return cast(Box, self._gym_env.action_space)

    @classmethod
    def object_name_to_object(cls, obj_name: str) -> Object:
        """Made public for perceiver."""
        return Object(obj_name, cls.obj_name_to_type[obj_name])

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Resets the current state to the train or test task initial state."""
        self._current_task = self.get_task(train_or_test, task_idx)
        # We now need to reset the underlying gym environment to the correct
        # state.
        seed = utils.get_task_seed(train_or_test, task_idx)
        self._current_observation = self._reset_initial_state_from_seed(seed)
        return self._copy_observation(self._current_observation)

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for gym envs. " +
                                  "Try using --bilevel_plan_without_sim True")

    def step(self, action: Action) -> Observation:
        self._gym_env.step(action.arr)
        if self._using_gui:
            self._gym_env.render()
        self._current_observation = {
            "state_info": self.get_object_centric_state_info()
        }
        return self._copy_observation(self._current_observation)

    @classmethod
    def state_info_to_state(cls, state_info: Dict[str, Any]) -> State:
        """Get state from state info dictionary."""
        assert "EEF" in state_info  # sanity check
        state_dict = {}
        for key, val in state_info.items():
            if key == "EEF":
                obj = cls.object_name_to_object("gripper")
                state_dict[obj] = {
                    "x": val[0],
                    "y": val[1],
                    "z": val[2],
                    "qw": val[3],
                    "qx": val[4],
                    "qy": val[5],
                    "qz": val[6],
                }
            elif key in _TRACKED_SITE_TO_JOINT.values():
                continue  # used below
            else:
                obj_name = key.replace("_site", "").replace(" ", "").lower()
                obj = cls.object_name_to_object(obj_name)
                if key in _TRACKED_SITE_TO_JOINT:
                    joint = _TRACKED_SITE_TO_JOINT[key]
                    angle = state_info[joint][0]
                else:
                    angle = 0
                state_dict[obj] = {
                    "x": val[0],
                    "y": val[1],
                    "z": val[2],
                    "angle": angle
                }
        state = utils.create_state_from_dict(state_dict)
        return state

    def goal_reached(self) -> bool:
        state = self.state_info_to_state(
            self._current_observation["state_info"])
        kettle = self.object_name_to_object("kettle")
        burner4 = self.object_name_to_object("burner4")
        knob4 = self.object_name_to_object("knob4")
        light = self.object_name_to_object("light")
        goal_desc = self._current_task.goal_description
        kettle_on_burner = self._OnTop_holds(state, [kettle, burner4])
        knob4_turned_on = self.On_holds(state, [knob4])
        light_turned_on = self.On_holds(state, [light])
        if goal_desc == ("Move the kettle to the back burner and turn it on; "
                         "also turn on the light"):
            return kettle_on_burner and knob4_turned_on and light_turned_on
        if goal_desc == "Move the kettle to the back burner":
            return kettle_on_burner
        if goal_desc == "Turn on the back burner":
            return knob4_turned_on
        if goal_desc == "Turn on the light":
            return light_turned_on
        raise NotImplementedError(f"Unrecognized goal: {goal_desc}")

    def _get_tasks(self, num: int,
                   train_or_test: str) -> List[EnvironmentTask]:
        tasks = []

        assert CFG.kitchen_goals in [
            "all", "kettle_only", "knob_only", "light_only"
        ]
        goal_descriptions: List[str] = []
        if CFG.kitchen_goals in ["all", "kettle_only"]:
            goal_descriptions.append("Move the kettle to the back burner")
        if CFG.kitchen_goals in ["all", "knob_only"]:
            goal_descriptions.append("Turn on the back burner")
        if CFG.kitchen_goals in ["all", "light_only"]:
            goal_descriptions.append("Turn on the light")
        if CFG.kitchen_goals == "all":
            desc = ("Move the kettle to the back burner and turn it on; also "
                    "turn on the light")
            goal_descriptions.append(desc)

        for task_idx in range(num):
            seed = utils.get_task_seed(train_or_test, task_idx)
            init_obs = self._reset_initial_state_from_seed(seed)
            goal_idx = task_idx % len(goal_descriptions)
            goal_description = goal_descriptions[goal_idx]
            task = EnvironmentTask(init_obs, goal_description)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int) -> Observation:
        self._gym_env.reset(seed=seed)
        return {"state_info": self.get_object_centric_state_info()}

    @classmethod
    def _AtPreTurn_holds(cls, state: State, objects: Sequence[Object],
                         on_or_off: str) -> bool:
        """Helper for _AtPreTurnOn_holds() and _AtPreTurnOff_holds()."""
        gripper, obj = objects
        obj_xyz = np.array(
            [state.get(obj, "x"),
             state.get(obj, "y"),
             state.get(obj, "z")])
        # On refers to Open and Off to Close
        dpos = cls.get_pre_push_delta_pos(obj, on_or_off)
        gripper_xyz = np.array([
            state.get(gripper, "x"),
            state.get(gripper, "y"),
            state.get(gripper, "z")
        ])
        return np.allclose(obj_xyz + dpos,
                           gripper_xyz,
                           atol=cls.at_pre_turn_atol)

    @classmethod
    def _AtPreTurnOn_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        return cls._AtPreTurn_holds(state, objects, "on")

    @classmethod
    def _AtPreTurnOff_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        return cls._AtPreTurn_holds(state, objects, "off")

    @classmethod
    def _AtPrePushOnTop_holds(cls, state: State,
                              objects: Sequence[Object]) -> bool:
        # The main thing that's different from _AtPreTurnOn_holds is that the
        # x position has a much higher range of allowed values, since it can
        # be anywhere behind the object.
        gripper, obj = objects
        obj_xyz = np.array(
            [state.get(obj, "x"),
             state.get(obj, "y"),
             state.get(obj, "z")])
        dpos = cls.get_pre_push_delta_pos(obj, "on")
        target_x, target_y, target_z = obj_xyz + dpos
        gripper_x, gripper_y, gripper_z = [
            state.get(gripper, "x"),
            state.get(gripper, "y"),
            state.get(gripper, "z")
        ]
        if not np.allclose([target_y, target_z], [gripper_y, gripper_z],
                           atol=cls.at_pre_pushontop_yz_atol):
            return False
        return np.isclose(target_x,
                          gripper_x,
                          atol=cls.at_pre_pushontop_x_atol)

    @classmethod
    def _AtPrePullKettle_holds(cls, state: State,
                               objects: Sequence[Object]) -> bool:
        gripper, obj = objects
        obj_xyz = np.array(
            [state.get(obj, "x"),
             state.get(obj, "y"),
             state.get(obj, "z")])
        dpos = cls.get_pre_push_delta_pos(obj, "off")
        target_x, target_y, target_z = obj_xyz + dpos
        gripper_x, gripper_y, gripper_z = [
            state.get(gripper, "x"),
            state.get(gripper, "y"),
            state.get(gripper, "z")
        ]
        if not np.allclose([target_y, target_z], [gripper_y, gripper_z],
                           atol=cls.at_pre_pullontop_yz_atol):
            return False
        return np.isclose(target_x,
                          gripper_x,
                          atol=cls.at_pre_pushontop_x_atol)

    @classmethod
    def _OnTop_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        obj1_xy = [state.get(obj1, "x"), state.get(obj1, "y")]
        obj2_xy = [
            state.get(obj2, "x"),
            state.get(obj2, "y"),
        ]
        return np.allclose(obj1_xy,
                           obj2_xy, atol=cls.ontop_atol) and state.get(
                               obj1, "z") > state.get(obj2, "z")

    @classmethod
    def _NotOnTop_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return not cls._OnTop_holds(state, objects)

    @classmethod
    def On_holds(cls,
                 state: State,
                 objects: Sequence[Object],
                 thresh_pad: float = 0.0) -> bool:
        """Made public for use in ground-truth options."""
        obj = objects[0]
        if obj.is_instance(cls.knob_type):
            return state.get(obj, "angle") < cls.on_angle_thresh - thresh_pad
        if obj.is_instance(cls.switch_type):
            return state.get(obj, "x") < cls.light_on_thresh - thresh_pad
        return False

    @classmethod
    def Off_holds(cls,
                  state: State,
                  objects: Sequence[Object],
                  thresh_pad: float = 0.0) -> bool:
        """Made public for use in ground-truth options."""
        # Can't do not On_holds() because of thresh_pad logic.
        obj = objects[0]
        if obj.is_instance(cls.knob_type):
            return state.get(obj, "angle") >= cls.on_angle_thresh + thresh_pad
        if obj.is_instance(cls.switch_type):
            return state.get(obj, "x") >= cls.light_on_thresh + thresh_pad
        return False

    @classmethod
    def Open_holds(cls,
                   state: State,
                   objects: Sequence[Object],
                   thresh_pad: float = 0.0) -> bool:
        """Made public for use in ground-truth options."""
        obj = objects[0]
        if obj.is_instance(cls.hinge_door_type):
            return state.get(obj,
                             "x") < cls.microhandle_open_thresh - thresh_pad
        return False

    @classmethod
    def Closed_holds(cls,
                     state: State,
                     objects: Sequence[Object],
                     thresh_pad: float = 0.0) -> bool:
        """Made public for use in ground-truth options."""
        # Can't do not Open_holds() because of thresh_pad logic.
        obj = objects[0]
        if obj.is_instance(cls.hinge_door_type):
            return state.get(obj,
                             "x") >= cls.microhandle_open_thresh + thresh_pad
        return False

    def _copy_observation(self, obs: Observation) -> Observation:
        return copy.deepcopy(obs)
