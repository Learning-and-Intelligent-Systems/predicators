"""A Kitchen environment wrapping kitchen from https://github.com/google-
research/relay-policy-learning."""
import copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
from gym.spaces import Box

try:
    import gymnasium as mujoco_kitchen_gym
    from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, get_site_xpos, get_site_xmat
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
    "knob1_site", "knob2_site", "knob3_site", "knob4_site",
    "light_site", "slide_site", "EEF"
]

_TRACKED_SITE_TO_JOINT = {
    "knob1_site": "knob_Joint_1",
    "knob2_site": "knob_Joint_2",
    "knob3_site": "knob_Joint_3",
    "knob4_site": "knob_Joint_4"
}

_TRACKED_BODIES = [
    "Burner 1", "Burner 2", "Burner 3", "Burner 4"
]


class KitchenEnv(BaseEnv):
    """Kitchen environment wrapping dm_control Kitchen."""

    gripper_type = Type("gripper", ["x", "y", "z", "qw", "qx", "qy", "qz"])
    object_type = Type("obj", ["x", "y", "z", "angle"])

    at_atol = 0.2  # tolerance for At classifier
    ontop_atol = 0.15  # tolerance for OnTop classifier
    on_angle_thresh = -0.8  # dial is On if less than this threshold
    light_on_thresh = -0.4  # light is On if less than this threshold

    obj_name_to_pre_push_dpos = {
        "kettle": (0.0, -0.3, -0.15),  # need to push from behind kettle
        "knob4": (-0.08, -0.12, 0.05),  # need to push from left to right
        "light": (0.1, 0.05, -0.2),  # need to push from right to left
    }

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        assert _MJKITCHEN_IMPORTED, "Failed to import mujoco_kitchen. \
Install from https://github.com/SiddarGu/Gymnasium-Robotics.git"

        # Predicates
        self._At, self._OnTop, self._TurnedOn = self.get_goal_at_predicates()

        render_mode = "human" if self._using_gui else "rgb_array"
        self._gym_env = mujoco_kitchen_gym.make("FrankaKitchen-v1", render_mode=render_mode)

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
        state_info = {}
        for site in _TRACKED_SITES:
            state_info[site] = get_site_xpos(self._gym_env.model, self._gym_env.data, site).copy()
            # Include rotation for gripper.
            if site == "EEF":
                xmat = get_site_xmat(self._gym_env.model, self._gym_env.data, site).copy()
                quat = mat2quat(xmat)
                state_info[site] = np.concatenate([state_info[site], quat])                             
        for joint in _TRACKED_SITE_TO_JOINT.values():
            state_info[joint] = get_joint_qpos(self._gym_env.model, self._gym_env.data, joint).copy()
        for body in _TRACKED_BODIES:
            body_id = self._gym_env.robot_env.model_names.body_name2id[body]
            state_info[body] = self._gym_env.data.xpos[body_id].copy()
        return state_info

    @classmethod
    def get_pre_push_delta_pos(cls, obj: Object) -> Tuple[float, float, float]:
        """Get dx, dy, dz offset for pushing."""
        try:
            return cls.obj_name_to_pre_push_dpos[obj.name]
        except KeyError:
            return (0.0, 0.0, 0.0)

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
        return {self._At, self._TurnedOn, self._OnTop}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._At, self._TurnedOn, self._OnTop}

    @property
    def types(self) -> Set[Type]:
        return {self.gripper_type, self.object_type}

    @property
    def action_space(self) -> Box:
        return self._gym_env.action_space

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
                obj = Object("gripper", cls.gripper_type)
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
                obj = Object(obj_name, cls.object_type)
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
        kettle = Object("kettle", self.object_type)
        burner = Object("burner4", self.object_type)
        knob = Object("knob4", self.object_type)
        light = Object("light", self.object_type)
        goal_desc = self._current_task.goal_description
        kettle_on_burner = self._OnTop_holds(state, [kettle, burner])
        knob_turned_on = self._On_holds(state, [knob])
        light_turned_on = self._On_holds(state, [light])
        if goal_desc == ("Move the kettle to the back burner and turn it on; "
                         "also turn on the light"):
            return kettle_on_burner and knob_turned_on and light_turned_on
        if goal_desc == "Move the kettle to the back burner":
            return kettle_on_burner
        if goal_desc == "Turn on the back burner":
            return knob_turned_on
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
    def _At_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        gripper, obj = objects
        obj_xyz = np.array(
            [state.get(obj, "x"),
             state.get(obj, "y"),
             state.get(obj, "z")])
        # We care about whether we're "at" the pre-push position for obj.
        dpos = cls.get_pre_push_delta_pos(obj)
        gripper_xyz = np.array([
            state.get(gripper, "x"),
            state.get(gripper, "y"),
            state.get(gripper, "z")
        ])
        return np.allclose(obj_xyz + dpos, gripper_xyz, atol=cls.at_atol)

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
    def _On_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj = objects[0]
        if "knob" in obj.name:
            return state.get(obj, "angle") < cls.on_angle_thresh
        if obj.name == "light":
            return state.get(obj, "x") < cls.light_on_thresh
        return False

    def _copy_observation(self, obs: Observation) -> Observation:
        return copy.deepcopy(obs)

    def get_goal_at_predicates(self: Any) -> Sequence[Predicate]:
        """Defined public so that the perceiver can use it."""
        return [
            Predicate("At", [self.gripper_type, self.object_type],
                      self._At_holds),
            Predicate("OnTop", [self.object_type, self.object_type],
                      self._OnTop_holds),
            Predicate("TurnedOn", [self.object_type], self._On_holds),
        ]
