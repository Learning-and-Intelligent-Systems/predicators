"""A Kitchen environment wrapping kitchen from https://github.com/google-
research/relay-policy-learning."""
import copy
from typing import Any, Dict, List, Optional, Sequence, Set

import matplotlib
import numpy as np
from gym.spaces import Box
from mujoco_kitchen.kitchen_envs import OBS_ELEMENT_INDICES
from mujoco_kitchen.utils import make_env

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Image, Object, \
    Observation, Predicate, State, Type, Video


class KitchenEnv(BaseEnv):
    """Kitchen environment wrapping dm_control Kitchen."""

    gripper_type = Type("gripper", ["x", "y", "z"])
    object_type = Type("obj", ["x", "y", "z"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates
        self._At = self.get_goal_at_predicate()

        # NOTE: we can change the level by modifying what we pass
        # into gym.make here.
        self._gym_env = make_env(
            "kitchen", "microwave", {
                "usage_kwargs": {
                    "max_path_length": 50,
                    "use_raw_action_wrappers": False,
                    "unflatten_images": False
                }
            })

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
        state = self._gym_env.env.get_env_state()[0]
        state_info = {}
        for key, val in OBS_ELEMENT_INDICES.items():
            state_info[key] = [state[i] for i in val]
        important_sites = [
            "hinge_site1", "hinge_site2", "kettle_site", "microhandle_site",
            "knob1_site", "knob2_site", "knob3_site", "knob4_site",
            "light_site", "slide_site", "end_effector"
        ]
        for site in important_sites:
            state_info[site] = self._gym_env.get_site_xpos(site)
            # Potentially can get this ^ from state
        return state_info

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
        arr: Image = self._gym_env.render('rgb_array')  # type: ignore
        return [arr]

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._At}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._At}

    @property
    def types(self) -> Set[Type]:
        return {self.gripper_type, self.object_type}

    @property
    def action_space(self) -> Box:
        # One-hot encoding of discrete action space with parameters.
        assert (self._gym_env.num_primitives +
                self._gym_env.max_arg_len, ) == (29, )  # type: ignore
        assert self._gym_env.action_space.shape == (29, )
        return self._gym_env.action_space

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Resets the current state to the train or test task initial state."""
        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_observation = {
            "obs": self._current_task.init_obs,
            "state_info": self.get_object_centric_state_info()
        }
        # We now need to reset the underlying gym environment to the correct
        # state.
        seed = self._get_task_seed(train_or_test, task_idx)
        self._reset_initial_state_from_seed(seed)
        return self._copy_observation(self._current_observation)

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for gym envs. " +
                                  "Try using --bilevel_plan_without_sim True")

    def step(self, action: Action) -> Observation:
        self._gym_env.step(action.arr)
        obs = self._gym_env.render(mode='rgb_array')  # type: ignore
        self._current_observation = {
            "obs": obs,
            "state_info": self.get_object_centric_state_info()
        }
        return self._copy_observation(self._current_observation)

    def state_info_to_state(self: Any, state_info: Any) -> State:
        """Get state from state info dictionary."""
        state_dict = {}
        for key, val in state_info.items():
            if "_site" in key:
                obj_name = key.replace("_site", "")
                obj = Object(obj_name, self.object_type)
                state_dict[obj] = {"x": val[0], "y": val[1], "z": val[2]}
            elif key == "end_effector":
                obj = Object("gripper", self.gripper_type)
                state_dict[obj] = {"x": val[0], "y": val[1], "z": val[2]}
        state = utils.create_state_from_dict(state_dict)
        return state

    def goal_reached(self) -> bool:
        state = self.state_info_to_state(
            self._current_observation["state_info"])
        gripper = Object("gripper", self.gripper_type)
        obj = Object("knob1", self.object_type)
        return self._At_holds(state=state, objects=[gripper, obj])

    def _get_tasks(self, num: int,
                   train_or_test: str) -> List[EnvironmentTask]:
        tasks = []
        for task_idx in range(num):
            seed = self._get_task_seed(train_or_test, task_idx)
            init_obs = self._reset_initial_state_from_seed(seed)
            goal_description = "Move Gripper to Knob1"
            task = EnvironmentTask(init_obs, goal_description)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int) -> Observation:
        self._gym_env.seed(seed)  # type: ignore
        self._gym_env.reset()
        obs = self._gym_env.render(mode='rgb_array')  # type: ignore
        return {"obs": obs, "state_info": self.get_object_centric_state_info()}

    @classmethod
    def _At_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        gripper, obj = objects
        obj_xyz = [
            state.get(obj, "x"),
            state.get(obj, "y"),
            state.get(obj, "z")
        ]
        gripper_xyz = [
            state.get(gripper, "x"),
            state.get(gripper, "y"),
            state.get(gripper, "z")
        ]
        return np.allclose(obj_xyz, gripper_xyz, atol=0.09)

    def _copy_observation(self, obs: Observation) -> Observation:
        return copy.deepcopy(obs)

    def get_goal_at_predicate(self: Any) -> Predicate:
        """Defined public so that the perceiver can use it."""
        return Predicate("At", [self.gripper_type, self.object_type],
                         self._At_holds)

    @staticmethod
    def _get_task_seed(train_or_test: str, task_idx: int) -> int:
        assert task_idx < CFG.test_env_seed_offset
        # SeedSequence generates a sequence of random values given an integer
        # "entropy". We use CFG.seed to define the "entropy" and then get the
        # n^th generated random value and use that to seed the gym environment.
        # This is all to avoid unintentional dependence between experiments
        # that are conducted with consecutive random seeds. For example, if
        # we used CFG.seed + task_idx to seed the gym environment, there would
        # be overlap between experiments when CFG.seed = 1, CFG.seed = 2, etc.
        entropy = CFG.seed
        if train_or_test == "test":
            entropy += CFG.test_env_seed_offset
        seed_sequence = np.random.SeedSequence(entropy)
        # Need to cast to int because generate_state() returns a numpy int.
        task_seed = int(seed_sequence.generate_state(task_idx + 1)[-1])
        return task_seed
