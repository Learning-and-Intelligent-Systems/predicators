"""A minigrid-specific perceiver."""

import sys
from typing import Dict, Tuple

import numpy as np

from predicators import utils
from predicators.settings import CFG
from predicators.envs.minigrid_env import MiniGridEnv
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, GroundAtom, Object, \
    Observation, State, Task, Video

from minigrid.core.constants import (
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
)

class MiniGridPerceiver(BasePerceiver):
    """A minigrid-specific perceiver."""

    def __init__(self) -> None:
        super().__init__()
        # Ditection: 0 - right, 1 - down, 2 - left, 3 - up
        self.state_dict = {}
        self.agent_pov_pos = (3,6) # agent's point of view is always at (3,6)
        self.agent_pos = (0,0) # starts at origin
        self.direction = 0 # starts facing right
        self.last_obs = None

    @classmethod
    def get_name(cls) -> str:
        return "minigrid_env"

    def reset(self, env_task: EnvironmentTask) -> Task:
        self.state_dict.clear()
        state = self._observation_to_state(env_task.init_obs)
        if env_task.goal_description == "Get to the goal":
            IsAgent, At, IsGoal, IsBall, IsKey, IsBox, \
            IsRed, IsGreen, IsBlue, IsPurple, IsYellow, IsGrey = MiniGridEnv.get_goal_predicates()
            assert len(MiniGridEnv.get_objects_of_enum(state, "agent")) == 1
            assert len(MiniGridEnv.get_objects_of_enum(state, "goal")) == 1
            agent_obj = list(MiniGridEnv.get_objects_of_enum(state, "agent"))[0]
            goal_obj = list(MiniGridEnv.get_objects_of_enum(state, "goal"))[0]
            goal = {GroundAtom(IsAgent, [agent_obj]),
                    GroundAtom(At, [agent_obj, goal_obj]),
                    GroundAtom(IsGoal, [goal_obj])}
        elif "go to the " in env_task.goal_description:
            color, obj_type = env_task.goal_description.split("go to the ")[1].split(" ")[0:2]
            obj_name = f"{color}_{obj_type}"
            IsAgent, At, IsGoal, IsBall, IsKey, IsBox, \
            IsRed, IsGreen, IsBlue, IsPurple, IsYellow, IsGrey = MiniGridEnv.get_goal_predicates()
            assert len(MiniGridEnv.get_objects_of_enum(state, "agent")) == 1
            assert len(MiniGridEnv.get_objects_of_enum(state, obj_type)) > 1
            agent_obj = list(MiniGridEnv.get_objects_of_enum(state, "agent"))[0]
            for obj in MiniGridEnv.get_objects_of_enum(state, obj_type):
                if obj.name == obj_name:
                    goal_obj = obj
            obj_type_to_predicate = {
                "ball": IsBall,
                "key": IsKey,
                "box": IsBox
            }  
            color_to_predicate = {
                "red": IsRed,
                "green": IsGreen,
                "blue": IsBlue,
                "purple": IsPurple,
                "yellow": IsYellow,
                "grey": IsGrey
            } 
            goal = {GroundAtom(IsAgent, [agent_obj]),
                    GroundAtom(At, [agent_obj, goal_obj]),
                    GroundAtom(obj_type_to_predicate[obj_type], [goal_obj]),
                    GroundAtom(color_to_predicate[color], [goal_obj]),
                    }
        elif env_task.goal_description == "get to the green goal square":
            IsAgent, At, IsGoal, IsBall, IsKey, IsBox, \
            IsRed, IsGreen, IsBlue, IsPurple, IsYellow, IsGrey = MiniGridEnv.get_goal_predicates()
            assert len(MiniGridEnv.get_objects_of_enum(state, "agent")) == 1
            assert len(MiniGridEnv.get_objects_of_enum(state, "goal")) == 1
            agent_obj = list(MiniGridEnv.get_objects_of_enum(state, "agent"))[0]
            goal_obj = list(MiniGridEnv.get_objects_of_enum(state, "goal"))[0]
            goal = {GroundAtom(IsAgent, [agent_obj]),
                    GroundAtom(At, [agent_obj, goal_obj]),
                    GroundAtom(IsGoal, [goal_obj])}
        elif env_task.goal_description.startswith("get a") or \
                env_task.goal_description.startswith("go get a") or \
                env_task.goal_description.startswith("fetch a") or \
                env_task.goal_description.startswith("go fetch a") or \
                env_task.goal_description.startswith("you must fetch a") or \
                env_task.goal_description.startswith("pick up the"):
                color, obj_type = env_task.goal_description.split(" ")[-2:]
                obj_name = f"{color}_{obj_type}"
                IsAgent, At, IsGoal, IsBall, IsKey, IsBox, \
                IsRed, IsGreen, IsBlue, IsPurple, IsYellow, IsGrey, \
                Holding, Near = MiniGridEnv.get_goal_predicates()
                assert len(MiniGridEnv.get_objects_of_enum(state, "agent")) == 1
                assert len(MiniGridEnv.get_objects_of_enum(state, obj_type)) > 1
                agent_obj = list(MiniGridEnv.get_objects_of_enum(state, "agent"))[0]
                for obj in MiniGridEnv.get_objects_of_enum(state, obj_type):
                    if obj.name == obj_name:
                        goal_obj = obj
                obj_type_to_predicate = {
                    "ball": IsBall,
                    "key": IsKey,
                    "box": IsBox
                }  
                color_to_predicate = {
                    "red": IsRed,
                    "green": IsGreen,
                    "blue": IsBlue,
                    "purple": IsPurple,
                    "yellow": IsYellow,
                    "grey": IsGrey
                } 
                goal = {GroundAtom(Holding, [goal_obj]),
                        GroundAtom(obj_type_to_predicate[obj_type], [goal_obj]),
                        GroundAtom(color_to_predicate[color], [goal_obj])}
        else:
            raise NotImplementedError(f"Goal description {env_task.goal_description} not supported")
        return Task(state, goal)

    def step(self, observation: Observation) -> State:
        return self._observation_to_state(observation)
    
    def _observation_to_objects(self, obs: Observation) -> Dict[str, Tuple[int, int]]:
        objs = []
        visual = obs[0]['image']
        direction = obs[0]['direction']
        objs.append(('agent',
                     None, 
                     direction,
                     0,
                     0))
        objs.append(('empty',
                     'black', 
                     0,
                     0,
                     0))
        for r in range(visual.shape[0]):
            for c in range(visual.shape[1]):
                obj = [IDX_TO_OBJECT[visual[r, c][0]], IDX_TO_COLOR[visual[r, c][1]], visual[r, c][2], r - self.agent_pov_pos[0], c - self.agent_pov_pos[1]]
                if obj[0] == 'empty':
                    obj[1] = 'black'
                objs.append(tuple(obj))
        return objs
    
    def transform_point(self, x1, y1, o1, x2, y2):
        # Compute global coordinates directly
        x_prime = x1 + x2 * np.cos(o1) - y2 * np.sin(o1)
        y_prime = y1 + x2 * np.sin(o1) + y2 * np.cos(o1)
        return x_prime, y_prime
    
    # Updated function with mathematically correct direction-to-radians mapping
    def _globalize_coords(self, r: int, c: int) -> Tuple[int, int]:
        # Adjusted direction-to-radian mapping
        direction_to_radian = {
            0: 0,                # right
            1: -np.pi / 2,       # down
            2: np.pi,            # left
            3: np.pi / 2         # up
        }
        o1 = direction_to_radian[self.direction]
        x1, y1 = self.agent_pos[0], self.agent_pos[1]
        x2, y2 = r, -c  # Use c directly
        x_prime, y_prime = self.transform_point(x1, y1, o1, x2, y2)
        return int(round(x_prime)), int(round(y_prime))

    def _observation_to_state(self, obs: Observation) -> State:
        import numpy as np

        self.direction = obs[0]['direction']
        if len(obs) == 5:
            if obs[4]['last_action'] == 2: # Moved Forward
                if (not np.array_equal(self.last_obs[0]['image'], obs[0]['image'])) or \
                    not np.array_equal(obs[0]['image'][self.agent_pov_pos[0], self.agent_pov_pos[1]-1], np.array([2, 5, 0], dtype=np.uint8)):
                    if self.direction == 0: # right (0, 1)
                        self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
                    elif self.direction == 1: # down (1, 0)
                        self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
                    elif self.direction == 2: # left (0, -1)
                        self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
                    elif self.direction == 3: # up (-1, 0)
                        self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        self.last_obs = obs

        objs = self._observation_to_objects(obs)

        def _get_object_name(r: int, c: int, type_name: str, color: str) -> str:
            # Put the location of the static objects in their names for easier
            # debugging.
            if type_name == "agent":
                return "agent"
            if type_name in ["empty", "wall"]:
                return f"{type_name}_{r}_{c}"
            else:
                return f"{color}_{type_name}"

        for type_name, color, obj_state, r, c in objs:
            enum = MiniGridEnv.name_to_enum[type_name]
            if CFG.minigrid_gym_fully_observable:
                global_r, global_c = r, c
            else:
                global_r, global_c = self._globalize_coords(r, c)
            if type_name in ["goal", "agent"]:
                object_name = type_name
                if type_name == "agent" and not CFG.minigrid_gym_fully_observable:
                    assert (global_r, global_c) == self.agent_pos
            else:
                object_name = _get_object_name(global_r, global_c, type_name, color)
            obj = Object(object_name, MiniGridEnv.object_type)
            self.state_dict[obj] = {
                "row": global_r,
                "column": global_c,
                "type": enum,
                "state": obj_state,
                "color": color,
            }

        if all([val["type"] != MiniGridEnv.name_to_enum['goal'] for key, val in self.state_dict.items()]):
            enum = MiniGridEnv.name_to_enum["goal"]
            object_name = "goal"
            obj = Object(object_name, MiniGridEnv.object_type)
            self.state_dict[obj] = {
                "row": sys.maxsize,
                "column": sys.maxsize,
                "type": enum,
                "state": -1,
                "color": 'green',
            }

        for color in ['blue', 'green', 'grey', 'purple', 'red', 'yellow']:
            for obj_type in ['key', 'ball', 'box']:
                if all([not (val["type"] == MiniGridEnv.name_to_enum[obj_type] and val["color"] == color) for key, val in self.state_dict.items()]):
                    enum = MiniGridEnv.name_to_enum[obj_type]
                    object_name = f"{color}_{obj_type}"
                    obj = Object(object_name, MiniGridEnv.object_type)
                    self.state_dict[obj] = {
                        "row": sys.maxsize,
                        "column": sys.maxsize,
                        "type": enum,
                        "state": -1,
                        "color": color,
                    }

        state = utils.create_state_from_dict(self.state_dict)
        return state

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        raise NotImplementedError("Mental images not implemented for minigrid")
