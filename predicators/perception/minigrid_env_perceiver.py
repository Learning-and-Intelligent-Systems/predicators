"""A minigrid-specific perceiver."""

from typing import Dict, Tuple

import numpy as np

from predicators import utils
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
        # Used for object tracking of the boxes, which are the only objects
        # with ambiguity. The keys are the object names.
        self._box_loc_to_name: Dict[Tuple[int, int], str] = {}

    @classmethod
    def get_name(cls) -> str:
        return "minigrid_env"

    def reset(self, env_task: EnvironmentTask) -> Task:
        self._box_loc_to_name.clear()  # reset the object tracking dictionary
        state = self._observation_to_state(env_task.init_obs)
        assert env_task.goal_description == "Get to the goal"
        IsAgent, At, IsGoal = MiniGridEnv.get_goal_predicates()
        assert len(MiniGridEnv.get_objects_of_enum(state, "agent")) == 1
        assert len(MiniGridEnv.get_objects_of_enum(state, "goal")) == 1
        agent_obj = list(MiniGridEnv.get_objects_of_enum(state, "agent"))[0]
        goal_obj = list(MiniGridEnv.get_objects_of_enum(state, "goal"))[0]
        goal = {GroundAtom(IsAgent, [agent_obj]),
                GroundAtom(At, [agent_obj, goal_obj]),
                GroundAtom(IsGoal, [goal_obj])}
        return Task(state, goal)

    def step(self, observation: Observation) -> State:
        return self._observation_to_state(observation)
    
    def _observation_to_objects(self, obs: Observation) -> Dict[str, Tuple[int, int]]:
        objs = []
        visual = obs[0]['image']
        direction = obs[0]['direction']
        agent_pos = (3,6)
        # from PIL import Image
        # import numpy as np
        # visual_rgb = np.array([[COLORS[IDX_TO_COLOR[visual[r][c][1]]] if IDX_TO_OBJECT[visual[r, c][0]] != 'empty' else [0,0,0] for c in range(visual.shape[1])] for r in range(visual.shape[0])], dtype=np.uint8)
        # img = Image.fromarray(visual_rgb)
        # img.show()
        objs.append(('agent',
                     None, 
                     direction,
                     agent_pos[0],
                     agent_pos[1]))
        objs.append(('empty',
                     'black', 
                     0,
                     agent_pos[0],
                     agent_pos[1]))
        for r in range(visual.shape[0]):
            for c in range(visual.shape[1]):
                obj = [IDX_TO_OBJECT[visual[r, c][0]], IDX_TO_COLOR[visual[r, c][1]], visual[r, c][2], r, c]
                if obj[0] == 'empty':
                    obj[1] = 'black'
                objs.append(tuple(obj))
        return objs

    def _observation_to_state(self, obs: Observation) -> State:
        state_dict = {}
        objs = self._observation_to_objects(obs)

        def _get_object_name(r: int, c: int, type_name: str) -> str:
            # Put the location of the static objects in their names for easier
            # debugging.
            if type_name == "agent":
                return "agent"
            return f"{type_name}_{r}_{c}"

        for type_name, color, obj_state, r, c in objs:
            enum = MiniGridEnv.name_to_enum[type_name]
            object_name = _get_object_name(r, c, type_name)
            obj = Object(object_name, MiniGridEnv.object_type)
            state_dict[obj] = {
                "row": r,
                "column": c,
                "type": enum,
                "state": obj_state,
            }

        state = utils.create_state_from_dict(state_dict)
        return state

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        raise NotImplementedError("Mental images not implemented for minigrid")
