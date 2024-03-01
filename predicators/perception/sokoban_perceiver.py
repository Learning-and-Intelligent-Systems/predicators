"""A sokoban-specific perceiver."""

from typing import Dict, Tuple

import numpy as np

from predicators import utils
from predicators.envs.sokoban import SokobanEnv
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, GroundAtom, Object, \
    Observation, State, Task, Video

# Each observation is a tuple of four 2D boolean masks (numpy arrays).
# The order is: free, goals, boxes, player.


class SokobanPerceiver(BasePerceiver):
    """A sokoban-specific perceiver."""

    def __init__(self) -> None:
        super().__init__()
        # Used for object tracking of the boxes, which are the only objects
        # with ambiguity. The keys are the object names.
        self._box_loc_to_name: Dict[Tuple[int, int], str] = {}

    @classmethod
    def get_name(cls) -> str:
        return "sokoban"

    def reset(self, env_task: EnvironmentTask) -> Task:
        self._box_loc_to_name.clear()  # reset the object tracking dictionary
        state = self._observation_to_state(env_task.init_obs)
        assert env_task.goal_description == "Cover all the goals with boxes"
        GoalCovered = SokobanEnv.get_goal_covered_predicate()
        goal_objs = SokobanEnv.get_objects_of_enum(state, "goal")
        goal = {GroundAtom(GoalCovered, [b]) for b in goal_objs}
        return Task(state, goal)

    def step(self, observation: Observation) -> State:
        return self._observation_to_state(observation)

    def _observation_to_state(self, obs: Observation) -> State:
        state_dict = {}

        walls, goals, boxes, player = obs
        type_to_mask = {
            "free": np.logical_not(walls | goals),
            "goal": goals,
            "box": boxes,
            "player": player
        }
        assert set(type_to_mask) == set(SokobanEnv.name_to_enum)

        # Handle moving boxes.
        new_locs = set((r, c) for r, c in np.argwhere(boxes))
        if not self._box_loc_to_name:
            # First time, so name the boxes arbitrarily.
            for i, (r, c) in enumerate(sorted(new_locs)):
                self._box_loc_to_name[(r, c)] = f"box_{i}"
        else:
            # Assume that at most one box has changed.
            old_locs = set(self._box_loc_to_name)
            changed_new_locs = new_locs - old_locs
            changed_old_locs = old_locs - new_locs
            if changed_new_locs:
                assert len(changed_new_locs) == 1
                assert len(changed_old_locs) == 1
                new_loc, = changed_new_locs
                old_loc, = changed_old_locs
                moved_box_name = self._box_loc_to_name.pop(old_loc)
                self._box_loc_to_name[new_loc] = moved_box_name

        def _get_object_name(r: int, c: int, type_name: str) -> str:
            # Put the location of the static objects in their names for easier
            # debugging.
            if type_name in {"free", "goal"}:
                return f"{type_name}_{r}_{c}"
            if type_name == "player":
                return "player"
            assert type_name == "box"
            return self._box_loc_to_name[(r, c)]

        for type_name, mask in type_to_mask.items():
            enum = SokobanEnv.name_to_enum[type_name]
            i = 0
            for r, c in np.argwhere(mask):
                object_name = _get_object_name(r, c, type_name)
                obj = Object(object_name, SokobanEnv.object_type)
                state_dict[obj] = {
                    "row": r,
                    "column": c,
                    "type": enum,
                }
                i += 1

        state = utils.create_state_from_dict(state_dict)
        return state

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        raise NotImplementedError("Mental images not implemented for sokoban")
