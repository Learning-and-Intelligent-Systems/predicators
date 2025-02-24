"""An approach that wraps a base approach for a spot environment. Detects when
objects have been "lost" and executes a specific controller to "find" them.
Passes control back to the main approach once all objects are not lost.

Assumes that some objects in the environment have a feature called "lost" that
is 1.0 if the object is lost and 0.0 otherwise. This feature should be tracked
by a perceiver.

For now, the "find" policy is represented with a single action that is
extracted from the environment.
"""

import logging
from typing import Any, Callable, List, Optional, Set

from gym.spaces import Box

from predicators import utils
from predicators.approaches import BaseApproach, BaseApproachWrapper
from predicators.envs.spot_env import get_detection_id_for_object, get_robot
from predicators.spot_utils.skills.spot_find_objects import find_objects
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.utils import get_allowed_map_regions
from predicators.structs import Action, Object, ParameterizedOption, \
    Predicate, SpotActionExtraInfo, State, Task, Type


class SpotWrapperApproach(BaseApproachWrapper):
    """Always "find" if some object is lost."""

    def __init__(self, base_approach: BaseApproach,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(base_approach, initial_predicates, initial_options,
                         types, action_space, train_tasks)
        self._base_approach_has_control = False  # for execution monitoring
        self._allowed_regions = get_allowed_map_regions()

    @classmethod
    def get_name(cls) -> str:
        return "spot_wrapper"

    @property
    def is_learning_based(self) -> bool:
        return self._base_approach.is_learning_based

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        # Maintain policy from the base approach.
        base_approach_policy: Optional[Callable[[State], Action]] = None
        self._base_approach_has_control = False
        need_stow = False

        def _policy(state: State) -> Action:
            nonlocal base_approach_policy, need_stow
            # If we think that we're done, return the done action.
            if task.goal_holds(state, self._vlm):
                extra_info = SpotActionExtraInfo("done", [], None, tuple(),
                                                 None, tuple())
                return utils.create_spot_env_action(extra_info)
            # If some objects are lost, find them.
            lost_objects: Set[Object] = set()
            for obj in state:
                if "lost" in obj.type.feature_names and \
                    state.get(obj, "lost") > 0.5:
                    lost_objects.add(obj)
            # Need to find the objects.
            if lost_objects:
                logging.info(f"[Spot Wrapper] Lost objects: {lost_objects}")
                # Reset the base approach policy.
                base_approach_policy = None
                need_stow = True
                self._base_approach_has_control = False
                robot, localizer, lease_client = get_robot()
                lost_object_ids = {
                    get_detection_id_for_object(o)
                    for o in lost_objects
                }
                allowed_regions = self._allowed_regions
                extra_info = SpotActionExtraInfo(
                    "find-objects", [], find_objects,
                    (state, self._rng, robot, localizer, lease_client,
                     lost_object_ids, allowed_regions), None, tuple())
                return utils.create_spot_env_action(extra_info)
            # Found the objects. Stow the arm before replanning.
            if need_stow:
                logging.info("[Spot Wrapper] Lost objects found, stowing.")
                base_approach_policy = None
                need_stow = False
                self._base_approach_has_control = False
                robot, _, _ = get_robot()
                extra_info = SpotActionExtraInfo("stow-arm", [], stow_arm,
                                                 (robot, ), None, tuple())
                return utils.create_spot_env_action(extra_info)
            # Check if we need to re-solve.
            if base_approach_policy is None:
                logging.info("[Spot Wrapper] Replanning with base approach.")
                cur_task = Task(state, task.goal)
                base_approach_policy = self._base_approach.solve(
                    cur_task, timeout)
                self._base_approach_has_control = True
                # Need to call this once here to fix off-by-one issue.
                _ = self._base_approach.get_execution_monitoring_info()
                # NOTE: might be worth reinstating the above line to check for
                # weird issues that might come up.
                # atom_seq = self._base_approach.get_execution_monitoring_info()
                # assert all(a.holds(state) for a in atom_seq[0])
            # Use the base policy.
            return base_approach_policy(state)

        return _policy

    def get_execution_monitoring_info(self) -> List[Any]:
        if self._base_approach_has_control:
            return self._base_approach.get_execution_monitoring_info()
        return []
