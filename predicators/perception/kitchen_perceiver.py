"""A Kitchen-specific perceiver."""

from predicators.envs.kitchen import KitchenEnv
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, GroundAtom, Object, \
    Observation, State, Task

# Each observation is a tuple of four 2D boolean masks (numpy arrays).
# The order is: free, goals, boxes, player.


class KitchenPerceiver(BasePerceiver):
    """A Kitchen-specific perceiver."""

    @classmethod
    def get_name(cls) -> str:
        return "kitchen"

    def reset(self, env_task: EnvironmentTask) -> Task:
        state = self._observation_to_state(env_task.init_obs)
        assert env_task.goal_description == "Move Gripper to Knob1"
        At = KitchenEnv.get_goal_at_predicate(KitchenEnv)
        gripper = Object("gripper", KitchenEnv.gripper_type)
        obj = Object("knob1", KitchenEnv.object_type)
        goal = {GroundAtom(At, [gripper, obj])}
        return Task(state, goal)

    def step(self, observation: Observation) -> State:
        return self._observation_to_state(observation)

    def _observation_to_state(self, obs: Observation) -> State:
        return KitchenEnv.state_info_to_state(KitchenEnv, obs["state_info"])
