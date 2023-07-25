"""A Kitchen-specific perceiver."""

from predicators.envs.kitchen import KitchenEnv
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, GroundAtom, Object, \
    Observation, State, Task


class KitchenPerceiver(BasePerceiver):
    """A Kitchen-specific perceiver."""

    @classmethod
    def get_name(cls) -> str:
        return "kitchen"

    def reset(self, env_task: EnvironmentTask) -> Task:
        state = self._observation_to_state(env_task.init_obs)
        OnTop = KitchenEnv.get_goal_at_predicates(KitchenEnv)[1]
        TurnedOn = KitchenEnv.get_goal_at_predicates(KitchenEnv)[2]
        kettle = Object("kettle", KitchenEnv.object_type)
        knob = Object("knob3", KitchenEnv.object_type)
        burner = Object("burner2", KitchenEnv.object_type)
        goal_desc = env_task.goal_description
        if goal_desc == "Move the kettle to the back burner and turn it on":
            goal = {
                GroundAtom(TurnedOn, [knob]),
                GroundAtom(OnTop, [kettle, burner])
            }
        elif goal_desc == "Move the kettle to the back burner":
            goal = {GroundAtom(OnTop, [kettle, burner])}
        elif goal_desc == "Turn on the back burner":
            goal = {
                GroundAtom(TurnedOn, [knob]),
            }
        else:
            raise NotImplementedError(f"Unrecognized goal: {goal_desc}")
        return Task(state, goal)

    def step(self, observation: Observation) -> State:
        return self._observation_to_state(observation)

    def _observation_to_state(self, obs: Observation) -> State:
        return KitchenEnv.state_info_to_state(obs["state_info"])
