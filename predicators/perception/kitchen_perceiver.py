"""A Kitchen-specific perceiver."""

from predicators.envs.kitchen import KitchenEnv
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, GroundAtom, Observation, \
    State, Task, Video


class KitchenPerceiver(BasePerceiver):
    """A Kitchen-specific perceiver."""

    @classmethod
    def get_name(cls) -> str:
        return "kitchen"

    def reset(self, env_task: EnvironmentTask) -> Task:
        state = self._observation_to_state(env_task.init_obs)
        pred_name_to_pred = KitchenEnv.create_predicates()
        OnTop = pred_name_to_pred["OnTop"]
        TurnedOn = pred_name_to_pred["TurnedOn"]
        kettle = KitchenEnv.object_name_to_object("kettle")
        knob4 = KitchenEnv.object_name_to_object("knob4")
        burner4 = KitchenEnv.object_name_to_object("burner4")
        light = KitchenEnv.object_name_to_object("light")
        goal_desc = env_task.goal_description
        if goal_desc == ("Move the kettle to the back burner and turn it on; "
                         "also turn on the light"):
            goal = {
                GroundAtom(TurnedOn, [knob4]),
                GroundAtom(OnTop, [kettle, burner4]),
                GroundAtom(TurnedOn, [light]),
            }
        elif goal_desc == "Move the kettle to the back burner":
            goal = {GroundAtom(OnTop, [kettle, burner4])}
        elif goal_desc == "Turn on the back burner":
            goal = {
                GroundAtom(TurnedOn, [knob4]),
            }
        elif goal_desc == "Turn on the light":
            goal = {
                GroundAtom(TurnedOn, [light]),
            }
        else:
            raise NotImplementedError(f"Unrecognized goal: {goal_desc}")
        return Task(state, goal)

    def step(self, observation: Observation) -> State:
        return self._observation_to_state(observation)

    def _observation_to_state(self, obs: Observation) -> State:
        return KitchenEnv.state_info_to_state(obs["state_info"])

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        raise NotImplementedError("Mental images not implemented for kitchen")
