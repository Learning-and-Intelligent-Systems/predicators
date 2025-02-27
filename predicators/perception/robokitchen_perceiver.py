"""A RoboKitchen-specific perceiver."""

from predicators.envs.robo_kitchen import RoboKitchenEnv
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, GroundAtom, Observation, \
    State, Task, Video


class RoboKitchenPerceiver(BasePerceiver):
    """A Kitchen-specific perceiver."""

    @classmethod
    def get_name(cls) -> str:
        return "robo_kitchen"

    def reset(self, env_task: EnvironmentTask) -> Task:
        state = self._observation_to_state(env_task.init_obs)
        pred_name_to_pred = RoboKitchenEnv.create_predicates()
        # OnTop = pred_name_to_pred["OnTop"]
        # TurnedOn = pred_name_to_pred["TurnedOn"]
        # KettleBoiling = pred_name_to_pred["KettleBoiling"]
        HingeOpen = pred_name_to_pred["HingeOpen"]
        # kettle = RoboKitchenEnv.object_name_to_object("kettle")
        # knob4 = RoboKitchenEnv.object_name_to_object("knob4")
        # knob3 = RoboKitchenEnv.object_name_to_object("knob3")
        # burner4 = RoboKitchenEnv.object_name_to_object("burner4")
        # burner3 = RoboKitchenEnv.object_name_to_object("burner3")
        # light = RoboKitchenEnv.object_name_to_object("light")
        door = RoboKitchenEnv.object_name_to_object("door_angle")
        goal_desc = env_task.goal_description
        if goal_desc == 'OpenSingleDoor':
            goal = {
                GroundAtom(HingeOpen, [door]),
            }
        # elif goal_desc == "Move the kettle to the back left burner":
        #     goal = {GroundAtom(OnTop, [kettle, burner4])}
        # elif goal_desc == "Move the kettle to the back right burner":
        #     goal = {GroundAtom(OnTop, [kettle, burner3])}
        # elif goal_desc == "Turn on the back left burner":
        #     goal = {
        #         GroundAtom(TurnedOn, [knob4]),
        #     }
        # elif goal_desc == "Turn on the back right burner":
        #     goal = {
        #         GroundAtom(TurnedOn, [knob3]),
        #     }
        # elif goal_desc == "Turn on the light":
        #     goal = {
        #         GroundAtom(TurnedOn, [light]),
        #     }
        # elif goal_desc == ("Move the kettle to the back left burner "
        #                    "and turn it on"):
        #     goal = {GroundAtom(KettleBoiling, [kettle, burner4, knob4])}
        # elif goal_desc == ("Move the kettle to the back right burner "
        #                    "and turn it on"):
        #     goal = {GroundAtom(KettleBoiling, [kettle, burner3, knob3])}
        else:
            raise NotImplementedError(f"Unrecognized goal: {goal_desc}")
        return Task(state, goal)

    def step(self, observation: Observation) -> State:
        return self._observation_to_state(observation)

    def _observation_to_state(self, obs: Observation) -> State:
        # Get contact set from observation, or use empty set if not provided
        contact_set = obs.get("contact_set", set())

        # Convert state_info to state, passing in the contact set
        state = RoboKitchenEnv.state_info_to_state(obs["state_info"], contact_set)

        assert state.simulator_state is not None
        state.simulator_state["images"] = obs["obs_images"]
        return state

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        raise NotImplementedError("Mental images not implemented for kitchen")
