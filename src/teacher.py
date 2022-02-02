"""Contains classes defining a teacher that can supply additional, oracular
information to assist an agent during online learning."""

from __future__ import annotations
from predicators.src.structs import State, Task, Query, Response,\
    GroundAtomsHoldQuery, GroundAtomsHoldResponse, DemonstrationQuery,\
        DemonstrationResponse
from predicators.src.settings import CFG, get_allowed_query_type_names
from predicators.src.envs import create_env
from predicators.src.approaches import ApproachTimeout, ApproachFailure
from predicators.src.approaches.oracle_approach import OracleApproach
from predicators.src import utils

################################################################################
#                                Core classes                                  #
################################################################################


class Teacher:
    """The teacher can respond to queries of various types."""

    def __init__(self) -> None:
        env = create_env(CFG.env)
        self._pred_name_to_pred = {pred.name: pred for pred in env.predicates}
        self._allowed_query_type_names = get_allowed_query_type_names()
        env_preds, _ = utils.parse_config_excluded_predicates(env)
        # NOTE: this below var is not actually used, just passed in out
        # of necessity
        env_train_tasks = env.get_train_tasks()
        self._oracle_approach = OracleApproach(env_preds, env.options,
                                               env.types, env.action_space,
                                               env_train_tasks)
        self._simulator = env.simulate

    def answer_query(self, state: State, query: Query) -> Response:
        """The key method that the teacher defines."""
        assert query.__class__.__name__ in self._allowed_query_type_names, \
            f"Disallowed query: {query}"
        if isinstance(query, GroundAtomsHoldQuery):
            return self._answer_GroundAtomsHold_query(state, query)
        assert isinstance(query, DemonstrationQuery)
        return self._answer_Demonstration_query(state, query)

    def _answer_GroundAtomsHold_query(
            self, state: State,
            query: GroundAtomsHoldQuery) -> GroundAtomsHoldResponse:
        holds = {}
        for ground_atom in query.ground_atoms:
            pred = self._pred_name_to_pred[ground_atom.predicate.name]
            holds[ground_atom] = pred.holds(state, ground_atom.objects)
        return GroundAtomsHoldResponse(query, holds)

    def _answer_Demonstration_query(
            self, state: State,
            query: DemonstrationQuery) -> DemonstrationResponse:
        task = Task(state, query.goal)
        try:
            policy = self._oracle_approach.solve(task, CFG.timeout)
        except (ApproachTimeout, ApproachFailure):
            return DemonstrationResponse(query, None)

        def goal_reached(s: State) -> bool:
            for ground_atom in query.goal:
                pred = self._pred_name_to_pred[ground_atom.predicate.name]
                if not pred.holds(s, ground_atom.objects):
                    return False
            return True

        teacher_traj = utils.run_policy_until(policy, self._simulator, state,
                                              goal_reached,
                                              CFG.max_num_steps_option_rollout)
        return DemonstrationResponse(query, teacher_traj)
