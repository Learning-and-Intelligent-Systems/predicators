"""Contains classes defining a teacher that can supply additional, oracular
information to assist an agent during online learning."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, List, Optional
import numpy as np
from predicators.src.structs import State, Task, Query, Response, \
    GroundAtomsHoldQuery, GroundAtomsHoldResponse, DemonstrationQuery, \
    DemonstrationResponse, LowLevelTrajectory, InteractionRequest, \
    Action, StateBasedDemonstrationQuery
from predicators.src.settings import CFG, get_allowed_query_type_names
from predicators.src.envs import get_or_create_env
from predicators.src.approaches import OracleApproach, ApproachTimeout, \
    ApproachFailure
from predicators.src.ground_truth_nsrts import _get_types_by_names, \
    _get_options_by_names
from predicators.src import utils


class Teacher:
    """The teacher can respond to queries of various types."""

    def __init__(self, train_tasks: Sequence[Task]) -> None:
        self._train_tasks = train_tasks
        env = get_or_create_env(CFG.env)
        self._pred_name_to_pred = {pred.name: pred for pred in env.predicates}
        self._allowed_query_type_names = get_allowed_query_type_names()
        self._oracle_approach = OracleApproach(
            env.predicates,
            env.options,
            env.types,
            env.action_space, [],
            task_planning_heuristic=CFG.offline_data_task_planning_heuristic,
            max_skeletons_optimized=CFG.offline_data_max_skeletons_optimized)
        self._simulator = env.simulate

    def answer_query(self, state: State, query: Query) -> Response:
        """The key method that the teacher defines."""
        assert query.__class__.__name__ in self._allowed_query_type_names, \
            f"Disallowed query: {query}"
        if isinstance(query, GroundAtomsHoldQuery):
            return self._answer_GroundAtomsHold_query(state, query)
        if isinstance(query, DemonstrationQuery):
            return self._answer_Demonstration_query(state, query)
        if isinstance(query, StateBasedDemonstrationQuery):
            return self._answer_StateBasedDemonstration_query(state, query)
        raise NotImplementedError(f"Unrecognized query: {query}.")

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
        # The query is asking for a demonstration from the current state to
        # the goal from the train task.
        goal = self._train_tasks[query.train_task_idx].goal
        task = Task(state, goal)
        try:
            policy = self._oracle_approach.solve(task, CFG.timeout)
        except (ApproachTimeout, ApproachFailure):
            return DemonstrationResponse(query, teacher_traj=None)

        traj = utils.run_policy_with_simulator(
            policy,
            self._simulator,
            task.init,
            task.goal_holds,
            max_num_steps=CFG.max_num_steps_option_rollout)
        assert task.goal_holds(traj.states[-1])
        teacher_traj = LowLevelTrajectory(traj.states,
                                          traj.actions,
                                          _is_demo=True,
                                          _train_task_idx=query.train_task_idx)
        return DemonstrationResponse(query, teacher_traj)

    def _answer_StateBasedDemonstration_query(
            self, state: State,
            query: StateBasedDemonstrationQuery) -> DemonstrationResponse:
        # The query is asking for a demonstration from the current state to
        # the goal state. Planning to a low-level state is hard. This
        # implementation currently assumes that only one option is required
        # to get from the state to the goal state. Furthermore, the option
        # is identified in an environment-specific way. This is all a stand-in
        # for a human teacher. Note that although these trajectories are good,
        # they are not demonstrations per se, because they do not reach task
        # goals (and are not necessarily associated with any particular task).
        trajectory = None
        goal_state = query.goal_state
        if CFG.env == "cover_multistep_options":
            assert CFG.cover_multistep_use_learned_equivalents
            # Setup.
            block_type, target_type, robot_type = _get_types_by_names(
                CFG.env, ["block", "target", "robot"])
            Pick, Place = _get_options_by_names(
                CFG.env, ["LearnedEquivalentPick", "LearnedEquivalentPlace"])
            robot = state.get_objects(robot_type)[0]
            blocks = state.get_objects(block_type)
            targets = state.get_objects(target_type)
            # Case 1: Pick.
            if abs(goal_state.get(robot, "holding") - 1) < 1e-6:
                blocks_held = [
                    b for b in blocks
                    if abs(goal_state.get(b, "grasp") - 1) < 1e-6
                ]
                assert len(blocks_held) == 1
                block = blocks_held[0]
                rx = state.get(robot, "x")
                desired_rx = goal_state.get(robot, "x")
                ry = state.get(robot, "y")
                desired_ry = goal_state.get(robot, "y")
                # is_block, is_target, width, x, grasp, y, height
                # grasp changes from -1.0 to 1.0
                block_param = [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
                # x, y, grip, holding
                # grip changes from -1.0 to 1.0
                # holding changes from -1.0 to 1.0
                robot_param = [desired_rx - rx, desired_ry - ry, 2.0, 2.0]
                param = np.array(block_param + robot_param, dtype=np.float32)
                option = Pick.ground([block, robot], param)
            # Case 2: Place.
            else:
                import ipdb
                ipdb.set_trace()

            policy = utils.option_plan_to_policy([option])
            termination_function = lambda s: s.allclose(goal_state)
            trajectory = utils.run_policy_with_simulator(
                policy,
                self._simulator,
                state,
                termination_function,
                max_num_steps=CFG.max_num_steps_option_rollout)
        else:
            raise NotImplementedError("State-based demonstration queries not"
                                      f"supported for env {CFG.env}.")
        # Validate. If the goal state was not reached, the query is assumed
        # to be invalid, and a None trajectory is returned.
        if trajectory is None or not trajectory.states[-1].allclose(
                goal_state):
            return DemonstrationResponse(query, teacher_traj=None)
        # Success.
        return DemonstrationResponse(query, teacher_traj=trajectory)


@dataclass
class TeacherInteractionMonitor(utils.Monitor):
    """Wraps the interaction between agent and teacher to include generating
    and answering queries."""
    _request: InteractionRequest
    _teacher: Teacher
    _responses: List[Optional[Response]] = field(init=False,
                                                 default_factory=list)
    _query_cost: float = field(init=False, default=0.0)

    def observe(self, state: State, action: Optional[Action]) -> None:
        del action  # unused
        query = self._request.query_policy(state)
        if query is None:
            self._responses.append(None)
        else:
            self._responses.append(self._teacher.answer_query(state, query))
            self._query_cost += query.cost

    def get_responses(self) -> List[Optional[Response]]:
        """Return the responses."""
        return self._responses

    def get_query_cost(self) -> float:
        """Return the query cost."""
        return self._query_cost
