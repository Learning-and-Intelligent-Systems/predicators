"""Contains classes defining a teacher that can supply additional, oracular
information to assist an agent during online learning."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, List, Optional, Callable
from predicators.src.structs import State, Task, Query, Response, \
    GroundAtomsHoldQuery, GroundAtomsHoldResponse, DemonstrationQuery, \
    DemonstrationResponse, LowLevelTrajectory, InteractionRequest, \
    Action, Image, Video
from predicators.src.settings import CFG, get_allowed_query_type_names
from predicators.src.envs import get_or_create_env
from predicators.src.approaches import OracleApproach, ApproachTimeout, \
    ApproachFailure
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


@dataclass
class TeacherInteractionVideoMonitor(TeacherInteractionMonitor,
                                     utils.VideoMonitor):
    """A monitor that renders each state and action encountered and queries and
    responses during interaction with the teacher.

    The render_fn is generally env.render.
    """
    _request: InteractionRequest
    _teacher: Teacher
    _render_fn: Callable[[Optional[Action], Optional[str]], List[Image]]
    _responses: List[Optional[Response]] = field(init=False,
                                                 default_factory=list)
    _query_cost: float = field(init=False, default=0.0)
    _video: Video = field(init=False, default_factory=list)

    def observe(self, state: State, action: Optional[Action]) -> None:
        query = self._request.query_policy(state)
        if query is None:
            response = None
            caption = "No queries"
        else:
            response = self._teacher.answer_query(state, query)
            self._query_cost += query.cost
            caption = f"{response}, cost={query.cost}"
        self._responses.append(response)
        self._video.extend(self._render_fn(action, caption))
