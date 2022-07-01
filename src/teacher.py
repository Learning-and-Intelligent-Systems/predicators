"""Contains classes defining a teacher that can supply additional, oracular
information to assist an agent during online learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.approaches.oracle_approach import OracleApproach
from predicators.src.envs import get_or_create_env
from predicators.src.ground_truth_nsrts import _get_options_by_names, \
    _get_types_by_names
from predicators.src.settings import CFG, get_allowed_query_type_names
from predicators.src.structs import Action, DemonstrationQuery, \
    DemonstrationResponse, GroundAtomsHoldQuery, GroundAtomsHoldResponse, \
    InteractionRequest, LowLevelTrajectory, PathToStateQuery, \
    PathToStateResponse, Query, Response, State, Task


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
        if isinstance(query, PathToStateQuery):
            return self._answer_PathToState_query(state, query)
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

    def _answer_PathToState_query(
            self, state: State,
            query: PathToStateQuery) -> PathToStateResponse:
        # The query is asking for a trajectory from the current state to
        # the goal state. Planning to a low-level state is hard. This
        # implementation currently assumes that only one option is required
        # to get from the state to the goal state. Furthermore, the option
        # is identified in an environment-specific way. This is all a stand-in
        # for a human teacher. Note that although these trajectories are good,
        # they are not demonstrations per se, because they do not reach task
        # goals (and are not necessarily associated with any particular task).
        trajectory = None
        null_response = PathToStateResponse(query, teacher_traj=None)
        goal_state = query.goal_state
        if CFG.env == "cover_multistep_options":
            # Setup.
            block_type, target_type, robot_type = _get_types_by_names(
                CFG.env, ["block", "target", "robot"])
            Pick, Place = _get_options_by_names(CFG.env, ["Pick", "Place"])
            robot = state.get_objects(robot_type)[0]
            blocks = state.get_objects(block_type)
            targets = state.get_objects(target_type)
            state_blocks_held = [
                b for b in blocks if abs(state.get(b, "grasp") - 1) < 1e-6
            ]
            goal_blocks_held = [
                b for b in blocks if abs(goal_state.get(b, "grasp") - 1) < 1e-6
            ]
            # Case 1: Pick. Note that if there is a block held in the goal
            # state, the only possible option that will lead to that goal state
            # being reached in exactly one step is a Pick option. If the option
            # does not succeed in achieving the goal state, "Validate" will
            # fail, and null_response will be returned.
            if len(goal_blocks_held) == 1:
                parameterized_option = Pick
                block = goal_blocks_held[0]
                arguments = [block, robot]
                # Parameters are non-static features only.
                changing_obj_feats = [(block, "grasp"), (robot, "x"),
                                      (robot, "y"), (robot, "grip"),
                                      (robot, "holding")]
            # Case 2: Place. Note that we only know how to do two things in
            # this environment, Pick and Place. In this Case 2, we have already
            # established that we're not going to be picking, so we must be
            # placing. That means there should be a block that we're holding.
            # If there's not, we proceed to Case 3 (return null_response). If
            # instead, we're holding a different block from the one that we
            # want to see changed in the goal state, then "Validate" will fail,
            # and we will also return null.
            elif len(state_blocks_held) == 1:
                parameterized_option = Place
                block = state_blocks_held[0]
                # The target does not actually matter, because it is not
                # involved in the definition of the Place policy. The only
                # reason that it's included in the ParameterizedOption is
                # that there is a change in the symbolic effects for the
                # target (it becomes covered).
                target = targets[0]
                arguments = [block, robot, target]
                # Parameters are non-static features only.
                changing_obj_feats = [(block, "x"), (block, "grasp"),
                                      (robot, "x"), (robot, "grip"),
                                      (robot, "holding")]
            # Case 3: invalid or unsupported request.
            else:
                return null_response
            goal_vec = [goal_state.get(o, f) for o, f in changing_obj_feats]
            state_vec = [state.get(o, f) for o, f in changing_obj_feats]
            params = np.subtract(goal_vec, state_vec)
            option = parameterized_option.ground(arguments, params)
            policy = utils.option_plan_to_policy([option])
            termination_function = lambda s: s.allclose(goal_state)
            trajectory = utils.run_policy_with_simulator(
                policy,
                self._simulator,
                state,
                termination_function,
                max_num_steps=CFG.max_num_steps_option_rollout)
        else:
            raise NotImplementedError("PathToStateQuery is not supported for "
                                      f"env {CFG.env}.")
        # Validate. If the goal state was not reached, the query is assumed
        # to be invalid, and a None trajectory is returned.
        if not trajectory.states[-1].allclose(goal_state):
            return null_response
        # Strip the trajectory of options to prevent cheating.
        for action in trajectory.actions:
            action.unset_option()
        # Success.
        return PathToStateResponse(query, teacher_traj=trajectory)


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
class TeacherInteractionMonitorWithVideo(TeacherInteractionMonitor,
                                         utils.VideoMonitor):
    """A monitor that wraps a TeacherInteractionMonitor to optionally also
    render every state and action encountered, if CFG.make_interaction_videos
    is True.

    The render_fn is generally env.render.
    """

    def observe(self, state: State, action: Optional[Action]) -> None:
        query = self._request.query_policy(state)
        if query is None:
            response = None
            caption = "None"
        else:
            response = self._teacher.answer_query(state, query)
            self._query_cost += query.cost
            caption = f"{response}, cost={query.cost}"
        self._responses.append(response)
        if CFG.make_interaction_videos:
            self._video.extend(self._render_fn(action, caption))
