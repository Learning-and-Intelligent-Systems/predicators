"""A bilevel planning approach that learns NSRTs from an offline dataset, and
continues learning options through reinforcement learning."""

from typing import Dict, List, Sequence, Set, cast, Tuple

import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches.base_approach import ApproachFailure, \
    ApproachTimeout
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.nsrt_learning.option_learning import \
    _LearnedNeuralParameterizedOption, _RLOptionLearnerBase, \
    create_rl_option_learner
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Dataset, InteractionRequest, \
    InteractionResult, LowLevelTrajectory, ParameterizedOption, Predicate, \
    Task, Type, _Option, State, Action, Array


class NSRTReinforcementLearningApproach(NSRTLearningApproach):
    """A bilevel planning approach that learns NSRTs from an offline dataset,
    and continues learning options through reinforcement learning."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._nsrts: Set[NSRT] = set()
        self._online_learning_cycle = 0
        self._initial_trajectories: List[LowLevelTrajectory] = []
        self._option_learners: Dict[NSRT, _RLOptionLearnerBase] = {}
        self._train_task_to_online_traj: Dict[int, LowLevelTrajectory] = {}
        self._train_task_to_option_plan: Dict[int, List[_Option]] = {}
        self._reward_epsilon = CFG.reward_epsilon
        self._pos_reward = CFG.pos_reward
        self._neg_reward = CFG.neg_reward

    @classmethod
    def get_name(cls) -> str:
        return "nsrt_rl"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        self._initial_trajectories = dataset.trajectories
        super().learn_from_offline_dataset(dataset)
        # We need to create a separate RL option learner for each option because
        # each one will maintain its own unique state associated with the
        # learning process.
        self._option_learners = {
            n: create_rl_option_learner()
            for n in self._nsrts
        }

    def get_interaction_requests(self) -> List[InteractionRequest]:
        # For each training task, try to solve the task to get a policy. If the
        # task can't be solved, construct a policy from the sequence of _Option
        # objects that achieves the longest partial refinement of a valid plan
        # skeleton. The teacher will collect a trajectory on the training task
        # using this policy.
        requests = []
        for i in range(len(self._train_tasks)):
            task = self._train_tasks[i]
            try:
                _act_policy = self.solve(task, CFG.timeout)
                self._train_task_to_option_plan[i] = self._last_plan
            except (ApproachTimeout, ApproachFailure) as e:
                partial_refinements = e.info.get("partial_refinements")
                assert partial_refinements is not None
                _, plan = max(partial_refinements, key=lambda x: len(x[1]))
                _act_policy = utils.option_plan_to_policy(plan)
                self._train_task_to_option_plan[i] = plan
            request = InteractionRequest(train_task_idx=i,
                                         act_policy=_act_policy,
                                         query_policy=lambda s: None,
                                         termination_function=task.goal_holds)
            requests.append(request)
        return requests

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        self._online_learning_cycle += 1
        # We get one result per training task.
        for i, result in enumerate(results):
            traj = LowLevelTrajectory(result.states,
                                      result.actions,
                                      _is_demo=False,
                                      _train_task_idx=i)
            self._train_task_to_online_traj[i] = traj

        # This maps each unique _Option to its experience data from the previous
        # online learning cycle. The experience data is a List of (states,
        # actions, rewards, objects, relative_parameters) tuples. Each tuple
        # contains information from a single instance the _Option was run. Each
        # element of the tuple is a list: 'states' is the list of states visited
        # ,'actions' is the list of actions taken, 'rewards' is the list of the
        # rewards acculumated, and 'objects' and 'relative_params' are the list
        # of objects and (relative) parameters that were passed into the _LearnedNeuralParameterizedOption
        # regressor. The relative parameter is necessary to store because during
        # learning, the RLOptionLearner will need to reconstruct the input to
        # the regressor.
        option_to_data: Dict[str, List[Tuple[List[State], List[Action],
                                             List[int], , List[Object], List[Array]]]] = {}

        # For each training task, compute the experience data for each _Option
        # we see used in that training task.
        for i in range(len(self._train_tasks)):
            plan = self._train_task_to_option_plan[i]
            traj = self._train_task_to_online_traj[i]

            curr_option_idx = 0
            curr_option = plan[curr_option_idx]
            if curr_option not in option_to_data:
                option_to_data[curr_option] = []
            curr_states = [traj.states[0]]
            curr_actions = []
            curr_rewards = []
            curr_objects_vec = []
            curr_relative_param = []
            actions = (a for a in traj.actions)

            # for j, s in enumerate(traj.states[1:]):
            for s, s_prime in zip(traj.states, traj.states[1:]):
                # Store the state, action, and relative parameter.
                curr_states.append(s)
                curr_actions.append(next(actions))
                curr_objects_vec =
                relative_param = curr_option.parent.get_option_param_from_state(
                    s,
                    curr_option.memory,
                    curr_option.objects,
                )
                curr_relative_params.append(relative_param)

                # Add pos_reward if we got within epsilon of the option's
                # subgoal, otherwise we add neg_reward.
                if np.allclose(relative_param, 0, atol=self._reward_epsilon):
                    subgoal_reached = True
                    curr_rewards.append(self._pos_reward)
                else:
                    subgoal_reached = False
                    curr_rewards.append(self._neg_reward)

                # Store transition data. If this is the last state and we didn't
                # reach the subgoal, only assign a reward if this option had
                # had sufficient time (steps) to succeed.
                if not subgoal_reached and j + 1 == len(
                        traj.states
                ) - 1 and CFG.max_num_steps_interaction_request - (
                        j + 1) < CFG.last_option_steps_threshold:
                    pass
                else:
                    option_to_data[curr_option].append(
                        (curr_states, curr_actions, curr_rewards, curr_objects_vec,
                         curr_relative_param))

                # Advance to next option.
                curr_option_idx += 1
                if curr_option_idx < len(plan):
                    curr_option = plan[curr_option_idx]
                    if curr_option not in option_to_data:
                        option_to_data[curr_option] = []
                else:
                    # If we run out of options in the plan, there should be
                    # an _OptionPlanExhausted exception, and so there is
                    # nothing more in the trajectory that we have not yet
                    # assigned to an _Option already. That is, the current
                    # state in this loop is the last state.
                    assert s.allclose(traj.states[-1])

                # Initialize trajectory for next option.
                curr_states = [s]
                curr_actions = []
                curr_rewards = []
                curr_objects_vec = []
                curr_relative_param = []

        # Call the RL option learner on each option.
        for option, experience in option_to_data.items():
            corresponding_nsrts = [
                nsrt for nsrt in self._nsrts if nsrt.option.name == option.name
            ]
            assert len(corresponding_nsrts) == 1
            corresponding_nsrt = corresponding_nsrts[0]
            corresponding_parent_option = cast(
                _LearnedNeuralParameterizedOption, corresponding_nsrt.option)
            updated_option = self._option_learners[corresponding_nsrt].update(
                corresponding_parent_option, experience)
            replaced_nsrt = NSRT(
                corresponding_nsrt.name, corresponding_nsrt.parameters,
                corresponding_nsrt.preconditions,
                corresponding_nsrt.add_effects,
                corresponding_nsrt.delete_effects,
                corresponding_nsrt.side_predicates, updated_option,
                corresponding_nsrt.option_vars, corresponding_nsrt.sampler)
            self._nsrts.remove(corresponding_nsrt)
            self._nsrts.add(replaced_nsrt)
