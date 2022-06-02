"""A bilevel planning approach that learns NSRTs from an offline dataset, and
continues learning options through reinforcement learning."""

from typing import Dict, List, Sequence, Set, Tuple, cast

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
from predicators.src.structs import NSRT, Action, Array, Dataset, \
    InteractionRequest, InteractionResult, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type, _Option


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
        # online learning cycle. The experience data is a list of list of tuples
        # , where the inner list contains experience from one execution of the
        # option. The tuple is a standard RL experience tuple of (state, action
        # reward, next_state), with one additional element: the input vector
        # that the option's regressor received for that state, which is the
        # contacenation of the state vector for the option's objects and the
        # relative parameter that the sampler provided. This input vector is
        # necessary during learning to update the option's regressor.
        option_to_data: Dict[ParameterizedOption,
                             List[List[Tuple[State, Array, Action, int,
                                             State]]]] = {}

        # For each training task, compute the experience data for each _Option
        # we see used in that training task.
        for i in range(len(self._train_tasks)):
            plan = self._train_task_to_option_plan[i]
            traj = self._train_task_to_online_traj[i]

            curr_option_idx = 0
            curr_option = plan[curr_option_idx]
            parent_option = cast(_LearnedNeuralParameterizedOption,
                                 curr_option.parent)
            if parent_option not in option_to_data:
                option_to_data[parent_option] = []
            actions = ((i, a) for i, a in enumerate(traj.actions))
            experience = []

            # Loop through the trajectory and compute the experience tuples for
            # each _Option in the plan.
            for state, next_state in zip(traj.states[:-1], traj.states[1:]):
                idx, action = next(actions)
                state_features = state.vec(curr_option.objects)
                relative_param = parent_option.get_rel_option_param_from_state(
                    state,
                    curr_option.memory,
                    curr_option.objects,
                )
                # The RLOptionLearner will need the input to the option's
                # regressor for training.
                input_vec = cast(
                    Array, np.hstack(([1.0], state_features, relative_param)))

                # Add pos_reward if we got within epsilon of the option's
                # subgoal, otherwise we add neg_reward. To do this, we can check
                # the relative parameter for the next state.
                next_rel_param = parent_option.get_rel_option_param_from_state(
                    next_state,
                    curr_option.memory,
                    curr_option.objects,
                )
                reward: int = 0
                if np.allclose(next_rel_param, 0, atol=self._reward_epsilon):
                    reward += self._pos_reward
                else:
                    reward += self._neg_reward

                experience.append(
                    (state, input_vec, action, reward, next_state))

                # Store experience data in two cases: (1) the next state is a
                # terminal state for the current option and (2) the current
                # option did not reach its terminal state but had a sufficient
                # number of steps to reach it so that it is reasonable to assign
                # it a reward.
                terminate = parent_option.effect_based_terminal(
                    next_state, curr_option.objects)
                had_sufficient_steps = next_state.allclose(
                    traj.states[-1]) and (
                        CFG.max_num_steps_interaction_request -
                        (idx + 1) > CFG.valid_reward_steps_threshold)
                if terminate:
                    option_to_data[parent_option].append(experience)
                    curr_option_idx += 1
                    if curr_option_idx < len(plan):
                        curr_option = plan[curr_option_idx]
                        parent_option = cast(_LearnedNeuralParameterizedOption,
                                             curr_option.parent)
                        experience = []  # Reset for next option in the plan.
                        if parent_option not in option_to_data:
                            option_to_data[parent_option] = []
                    else:
                        # If we run out of options in the plan, there should be
                        # an _OptionPlanExhausted exception, and so there is
                        # nothing more in the trajectory that we have not yet
                        # assigned to an _Option already. That is, the current
                        # state in this loop is the last state.
                        assert next_state.allclose(traj.states[-1])
                elif had_sufficient_steps:
                    assert curr_option == plan[-1]
                    option_to_data[parent_option].append(experience)

        # Call the RL option learner on each option.
        for option, experiences in option_to_data.items():
            corresponding_nsrts = [
                nsrt for nsrt in self._nsrts if nsrt.option.name == option.name
            ]
            assert len(corresponding_nsrts) == 1
            corresponding_nsrt = corresponding_nsrts[0]
            corresponding_parent_option = cast(
                _LearnedNeuralParameterizedOption, corresponding_nsrt.option)
            updated_option = self._option_learners[corresponding_nsrt].update(
                corresponding_parent_option, experiences)
            replaced_nsrt = NSRT(
                corresponding_nsrt.name, corresponding_nsrt.parameters,
                corresponding_nsrt.preconditions,
                corresponding_nsrt.add_effects,
                corresponding_nsrt.delete_effects,
                corresponding_nsrt.side_predicates, updated_option,
                corresponding_nsrt.option_vars, corresponding_nsrt.sampler)
            self._nsrts.remove(corresponding_nsrt)
            self._nsrts.add(replaced_nsrt)
