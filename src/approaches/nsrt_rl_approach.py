"""A bilevel planning approach that learns NSRTs from an offline dataset, and
continues learning options through reinforcement learning."""

from typing import Dict, List, Sequence, Set, cast

import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches.base_approach import ApproachFailure, \
    ApproachTimeout
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.nsrt_learning.option_learning import \
    _create_absolute_option_param, _LearnedNeuralParameterizedOption, \
    create_rl_option_learner
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Action, Array, Dataset, \
    InteractionRequest, InteractionResult, LowLevelTrajectory, Object, \
    ParameterizedOption, Predicate, State, Task, Type, Variable, VarToObjSub, \
    _Option


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
            n.name: create_rl_option_learner()
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
                # Store the list of _Option objects corresponding to this policy.
                self._train_task_to_option_plan[i] = self._last_plan
            except (ApproachTimeout, ApproachFailure) as e:
                partial_refinements = e.info.get("partial_refinements")
                assert partial_refinements is not None
                _, plan = max(partial_refinements, key=lambda x: len(x[1]))
                _act_policy = utils.option_plan_to_policy(plan)
                # Store the list of _Option objects corresponding to this policy.
                self._train_task_to_option_plan[i] = plan
            request = InteractionRequest(train_task_idx=i,
                                         act_policy=_act_policy,
                                         query_policy=lambda s: None,
                                         termination_function=task.goal_holds)
            requests.append(request)
        return requests

    @staticmethod
    def _get_unchanging_features(
            changing_var_to_feat: Dict[Variable, List[int]],
            var_to_obj: VarToObjSub) -> Dict[Variable, List[int]]:
        var_to_unchanging_feat_ind = {}
        for var, changing_indices in changing_var_to_feat.items():
            dim = var_to_obj[var].type.dim
            unchanging_indices = [
                i for i in range(dim) if i not in changing_indices
            ]
            var_to_unchanging_feat_ind[var] = unchanging_indices
        return var_to_unchanging_feat_ind

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
        # actions, rewards, relative_parameters) tuples. Each tuple contains
        # information from a single instance the _Option was run. Each element
        # of the tuple is a list: 'states' is the list of states visited,
        # 'actions' is the list of actions taken, 'rewards' is the list of the
        # rewards acculumated, and 'relative_params' is a list of (relative)
        # parameter that was passed into the _LearnedNeuralParameterizedOption's
        # regressor. The relative parameter is necessary to store because during
        # learning, the RLOptionLearner will need to reconstruct the input to
        # the regressor.
        option_to_data: Dict[str,
                             List] = {}  # option_name -> online experience

        # For each training task, compute the experience data for each _Option
        # we see used in that training task.
        for i in range(len(self._train_tasks)):
            plan = self._train_task_to_option_plan[i]
            traj = self._train_task_to_online_traj[i]

            curr_option_idx = 0
            curr_option = plan[curr_option_idx]
            parent = cast(_LearnedNeuralParameterizedOption,
                          curr_option.parent)
            var_to_obj = dict(
                zip(parent._operator.parameters, curr_option.objects))
            var_to_unchanging_feat_ind = NSRTReinforcementLearningApproach._get_unchanging_features(
                parent._changing_var_to_feat, var_to_obj)
            if curr_option.name not in option_to_data:
                option_to_data[curr_option.name] = []
            curr_states = [traj.states[0]]
            curr_actions = []
            curr_rewards = []
            curr_relative_params = []
            actions = (a for a in traj.actions)

            for j, s in enumerate(traj.states[1:]):
                # Store the state, action, and relative parameter.
                curr_states.append(s)
                curr_actions.append(next(actions))
                curr_state_changing_feat = _create_absolute_option_param(
                    s,
                    parent._changing_var_to_feat,
                    parent._changing_var_order,
                    var_to_obj,
                )
                subgoal_state_changing_feat = curr_option.memory[
                    "absolute_params"]
                relative_param = subgoal_state_changing_feat - curr_state_changing_feat
                curr_relative_params.append(relative_param)

                # In the remaining code in this loop we compute the reward.
                reward = 0

                # Add large negative reward if any object's features that are
                # not supposed to change, do change. This can happen in two
                # ways: 1) if any of the features of the option's objects that
                # aren't supposed to change, change, and 2) if any feature of
                # any other objects in the state, change.
                prev_state_unchanging_feat = _create_absolute_option_param(
                    curr_states[-2],
                    var_to_unchanging_feat_ind,
                    parent._changing_var_order,
                    var_to_obj,
                )  # concatenated unchanging features of the option's objects
                curr_state_unchanging_feat = _create_absolute_option_param(
                    curr_states[-1],
                    var_to_unchanging_feat_ind,
                    parent._changing_var_order,
                    var_to_obj,
                )  # concatenated unchanging features of the option's objects
                other_objects = [
                    o for o in s.data.keys() if o not in curr_option.objects
                ]
                # The first term checks the current option's objects, and the
                # second term checks all other objects.
                option_objects_unchanged = np.allclose(
                    prev_state_unchanging_feat - curr_state_unchanging_feat,
                    0,
                    atol=1e-5)
                other_objects_unchanged = np.allclose(
                    curr_states[-2].vec(other_objects),
                    curr_states[-1].vec(other_objects),
                    atol=1e-5)
                if not option_objects_unchanged or not other_objects_unchanged:
                    reward += self._neg_reward * 10

                # Add pos_reward if we got within epsilon of the option's
                # subgoal, otherwise we add neg_reward.
                # Ignore the optimization that checks for repeated states, which
                # messes with the terminal state check.
                _ = curr_option.memory.pop("last_state", None)
                if curr_option.terminal(s):
                    # Check if we reached our subgoal within a tolerance.
                    if np.allclose(relative_param,
                                   0,
                                   atol=self._reward_epsilon):
                        curr_rewards.append(reward + self._pos_reward)
                    else:
                        curr_rewards.append(reward + self._neg_reward)

                    # Store transition data.
                    option_to_data[curr_option.name].append(
                        (curr_states, curr_actions, curr_rewards,
                         curr_relative_params))

                    # Advance to next option.
                    curr_option_idx += 1
                    if curr_option_idx < len(plan):
                        curr_option = plan[curr_option_idx]
                        parent = cast(_LearnedNeuralParameterizedOption,
                                      curr_option.parent)
                        var_to_obj = dict(
                            zip(parent._operator.parameters,
                                curr_option.objects))
                        var_to_unchanging_feat_ind = NSRTReinforcementLearningApproach._get_unchanging_features(
                            parent._changing_var_to_feat, var_to_obj)
                        if curr_option.name not in option_to_data:
                            option_to_data[curr_option.name] = []
                    else:
                        # If we run out of options in the plan, there should be
                        # an _OptionPlanExhausted exception, and so there is
                        # nothing more in the trajectory that we have not yet
                        # assigned to an _Option already. That is, the current
                        # state in this loop is the last state.
                        assert s == traj.states[-1]

                    # Initialize trajectory for next option.
                    curr_states = [s]
                    curr_actions = []
                    curr_rewards = []
                    curr_relative_params = []

                else:  # if this transition did not get us to the terminal state
                    curr_rewards.append(reward + self._neg_reward)
                    # Handle the case where we are at the last state in the
                    # trajectory, but it is not a terminal state of the current
                    # option. This occurs when the plan we are executing is a
                    # partial refinement.
                    if j + 1 == len(traj.states) - 1:
                        # Store transition data if this option had enough steps
                        # to run. Note that j+1 actions were taken to get to
                        # the current state.
                        if CFG.max_num_steps_interaction_request - (
                                j + 1) >= CFG.last_option_steps_threshold:
                            option_to_data[curr_option.name].append(
                                (curr_states, curr_actions, curr_rewards,
                                 curr_relative_params))

        # Call the RL option learner on each option.
        for option_name, experience in option_to_data.items():
            corresponding_nsrts = [
                nsrt for nsrt in self._nsrts if nsrt.option.name == option_name
            ]
            assert len(corresponding_nsrts) == 1
            corresponding_nsrt = corresponding_nsrts[0]
            corresponding_parent_option = cast(
                _LearnedNeuralParameterizedOption, corresponding_nsrt.option)
            updated_option = self._option_learners[
                corresponding_nsrt.name].update(corresponding_parent_option,
                                                experience)
            replaced_nsrt = NSRT(
                corresponding_nsrt.name, corresponding_nsrt.parameters,
                corresponding_nsrt.preconditions,
                corresponding_nsrt.add_effects,
                corresponding_nsrt.delete_effects,
                corresponding_nsrt.side_predicates, updated_option,
                corresponding_nsrt.option_vars, corresponding_nsrt.sampler)
            self._nsrts.remove(corresponding_nsrt)
            self._nsrts.add(replaced_nsrt)
