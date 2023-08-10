"""An approach that doesn't use a planner, but rather uses RL to learn both how
to task-plan (sequence parameterized options together) and how to refine plans
via sampling."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
import torch
from gym.spaces import Box

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.envs import get_or_create_env
from predicators.ml_models import ConcatMLP
from predicators.rl.policies import PAMDPPolicy, MakeDeterministic
from predicators.rl.rl_utils import EnvReplayBuffer
from predicators.rl.training_functions import SACHybridTrainer
from predicators.settings import CFG
from predicators.structs import NSRT, Action, Array, Dataset, GroundAtom, \
    InteractionResult, LowLevelTrajectory, Metrics, NSRTSampler, Object, \
    ParameterizedOption, Predicate, Segment, State, Task, Type, _GroundNSRT, \
    _GroundSTRIPSOperator, _Option


class OnlineRLApproach(OnlineNSRTLearningApproach):
    """Performs online reinforcement learning to learn both how to plan and how
    to sample."""
    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # The current implementation assumes that NSRTs are not changing.
        assert CFG.strips_learner == "oracle"
        # The base sampler should also be unchanging and from the oracle.
        assert CFG.sampler_learner == "oracle"
        # As we collect more online data, the self._learn_nsrts function
        # will update self._segmented_trajs. We need an index to keep track
        # of what is new.
        self._last_seen_segment_traj_idx = -1
        # Construct all information necessary to setup, train and eval
        # RL models.
        curr_env = get_or_create_env(CFG.env)
        # NOTE: we assume for this simple approach that the number of objects doesn't 
        # change when we change the task (i.e, all train tasks and test tasks have the
        # same objects).
        # TODO: This currently only works if the observation is secretly a State. We probably
        # want to call the perceiver or something to actually get the state.
        init_obs = curr_env.reset('train', 0)
        self._observation_size = init_obs.vec(sorted(list(init_obs))).shape[0]
        # We need to know the NSRTs in order to get the size of the discrete and continuous
        # action spaces. This can't happen until the first round of NSRT learning is called.
        # Thus, for now, we will set these to be None.
        self._discrete_actions_size = None
        self._continuous_actions_size = max(opt.params_space.shape[0] for opt in self._initial_options)
        self._learned_policy = None
        self._qf1 = None
        self._qf2 = None
        self._target_qf1 = None
        self._target_qf2 = None
        self._trainer_function = None
        self._replay_buffer = None
        self._sorted_ground_nsrts: Optional[List[_GroundNSRT]] = None
        # Seed torch.
        torch.manual_seed(self._seed)
        

    @classmethod
    def get_name(cls) -> str:
        return "online_rl"
    
    # TODO: add the necessary inputs and then actually finish implementing
    # this.
    def get_reward(self) -> float:
        """
        Given transition data, returns the corresponding reward value.

        Used to construct the dataset for the learner.
        """

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Update the dataset with the offline data.
        for traj in dataset.trajectories:
            self._update_dataset(traj)
        super().learn_from_offline_dataset(dataset)
        if self._discrete_actions_size is None:
            # We haven't yet correctly set the discrete and
            # continuous action sizes, so do this now that we have
            # the data.
            assert self._nsrts is not None
            curr_env = get_or_create_env(CFG.env)
            objects = sorted(list(curr_env.reset('train', 0)))
            ground_nsrts = []
            for nsrt in sorted(self._nsrts):
                for ground_nsrt in utils.all_ground_nsrts(nsrt, objects):
                    ground_nsrts.append(ground_nsrt)
            self._discrete_actions_size = len(ground_nsrts)
            # Additionally, we can now setup the precise ground NSRTs list,
            # the learned policy, the q-function networks, the model trainer,
            # and the replay buffer.
            self._sorted_ground_nsrts = ground_nsrts
            # TODO: not really sure what the 'one_hot_s' setting is about...
            # Also, not really sure about setting all the additional policy_kwargs
            # that the robosuite_launcher script in the original codebase sets.
            self._learned_policy = PAMDPPolicy(obs_dim=self._observation_size, action_dim_s=self._discrete_actions_size, action_dim_p=self._continuous_actions_size, one_hot_s=True, hidden_sizes=CFG.online_rl_qnetwork_hidden_sizes,)
            self._qf1 = ConcatMLP(
                input_size=self._observation_size + self._discrete_actions_size + self._continuous_actions_size,
                output_size=1,
                hidden_sizes=CFG.online_rl_qnetwork_hidden_sizes,
            )
            self._qf2 = ConcatMLP(
                input_size=self._observation_size + self._discrete_actions_size + self._continuous_actions_size,
                output_size=1,
                hidden_sizes=CFG.online_rl_qnetwork_hidden_sizes,
            )
            self._target_qf1 = ConcatMLP(
                input_size=self._observation_size + self._discrete_actions_size + self._continuous_actions_size,
                output_size=1,
                hidden_sizes=CFG.online_rl_qnetwork_hidden_sizes,
            )
            self._target_qf2 = ConcatMLP(
                input_size=self._observation_size + self._discrete_actions_size + self._continuous_actions_size,
                output_size=1,
                hidden_sizes=CFG.online_rl_qnetwork_hidden_sizes,
            )
            self._trainer_function = SACHybridTrainer(env_action_space=curr_env.action_space, policy=self._learned_policy, qf1=self._qf1, qf2=self._qf2, target_qf1=self._target_qf1, target_qf2=self._target_qf2)
            self._replay_buffer = EnvReplayBuffer(CFG.online_rl_max_replay_buffer_size, self._observation_size, self._discrete_actions_size + self._continuous_actions_size)

    
    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        # Add the new data to the cumulative dataset.
        for result in results:
            traj = LowLevelTrajectory(result.states, result.actions)
            self._update_dataset(traj)
        # Update the RL policy.
        annotations = None
        if self._dataset.has_annotations:
            annotations = self._dataset.annotations  # pragma: no cover
        super()._learn_nsrts(self._dataset.trajectories,
                          self._online_learning_cycle,
                          annotations=annotations)
        # Check the assumption that operators and options are 1:1.
        # This is just an implementation convenience.
        assert len({nsrt.option for nsrt in self._nsrts}) == len(self._nsrts)
        for nsrt in self._nsrts:
            assert nsrt.option_vars == nsrt.parameters
        # Now, loop thru newly-collected trajectories and add their
        # corresponding transitions to the replay buffer.
        start_idx = self._last_seen_segment_traj_idx + 1
        new_trajs = self._segmented_trajs[start_idx:]
        import ipdb; ipdb.set_trace()
        # TODO: add this new data to the replay buffer, then call
        # a model update.
        
        # Advance the online learning cycle.
        self._online_learning_cycle += 1

    
    # TODO: override the explorer. Be sure to put the policy into deterministic mode
    # and then sample actions from it.
    

    
    def _update_replay_buffer() -> None:
        # TODO: implement after exploration.
        pass


    # TODO: figure out how to call training.
    def _train(self, batch) -> None:
        pass


    def _convert_policy_action_to_env_action(self, policy_action: Array) -> _Option:
        """Convert the output of our learned policy into an environment
        action by selecting the correct operator and grounding it with the
        correct parameters."""
        assert self._sorted_ground_nsrts is not None
        assert policy_action.shape[0] == self._discrete_actions_size + self._continuous_actions_size
        # Discrete actions output should be 0 everywhere except for one place.
        discrete_actions_output = policy_action[:self._discrete_actions_size]
        discrete_action_idx = np.argmax(discrete_actions_output)
        assert discrete_actions_output[discrete_action_idx] == 1.0
        ground_nsrt = self._sorted_ground_nsrts[discrete_action_idx]
        continuous_params_output = policy_action[-self._continuous_actions_size:]
        continuous_params_for_option = continuous_params_output[ground_nsrt.option.params_space.shape[0]]
        output_ground_option = ground_nsrt.option.ground(ground_nsrt.option_objs, continuous_params_for_option)
        return output_ground_option



    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        eval_policy = MakeDeterministic(self._learned_policy)
        curr_option = None
        num_curr_option_steps = 0

        def _rollout_rl_policy(state: State) -> Action:
            """Execute the option policy until we get an option termination or timeout (i.e, we exceed the max steps for the option)
            and then get a new output from the model."""

            # TODO: finish and test; might need to catch option execution failures.
            nonlocal self, eval_policy, curr_option, num_curr_option_steps
            state_vec = state.vec(sorted(list(state)))
            if curr_option is None:
                # We need to produce a new ground option from the network.
                assert state_vec.shape[0] == self._observation_size
                policy_action = eval_policy.get_action(state_vec)[0]
                curr_option = self._convert_policy_action_to_env_action(policy_action)
    
            if not curr_option.initiable(state):
                num_cur_option_steps = 0
                raise OptionExecutionFailure(
                    "Unsound option policy.",
                    info={"last_failed_option": curr_option})
                
    
            if CFG.max_num_steps_option_rollout is not None and \
                num_cur_option_steps >= CFG.max_num_steps_option_rollout:
                raise OptionTimeoutFailure(
                    "Exceeded max option steps.",
                    info={"last_failed_option": curr_option})

            if curr_option.terminal(state):
                curr_option = None

                

        return _rollout_rl_policy
