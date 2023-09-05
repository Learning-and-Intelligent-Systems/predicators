"""An approach that doesn't use a planner, but rather uses RL to learn both how
to task-plan (sequence parameterized options together) and how to refine plans
via sampling."""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
import torch
from gym.spaces import Box

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.envs import get_or_create_env
from predicators.explorers import BaseExplorer, create_explorer
from predicators.ml_models import ConcatMLP
from predicators.rl.policies import MakeDeterministic, PAMDPPolicy
from predicators.rl.rl_utils import EnvReplayBuffer, \
    env_state_to_maple_input, make_executable_maple_policy, \
    np_to_pytorch_batch
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
        self._observation_size = env_state_to_maple_input(init_obs).shape[0]
        # TODO: Also, this only works if every single train and test task has the same objects
        # and goal, though the initial state can vary.
        self._goal_atoms = curr_env.get_task("train", 0).goal
        # We need to know the NSRTs in order to get the size of the discrete and continuous
        # action spaces. This can't happen until the first round of NSRT learning is called.
        # Thus, for now, we will set these to be None.
        self._discrete_actions_size = None
        self._continuous_actions_size = max(opt.params_space.shape[0]
                                            for opt in self._initial_options)
        self._learned_policy = None
        self._qf1 = None
        self._qf2 = None
        self._target_qf1 = None
        self._target_qf2 = None
        self._trainer_function = None
        self._replay_buffer = None
        self._sorted_ground_nsrts: Optional[List[_GroundNSRT]] = None
        self._param_opt_to_nsrt: Dict[ParameterizedOption, _GroundNSRT] = {}
        self._ground_nsrt_to_idx: Dict[_GroundNSRT, int] = {}

        # Seed torch.
        torch.manual_seed(self._seed)

    @classmethod
    def get_name(cls) -> str:
        return "online_rl"

    def get_reward(self, segment: Segment) -> float:
        """Given transition data, returns the corresponding reward value.

        Used to construct the dataset for the learner.
        """
        # For now, just check if the goal atoms are a subset of
        # the segment's final atoms.
        # TODO: we'll likely need a more dense and fine-grained reward
        # function.
        if self._goal_atoms.issubset(
                segment.final_atoms) and not self._goal_atoms.issubset(
                    segment.init_atoms):
            return 1.0
        return 0.0

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
                self._param_opt_to_nsrt[nsrt.option] = nsrt
                for ground_nsrt in utils.all_ground_nsrts(nsrt, objects):
                    ground_nsrts.append(ground_nsrt)
            for i, ground_nsrt in enumerate(ground_nsrts):
                self._ground_nsrt_to_idx[ground_nsrt] = i
            self._discrete_actions_size = len(ground_nsrts)
            # Additionally, we can now setup the precise ground NSRTs list,
            # the learned policy, the q-function networks, the model trainer,
            # and the replay buffer.
            self._sorted_ground_nsrts = ground_nsrts
            # TODO: not really sure what the 'one_hot_s' setting is about...
            # Also, not really sure about setting all the additional policy_kwargs
            # that the robosuite_launcher script in the original codebase sets.
            self._learned_policy = PAMDPPolicy(
                obs_dim=self._observation_size,
                action_dim_s=self._discrete_actions_size,
                action_dim_p=self._continuous_actions_size,
                one_hot_s=True,
                hidden_sizes=CFG.online_rl_qnetwork_hidden_sizes,
            )
            self._qf1 = ConcatMLP(
                input_size=self._observation_size +
                self._discrete_actions_size + self._continuous_actions_size,
                output_size=1,
                hidden_sizes=CFG.online_rl_qnetwork_hidden_sizes,
            )
            self._qf2 = ConcatMLP(
                input_size=self._observation_size +
                self._discrete_actions_size + self._continuous_actions_size,
                output_size=1,
                hidden_sizes=CFG.online_rl_qnetwork_hidden_sizes,
            )
            self._target_qf1 = ConcatMLP(
                input_size=self._observation_size +
                self._discrete_actions_size + self._continuous_actions_size,
                output_size=1,
                hidden_sizes=CFG.online_rl_qnetwork_hidden_sizes,
            )
            self._target_qf2 = ConcatMLP(
                input_size=self._observation_size +
                self._discrete_actions_size + self._continuous_actions_size,
                output_size=1,
                hidden_sizes=CFG.online_rl_qnetwork_hidden_sizes,
            )
            self._trainer_function = SACHybridTrainer(
                env_action_space=curr_env.action_space,
                policy=self._learned_policy,
                qf1=self._qf1,
                qf2=self._qf2,
                target_qf1=self._target_qf1,
                target_qf2=self._target_qf2)
            self._replay_buffer = EnvReplayBuffer(
                CFG.online_rl_max_replay_buffer_size, self._observation_size,
                self._discrete_actions_size + self._continuous_actions_size)

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

        num_place_on_bumpy_actions = 0.0

        # Now, loop thru newly-collected trajectories and add their
        # corresponding transitions to the replay buffer.
        start_idx = self._last_seen_segment_traj_idx + 1
        new_trajs = self._segmented_trajs[start_idx:]
        num_positive_trajs = 0
        for seg_traj in new_trajs:
            self._last_seen_segment_traj_idx += 1
            for i, segment in enumerate(seg_traj):
                init_ll_state = segment.trajectory.states[0]
                final_ll_state = segment.trajectory.states[-1]
                init_maple_state = env_state_to_maple_input(init_ll_state)
                final_maple_state = env_state_to_maple_input(final_ll_state)
                assert segment.has_option()
                nsrt = self._param_opt_to_nsrt[segment.get_option().parent]
                ground_nsrt = nsrt.ground(tuple(segment.get_option().objects))
                discrete_action = np.zeros(self._discrete_actions_size)
                discrete_action[self._ground_nsrt_to_idx[ground_nsrt]] = 1.0
                continuous_action = np.zeros(self._continuous_actions_size)
                continuous_action[:len(segment.get_option(
                ).params)] = np.array(segment.get_option().params)
                maple_action = np.concatenate(
                    (discrete_action, continuous_action), axis=0)
                reward = self.get_reward(segment)
                terminal = False

                if reward == 1.0:
                    num_positive_trajs += 1

                if i == len(seg_traj) - 1:
                    terminal = True

                self._replay_buffer.add_sample(init_maple_state, maple_action,
                                               reward, terminal,
                                               final_maple_state)

                # if np.allclose(init_maple_state, np.array([1.0, 0.0, 0.01, 0.90651945, -1.0, 1.0, 0.5, 1.35, 0.65, 0.0, 1.0, 0.008, 0.66038981]), rtol=0.1):
                # if not ground_nsrt.option.params_space.contains(continuous_action.astype(np.float32)):
                #     import ipdb; ipdb.set_trace()

                if terminal == 1.0:
                    break
        logging.info(
            f"{num_positive_trajs} goal-achieving trajectories out of {len(new_trajs)}"
        )
        logging.info(f"{num_place_on_bumpy_actions} place onto bumpy actions")
        import ipdb; ipdb.set_trace()

        # Call training on data from the updated replay buffer.
        self._train()

        # Advance the online learning cycle.
        self._online_learning_cycle += 1

    def _create_explorer(self) -> BaseExplorer:
        assert CFG.explorer == "maple_explorer"
        # Geometrically increase the length of exploration.
        b = CFG.active_sampler_learning_explore_length_base
        # b * 5 * (1 + self._online_learning_cycle
        max_steps = 5 #b**(1 + self._online_learning_cycle)
        preds = self._get_current_predicates()
        explorer = create_explorer(
            CFG.explorer,
            preds,
            self._initial_options,
            self._types,
            self._action_space,
            self._train_tasks,
            self._get_current_nsrts(),
            self._option_model,
            max_steps_before_termination=max_steps,
            ground_nsrts=self._sorted_ground_nsrts,
            exploration_policy=self._learned_policy,
            observations_size=self._observation_size,
            discrete_actions_size=self._discrete_actions_size,
            continuous_actions_size=self._continuous_actions_size)
        return explorer

    def _train(self) -> None:
        for i in range(CFG.online_rl_num_trains_per_train_loop):
            np_batch = self._replay_buffer.random_batch(
                CFG.online_rl_batch_size)
            torch_batch = np_to_pytorch_batch(np_batch)
            train_stats = self._trainer_function.train_from_torch(torch_batch)
            logging.info(
                f"Training iter: {i}/{CFG.online_rl_num_trains_per_train_loop}"
            )
            # if i % 25 == 0:
            logging.info(train_stats)

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        eval_policy = MakeDeterministic(self._learned_policy)
        return make_executable_maple_policy(eval_policy,
                                            self._sorted_ground_nsrts,
                                            self._observation_size,
                                            self._discrete_actions_size,
                                            self._continuous_actions_size)
