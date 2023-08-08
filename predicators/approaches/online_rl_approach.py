"""An approach that doesn't use a planner, but rather uses RL to learn
both how to task-plan (sequence parameterized options together) and
how to refine plans via sampling.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
import torch
from gym.spaces import Box

from predicators import utils
from predicators.rl.rl_utils import ReplayBuffer, SimpleReplayBuffer
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.structs import NSRT, Array, GroundAtom, LowLevelTrajectory, \
    Metrics, NSRTSampler, Object, ParameterizedOption, Predicate, Segment, \
    State, Task, Type, _GroundNSRT, _GroundSTRIPSOperator, _Option, InteractionResult
from predicators.settings import CFG

class OnlineRLApproach(OnlineNSRTLearningApproach):
    """Performs online reinforcement learning to learn both how to plan and 
    how to sample."""
    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task], max_replay_buffer_size, observation_dim, action_dim, env_info_sizes) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # The current implementation assumes that NSRTs are not changing.
        assert CFG.strips_learner == "oracle"
        # The base sampler should also be unchanging and from the oracle.
        assert CFG.sampler_learner == "oracle"
        # Construct all information necessary to setup, train and eval
        # RL models.
        self._rl_model = None
        self._replay_buffer: ReplayBuffer = SimpleReplayBuffer(max_replay_buffer_size=max_replay_buffer_size, observation_dim=observation_dim, action_dim=action_dim, env_info_sizes=env_info_sizes)
        # Seed torch.
        torch.manual_seed(self._seed)

    @classmethod
    def get_name(cls) -> str:
        return "online_rl"
    
    # TODO: Figure out where to call learning here
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
        # TODO: using this dataset, learn the MAPLE policy.
        # Advance the online learning cycle.
        self._online_learning_cycle += 1

    
    # TODO: override the explorer.
    

    
    def _update_replay_buffer() -> None:
        # TODO: implement after exploration.
        pass


    # TODO: figure out how to call training.
    def _train(self, batch) -> None:
        pass


    # TODO: override _solve.
