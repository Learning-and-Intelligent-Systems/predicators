"""Maple Q approach. TODO describe."""

from __future__ import annotations

import abc
import logging
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Optional, \
    Sequence, Set, Tuple

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.explorers import BaseExplorer, create_explorer
from predicators.ml_models import QFunction, QFunctionData
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LowLevelTrajectory, \
    Metrics, NSRTSampler, Object, ParameterizedOption, Predicate, Segment, \
    State, Task, Type, _GroundNSRT, _GroundSTRIPSOperator, _Option


class MapleQ(OnlineNSRTLearningApproach):
    """TODO document."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)

        # The current implementation assumes that NSRTs are not changing.
        assert CFG.strips_learner == "oracle"
        # The base sampler should also be unchanging and from the oracle.
        assert CFG.sampler_learner == "oracle"

        # Log all transition data.
        self._maple_data: QFunctionData = []
        self._last_seen_segment_traj_idx = -1

        # Store the Q function.
        self._q_function = QFunction()

    @classmethod
    def get_name(cls) -> str:
        return "maple_q"
    
    def _create_explorer(self) -> BaseExplorer:
        """Create a new explorer at the beginning of each interaction cycle."""
        # Note that greedy lookahead is not yet supported.
        preds = self._get_current_predicates()
        assert CFG.explorer == "maple_q"
        explorer = create_explorer(CFG.explorer,
                                   preds,
                                   self._initial_options,
                                   self._types,
                                   self._action_space,
                                   self._train_tasks,
                                   self._get_current_nsrts(),
                                   self._option_model,
                                   maple_q_function=self._q_function)
        return explorer
    
    def load(self, online_learning_cycle: Optional[int]) -> None:
        super().load(online_learning_cycle)
        # TODO

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     online_learning_cycle: Optional[int],
                     annotations: Optional[List[Any]]) -> None:
        # Start by learning NSRTs in the usual way.
        super()._learn_nsrts(trajectories, online_learning_cycle, annotations)
        # Check the assumption that operators and options are 1:1.
        # This is just an implementation convenience.
        assert len({nsrt.option for nsrt in self._nsrts}) == len(self._nsrts)
        for nsrt in self._nsrts:
            assert nsrt.option_vars == nsrt.parameters
        # Update the data using the updated self._segmented_trajs.
        self._update_maple_data()
        # Re-learn Q function.
        self._q_function.train(self._maple_data)
        # Save the things we need other than the NSRTs, which were already
        # saved in the above call to self._learn_nsrts()
        # TODO

    def _update_maple_data(self) -> None:
        start_idx = self._last_seen_segment_traj_idx + 1
        new_trajs = self._segmented_trajs[start_idx:]
        for segmented_traj in new_trajs:
            self._last_seen_segment_traj_idx += 1
            for segment in segmented_traj:
                s = segment.states[0]
                o = segment.get_option()
                ns = segment.states[-1]
                import ipdb; ipdb.set_trace()
                reward = ...
                terminal = ...
                self._maple_data.append((s, o, ns, reward, terminal))
    
