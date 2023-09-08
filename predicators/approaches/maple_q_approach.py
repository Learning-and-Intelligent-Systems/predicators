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
    State, Task, Type, _GroundNSRT, _GroundSTRIPSOperator, _Option, Action


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
        self._q_function = QFunction(seed=CFG.seed,
            hid_sizes=CFG.mlp_regressor_hid_sizes,
            max_train_iters=CFG.mlp_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
            use_torch_gpu=CFG.use_torch_gpu,
            train_print_every=CFG.pytorch_train_print_every,
            n_iter_no_change=CFG.active_sampler_learning_n_iter_no_change)

    @classmethod
    def get_name(cls) -> str:
        return "maple_q"
    
    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        # TODO remove copied code (from explorer)
        goal = task.goal
        objects = set(task.init)
        all_ground_nsrts: List[_GroundNSRT] = []
        for nsrt in self._nsrts:
            all_ground_nsrts.extend(utils.all_ground_nsrts(nsrt, objects))

        def _option_policy(state: State) -> _Option:
            candidates: List[_Option] = []
            # Find all applicable ground NSRTs.
            atoms = utils.abstract(state, self._get_current_predicates())
            applicable_ground_nsrts = utils.get_applicable_operators(all_ground_nsrts, atoms)
            # Sample candidate options.
            for ground_nsrt in applicable_ground_nsrts:
                for _ in range(CFG.active_sampler_learning_num_samples):
                    option = ground_nsrt.sample_option(state, goal, self._rng)
                    candidates.append(option)
            # Score the candidates using the BEST Q function (exploit only).
            scores: List[float] = []
            for option in candidates:
                score = self._q_function.predict_q_value(state, option)
                scores.append(score)
            # Select the best-scoring candidate.
            idx = np.argmax(scores)
            selected_option = candidates[idx]
            return selected_option

        return utils.option_policy_to_policy(_option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout)
    
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
        if not online_learning_cycle:
            all_ground_nsrts: Set[_GroundNSRT] = set()
            objects = {o for t in self._train_tasks for o in t.init}
            for nsrt in self._nsrts:
                all_ground_nsrts.update(utils.all_ground_nsrts(nsrt, objects))
            self._q_function.set_grounding(objects, all_ground_nsrts)
        self._update_maple_data()
        # Re-learn Q function.
        self._q_function.train_from_q_data(self._maple_data)
        # Save the things we need other than the NSRTs, which were already
        # saved in the above call to self._learn_nsrts()
        # TODO

    def _update_maple_data(self) -> None:
        start_idx = self._last_seen_segment_traj_idx + 1
        new_trajs = self._segmented_trajs[start_idx:]

        # TODO!!!
        goal = self._train_tasks[0].goal
        assert all(task.goal == goal for task in self._train_tasks)

        for segmented_traj in new_trajs:
            self._last_seen_segment_traj_idx += 1
            for i, segment in enumerate(segmented_traj):
                s = segment.states[0]
                o = segment.get_option()
                ns = segment.states[-1]
                reward = 1.0 if goal.issubset(segment.final_atoms) else 0.0
                terminal = reward > 0 or i == len(segmented_traj) - 1
                self._maple_data.append((s, o, ns, reward, terminal))
