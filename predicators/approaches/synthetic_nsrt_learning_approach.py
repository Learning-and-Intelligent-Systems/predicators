"""TODO

python predicators/main.py --env painting --approach synthetic \
        --seed 0 --num_train_tasks 10  --num_test_tasks 10\
        --painting_lid_open_prob 0.5 \
        --painting_initial_holding_prob 1.0 \
        --painting_num_objs_train '[1]' \
        --painting_num_objs_test '[1]' \
        --painting_num_objs_test '[1]' \
        --painting_goal_receptacles 'box'
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Set, Sequence

import dill as pkl
from pg3.policy_search import learn_policy

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.planning import PlanningFailure, run_low_level_search
from predicators.settings import CFG
from predicators.structs import Action, Box, Dataset, GroundAtom, \
    LiftedDecisionList, Object, ParameterizedOption, Predicate, State, Task, \
    Type, _GroundNSRT


class SyntheticNSRTLearningApproach(NSRTLearningApproach):
    """TODO."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._current_ldl = LiftedDecisionList([])

    @classmethod
    def get_name(cls) -> str:
        return "synthetic"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        # TODO override
        return super()._solve(task, timeout)

    def _create_ground_atom_dataset(self, trajectories: List[LowLevelTrajectory]) -> List[GroundAtomTrajectory]:
        # TODO should we be learning NRSTs first, then using ground NSRTs instead of options here??
        ground_atom_trajs = super()._create_ground_atom_dataset(trajectories)
        for ll_traj, ground_atom_seq in ground_atom_trajs:
            positive_synthetic_atoms_to_add = set()
            negative_synthetic_atoms_to_remove = set()
            assert len(ll_traj.states) == len(ground_atom_seq), "TODO handle multi-step options"
            for t, act in enumerate(ll_traj.actions):
                ground_atom_seq[t] |= positive_synthetic_atoms_to_add
                ground_atom_seq[t] -= negative_synthetic_atoms_to_remove
                option = act.get_option()
                new_synthetic_atom = self._option_spec_to_synthetic_atom(option.parent, option.objects)
                old_synthetic_atom = self._option_spec_to_synthetic_atom(option.parent, option.objects, negate=True)
                positive_synthetic_atoms_to_add.add(new_synthetic_atom)
                negative_synthetic_atoms_to_remove.add(old_synthetic_atom)
            last_t = len(ll_traj.actions)
            ground_atom_seq[last_t] |= positive_synthetic_atoms_to_add
            ground_atom_seq[last_t] -= negative_synthetic_atoms_to_remove
        return ground_atom_trajs

    def _get_current_predicates(self) -> Set[Predicate]:
        primitive_preds = super()._get_current_predicates()
        synthetic_preds = self._get_synthetic_predicates()
        return primitive_preds | synthetic_preds

    def _get_synthetic_predicates(self) -> Set[Predicate]:
        synthetic_preds = set()
        for opt in self._initial_options:
            pred = self._parameterized_option_to_predicate(opt)
            synthetic_preds.add(pred)
            pred = self._parameterized_option_to_predicate(opt, negate=True)
            synthetic_preds.add(pred)
        return synthetic_preds

    def _parameterized_option_to_predicate(self, param_opt: ParameterizedOption, negate: bool = False) -> Predicate:
        if negate:
            name = f"{param_opt.name}-HAS-NOT-happened"
        else:
            name = f"{param_opt.name}-happened"
        return Predicate(name, param_opt.types, lambda s,o: negate)

    def _option_spec_to_synthetic_atom(self, param_opt: ParameterizedOption, objects: Sequence[Object], negate: bool = False) -> GroundAtom:
        pred = self._parameterized_option_to_predicate(param_opt, negate=negate)
        atom = GroundAtom(pred, objects)
        return atom
