"""Oracle for STRIPS learning."""

import logging
from typing import List, Set

from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import PNAD, Datastore, DummyOption, LiftedAtom, Segment, Predicate


class OracleSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a STRIPS learner that uses oracle operators
    but re-learns all the components via currently-implemented
    methods in the base class."""

    def _induce_add_effects_by_intersection(self, pnad: PNAD) -> Set[LiftedAtom]:
        """Given a PNAD with a nonempty datastore, compute the add effects
        for the PNAD's operator by intersecting all lifted add effects."""
        assert len(pnad.datastore) > 0
        for i, (segment, var_to_obj) in enumerate(pnad.datastore):
            objects = set(var_to_obj.values())
            obj_to_var = {o: v for v, o in var_to_obj.items()}
            atoms = {
                atom
                for atom in segment.add_effects
                if all(o in objects for o in atom.objects)
            }
            lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}
            if i == 0:
                add_effects = lifted_atoms
            else:
                add_effects &= lifted_atoms
        return add_effects

    def _compute_datastore_given_pnad_name(self, pnad_name: str, dummy_pnad: PNAD) -> Datastore:
        datastore = []
        for seg_traj, op_name_list in zip(self._segmented_trajs, self._annotations):
            assert len(seg_traj) == len(op_name_list)
            for segment, op_name in zip(seg_traj, op_name_list):
                if op_name == pnad_name:
                    objs = set(segment.trajectory.states[0])
                    pnad, sub = self._find_best_matching_pnad_and_sub(segment, objs, [dummy_pnad])
                    assert pnad == dummy_pnad
                    datastore.append((segment, sub))
        return datastore

    def _find_add_effect_intersection_preds(self, segments: List[Segment]) -> Set[Predicate]:
        unique_add_effect_preds: Set[Predicate] = set()
        for seg in segments:
            if len(unique_add_effect_preds) == 0:
                unique_add_effect_preds = set(
                    atom.predicate for atom in seg.add_effects)
            else:
                unique_add_effect_preds &= set(
                    atom.predicate for atom in seg.add_effects)
        return unique_add_effect_preds

    def _learn(self) -> List[PNAD]:
        # For this learner to work, we need annotations to be non-null
        # and the length of the annotations to match the length of
        # the trajectory set.
        assert self._annotations is not None
        assert len(self._annotations) == len(self._trajectories)
        assert CFG.offline_data_method == "demo+gt_operators"

        env = get_or_create_env(CFG.env)
        env_options = get_gt_options(env.get_name())
        gt_nsrts = get_gt_nsrts(env.get_name(), env.predicates, env_options)
        pnads: List[PNAD] = []
        for nsrt in gt_nsrts:
            # If options are unknown, use a dummy option spec.
            if CFG.option_learner == "no_learning":
                option_spec = (nsrt.option, list(nsrt.option_vars))
            else:
                option_spec = (DummyOption.parent, [])

            dummy_op = nsrt.op.copy_with(preconditions=set(), add_effects=set(), delete_effects=set(), ignore_effects=self._predicates)
            initial_dummy_pnad = PNAD(dummy_op, [], option_spec)
            datastore = self._compute_datastore_given_pnad_name(nsrt.name, initial_dummy_pnad)

            assert(len(datastore) > 0)

            dummy_pnad_with_datastore = PNAD(dummy_op, datastore, option_spec)
            add_effects = self._induce_add_effects_by_intersection(dummy_pnad_with_datastore)
            preconditions = self._induce_preconditions_via_intersection(dummy_pnad_with_datastore)
            correct_pnad = PNAD(dummy_pnad_with_datastore.op.copy_with(preconditions=preconditions, add_effects=add_effects), datastore, option_spec)
            self._compute_pnad_delete_effects(correct_pnad)
            self._compute_pnad_ignore_effects(correct_pnad)
            
            pnads.append(correct_pnad)
                
        return pnads

    @classmethod
    def get_name(cls) -> str:
        return "oracle_clustering"
