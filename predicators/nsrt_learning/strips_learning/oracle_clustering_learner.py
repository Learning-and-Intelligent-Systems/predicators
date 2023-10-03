"""Oracle for STRIPS learning."""

from typing import List, Set

from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import NSRT, PNAD, Datastore, DummyOption, \
    LiftedAtom, Predicate, Segment


class OracleSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a STRIPS learner that uses oracle operators but re-learns
    all the components via currently-implemented methods in the base class."""

    def _induce_add_effects_by_intersection(self,
                                            pnad: PNAD) -> Set[LiftedAtom]:
        """Given a PNAD with a nonempty datastore, compute the add effects for
        the PNAD's operator by intersecting all lifted add effects."""
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

    def _compute_datastores_given_nsrt(self, nsrt: NSRT) -> Datastore:
        datastore = []
        assert self._annotations is not None
        for seg_traj, ground_nsrt_list in zip(self._segmented_trajs,
                                              self._annotations):
            assert len(seg_traj) == len(ground_nsrt_list)
            for segment, ground_nsrt in zip(seg_traj, ground_nsrt_list):
                if ground_nsrt.parent == nsrt:
                    op_vars = nsrt.op.parameters
                    obj_sub = ground_nsrt.objects
                    var_to_obj_sub = dict(zip(op_vars, obj_sub))
                    datastore.append((segment, var_to_obj_sub))
        return datastore

    def _find_add_effect_intersection_preds(
            self, segments: List[Segment]) -> Set[Predicate]:
        unique_add_effect_preds: Set[Predicate] = set()
        for seg in segments:
            if len(unique_add_effect_preds) == 0:
                unique_add_effect_preds = set(atom.predicate
                                              for atom in seg.add_effects)
            else:
                unique_add_effect_preds &= set(atom.predicate
                                               for atom in seg.add_effects)
        return unique_add_effect_preds

    def _learn(self) -> List[PNAD]:
        # For this learner to work, we need annotations to be non-null
        # and the length of the annotations to match the length of
        # the trajectory set.
        # assert self._annotations is not None
        # assert len(self._annotations) == len(self._trajectories)
        # assert CFG.offline_data_method == "demo+gt_operators"
        assert self._clusters is not None

        # env = get_or_create_env(CFG.env)
        # env_options = get_gt_options(env.get_name())
        # gt_nsrts = get_gt_nsrts(env.get_name(), env.predicates, env_options)
        pnads: List[PNAD] = []

        for name, v in clusters.items():
            preconds, add_effects, segments = v
            seg_0 = segments[0]
            opt_objs = tuple(seg_0.get_option().objects)
            revelant_add_effects = [a for a in seg_0.add_effects if a.predicate in add_effects]
            relevant_preconds = [a for a in seg_0.init_atoms if a.predicate in preconds]
            objects = {o for atom in relevant_add_effects for o in atom.objects} | set(opt_objs)
            objects_list = sorted(objects)
            params = utils.create_new_variables([o.type for o in objects_list])
            obj_to_var = dict(zip(objects_list, params))
            var_to_obj = dict(zip(params, objects_list))
            op_add_effects = {atom.lift(obj_to_var) for atom in relevant_add_effects}

            op_preconds = {atom.lift(obj_to_var) for atom in relevant_preconds}
            op_ignore_effects = set()

            op = STRIPSOperator(name, params, op_preconds, op_add_effects, ignore_effects)
            datastore = []
            for seg in segments:
                seg_opt_objs = tuple(seg.get_option().objects)
                relevant_add_effects = [a for a in seg.add_effects if a.predicate in add_effects]
                seg_objs = {o for atom in relevant_add_effects for o in atom.objects} | set(seg_opt_objs)
                seg_objs_list = sorted(seg_objs)
                seg_params = utils.create_new_variables(
                    [o.type for o in seg_objs_list])
                var_to_obj = dict(zip(seg_params, seg_objs_list))
                datastore.append((seg, var_to_obj))
            option_vars = [obj_to_var[o] for o in opt_objs]
            option_spec = [seg_0.get_option().parent, option_vars]
            pnads.append(PNAD(op, datastore, option_spec))

            # might be able to handle ignore effects by something like what
            # _compute_pnad_ignore_effects in base_strips_learner.py does.
            # basically you can try to see if the ops would be used anywhere in
            # the demos, and and narrow it down via the demos. but you might
            # pick up some extraneous predicates in some of them, so you would
            # have to be careful and do some inference when trying them out on
            # the demos.

        return pnads

    # def _learn(self) -> List[PNAD]:
    #     # For this learner to work, we need annotations to be non-null
    #     # and the length of the annotations to match the length of
    #     # the trajectory set.
    #     assert self._annotations is not None
    #     assert len(self._annotations) == len(self._trajectories)
    #     assert CFG.offline_data_method == "demo+gt_operators"
    #
    #     env = get_or_create_env(CFG.env)
    #     env_options = get_gt_options(env.get_name())
    #     gt_nsrts = get_gt_nsrts(env.get_name(), env.predicates, env_options)
    #     pnads: List[PNAD] = []
    #     for nsrt in gt_nsrts:
    #         # If options are unknown, use a dummy option spec.
    #         if CFG.option_learner == "no_learning":
    #             option_spec = (nsrt.option, list(nsrt.option_vars))
    #         else:
    #             option_spec = (DummyOption.parent, [])
    #
    #         datastore = self._compute_datastores_given_nsrt(nsrt)
    #         pnad = PNAD(nsrt.op, datastore, option_spec)
    #         add_effects = self._induce_add_effects_by_intersection(pnad)
    #         preconditions = self._induce_preconditions_via_intersection(pnad)
    #         pnad = PNAD(
    #             pnad.op.copy_with(preconditions=preconditions,
    #                               add_effects=add_effects), datastore,
    #             option_spec)
    #         self._compute_pnad_delete_effects(pnad)
    #         self._compute_pnad_ignore_effects(pnad)
    #         pnads.append(pnad)
    #
    #     return pnads

    @classmethod
    def get_name(cls) -> str:
        return "oracle_clustering"
