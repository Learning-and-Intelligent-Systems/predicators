"""Algorithms for STRIPS learning that rely on clustering to obtain effects."""

import abc
import functools
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, FrozenSet, Iterator, List, Set, Tuple, cast

from predicators import utils
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import PNAD, Datastore, DummyOption, LiftedAtom, \
    ParameterizedOption, Predicate, STRIPSOperator, VarToObjSub, State, \
    GroundAtom, GroundOptionRecord, Object


class ClusteringSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a clustering-based STRIPS learner."""

    def _learn(self) -> List[PNAD]:
        segments = [seg for segs in self._segmented_trajs for seg in segs]
        # Cluster the segments according to common option and effects.
        pnads: List[PNAD] = []
        for segment in segments:
            if segment.has_option():
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
            else:
                segment_param_option = DummyOption.parent
                segment_option_objs = tuple()
            for pnad in pnads:
                # Try to unify this transition with existing effects.
                # Note that both add and delete effects must unify,
                # and also the objects that are arguments to the options.
                (pnad_param_option, pnad_option_vars) = pnad.option_spec
                # ent_to_ent_sub:: {object: variable}
                suc, ent_to_ent_sub = utils.unify_preconds_effects_options(
                    frozenset(),  # no preconditions
                    frozenset(),  # no preconditions
                    frozenset(segment.add_effects),
                    frozenset(pnad.op.add_effects),
                    # frozenset(),  # test
                    # frozenset(),  # test
                    frozenset(segment.delete_effects),
                    frozenset(pnad.op.delete_effects),
                    segment_param_option,
                    pnad_param_option,
                    segment_option_objs,
                    tuple(pnad_option_vars),
                )
                # breakpoint()
                sub = cast(VarToObjSub,
                           {v: o
                            for o, v in ent_to_ent_sub.items()})
                if suc:
                    # Add to this PNAD.
                    assert set(sub.keys()) == set(pnad.op.parameters)
                    pnad.add_to_datastore((segment, sub))
                    break
            else:
                # Otherwise, create a new PNAD.
                objects = {o for atom in segment.add_effects |
                           segment.delete_effects for o in atom.objects} | \
                          set(segment_option_objs)
                objects_lst = sorted(objects)
                # a hack: should change to the types of the predicate/option
                # e.g. Holding(?i:item) won't work here
                # For each object, get the least general type from the types of
                #   the predicates, options that mentions it.
                if CFG.use_least_generalization_types_in_clustering:
                    least_generalization_types = []
                    for o in objects_lst:
                        types = set()
                        for atom in segment.add_effects | segment.delete_effects:
                            for i, obj in enumerate(atom.objects):
                                if obj == o:
                                    types.add(atom.predicate.types[i])
                        for i, obj in enumerate(segment_option_objs):
                            if obj == o:
                                types.add(segment_param_option.types[i])
                        # find the most specific type in types
                        #   (i.e. the type that is not a supertype of any other type)
                        types_copy = deepcopy(types)
                        # for example if types{bun, item, object} return bun
                        # if types{item, object} return item
                        for t in types:
                            for t2 in types:
                                try:
                                    if t != t2 and t in t2.all_ancestors():
                                        types_copy.discard(t)
                                except:
                                    breakpoint()
                        try:
                            assert len(types_copy) == 1
                        except:
                            breakpoint()
                        least_generalization_types.append(types_copy.pop())

                params = utils.create_new_variables(
                    least_generalization_types if CFG.
                    use_least_generalization_types_in_clustering else
                    [o.type for o in objects_lst])
                preconds: Set[LiftedAtom] = set()  # will be learned later
                obj_to_var = dict(zip(objects_lst, params))
                var_to_obj = dict(zip(params, objects_lst))
                add_effects = {
                    atom.lift(obj_to_var)
                    for atom in segment.add_effects
                }
                delete_effects = {
                    atom.lift(obj_to_var)
                    for atom in segment.delete_effects
                }
                ignore_effects: Set[Predicate] = set()  # will be learned later
                op = STRIPSOperator(f"Op{len(pnads)}", params, preconds,
                                    add_effects, delete_effects,
                                    ignore_effects)
                datastore = [(segment, var_to_obj)]
                option_vars = [obj_to_var[o] for o in segment_option_objs]
                option_spec = (segment_param_option, option_vars)
                pnads.append(PNAD(op, datastore, option_spec))

        # Learn the preconditions of the operators in the PNADs. This part
        # is flexible; subclasses choose how to implement it.
        pnads = self._learn_pnad_preconditions(pnads)

        # Handle optional postprocessing to learn ignore effects.
        pnads = self._postprocessing_learn_ignore_effects(pnads)

        # Log and return the PNADs.
        if self._verbose:
            logging.info("Learned operators (before option learning):")
            for pnad in pnads:
                logging.info(pnad)
        return pnads

    @abc.abstractmethod
    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        """Subclass-specific algorithm for learning PNAD preconditions.

        Returns a list of new PNADs. Should NOT modify the given PNADs.
        """
        raise NotImplementedError("Override me!")

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        """Optionally postprocess to learn ignore effects."""
        _ = self  # unused, but may be used in subclasses
        return pnads


class ClusterAndIntersectSTRIPSLearner(ClusteringSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via
    intersection."""

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        new_pnads = []
        for pnad in pnads:
            preconditions = self._induce_preconditions_via_intersection(pnad)
            # Since we are taking an intersection, we're guaranteed that the
            # datastore can't change, so we can safely use pnad.datastore here.
            new_pnads.append(
                PNAD(pnad.op.copy_with(preconditions=preconditions),
                     pnad.datastore, pnad.option_spec))
        return new_pnads

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect"

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        """Prune PNADs whose datastores are too small.

        Specifically, keep PNADs that have at least
        CFG.cluster_and_intersect_min_datastore_fraction fraction of the
        segments produced by the option in their NSRT.
        """
        if not CFG.cluster_and_intersect_prune_low_data_pnads:
            return pnads
        option_to_dataset_size: Dict[ParameterizedOption,
                                     int] = defaultdict(int)
        for pnad in pnads:
            option = pnad.option_spec[0]
            option_to_dataset_size[option] += len(pnad.datastore)
        ret_pnads: List[PNAD] = []
        for pnad in pnads:
            option = pnad.option_spec[0]
            fraction = len(pnad.datastore) / option_to_dataset_size[option]
            if fraction >= CFG.cluster_and_intersect_min_datastore_fraction:
                ret_pnads.append(pnad)
        return ret_pnads

class ClusterIntersectAndSearchSTRIPSLearner(ClusterAndIntersectSTRIPSLearner):
    """A clustering-based STRIPS learner that learns preconditions via 
    intersection and search over predicate subsets w.r.t. the classification
    accuracy + simplicity objective that uses the ground truth positive and
    negative states for each option.
    Note that in the standard intersection step, Atom with objects not in effect
    and option is remove (presumbly as a form of regularization) but this is not
    need here and we can let search do the work.
    """
    def __init__(self, *args, fail_optn_dict: Dict[str, GroundOptionRecord],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_optn_dict = fail_optn_dict

    @classmethod
    def get_name(cls) -> str:
        return "cluster_intersect_and_search"

    # @staticmethod
    # def _induce_preconditions_via_intersection(pnad: PNAD) -> Set[LiftedAtom]:
    #     """Given a PNAD with a nonempty datastore, compute the preconditions
    #     for the PNAD's operator by intersecting all lifted preimages."""
    #     assert len(pnad.datastore) > 0
    #     for i, (segment, var_to_obj) in enumerate(pnad.datastore):
    #         objects = set(var_to_obj.values())
    #         obj_to_var = {o: v for v, o in var_to_obj.items()}

    #         # relax the assumption of only objects in the effect
    #         atoms = segment.init_atoms
    #         # atoms = {
    #         #     atom
    #         #     for atom in segment.init_atoms
    #         #     if all(o in objects for o in atom.objects)
    #         # }
    #         # if atom's object is in obj_to_var
    #         lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}
    #         # if atom's object is not in obj_to_var

    #         if i == 0:
    #             preconditions = lifted_atoms
    #         else:
    #             preconditions &= lifted_atoms
    #     return preconditions
    
    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        new_pnads = []
        
        # Cluster the states in the fail_optn_dict sorted with by option name
        optn_to_fail_data: Dict[str, 
            List[Tuple[State, Set[GroundAtom], List[Object]]]] = defaultdict(
                list)
        for gop_str, optn_rec in self.fail_optn_dict.items():
            # logging.debug(f"Accessing option record for {gop_str}")
            for s, ab_s in zip(optn_rec.states, optn_rec.abstract_states):
                # logging.debug(f"Adding neg states for {optn_rec.option.name}")
                optn_to_fail_data[optn_rec.option.name].append(
                    (s, ab_s, optn_rec.optn_objs)
                )

        for pnad in pnads:
            init_preconditions = self._induce_preconditions_via_intersection(
                pnad)
            option_name = pnad.option_spec[0].name
            # logging.debug(f"fetching neg states for {option_name}")
            logging.debug(f"Search with "
                    f"{len(optn_to_fail_data[option_name])} negative and "
                    f"{len(pnad.datastore)} positive data for: {pnad}")
            refined_preconditions = self._run_search(pnad, init_preconditions,
                fail_data=optn_to_fail_data[option_name])
            logging.debug(f"Precondition before search {init_preconditions}")

            new_pnads.append(
                PNAD(pnad.op.copy_with(preconditions=refined_preconditions),
                     pnad.datastore, pnad.option_spec))

        return new_pnads
    
    def _run_search(self, pnad: PNAD, init_preconditions: Set[LiftedAtom],
                    fail_data: List[Tuple[State, Set[GroundAtom], List[Object]]]
                    ) -> FrozenSet[LiftedAtom]:
        """Run search to find a single precondition set for the pnad operator.
        """
        initial_state = frozenset(init_preconditions)
        check_goal = lambda s: False
        # the classification function
        # states in the datastore the option is successfully executed
        succ_data = pnad.datastore
        score_func = functools.partial(self._score_preconditions, pnad,
                                       succ_data, fail_data)
        path, _ = utils.run_gbfs(initial_state,
                                 check_goal,
                                 self._get_precondition_successors,
                                 score_func)

        # log info
        ret_precon = path[-1]
        logging.debug(f"Search finished: selected:")
        score_func(ret_precon)
        return ret_precon
    
    @staticmethod
    def _score_preconditions(
            pnad: PNAD, 
            succ_data: Datastore, 
            fail_data: List[Tuple[State, Set[GroundAtom], List[Object]]], 
            preconditions: FrozenSet[LiftedAtom]):
        '''Score a precondition based on the succ_states in the datastore and
        failed states in the fail_optn_dict.'''
        # The positive states are the states in the pnad, the negative states
        # are all the states in fail_optn_dict with the same option.
        # Get the tp, fn, tn, fp states for each ground_option
        candidate_op = pnad.op.copy_with(preconditions=preconditions)
        option_spec = pnad.option_spec
        optn_var = option_spec[1]
        n_succ_states, n_fail_states = len(succ_data), len(fail_data)
        tp_states, fn_states, tn_states, fp_states = [], [], [], []
        n_tot = n_succ_states + n_fail_states

        # TP: states that satisfy the preconditions given the partial varToObj
        # substituion of the option
        # assume succ_states is a Datastore here
        # For succ_states, we only need to ground the operator with the
        # var_to_obj substitution in the state because we've assumed the the 
        # precondition doesn't have any variables that are not in the effect
        # and option. 
        # If we want to remove this assumption then we can try to ground them
        # with varToObj substitution consistent with the option and effect..
        for seg, var_to_obj in succ_data:
            # alternatively:
            # ground_op = candidate_op.ground(var_to_obj)
            # if ground_op.preconditions.issubset(seg.init_atoms):
            g_pre = {a.ground(var_to_obj) for a in preconditions}
            if g_pre.issubset(seg.init_atoms):
                tp_states.append(seg.states[0])
            else:
                fn_states.append(seg.states[0])
        
        # For the fail_states, we only have a partial sub. for the options
        # and we need to compute the false positives and true negatives based on
        # the ground nsrts with substitutions consistent with the option.
        for state, atom_state, optn_objs in fail_data:
            # we need to use optn_var from the pnad instead of the from the 
            # self.fail_optn_dict because the var name in the pnad (which is
            # the same as in candidate_op) is different from the var name in
            # the fail_optn_dict.
            ground_nsrts = utils.all_ground_operators_given_partial(
                candidate_op, set(state), dict(zip(optn_var, optn_objs)))
            
            if any([
                    gnsrt.preconditions.issubset(atom_state)
                    for gnsrt in ground_nsrts
            ]):
                fp_states.append(state)
            else:
                tn_states.append(state)
        
        # Convert the states to string
        n_tp, n_fn = len(tp_states), len(fn_states)
        n_tn, n_fp = len(tn_states), len(fp_states)
        acc = (n_tp + n_tn) / n_tot
        # If there are a lot of failed states and a lot of success states, then
        # fp would be very small even with a weak precondition.
        # In other word, a weak precondition would have large tp but small tn
        # and would be accepted under this metric.

        complexity_penalty = CFG.grammar_search_pred_complexity_weight *\
                            len(preconditions)
        cost = -acc + complexity_penalty
        logging.debug(f"{preconditions} gets score {cost}: tot={n_tot}, "
                      f"tp={n_tp}, tn={n_tn}, fp={n_fp}")
        return cost

    @staticmethod
    def _get_precondition_successors(
        preconditions: FrozenSet[LiftedAtom]
    ) -> Iterator[Tuple[int, FrozenSet[LiftedAtom], float]]:
        """The successors remove each atom in the preconditions."""
        preconditions_sorted = sorted(preconditions)
        for i in range(len(preconditions_sorted)):
            successor = preconditions_sorted[:i] + preconditions_sorted[i + 1:]
            yield i, frozenset(successor), 1.0


class ClusterAndSearchSTRIPSLearner(ClusteringSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via search,
    following the LOFT algorithm: https://arxiv.org/abs/2103.00589."""

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        new_pnads = []
        for i, pnad in enumerate(pnads):
            positive_data = pnad.datastore
            # Construct negative data by merging the datastores of all
            # other PNADs that have the same option.
            negative_data = []
            for j, other_pnad in enumerate(pnads):
                if i == j:
                    continue
                if pnad.option_spec[0] != other_pnad.option_spec[0]:
                    continue
                negative_data.extend(other_pnad.datastore)
            # Run the top-level search to find sets of precondition sets. This
            # also produces datastores, letting us avoid making a potentially
            # expensive call to recompute_datastores_from_segments().
            all_preconditions_to_datastores = self._run_outer_search(
                pnad, positive_data, negative_data)
            for j, preconditions in enumerate(all_preconditions_to_datastores):
                datastore = all_preconditions_to_datastores[preconditions]
                new_pnads.append(
                    PNAD(
                        pnad.op.copy_with(name=f"{pnad.op.name}-{j}",
                                          preconditions=preconditions),
                        datastore, pnad.option_spec))
        return new_pnads

    def _run_outer_search(
            self, pnad: PNAD, positive_data: Datastore,
            negative_data: Datastore
    ) -> Dict[FrozenSet[LiftedAtom], Datastore]:
        """Run outer-level search to find a set of precondition sets and
        associated datastores.

        Each precondition set will produce one operator.
        """
        all_preconditions_to_datastores = {}
        # We'll remove positives as they get covered.
        remaining_positives = list(positive_data)
        while remaining_positives:
            new_preconditions = self._run_inner_search(pnad,
                                                       remaining_positives,
                                                       negative_data)
            # Compute the datastore and update the remaining positives.
            datastore = []
            new_remaining_positives = []
            for seg, var_to_obj in remaining_positives:
                ground_pre = {a.ground(var_to_obj) for a in new_preconditions}
                if not ground_pre.issubset(seg.init_atoms):
                    # If the preconditions ground with this substitution don't
                    # hold in this segment's init_atoms, this segment has yet
                    # to be covered, so we keep it in the positives.
                    new_remaining_positives.append((seg, var_to_obj))
                else:
                    # Otherwise, we can add this segment to the datastore and
                    # also move it to negative_data, for any future
                    # preconditions that get learned.
                    datastore.append((seg, var_to_obj))
                    negative_data.append((seg, var_to_obj))
            # Special case: if the datastore is empty, that means these
            # new_preconditions don't cover any positives, so the search
            # failed to find preconditions that have a better score than inf.
            # Therefore we give up, without including these new_preconditions
            # into all_preconditions_to_datastores.
            if len(datastore) == 0:
                break
            assert len(new_remaining_positives) < len(remaining_positives)
            remaining_positives = new_remaining_positives
            # Update all_preconditions_to_datastores.
            assert new_preconditions not in all_preconditions_to_datastores
            all_preconditions_to_datastores[new_preconditions] = datastore
        if not all_preconditions_to_datastores:
            # If we couldn't find any preconditions, default to empty.
            assert len(remaining_positives) == len(positive_data)
            all_preconditions_to_datastores[frozenset()] = positive_data
        return all_preconditions_to_datastores

    def _run_inner_search(self, pnad: PNAD, positive_data: Datastore,
                          negative_data: Datastore) -> FrozenSet[LiftedAtom]:
        """Run inner-level search to find a single precondition set."""
        initial_state = self._get_initial_preconditions(positive_data)
        check_goal = lambda s: False
        heuristic = functools.partial(self._score_preconditions, pnad,
                                      positive_data, negative_data)
        max_expansions = CFG.cluster_and_search_inner_search_max_expansions
        timeout = CFG.cluster_and_search_inner_search_timeout
        path, _ = utils.run_gbfs(initial_state,
                                 check_goal,
                                 self._get_precondition_successors,
                                 heuristic,
                                 max_expansions=max_expansions,
                                 timeout=timeout)
        return path[-1]

    @staticmethod
    def _get_initial_preconditions(
            positive_data: Datastore) -> FrozenSet[LiftedAtom]:
        """The initial preconditions are a UNION over all lifted initial states
        in the data.

        We filter out atoms containing any object that doesn't have a
        binding to the PNAD parameters.
        """
        initial_preconditions = set()
        for seg, var_to_obj in positive_data:
            obj_to_var = {v: k for k, v in var_to_obj.items()}
            for atom in seg.init_atoms:
                if not all(obj in obj_to_var for obj in atom.objects):
                    continue
                initial_preconditions.add(atom.lift(obj_to_var))
        return frozenset(initial_preconditions)

    @staticmethod
    def _get_precondition_successors(
        preconditions: FrozenSet[LiftedAtom]
    ) -> Iterator[Tuple[int, FrozenSet[LiftedAtom], float]]:
        """The successors remove each atom in the preconditions."""
        preconditions_sorted = sorted(preconditions)
        for i in range(len(preconditions_sorted)):
            successor = preconditions_sorted[:i] + preconditions_sorted[i + 1:]
            yield i, frozenset(successor), 1.0

    @staticmethod
    def _score_preconditions(pnad: PNAD, positive_data: Datastore,
                             negative_data: Datastore,
                             preconditions: FrozenSet[LiftedAtom]) -> float:
        candidate_op = pnad.op.copy_with(preconditions=preconditions)
        option_spec = pnad.option_spec
        del pnad  # unused after this
        # Count up the number of true positives and false positives.
        num_true_positives = 0
        num_false_positives = 0
        for seg, var_to_obj in positive_data:
            ground_pre = {a.ground(var_to_obj) for a in preconditions}
            if ground_pre.issubset(seg.init_atoms):
                num_true_positives += 1
        if num_true_positives == 0:
            # As a special case, if the number of true positives is 0, we
            # never want to accept these preconditions, so we can give up.
            return float("inf")
        for seg, _ in negative_data:
            # We don't want to use the substitution in the datastore for
            # negative_data, because in general the variables could be totally
            # different. So we consider all possible groundings that are
            # consistent with the option_spec. If, for any such grounding, the
            # preconditions hold in the segment's init_atoms, then this is a
            # false positive.
            objects = list(seg.states[0])
            option = seg.get_option()
            assert option.parent == option_spec[0]
            option_objs = option.objects
            isub = dict(zip(option_spec[1], option_objs))
            for idx, ground_op in enumerate(
                    utils.all_ground_operators_given_partial(
                        candidate_op, objects, isub)):
                # If the maximum number of groundings is reached, treat this
                # as a false positive. Doesn't really matter in practice
                # because the GBFS is going to time out anyway -- we just
                # want the code to not hang in this score function.
                if idx >= CFG.cluster_and_search_score_func_max_groundings or \
                   ground_op.preconditions.issubset(seg.init_atoms):
                    num_false_positives += 1
                    break
        tp_w = CFG.clustering_learner_true_pos_weight
        fp_w = CFG.clustering_learner_false_pos_weight
        score = fp_w * num_false_positives + tp_w * (-num_true_positives)
        # Penalize the number of variables in the preconditions.
        all_vars = {v for atom in preconditions for v in atom.variables}
        score += CFG.cluster_and_search_var_count_weight * len(all_vars)
        # Penalize the number of preconditions.
        score += CFG.cluster_and_search_precon_size_weight * len(preconditions)
        return score

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_search"


class ClusterAndIntersectSidelineSTRIPSLearner(ClusterAndIntersectSTRIPSLearner
                                               ):
    """Base class for a clustering-based STRIPS learner that does sidelining
    via hill climbing, after operator learning."""

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        # Run hill climbing search, starting from original PNADs.
        path, _, _ = utils.run_hill_climbing(
            tuple(pnads), self._check_goal, self._get_sidelining_successors,
            functools.partial(self._evaluate, pnads))
        # The last state in the search holds the final PNADs.
        pnads = list(path[-1])
        # Because the PNADs have been modified, recompute the datastores.
        self._recompute_datastores_from_segments(pnads)
        # Filter out PNADs that have an empty datastore.
        pnads = [pnad for pnad in pnads if pnad.datastore]
        return pnads

    @abc.abstractmethod
    def _evaluate(self, initial_pnads: List[PNAD], s: Tuple[PNAD,
                                                            ...]) -> float:
        """Abstract evaluation/score function for search.

        Lower is better.
        """
        raise NotImplementedError("Override me!")

    @staticmethod
    def _check_goal(s: Tuple[PNAD, ...]) -> bool:
        del s  # unused
        # There are no goal states for this search; run until exhausted.
        return False

    @staticmethod
    def _get_sidelining_successors(
        s: Tuple[PNAD,
                 ...], ) -> Iterator[Tuple[None, Tuple[PNAD, ...], float]]:
        # For each PNAD/operator...
        for i in range(len(s)):
            pnad = s[i]
            _, option_vars = pnad.option_spec
            # ...consider changing each of its add effects to an ignore effect.
            for effect in pnad.op.add_effects:
                if len(pnad.op.add_effects) > 1:
                    # We don't want sidelining to result in a noop.
                    new_pnad = PNAD(
                        pnad.op.effect_to_ignore_effect(
                            effect, option_vars, "add"), pnad.datastore,
                        pnad.option_spec)
                    sprime = list(s)
                    sprime[i] = new_pnad
                    yield (None, tuple(sprime), 1.0)

            # ...consider removing it.
            sprime = list(s)
            del sprime[i]
            yield (None, tuple(sprime), 1.0)


class ClusterAndIntersectSidelinePredictionErrorSTRIPSLearner(
        ClusterAndIntersectSidelineSTRIPSLearner):
    """A STRIPS learner that uses hill climbing with a prediction error score
    function for ignore effect learning."""

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect_sideline_prederror"

    def _evaluate(self, initial_pnads: List[PNAD], s: Tuple[PNAD,
                                                            ...]) -> float:
        segments = [seg for traj in self._segmented_trajs for seg in traj]
        strips_ops = [pnad.op for pnad in s]
        option_specs = [pnad.option_spec for pnad in s]
        max_groundings = CFG.cluster_and_intersect_prederror_max_groundings
        num_true_positives, num_false_positives, _, _ = \
            utils.count_positives_for_ops(strips_ops, option_specs, segments,
                                          max_groundings=max_groundings)
        # Note: lower is better! We want more true positives and fewer
        # false positives.
        tp_w = CFG.clustering_learner_true_pos_weight
        fp_w = CFG.clustering_learner_false_pos_weight
        return fp_w * num_false_positives + tp_w * (-num_true_positives)


class ClusterAndIntersectSidelineHarmlessnessSTRIPSLearner(
        ClusterAndIntersectSidelineSTRIPSLearner):
    """A STRIPS learner that uses hill climbing with a harmlessness score
    function for ignore effect learning."""

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect_sideline_harmlessness"

    def _evaluate(self, initial_pnads: List[PNAD], s: Tuple[PNAD,
                                                            ...]) -> float:
        preserves_harmlessness = self._check_harmlessness(list(s))
        if preserves_harmlessness:
            # If harmlessness is preserved, the score is the number of
            # operators that we have, minus the number of ignore effects.
            # This means we prefer fewer operators and more ignore effects.
            score = 2 * len(s)
            for pnad in s:
                score -= len(pnad.op.ignore_effects)
        else:
            # If harmlessness is not preserved, the score is an arbitrary
            # constant bigger than the total number of operators at the
            # start of the search. This is guaranteed to be worse (higher)
            # than any score that occurs if harmlessness is preserved.
            score = 10 * len(initial_pnads)
        return score
