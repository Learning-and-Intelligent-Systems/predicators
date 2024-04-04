"""Algorithms for STRIPS learning that rely on clustering to obtain effects."""

import abc
import functools
import logging
from typing import Dict, FrozenSet, Iterator, List, Set, Tuple, cast

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.src.settings import CFG
from predicators.src.structs import Datastore, DummyOption, LiftedAtom, \
    PartialNSRTAndDatastore, Predicate, STRIPSOperator, VarToObjSub


class ClusteringSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a clustering-based STRIPS learner."""

    def _learn(self) -> List[PartialNSRTAndDatastore]:
        segments = [seg for segs in self._segmented_trajs for seg in segs]
        # Cluster the segments according to common option and effects.
        pnads: List[PartialNSRTAndDatastore] = []
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
                suc, ent_to_ent_sub = utils.unify_preconds_effects_options(
                    frozenset(),  # no preconditions
                    frozenset(),  # no preconditions
                    frozenset(segment.add_effects),
                    frozenset(pnad.op.add_effects),
                    frozenset(segment.delete_effects),
                    frozenset(pnad.op.delete_effects),
                    segment_param_option,
                    pnad_param_option,
                    segment_option_objs,
                    tuple(pnad_option_vars))
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
                params = utils.create_new_variables(
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
                side_predicates: Set[Predicate] = set(
                )  # will be learned later
                op = STRIPSOperator(f"Op{len(pnads)}", params, preconds,
                                    add_effects, delete_effects,
                                    side_predicates)
                datastore = [(segment, var_to_obj)]
                option_vars = [obj_to_var[o] for o in segment_option_objs]
                option_spec = (segment_param_option, option_vars)
                pnads.append(
                    PartialNSRTAndDatastore(op, datastore, option_spec))

        # Learn the preconditions of the operators in the PNADs. This part
        # is flexible; subclasses choose how to implement it.
        pnads = self._learn_pnad_preconditions(pnads)

        # Handle optional postprocessing to learn side predicates.
        pnads = self._postprocessing_learn_side_predicates(pnads)

        # Log and return the PNADs.
        if self._verbose:
            logging.info("Learned operators (before option learning):")
            for pnad in pnads:
                logging.info(pnad)
        return pnads

    @abc.abstractmethod
    def _learn_pnad_preconditions(
            self, pnads: List[PartialNSRTAndDatastore]
    ) -> List[PartialNSRTAndDatastore]:
        """Subclass-specific algorithm for learning PNAD preconditions.

        Returns a list of new PNADs. Should NOT modify the given PNADs.
        """
        raise NotImplementedError("Override me!")

    def _postprocessing_learn_side_predicates(
            self, pnads: List[PartialNSRTAndDatastore]
    ) -> List[PartialNSRTAndDatastore]:
        """Optionally postprocess to learn side predicates."""
        _ = self  # unused, but may be used in subclasses
        return pnads


class ClusterAndIntersectSTRIPSLearner(ClusteringSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via
    intersection."""

    def _learn_pnad_preconditions(
            self, pnads: List[PartialNSRTAndDatastore]
    ) -> List[PartialNSRTAndDatastore]:
        new_pnads = []
        for pnad in pnads:
            preconditions = self._induce_preconditions_via_intersection(pnad)
            # Since we are taking an intersection, we're guaranteed that the
            # datastore can't change, so we can safely use pnad.datastore here.
            new_pnads.append(
                PartialNSRTAndDatastore(
                    pnad.op.copy_with(preconditions=preconditions),
                    pnad.datastore, pnad.option_spec))
        return new_pnads

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect"


class ClusterAndSearchSTRIPSLearner(ClusteringSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via search,
    following the LOFT algorithm: https://arxiv.org/abs/2103.00589."""

    def _learn_pnad_preconditions(
            self, pnads: List[PartialNSRTAndDatastore]
    ) -> List[PartialNSRTAndDatastore]:
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
                    PartialNSRTAndDatastore(
                        pnad.op.copy_with(name=f"{pnad.op.name}-{j}",
                                          preconditions=preconditions),
                        datastore, pnad.option_spec))
        return new_pnads

    def _run_outer_search(
            self, pnad: PartialNSRTAndDatastore, positive_data: Datastore,
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

    def _run_inner_search(self, pnad: PartialNSRTAndDatastore,
                          positive_data: Datastore,
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
    def _score_preconditions(pnad: PartialNSRTAndDatastore,
                             positive_data: Datastore,
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

    def _postprocessing_learn_side_predicates(
            self, pnads: List[PartialNSRTAndDatastore]
    ) -> List[PartialNSRTAndDatastore]:
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
    def _evaluate(self, initial_pnads: List[PartialNSRTAndDatastore],
                  s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        """Abstract evaluation/score function for search.

        Lower is better.
        """
        raise NotImplementedError("Override me!")

    @staticmethod
    def _check_goal(s: Tuple[PartialNSRTAndDatastore, ...]) -> bool:
        del s  # unused
        # There are no goal states for this search; run until exhausted.
        return False

    @staticmethod
    def _get_sidelining_successors(
        s: Tuple[PartialNSRTAndDatastore, ...],
    ) -> Iterator[Tuple[None, Tuple[PartialNSRTAndDatastore, ...], float]]:
        # For each PNAD/operator...
        for i in range(len(s)):
            pnad = s[i]
            _, option_vars = pnad.option_spec
            # ...consider changing each of its add effects to a side predicate.
            for effect in pnad.op.add_effects:
                if len(pnad.op.add_effects) > 1:
                    # We don't want sidelining to result in a no-op.
                    new_pnad = PartialNSRTAndDatastore(
                        pnad.op.effect_to_side_predicate(
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
    function for side predicate learning."""

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect_sideline_prederror"

    def _evaluate(self, initial_pnads: List[PartialNSRTAndDatastore],
                  s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
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
    function for side predicate learning."""

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect_sideline_harmlessness"

    def _evaluate(self, initial_pnads: List[PartialNSRTAndDatastore],
                  s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        preserves_harmlessness = self._check_harmlessness(list(s))
        if preserves_harmlessness:
            # If harmlessness is preserved, the score is the number of
            # operators that we have, minus the number of side predicates.
            # This means we prefer fewer operators and more side predicates.
            score = 2 * len(s)
            for pnad in s:
                score -= len(pnad.op.side_predicates)
        else:
            # If harmlessness is not preserved, the score is an arbitrary
            # constant bigger than the total number of operators at the
            # start of the search. This is guaranteed to be worse (higher)
            # than any score that occurs if harmlessness is preserved.
            score = 10 * len(initial_pnads)
        return score
