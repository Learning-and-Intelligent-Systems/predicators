"""Algorithms for STRIPS learning that rely on clustering to obtain effects.
"""

import abc
import logging
import functools
from typing import List, Set, cast, Tuple, Iterator

from predicators.src import utils
from predicators.src.structs import DummyOption, LiftedAtom, VarToObjSub, \
    PartialNSRTAndDatastore, Predicate, STRIPSOperator
from predicators.src.nsrt_learning.strips_learning import BaseSTRIPSLearner


class ClusteringSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a clustering-based STRIPS learner.
    """
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
                sub = cast(VarToObjSub, {v: o for o, v
                                         in ent_to_ent_sub.items()})
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
                side_predicates: Set[Predicate] = set()  # will be learned later
                op = STRIPSOperator(f"Op{len(pnads)}", params, preconds,
                                    add_effects, delete_effects,
                                    side_predicates)
                datastore = [(segment, var_to_obj)]
                option_vars = [obj_to_var[o] for o in segment_option_objs]
                option_spec = (segment_param_option, option_vars)
                pnads.append(PartialNSRTAndDatastore(
                    op, datastore, option_spec))

        # Learn the preconditions of the operators in the PNADs. This part
        # is flexible; subclasses choose how to implement it.
        for pnad in pnads:
            self._learn_pnad_preconditions(pnad)

        # Handle optional postprocessing to learn side predicates.
        pnads = self._postprocessing_learn_side_predicates(pnads)

        # Log and return the PNADs.
        if self._verbose:
            logging.info("Learned operators (before option learning):")
            for pnad in pnads:
                logging.info(pnad)
        return pnads

    @abc.abstractmethod
    def _learn_pnad_preconditions(self, pnad: PartialNSRTAndDatastore) -> None:
        """Subclass-specific algorithm for learning PNAD preconditions.

        Modifies the given PNAD.
        """
        raise NotImplementedError("Override me!")

    def _postprocessing_learn_side_predicates(
            self, pnads: List[PartialNSRTAndDatastore]
    ) -> List[PartialNSRTAndDatastore]:
        """Optionally postprocess to learn side predicates.
        """
        _ = self  # unused, but may be used in subclasses
        return pnads


class ClusterAndIntersectSTRIPSLearner(ClusteringSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via intersection.
    """
    def _learn_pnad_preconditions(self, pnad: PartialNSRTAndDatastore) -> None:
        preconditions = self._induce_preconditions_via_intersection(pnad)
        pnad.op = pnad.op.copy_with(preconditions=preconditions)


class ClusterAndSearchSTRIPSLearner(ClusteringSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via search,
    following the LOFT algorithm: https://arxiv.org/abs/2103.00589.
    """
    def _learn_pnad_preconditions(self, pnad: PartialNSRTAndDatastore) -> None:
        raise NotImplementedError


class ClusterAndIntersectSidelineSTRIPSLearner(
        ClusterAndIntersectSTRIPSLearner):
    """Base class for a clustering-based STRIPS learner that does sidelining
    via hill climbing, after operator learning.
    """
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

    def _evaluate(self, initial_pnads: List[PartialNSRTAndDatastore],
                  s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        segments = [seg for traj in self._segmented_trajs for seg in traj]
        strips_ops = [pnad.op for pnad in s]
        option_specs = [pnad.option_spec for pnad in s]
        num_true_positives, num_false_positives, _, _ = \
            utils.count_positives_for_ops(strips_ops, option_specs, segments)
        # Note: lower is better! We want more true positives and fewer
        # false positives.
        return num_false_positives + 10 * (-num_true_positives)

    @property
    def _should_satisfy_harmlessness(self) -> bool:
        # There are no guarantees that local search to improve prediction error
        # would satisfy harmlessness!
        return False


class ClusterAndIntersectSidelineHarmlessnessSTRIPSLearner(
        ClusterAndIntersectSidelineSTRIPSLearner):
    """A STRIPS learner that uses hill climbing with a harmlessness score
    function for side predicate learning."""

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
