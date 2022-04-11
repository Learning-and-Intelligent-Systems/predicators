"""Methods for learning to sideline predicates in NSRTs."""

import abc
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import \
    induce_pnad_preconditions
from predicators.src.planning import task_plan_grounding
from predicators.src.predicate_search_score_functions import \
    _PredictionErrorScoreFunction
from predicators.src.settings import CFG
from predicators.src.structs import GroundAtom, LiftedAtom, \
    LowLevelTrajectory, OptionSpec, ParameterizedOption, \
    PartialNSRTAndDatastore, Predicate, Segment, State, STRIPSOperator, Task, \
    Variable, _GroundNSRT, _GroundSTRIPSOperator


class HillClimbingSidePredicateLearner(SidePredicateLearner):
    """An abstract side predicate learning strategy that performs hill climbing
    over candidate sidelinings of add effects, one at a time.

    Leaves the evaluation function unspecified.
    """

    def _sideline(self) -> List[PartialNSRTAndDatastore]:
        # Run the search, starting from original PNADs.
        path, _, _ = utils.run_hill_climbing(tuple(self._initial_pnads),
                                             self._check_goal,
                                             self._get_successors,
                                             self._evaluate)
        # The last state in the search holds the final PNADs.
        return list(path[-1])

    @abc.abstractmethod
    def _evaluate(self, s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
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
    def _get_successors(
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


class PredictionErrorHillClimbingSidePredicateLearner(
        HillClimbingSidePredicateLearner):
    """A side predicate learning strategy that does hill climbing with a
    prediction error score function."""

    def __init__(self, pnads: List[PartialNSRTAndDatastore],
                 trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task], predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]]) -> None:
        super().__init__(pnads, trajectories, train_tasks, predicates,
                         segmented_trajs)
        self._score_func = _PredictionErrorScoreFunction(
            self._predicates, [], {}, self._train_tasks)

    def _evaluate(self, s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        strips_ops = [pnad.op for pnad in s]
        option_specs = [pnad.option_spec for pnad in s]
        score = self._score_func.evaluate_with_operators(
            frozenset(), self._trajectories, self._segmented_trajs, strips_ops,
            option_specs)
        return score


class PreserveSkeletonsHillClimbingSidePredicateLearner(
        HillClimbingSidePredicateLearner):
    """A side predicate learning strategy that does hill climbing with a
    skeleton preservation (harmlessness) score function."""

    def _evaluate(self, s: Tuple[PartialNSRTAndDatastore, ...]) -> float:
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
            score = 10 * len(self._initial_pnads)
        return score
