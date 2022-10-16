"""Learn operators by searching over sets of add effect sets."""

from __future__ import annotations

import abc
import functools
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from predicators import utils
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import GroundAtom, LiftedAtom, LowLevelTrajectory, \
    Object, OptionSpec, ParameterizedOption, PartialNSRTAndDatastore, \
    Predicate, Segment, STRIPSOperator, Task, _GroundSTRIPSOperator

_PNADMap = Dict[ParameterizedOption, List[PartialNSRTAndDatastore]]
# Necessary images and ground operators, in reverse order.
_Chain = Tuple[List[Set[GroundAtom]], List[_GroundSTRIPSOperator]]


@dataclass(frozen=True)
class _EffectSets:
    # Maps a parameterized option to a set of option specs (for that option)
    # and add effects.
    _param_option_to_groups: Dict[ParameterizedOption,
                                  List[Tuple[OptionSpec, Set[LiftedAtom]]]]

    @functools.cached_property
    def _hash(self) -> int:
        option_to_hashable_group = {
            o: (tuple(v), frozenset(a))
            for o in self._param_option_to_groups
            for ((_, v), a) in self._param_option_to_groups[o]
        }
        return hash(tuple(option_to_hashable_group))

    def __hash__(self) -> int:
        return self._hash

    def __iter__(self) -> Iterator[Tuple[OptionSpec, Set[LiftedAtom]]]:
        for o in sorted(self._param_option_to_groups):
            for (spec, atoms) in self._param_option_to_groups[o]:
                yield (spec, atoms)

    def __str__(self) -> str:
        s = ""
        for (spec, atoms) in self:
            opt = spec[0].name
            opt_args = ", ".join([str(a) for a in spec[1]])
            s += f"\n{opt}({opt_args}): {atoms}"
        s += "\n"
        return s

    def __repr__(self) -> str:
        return str(self)

    def add(self, option_spec: OptionSpec,
            add_effects: Set[LiftedAtom]) -> _EffectSets:
        """Create a new _EffectSets with this new entry added to existing."""
        param_option = option_spec[0]
        assert param_option in self._param_option_to_groups
        new_param_option_to_groups = {
            p: [(s, set(a)) for s, a in group]
            for p, group in self._param_option_to_groups.items()
        }
        new_param_option_to_groups[param_option].append(
            (option_spec, add_effects))
        return _EffectSets(new_param_option_to_groups)


class _EffectSearchOperator(abc.ABC):
    """An operator that proposes successor sets of effect sets."""

    def __init__(
        self,
        trajectories: List[LowLevelTrajectory],
        train_tasks: List[Task],
        predicates: Set[Predicate],
        segmented_trajs: List[List[Segment]],
        effect_sets_to_pnads: Callable[[_EffectSets], _PNADMap],
        backchain: Callable[[List[Segment], _PNADMap, Set[GroundAtom]],
                            _Chain],
    ) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._effect_sets_to_pnads = effect_sets_to_pnads
        self._backchain = backchain

    @abc.abstractmethod
    def get_successors(self,
                       effect_sets: _EffectSets) -> Iterator[_EffectSets]:
        """Generate zero or more successor effect sets."""
        raise NotImplementedError("Override me!")


class _BackChainingEffectSearchOperator(_EffectSearchOperator):
    """An operator that uses backchaining to propose a new effect set."""

    def get_successors(self,
                       effect_sets: _EffectSets) -> Iterator[_EffectSets]:
        pnads = self._effect_sets_to_pnads(effect_sets)
        uncovered_transition = self._get_first_uncovered_transition(pnads)
        if uncovered_transition is None:
            return
        param_option, option_objs, add_effs = uncovered_transition
        # Create a new effect set.
        all_objs = sorted(
            set(option_objs) | {o
                                for a in add_effs for o in a.objects})
        all_types = [o.type for o in all_objs]
        all_vars = utils.create_new_variables(all_types)
        obj_to_var = dict(zip(all_objs, all_vars))
        option_vars = [obj_to_var[o] for o in option_objs]
        option_spec = (param_option, option_vars)
        lifted_add_effs = {a.lift(obj_to_var) for a in add_effs}
        new_effect_sets = effect_sets.add(option_spec, lifted_add_effs)
        yield new_effect_sets

    def _get_first_uncovered_transition(
        self, pnads: _PNADMap
    ) -> Optional[Tuple[ParameterizedOption, List[Object], Set[GroundAtom]]]:
        # Find the first uncovered segment. Do this in a kind of breadth-first
        # backward search over trajectories. TODO: see whether this matters.
        # Compute all the chains once up front.
        backchaining_results = []
        max_chain_len = 0
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            atoms_seq = utils.segment_trajectory_to_atoms_sequence(seg_traj)
            traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
            chain = self._backchain(seg_traj, pnads, traj_goal)
            max_chain_len = max(max_chain_len, len(chain[1]))
            backchaining_results.append(
                (seg_traj, atoms_seq, traj_goal, chain))
        # Now look for an uncovered segment.
        for depth in range(max_chain_len + 1):
            for seg_traj, atoms_seq, traj_goal, chain in backchaining_results:
                image_chain, op_chain = chain
                if len(op_chain) > depth:
                    continue
                # We found an uncovered transition.
                # TODO make this less horrible.
                necessary_image = image_chain[-1]
                t = (len(seg_traj) - 1) - len(op_chain)
                segment = seg_traj[t]
                necessary_add_effects = necessary_image - atoms_seq[t]
                assert necessary_add_effects.issubset(segment.add_effects)
                option = segment.get_option()
                return (option.parent, option.objects, necessary_add_effects)
        # Everything was covered.
        return None


class _EffectSearchHeuristic(abc.ABC):
    """Given a set of effect sets, produce a score, with lower better."""

    def __init__(
        self,
        trajectories: List[LowLevelTrajectory],
        train_tasks: List[Task],
        predicates: Set[Predicate],
        segmented_trajs: List[List[Segment]],
        effect_sets_to_pnads: Callable[[_EffectSets], _PNADMap],
        backchain: Callable[[List[Segment], _PNADMap, Set[GroundAtom]],
                            _Chain],
    ) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._effect_sets_to_pnads = effect_sets_to_pnads
        self._backchain = backchain

    @abc.abstractmethod
    def __call__(self, effect_sets: _EffectSets) -> float:
        """Compute the heuristic value for the given effect sets."""
        raise NotImplementedError("Override me!")


class _BackChainingHeuristic(_EffectSearchHeuristic):
    """Counts the number of transitions that are not yet covered by some
    operator in the backchaining sense."""

    def __call__(self, effect_sets: _EffectSets) -> float:
        pnads = self._effect_sets_to_pnads(effect_sets)
        uncovered_transitions = 0
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
            _, chain = self._backchain(seg_traj, pnads, traj_goal)
            uncovered_transitions += (len(seg_traj) - 1) - len(chain)
        return uncovered_transitions


class EffectSearchSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a effect search STRIPS learner."""

    @classmethod
    def get_name(cls) -> str:
        return "effect_search"

    def _learn(self) -> List[PartialNSRTAndDatastore]:

        # Set up hill-climbing search over effect sets.
        _S: TypeAlias = _EffectSets
        # An "action" here is a search operator and an integer representing the
        # count of successors generated by that operator.
        _A: TypeAlias = Tuple[_EffectSearchOperator, int]

        # Create the search operators.
        search_operators = self._create_search_operators()

        # Create the heuristic.
        heuristic = self._create_heuristic()

        # Initialize the search.
        initial_state = self._create_initial_effect_sets()

        def get_successors(effs: _S) -> Iterator[Tuple[_A, _S, float]]:
            for op in search_operators:
                for i, child in enumerate(op.get_successors(effs)):
                    yield (op, i), child, 1.0  # cost always 1

        # Run hill-climbing search.
        path, _, _ = utils.run_hill_climbing(initial_state=initial_state,
                                             check_goal=lambda _: False,
                                             get_successors=get_successors,
                                             heuristic=heuristic)

        # Extract the best effect set.
        best_effect_sets = path[-1]

        # Convert into operators.
        import ipdb
        ipdb.set_trace()

    def _create_search_operators(self) -> List[_EffectSearchOperator]:
        backchaining_op = _BackChainingEffectSearchOperator(
            self._trajectories, self._train_tasks, self._predicates,
            self._segmented_trajs, self._effect_sets_to_pnads, self._backchain)
        return [backchaining_op]

    def _create_heuristic(self) -> _EffectSearchHeuristic:
        backchaining_heur = _BackChainingHeuristic(
            self._trajectories, self._train_tasks, self._predicates,
            self._segmented_trajs, self._effect_sets_to_pnads, self._backchain)
        return backchaining_heur

    def _create_initial_effect_sets(self) -> _EffectSets:
        param_option_to_groups = {}
        param_options = [
            s.get_option().parent for segs in self._segmented_trajs
            for s in segs
        ]
        for param_option in param_options:
            option_vars = utils.create_new_variables(param_option.types)
            option_spec = (param_option, option_vars)
            add_effects = set()
            group = (option_spec, add_effects)
            param_option_to_groups[param_option] = [group]
        return _EffectSets(param_option_to_groups)

    @functools.lru_cache(maxsize=None)
    def _effect_sets_to_pnads(
            self, effect_sets: _EffectSets) -> List[PartialNSRTAndDatastore]:
        pnads = self._effect_sets_to_pnads(effect_sets)
        # Add preconditions.
        for pnad in pnads:
            # If the PNAD is empty, that means we haven't yet computed good
            # effects, so none of the data is matching this PNAD. Just skip
            # in this case and wait for the search to find better effects.
            if not pnad.datastore:
                continue
            preconditions = self._induce_preconditions_via_intersection(pnad)
            pnad.op = pnad.op.copy_with(preconditions=preconditions)
        # Add delete and ignore effects.
        for pnad in pnads:
            self._compute_pnad_delete_effects(pnad)
            self._compute_pnad_ignore_effects(pnad)

        # TODO: handle keep effects in the outer search or here?

        return pnads

    @functools.lru_cache(maxsize=None)
    def _effect_sets_to_pnads(self, effect_sets: _EffectSets) -> _PNADMap:
        pnads = []
        for (option_spec, add_effects) in effect_sets:
            parameterized_option, parameters = option_spec
            op = STRIPSOperator(parameterized_option.name, parameters, set(),
                                add_effects, set(), set())
            pnad = PartialNSRTAndDatastore(op, [], option_spec)
            pnads.append(pnad)
        self._recompute_datastores_from_segments(pnads)
        pnad_map: _PNADMap = {pnad.option_spec[0]: [] for pnad in pnads}
        for pnad in pnads:
            pnad_map[pnad.option_spec[0]].append(pnad)
        return pnad_map

    def _backchain(self, segmented_traj: List[Segment], pnads: _PNADMap,
                   traj_goal: Set[GroundAtom]) -> _Chain:
        """Returns ground operators in REVERSE order."""
        image_chain: List[Set[GroundAtom]] = []
        operator_chain: List[_GroundSTRIPSOperator] = []
        atoms_seq = utils.segment_trajectory_to_atoms_sequence(segmented_traj)
        objects = set(segmented_traj[0].states[0])
        assert traj_goal.issubset(atoms_seq[-1])
        necessary_image = set(traj_goal)
        for t in range(len(atoms_seq) - 2, -1, -1):
            image_chain.append(necessary_image)
            segment = segmented_traj[t]
            param_option = segment.get_option().parent
            best_pnad, best_sub = self._find_best_matching_pnad_and_sub(
                segment, objects, pnads[param_option])
            # If no match found, terminate.
            if best_pnad is None:
                break
            # Otherwise, add to the chain.
            # TODO
            operator_chain.append(...)
            import ipdb
            ipdb.set_trace()
        return (image_chain, operator_chain)
