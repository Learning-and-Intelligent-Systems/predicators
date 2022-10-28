"""Learn operators by searching over sets of add effect sets."""

from __future__ import annotations

import abc
import functools
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, \
    Tuple

from predicators import utils
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.structs import GroundAtom, LiftedAtom, LowLevelTrajectory, \
    Object, OptionSpec, ParameterizedOption, PartialNSRTAndDatastore, \
    Predicate, Segment, STRIPSOperator, Task, _GroundSTRIPSOperator

# Necessary images and ground operators, in reverse order. Also, if the chain
# is not full-length, then the "best" operator that backchaining tried to use
# but couldn't.
_Chain = Tuple[List[Set[GroundAtom]], List[_GroundSTRIPSOperator],
               Optional[_GroundSTRIPSOperator]]


@dataclass(frozen=True)
class _EffectSets:
    # Maps a parameterized option to a set of option specs (for that option),
    # add effects, and keep effects.
    _param_option_to_groups: Dict[ParameterizedOption,
                                  List[Tuple[OptionSpec, Set[LiftedAtom],
                                             Set[LiftedAtom]]]]

    @functools.cached_property
    def _hash(self) -> int:
        option_to_hashable_group = {
            o: (tuple(v), frozenset(aa), frozenset(ka))
            for o in self._param_option_to_groups
            for ((_, v), aa, ka) in self._param_option_to_groups[o]
        }
        return hash(tuple(option_to_hashable_group))

    def __hash__(self) -> int:
        return self._hash

    def __iter__(self) -> Iterator[Tuple[OptionSpec, Set[LiftedAtom]]]:
        for o in sorted(self._param_option_to_groups):
            for (spec, add_atoms,
                 keep_atoms) in self._param_option_to_groups[o]:
                yield (spec, add_atoms, keep_atoms)

    def __str__(self) -> str:
        s = ""
        for (spec, add_atoms, keep_atoms) in self:
            opt = spec[0].name
            opt_args = ", ".join([str(a) for a in spec[1]])
            s += f"\n{opt}({opt_args}): {add_atoms}, {keep_atoms}"
        s += "\n"
        return s

    def __repr__(self) -> str:
        return str(self)

    def add(self, option_spec: OptionSpec, add_effects: Set[LiftedAtom],
            keep_effects: Set[LiftedAtom]) -> _EffectSets:
        """Create a new _EffectSets with this new entry added to existing."""
        param_option = option_spec[0]
        new_param_option_to_groups = {
            p: [(s, set(aa), set(ka)) for s, aa, ka in group]
            for p, group in self._param_option_to_groups.items()
        }
        if param_option not in new_param_option_to_groups:
            new_param_option_to_groups[param_option] = []
        new_param_option_to_groups[param_option].append(
            (option_spec, add_effects, keep_effects))
        return _EffectSets(new_param_option_to_groups)

    def remove(self, option_spec: OptionSpec, add_effects: Set[LiftedAtom],
               keep_effects: Set[LiftedAtom]) -> _EffectSets:
        """Create a new _EffectSets with this entry removed from existing."""
        param_option = option_spec[0]
        assert param_option in self._param_option_to_groups
        new_param_option_to_groups = {
            p: [(s, set(aa), set(ka)) for s, aa, ka in group]
            for p, group in self._param_option_to_groups.items()
        }
        new_param_option_to_groups[param_option].remove(
            (option_spec, add_effects, keep_effects))
        return _EffectSets(new_param_option_to_groups)


class _EffectSearchOperator(abc.ABC):
    """An operator that proposes successor sets of effect sets."""

    def __init__(
        self,
        trajectories: List[LowLevelTrajectory],
        train_tasks: List[Task],
        predicates: Set[Predicate],
        segmented_trajs: List[List[Segment]],
        effect_sets_to_pnads: Callable[[_EffectSets],
                                       List[PartialNSRTAndDatastore]],
        backchain: Callable[
            [List[Segment], List[PartialNSRTAndDatastore], Set[GroundAtom]],
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
        param_option, option_objs, add_effs, keep_effs = uncovered_transition
        option_spec, lifted_add_effs, lifted_keep_effs = self._create_new_effect_set(
            param_option, option_objs, add_effs, keep_effs)
        new_effect_sets = effect_sets.add(option_spec, lifted_add_effs,
                                          lifted_keep_effs)
        yield new_effect_sets

    def _create_new_effect_set(self, param_option, option_objs, add_effs,
                               keep_effs):
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
        lifted_keep_effs = {a.lift(obj_to_var) for a in keep_effs}
        return (option_spec, lifted_add_effs, lifted_keep_effs)

    def _get_first_uncovered_transition(
        self,
        pnads: List[PartialNSRTAndDatastore],
    ) -> Optional[Tuple[ParameterizedOption, Sequence[Object], Set[GroundAtom],
                        Set[GroundAtom]]]:
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
                image_chain, op_chain, last_used_op = chain
                if len(op_chain) > depth or len(op_chain) == len(seg_traj):
                    continue
                # We found an uncovered transition: we now need to return
                # the information necessary to induce a new operator to cover
                # it.
                # The timestep of the uncovered transition is the number of segments
                # - 1 - (numer of actions in our backchained plan)
                t = (len(seg_traj) - 1) - len(op_chain)
                assert t >= 0
                segment = seg_traj[t]
                necessary_image = image_chain[-1]
                # Necessary add effects are everything true in the necessary image
                # that was not true after calling the operator from atoms_seq[t].
                option = segment.get_option()
                if last_used_op is not None:
                    # This means that we're adding a new operator by specializing an existing one.
                    pred_next_atoms = utils.apply_operator(
                        last_used_op, atoms_seq[t])
                    missing_effects = (necessary_image - pred_next_atoms)
                    # These are just the missing effects + existing add effects.
                    necessary_add_effects = missing_effects | last_used_op.add_effects
                    # These are the missing effects that were ignore effects of the last_used_op.
                    necessary_keep_effects = {
                        a
                        for a in missing_effects if a in atoms_seq[t]
                        and a.predicate in last_used_op.ignore_effects
                    }
                else:
                    # This means this is the first time we're inducing an operator
                    # for this option.
                    necessary_add_effects = necessary_image - atoms_seq[t]
                    necessary_keep_effects = set()

                return (option.parent, option.objects, necessary_add_effects,
                        necessary_keep_effects)
        # Everything was covered.
        return None


class _PruningEffectSearchOperator(_EffectSearchOperator):
    """An operator that prunes effect sets."""

    def get_successors(self,
                       effect_sets: _EffectSets) -> Iterator[_EffectSets]:
        for (spec, add_effects, keep_effects) in effect_sets:
            yield effect_sets.remove(spec, add_effects, keep_effects)


class _EffectSearchHeuristic(abc.ABC):
    """Given a set of effect sets, produce a score, with lower better."""

    def __init__(
        self,
        trajectories: List[LowLevelTrajectory],
        train_tasks: List[Task],
        predicates: Set[Predicate],
        segmented_trajs: List[List[Segment]],
        effect_sets_to_pnads: Callable[[_EffectSets],
                                       List[PartialNSRTAndDatastore]],
        backchain: Callable[
            [List[Segment], List[PartialNSRTAndDatastore], Set[GroundAtom]],
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
            _, chain, _ = self._backchain(seg_traj, pnads, traj_goal)
            assert len(chain) <= len(seg_traj)
            uncovered_transitions += len(seg_traj) - len(chain)
        return uncovered_transitions


class EffectSearchSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a effect search STRIPS learner."""

    @classmethod
    def get_name(cls) -> str:
        return "effect_search"

    def _learn(self) -> List[PartialNSRTAndDatastore]:

        # Set up hill-climbing search over effect sets.
        # Create the search operators.
        search_operators = self._create_search_operators()

        # Create the heuristic.
        heuristic = self._create_heuristic()

        # Initialize the search.
        initial_state = self._create_initial_effect_sets()

        def get_successors(
            effs: _EffectSets
        ) -> Iterator[Tuple[Tuple[_EffectSearchOperator, int], _EffectSets,
                            float]]:
            for op in search_operators:
                for i, child in enumerate(op.get_successors(effs)):
                    yield (op, i), child, 1.0  # cost always 1

        # Run hill-climbing search.
        path, _, cost = utils.run_hill_climbing(initial_state=initial_state,
                                                check_goal=lambda _: False,
                                                get_successors=get_successors,
                                                heuristic=heuristic)

        # This effectively asserts harmlessness.
        assert cost[-1] == 0
        # Extract the best effect set.
        best_effect_sets = path[-1]
        # Convert into PNADs.
        final_pnads = self._effect_sets_to_pnads(best_effect_sets)
        return final_pnads

    def _create_search_operators(self) -> List[_EffectSearchOperator]:
        op_classes = [
            _BackChainingEffectSearchOperator, _PruningEffectSearchOperator
        ]
        ops = [
            cls(self._trajectories, self._train_tasks, self._predicates,
                self._segmented_trajs, self._effect_sets_to_pnads,
                self._backchain) for cls in op_classes
        ]
        return ops

    def _create_heuristic(self) -> _EffectSearchHeuristic:
        backchaining_heur = _BackChainingHeuristic(
            self._trajectories, self._train_tasks, self._predicates,
            self._segmented_trajs, self._effect_sets_to_pnads, self._backchain)
        return backchaining_heur

    def _create_initial_effect_sets(self) -> _EffectSets:
        param_option_to_groups = {}
        return _EffectSets(param_option_to_groups)

    @functools.lru_cache(maxsize=None)
    def _effect_sets_to_pnads(
            self, effect_sets: _EffectSets) -> List[PartialNSRTAndDatastore]:
        # Start with the add effects and option specs.
        pnads = []
        for (option_spec, add_effects, keep_effects) in effect_sets:
            parameterized_option, op_vars = option_spec
            effect_vars = {v
                           for a in add_effects for v in a.variables
                           } | {v
                                for a in keep_effects for v in a.variables}
            parameters = sorted(set(op_vars) | effect_vars)
            # Add all ignore effects initially so that precondition learning
            # works. Be sure that the keep effects are part of both the
            # preconditions and add effects.
            # TODO: better code way to do this?
            ignore_effects = self._predicates.copy()
            op = STRIPSOperator(parameterized_option.name, parameters,
                                keep_effects, add_effects | keep_effects,
                                set(), ignore_effects)
            pnad = PartialNSRTAndDatastore(op, [], option_spec)
            pnads.append(pnad)
        self._recompute_datastores_from_segments(pnads)
        # Prune any PNADs with empty datastores.
        pnads = [p for p in pnads if p.datastore]
        # Add preconditions.
        for pnad in pnads:
            preconditions = self._induce_preconditions_via_intersection(pnad)
            pnad.op = pnad.op.copy_with(preconditions=preconditions)
        # Add delete and ignore effects.
        for pnad in pnads:
            self._compute_pnad_delete_effects(pnad)
            self._compute_pnad_ignore_effects(pnad)
        # Fix naming.
        pnad_map: Dict[ParameterizedOption, List[PartialNSRTAndDatastore]] = {
            p.option_spec[0]: []
            for p in pnads
        }
        for p in pnads:
            pnad_map[p.option_spec[0]].append(p)
        pnads = self._get_uniquely_named_nec_pnads(pnad_map)
        return pnads

    def _backchain(self, segmented_traj: List[Segment],
                   pnads: List[PartialNSRTAndDatastore],
                   traj_goal: Set[GroundAtom]) -> _Chain:
        """Returns ground operators in REVERSE order."""
        image_chain: List[Set[GroundAtom]] = []
        operator_chain: List[_GroundSTRIPSOperator] = []
        final_failed_op: Optional[_GroundSTRIPSOperator] = None
        atoms_seq = utils.segment_trajectory_to_atoms_sequence(segmented_traj)
        objects = set(segmented_traj[0].states[0])
        assert traj_goal.issubset(atoms_seq[-1])
        necessary_image = set(traj_goal)
        image_chain = [necessary_image]
        for t in range(len(atoms_seq) - 2, -1, -1):
            segment = segmented_traj[t]
            pnad, var_to_obj = self._find_best_matching_pnad_and_sub(
                segment, objects, pnads)
            # If no match found, terminate.
            if pnad is None:
                break
            assert var_to_obj is not None
            obj_to_var = {v: k for k, v in var_to_obj.items()}
            assert len(var_to_obj) == len(obj_to_var)
            ground_op = pnad.op.ground(
                tuple(var_to_obj[var] for var in pnad.op.parameters))
            next_atoms = utils.apply_operator(ground_op, segment.init_atoms)
            # If we're missing something in the necessary image, terminate.
            if not necessary_image.issubset(next_atoms):
                final_failed_op = ground_op
                break
            # Otherwise, extend the chain.
            operator_chain.append(ground_op)
            # TODO remove copied code.
            # Update necessary_image for this timestep. It no longer
            # needs to include the ground add effects of this PNAD, but
            # must now include its ground preconditions.
            necessary_image = necessary_image.copy()
            necessary_image -= {
                a.ground(var_to_obj)
                for a in pnad.op.add_effects
            }
            necessary_image |= {
                a.ground(var_to_obj)
                for a in pnad.op.preconditions
            }
            image_chain.append(necessary_image)
        return (image_chain, operator_chain, final_failed_op)
