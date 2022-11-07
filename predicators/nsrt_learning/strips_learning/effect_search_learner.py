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
                                             Set[LiftedAtom], Set[Predicate]]]]

    @functools.cached_property
    def _hash(self) -> int:
        option_to_hashable_group = {
            o: (tuple(v), frozenset(aa), frozenset(ka), frozenset(p))
            for o in self._param_option_to_groups
            for ((_, v), aa, ka, p) in self._param_option_to_groups[o]
        }
        return hash(tuple(option_to_hashable_group))

    def __hash__(self) -> int:
        return self._hash

    def __iter__(
        self
    ) -> Iterator[Tuple[OptionSpec, Set[LiftedAtom], Set[LiftedAtom],
                        Set[Predicate]]]:
        for o in sorted(self._param_option_to_groups):
            for (spec, add_atoms, keep_atoms,
                 pred) in self._param_option_to_groups[o]:
                yield (spec, add_atoms, keep_atoms, pred)

    def __str__(self) -> str:
        s = ""
        for (spec, add_atoms, keep_atoms, unignore_preds) in self:
            opt = spec[0].name
            opt_args = ", ".join([str(a) for a in spec[1]])
            s += f"\n{opt}({opt_args}): {add_atoms}, {keep_atoms}, {unignore_preds}"
        s += "\n"
        return s

    def add(self, option_spec: OptionSpec, add_effects: Set[LiftedAtom],
            keep_effects: Set[LiftedAtom],
            unignorable_preds: Set[Predicate]) -> _EffectSets:
        """Create a new _EffectSets with this new entry added to existing."""
        param_option = option_spec[0]
        new_param_option_to_groups = {
            p: [(s, set(aa), set(ka), set(pred)) for s, aa, ka, pred in group]
            for p, group in self._param_option_to_groups.items()
        }
        if param_option not in new_param_option_to_groups:
            new_param_option_to_groups[param_option] = []
        new_param_option_to_groups[param_option].append(
            (option_spec, add_effects, keep_effects, unignorable_preds))
        return _EffectSets(new_param_option_to_groups)

    def remove(self, option_spec: OptionSpec, add_effects: Set[LiftedAtom],
               keep_effects: Set[LiftedAtom],
               unignorable_preds: Set[Predicate]) -> _EffectSets:
        """Create a new _EffectSets with this entry removed from existing."""
        param_option = option_spec[0]
        assert param_option in self._param_option_to_groups
        new_param_option_to_groups = {
            p: [(s, set(aa), set(ka), set(pred)) for s, aa, ka, pred in group]
            for p, group in self._param_option_to_groups.items()
        }
        new_param_option_to_groups[param_option].remove(
            (option_spec, add_effects, keep_effects, unignorable_preds))
        return _EffectSets(new_param_option_to_groups)


class _EffectSearchOperator(abc.ABC):
    """An operator that proposes successor sets of effect sets."""

    def __init__(
            self, trajectories: List[LowLevelTrajectory],
            train_tasks: List[Task], predicates: Set[Predicate],
            segmented_trajs: List[List[Segment]],
            effect_sets_to_pnads: Callable[[_EffectSets],
                                           List[PartialNSRTAndDatastore]],
            backchain: Callable[[
                List[Segment], List[PartialNSRTAndDatastore], Set[GroundAtom]
            ], _Chain], associated_heuristic: _EffectSearchHeuristic) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._effect_sets_to_pnads = effect_sets_to_pnads
        self._backchain = backchain
        self._associated_heuristic = associated_heuristic

    @abc.abstractmethod
    def get_successors(self,
                       effect_sets: _EffectSets) -> Iterator[_EffectSets]:
        """Generate zero or more successor effect sets."""
        raise NotImplementedError("Override me!")


class _BackChainingEffectSearchOperator(_EffectSearchOperator):
    """An operator that uses backchaining to propose a new effect set."""

    def get_successors(self,
                       effect_sets: _EffectSets) -> Iterator[_EffectSets]:
        initial_heuristic_val = self._associated_heuristic(effect_sets)

        def _get_uncovered_transition_from_effect_sets(
            curr_effect_sets: _EffectSets
        ) -> Optional[Tuple[ParameterizedOption, Sequence[Object],
                            Set[GroundAtom], Set[GroundAtom], Segment]]:
            pnads = self._effect_sets_to_pnads(curr_effect_sets)
            uncovered_transition = self._get_first_uncovered_transition(pnads)
            return uncovered_transition

        def _get_new_effect_sets_by_backchaining(
                curr_effect_sets: _EffectSets) -> _EffectSets:
            new_effect_sets = curr_effect_sets
            uncovered_transition = _get_uncovered_transition_from_effect_sets(
                curr_effect_sets)
            if uncovered_transition is not None:
                param_option, option_objs, add_effs, keep_effs, segment = \
                        uncovered_transition
                option_spec, lifted_add_effs, lifted_keep_effs, unignorable_preds = \
                    self._create_new_effect_set(param_option, option_objs, \
                        add_effs, keep_effs, segment)
                new_effect_sets = curr_effect_sets.add(option_spec,
                                                       lifted_add_effs,
                                                       lifted_keep_effs,
                                                       unignorable_preds)
                # We should have covered this transition with the new effect sets.
                new_uncovered_transition = _get_uncovered_transition_from_effect_sets(
                    new_effect_sets)

                # new_pnads = self._effect_sets_to_pnads(new_effect_sets)
                # for pnad in new_pnads:
                #     print(pnad)
                # print()
                # print(uncovered_transition)
                # print(new_uncovered_transition)
                # import ipdb; ipdb.set_trace()

                if new_uncovered_transition == uncovered_transition:
                    # If not, there is a keep effect issue!
                    # We want to add keep effects for any atom(s) that:
                    # (1) True in the segment.init_atoms.
                    # (2) Also in the necessary image.
                    # (3) Not in the current add or keep effects.
                    keep_effs |= {
                        a
                        for a in segment.necessary_image
                        if a in segment.init_atoms and a not in (add_effs
                                                                 | keep_effs)
                    }
                    option_spec, lifted_add_effs, lifted_keep_effs, unignorable_preds = \
                    self._create_new_effect_set(param_option, option_objs, \
                        add_effs, keep_effs, segment)
                    new_effect_sets = curr_effect_sets.add(
                        option_spec, lifted_add_effs, lifted_keep_effs,
                        unignorable_preds)
                    new_uncovered_transition = _get_uncovered_transition_from_effect_sets(
                        new_effect_sets)
                    # Now, we can assert that we've covered this datapoint!
                    assert new_uncovered_transition != uncovered_transition

            return new_effect_sets

        new_effect_sets = _get_new_effect_sets_by_backchaining(effect_sets)
        if initial_heuristic_val > 0:
            new_heuristic_val = self._associated_heuristic(new_effect_sets)
            if new_heuristic_val >= initial_heuristic_val:
                # This means there was a keep effect problem with the new add
                # effects we just induced. We need to call backchaining again
                # to fix this.
                
                # new_pnads = self._effect_sets_to_pnads(new_effect_sets)
                # for new_p in new_pnads:
                #     print(new_p)
                # print()
                # import ipdb; ipdb.set_trace()
                
                new_effect_sets = _get_new_effect_sets_by_backchaining(
                    new_effect_sets)
                new_heuristic_val = self._associated_heuristic(new_effect_sets)
                

        new_pnads = self._effect_sets_to_pnads(new_effect_sets)
        for new_p in new_pnads:
            print(new_p)
        print()
        import ipdb; ipdb.set_trace()
        
        yield new_effect_sets

    def _create_new_effect_set(
        self,
        param_option: ParameterizedOption,
        option_objs: Sequence[Object],
        add_effs: Set[GroundAtom],
        keep_effs: Set[GroundAtom],
        segment_to_cover: Segment,
    ) -> Tuple[OptionSpec, Set[LiftedAtom], Set[LiftedAtom], Set[Predicate]]:
        # Create a new effect set.
        all_objs = sorted(
            set(option_objs)
            | {o
               for a in (add_effs | keep_effs) for o in a.objects})
        all_types = [o.type for o in all_objs]
        all_vars = utils.create_new_variables(all_types)
        obj_to_var = dict(zip(all_objs, all_vars))
        option_vars = [obj_to_var[o] for o in option_objs]
        option_spec = (param_option, option_vars)
        lifted_add_effs = {a.lift(obj_to_var) for a in add_effs}
        lifted_keep_effs = {a.lift(obj_to_var) for a in keep_effs}
        # Since this segment is what caused us to create this new operator
        # in the first place, it must have a necessary image that we're
        # trying to achieve.
        assert segment_to_cover.necessary_image is not None
        # In order for this new operator to cover this segment, it cannot
        # ignore predicates that would cause it to violate the necessary
        # image.
        unignorable_preds = {
            a.predicate
            for a in segment_to_cover.necessary_image
            if a not in (add_effs | keep_effs)
        }
        return (option_spec, lifted_add_effs, lifted_keep_effs,
                unignorable_preds)

    def _get_first_uncovered_transition(
        self,
        pnads: List[PartialNSRTAndDatastore],
    ) -> Optional[Tuple[ParameterizedOption, Sequence[Object], Set[GroundAtom],
                        Set[GroundAtom], Segment]]:
        # Find the first uncovered segment. Do this in a kind of breadth-first
        # backward search over trajectories.
        # Compute all the chains once up front.
        backchaining_results = []
        max_chain_len = 0
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if ll_traj.is_demo:
                atoms_seq = utils.segment_trajectory_to_atoms_sequence(
                    seg_traj)
                traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
                chain = self._backchain(seg_traj, pnads, traj_goal)
                max_chain_len = max(max_chain_len, len(chain[1]))
                backchaining_results.append(
                    (seg_traj, atoms_seq, traj_goal, chain))
        # Now look for an uncovered segment. If one cannot be found, this
        # method will automatically return None.
        for depth in range(max_chain_len + 1):
            for seg_traj, atoms_seq, traj_goal, chain in backchaining_results:
                image_chain, op_chain, last_used_op = chain
                if not (len(op_chain) > depth
                        or len(op_chain) == len(seg_traj)):
                    # We found an uncovered transition: we now need to return
                    # the information necessary to induce a new operator to
                    # cover it.
                    # The timestep of the uncovered transition is the number of
                    # segments - 1 - (numer of actions in our backchained plan)
                    t = (len(seg_traj) - 1) - len(op_chain)
                    assert t >= 0
                    segment = seg_traj[t]
                    option = segment.get_option()
                    necessary_image = image_chain[-1]
                    # Necessary add effects are everything true in the
                    # necessary image that was not true after calling the
                    # operator from atoms_seq[t].
                    necessary_add_effects = necessary_image - atoms_seq[t]
                    necessary_keep_effects = set()
                    if last_used_op is not None and \
                        last_used_op.add_effects == necessary_add_effects:
                        # In this case, there exists some PNAD such that:
                        # (1) the preconditions hold in the pre-image state of
                        # the segment.
                        # (2) the effects yield a state that is a subset of the
                        # post-image state.
                        # (3) the PNAD's add effects already capture all the
                        # necessary add effects.
                        # This means that the only reason this PNAD failed to
                        # capture the necessary image is due to some issue
                        # with delete/ignore effects.
                        pred_next_atoms = utils.apply_operator(
                            last_used_op, atoms_seq[t])
                        missing_effects = (necessary_image - pred_next_atoms)
                        # These are just the missing effects + existing add
                        # effects.
                        necessary_add_effects = missing_effects | \
                            last_used_op.add_effects
                        # These are the missing effects that were ignore
                        # effects of the last_used_op.
                        necessary_keep_effects = {
                            a
                            for a in missing_effects if a in atoms_seq[t] and (
                                a.predicate in last_used_op.ignore_effects
                                or a in last_used_op.delete_effects)
                        }

                    return (option.parent, option.objects,
                            necessary_add_effects, necessary_keep_effects,
                            segment)
        return None


class _PruningEffectSearchOperator(_EffectSearchOperator):
    """An operator that prunes effect sets."""

    def get_successors(self,
                       effect_sets: _EffectSets) -> Iterator[_EffectSets]:
        for (spec, add_effects, keep_effects,
             unignorable_preds) in effect_sets:
            yield effect_sets.remove(spec, add_effects, keep_effects,
                                     unignorable_preds)


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
        # We compute the total number of segments, which is also the
        # maximum number of operators that we will induce (since, in
        # the worst case, we induce a different operator for every
        # segment).
        self._total_num_segments = 0
        for seg_traj in self._segmented_trajs:
            self._total_num_segments += len(seg_traj)

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
            if ll_traj.is_demo:
                traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
                _, chain, _ = self._backchain(seg_traj, pnads, traj_goal)
                assert len(chain) <= len(seg_traj)
                uncovered_transitions += len(seg_traj) - len(chain)
        # Our objective is such that covering more data is *always*
        # more important than creating a less complex set of operators.
        # Thus, we multiply the coverage term by the maximum number of
        # possible operators, so it will always be beneficial to
        # cover more data over deleting operators to make a less complex
        # hypothesis.
        coverage_term = uncovered_transitions * self._total_num_segments
        # NOTE: for now, we measure complexity by simply counting the number
        # of learned PNADs. We could come up with more intricate and
        # accurate measures that also take into account the add effects,
        # arity, etc. (though this might involve changing the weighting
        # of the coverage term).
        complexity_term = len(pnads)
        return coverage_term + complexity_term


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
        path, _, costs = utils.run_hill_climbing(initial_state=initial_state,
                                             check_goal=lambda _: False,
                                             get_successors=get_successors,
                                             heuristic=heuristic)

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
                self._segmented_trajs,
                self._effect_sets_to_pnads, self._backchain,
                self._create_heuristic()) for cls in op_classes
        ]
        return ops

    def _create_heuristic(self) -> _EffectSearchHeuristic:
        backchaining_heur = _BackChainingHeuristic(
            self._trajectories, self._train_tasks, self._predicates,
            self._segmented_trajs, self._effect_sets_to_pnads, self._backchain)
        return backchaining_heur

    def _create_initial_effect_sets(self) -> _EffectSets:
        param_option_to_groups: Dict[ParameterizedOption,
                                     List[Tuple[OptionSpec, Set[LiftedAtom],
                                                Set[LiftedAtom]]]] = {}
        return _EffectSets(param_option_to_groups)

    @functools.lru_cache(maxsize=None)
    def _effect_sets_to_pnads(
            self, effect_sets: _EffectSets) -> List[PartialNSRTAndDatastore]:
        # Start with the add effects and option specs.
        pnads = []
        for (option_spec, add_effects, keep_effects,
             unignorable_preds) in effect_sets:
            parameterized_option, op_vars = option_spec
            add_effect_vars = {v for a in add_effects for v in a.variables}
            keep_effect_vars = {v for a in keep_effects for v in a.variables}
            effect_vars = add_effect_vars | keep_effect_vars
            parameters = sorted(set(op_vars) | effect_vars)
            # Add all ignore effects (aside from unignorable predicates from
            # _EffectSet) initially so that precondition learning
            # works. Be sure that the keep effects are part of both the
            # preconditions and add effects.
            ignore_effects = self._predicates.copy()
            ignore_effects = ignore_effects - unignorable_preds
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
        """Returns chain of ground operators in REVERSE order."""
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
            segment.necessary_image = necessary_image
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
