"""Learn operators by searching over sets of PNADs."""

from __future__ import annotations

import abc
import time
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

from predicators import utils
from predicators.nsrt_learning.strips_learning.gen_to_spec_learner import \
    GeneralToSpecificSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import PNAD, GroundAtom, LowLevelTrajectory, \
    ParameterizedOption, Predicate, Segment, Task, _GroundSTRIPSOperator, NSRT, Variable, LiftedAtom
import re


class _PNADSearchOperator(abc.ABC):
    """An operator that proposes successor sets of PNAD sets."""

    def __init__(self, trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task], predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]],
                 learner: PNADSearchSTRIPSLearner,
                 associated_heuristic: _PNADSearchHeuristic) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._learner = learner
        self._associated_heuristic = associated_heuristic

    @abc.abstractmethod
    def get_successors(self,
                       pnads: FrozenSet[PNAD]) -> Iterator[FrozenSet[PNAD]]:
        """Generate zero or more successor PNAD sets."""
        raise NotImplementedError("Override me!")


class _BackChainingPNADSearchOperator(_PNADSearchOperator):
    """An operator that uses backchaining to propose a new PNAD set."""

    def get_successors(self,
                       pnads: FrozenSet[PNAD]) -> Iterator[FrozenSet[PNAD]]:
        init_heuristic_val = self._associated_heuristic(pnads)
        new_heuristic_val = float('inf')
        ret_pnads_list = sorted(pnads)
        uncovered_segment = self._get_first_uncovered_segment(ret_pnads_list)
        if uncovered_segment is not None:
            while uncovered_segment is not None and \
                new_heuristic_val >= init_heuristic_val:
                # We will need to induce an operator to cover this
                # segment, and thus it must have some necessary add effects.
                new_pnad = self._learner.spawn_new_pnad(uncovered_segment)
                ret_pnads_list = self._append_new_pnad_and_keep_effects(
                    new_pnad, ret_pnads_list)
                ret_pnads = frozenset(pnad.copy() for pnad in ret_pnads_list)
                new_heuristic_val = self._associated_heuristic(ret_pnads)
                uncovered_segment = self._get_first_uncovered_segment(
                    ret_pnads_list)

            yield ret_pnads

    def _append_new_pnad_and_keep_effects(
            self, new_pnad: PNAD, current_pnads: List[PNAD]) -> List[PNAD]:
        """Given some newly-created PNAD and a set of existing PNADs, correctly
        repartition data amongst all these PNADs and induce keep effects for
        the newly-created PNAD.

        Return the final set of all PNADs (current, newly-created, and
        ones with keep effects).
        """
        start_time = time.perf_counter()
        new_pnads = current_pnads + [new_pnad]
        # We first repartition data and ensure delete and ignore
        # effects for the newly-created PNAD are correct.
        new_pnads = self._learner.recompute_pnads_from_effects(new_pnads)
        # For certain predicates, the above call can eat up significant
        # time and memory. Thus, we check here if we exceed the preset
        # timeout.
        if time.perf_counter() - start_time > CFG.pnad_search_timeout:
            raise TimeoutError()
        # Ensure that the unnecessary keep effs sub and poss
        # keep effects are both cleared before backchaining. This is
        # important because we will be inducing keep effects after this
        # backchaining.
        for pnad in new_pnads:
            pnad.poss_keep_effects.clear()
            self._learner.clear_unnecessary_keep_effs(pnad)
        # We rerun backchaining to make sure the seg_to_keep_effects_sub
        # is up-to-date.
        self._get_backchaining_results(new_pnads)
        # Now we can induce keep effects for all PNADs.
        all_pnads_with_keep_effs = []
        for pnad in new_pnads:
            new_pnads_with_keep_effs = \
                self._learner.get_pnads_with_keep_effects(pnad)
            all_pnads_with_keep_effs += sorted(new_pnads_with_keep_effs)
        new_pnads += all_pnads_with_keep_effs

        # We recompute pnads again here to delete keep effect operators
        # that are unnecessary.
        new_pnads = self._learner.recompute_pnads_from_effects(
            sorted(new_pnads))
        print(len(new_pnads))
        return new_pnads

    def _get_backchaining_results(
        self, pnads: List[PNAD]
    ) -> Tuple[int, List[Tuple[List[Segment], List[_GroundSTRIPSOperator]]]]:
        backchaining_results = []
        max_chain_len = 0
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if ll_traj.is_demo:
                traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
                chain = self._learner.backchain(seg_traj, pnads, traj_goal)
                max_chain_len = max(max_chain_len, len(chain))
                backchaining_results.append((seg_traj, chain))
        return max_chain_len, backchaining_results

    def _get_first_uncovered_segment(
        self,
        pnads: List[PNAD],
    ) -> Optional[Segment]:
        # Find the first uncovered segment. Do this in a kind of breadth-first
        # backward search over trajectories.
        # Compute all the chains once up front.
        max_chain_len, backchaining_results = self._get_backchaining_results(
            pnads)
        # Now look for an uncovered segment. If one cannot be found, this
        # method will automatically return None.
        for depth in range(max_chain_len + 1):
            for seg_traj, op_chain in backchaining_results:
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
                    return segment
        return None


class _PruningPNADSearchOperator(_PNADSearchOperator):
    """An operator that prunes PNAD sets."""

    def get_successors(self,
                       pnads: FrozenSet[PNAD]) -> Iterator[FrozenSet[PNAD]]:
        sorted_pnad_list = sorted(pnads)
        for pnad_to_remove in sorted_pnad_list:
            pnads_after_removal = [
                pnad.copy() for pnad in sorted_pnad_list
                if pnad != pnad_to_remove
            ]
            recomp_pnads = self._learner.recompute_pnads_from_effects(
                pnads_after_removal)
            yield frozenset(recomp_pnads)


class _PNADSearchHeuristic(abc.ABC):
    """Given a set of PNAD sets, produce a score, with lower better."""

    def __init__(self, trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task], predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]],
                 learner: PNADSearchSTRIPSLearner) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._learner = learner
        # We compute the total number of segments, which is also the
        # maximum number of operators that we will induce (since, in
        # the worst case, we induce a different operator for every
        # segment).
        self._total_num_segments = 0
        for seg_traj in self._segmented_trajs:
            self._total_num_segments += len(seg_traj)

    @abc.abstractmethod
    def __call__(self, curr_pnads: FrozenSet[PNAD]) -> float:
        """Compute the heuristic value for the given PNAD sets."""
        raise NotImplementedError("Override me!")


class _BackChainingHeuristic(_PNADSearchHeuristic):
    """Counts the number of transitions that are not yet covered by some
    operator in the backchaining sense."""

    def __call__(self, curr_pnads: FrozenSet[PNAD]) -> float:
        # Next, run backchaining using these PNADs.
        uncovered_transitions = 0
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if ll_traj.is_demo:
                traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
                chain = self._learner.backchain(seg_traj, sorted(curr_pnads),
                                                traj_goal)
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
        complexity_term = len(curr_pnads)
        return coverage_term + complexity_term


class PNADSearchSTRIPSLearner(GeneralToSpecificSTRIPSLearner):
    """Base class for a effect search STRIPS learner."""

    @classmethod
    def get_name(cls) -> str:
        return "pnad_search"

    def recompute_pnads_from_effects(self, pnads: List[PNAD]) -> List[PNAD]:
        """Given some input PNADs, strips away everything except the add and
        keep effects, then re-partitions data amongst these and uses this to
        recompute these components."""
        # IMPORTANT: We need to make copies of the PNAD objects to avoid
        # modifying the existing PNAD objects.
        new_pnads = []

        # First, reset all PNADs to only maintain their add and
        # keep effects. Ensure they ignore all predicates in the domain.
        for pnad in pnads:
            keep_effects = pnad.op.preconditions & pnad.op.add_effects
            new_pnad_op = pnad.op.copy_with(
                preconditions=keep_effects,
                add_effects=pnad.op.add_effects,
                delete_effects=set(),
                ignore_effects=self._predicates.copy())
            new_pnad = PNAD(new_pnad_op, [], pnad.option_spec)
            new_pnad.poss_keep_effects = pnad.poss_keep_effects
            new_pnad.seg_to_keep_effects_sub = pnad.seg_to_keep_effects_sub
            new_pnads.append(new_pnad)

        # Repartition all data amongst these new PNADs.
        self._recompute_datastores_from_segments(new_pnads)
        # Prune any PNADs with empty datastores.
        new_pnads = [p for p in new_pnads if p.datastore]
        # Add new preconditions.
        for pnad in new_pnads:
            preconditions = self._induce_preconditions_via_intersection(pnad)
            pnad.op = pnad.op.copy_with(preconditions=preconditions)
        # Add delete and ignore effects.
        for pnad in new_pnads:
            self._compute_pnad_delete_effects(pnad)
            self._compute_pnad_ignore_effects(pnad)
        # Fix naming.
        pnad_map: Dict[ParameterizedOption,
                       List[PNAD]] = {p.option_spec[0]: []
                                      for p in new_pnads}
        for p in new_pnads:
            p.op = p.op.copy_with(name=p.option_spec[0].name)
            pnad_map[p.option_spec[0]].append(p)
        new_pnads = self._get_uniquely_named_nec_pnads(pnad_map)
        return new_pnads
    
    
    def parse_nsrt_block(self, block: str) -> PNAD:
        """Parses a single NSRT block into an PNAD object."""
        lines = block.strip().split("\n")
        
        name_match = re.match(r"(\S+):", lines[0])
        name = name_match.group(1) if name_match else ""

        parameters = re.findall(r"\?x\d+:\w+", lines[1])
        
        def extract_effects(label: str) -> Set[str]:
            """Extracts a list of predicates from labeled sections."""
            for line in lines:
                if line.strip().startswith(label):
                    return set(re.findall(r"\w+\(.*?\)", line))
            return set()
        
        preconditions = extract_effects("Preconditions")
        add_effects = extract_effects("Add Effects")
        delete_effects = extract_effects("Delete Effects")
        ignore_effects = extract_effects("Ignore Effects")

        option_spec_match = re.search(r"Option Spec:\s*(.*)", block)
        option_spec = option_spec_match.group(1) if option_spec_match else ""

        objects = set()
        atoms = set()
        option_specs = {}
        for traj in self._segmented_trajs:
            for segment in traj:
                for state in segment.states:
                    for k, v in state.items():
                        objects.add(k)
                atoms |= segment.init_atoms | segment.final_atoms
                option_specs[segment.get_option().parent.name] = segment.get_option().parent
        all_predicates_list = [(atom.predicate.name,atom.predicate) for atom in atoms]
        def get_predicate(name, entities):
            for pred_name, pred in all_predicates_list:
                if pred_name == pred_name and pred.arity == len(entities):
                    valid_types = True
                    for i, ent in enumerate(entities):
                        if ent.type != pred.types[i]:
                            valid_types = False
                    if valid_types:
                        return pred
            raise NotImplementedError
            
        types = {obj.type.name:obj.type for obj in objects}

        def extract_parameters(predicate: str) -> Set[str]:
            parameter_pattern = re.compile(r"\?x\d+:\w+")  # Matches variables like ?x0:obj_type
            matches = parameter_pattern.findall(predicate)
            return matches
        
        parameters = [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in parameters]
        preconditions = set([LiftedAtom(get_predicate(pre.split("(")[0], [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(pre)]), [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(pre)]) for pre in preconditions])
        add_effects = set([LiftedAtom(get_predicate(add.split("(")[0], [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(add)]), [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(add)]) for add in add_effects])
        delete_effects = set([LiftedAtom(get_predicate(dle.split("(")[0], [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(dle)]), [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(dle)]) for dle in delete_effects])
        ignore_effects = set([get_predicate(ige, None) for ige in ignore_effects])
        option_spec = (option_specs[option_spec.split("(")[0]], [])

        nsrt = NSRT(name, parameters, preconditions, add_effects, delete_effects, ignore_effects, option_spec, [], None)
        return PNAD(nsrt.op, [], option_spec)

    def _learn(self) -> List[PNAD]:
        # Set up hill-climbing search over PNAD sets.
        # Create the search operators.
        search_operators = self._create_search_operators()
        # Create the heuristic.
        heuristic = self._create_heuristic()
        # Initialize the search.
        initial_state: FrozenSet[PNAD] = frozenset()

        def get_successors(
            pnads: FrozenSet[PNAD]
        ) -> Iterator[Tuple[Tuple[_PNADSearchOperator, int], FrozenSet[PNAD],
                            float]]:
            for op in search_operators:
                for i, child in enumerate(op.get_successors(pnads)):
                    yield (op, i), child, 1.0  # cost always 1

        # Load initial pnad set
        if CFG.pnad_search_load_initial:
            initial_state = None
            with open("test_saved.NSRTs.txt", "r") as file:
                content = file.read()
            nsrt_strs = ["NSRT-" + nsrt_str for nsrt_str in content.split("NSRT-") if nsrt_str != '']
            pnads = [self.parse_nsrt_block(nsrt_str) for nsrt_str in nsrt_strs]
            self._recompute_datastores_from_segments(pnads)
            initial_state = frozenset(pnads)

        # Run hill-climbing search.
        path, _, _ = utils.run_hill_climbing(initial_state=initial_state,
                                             check_goal=lambda _: False,
                                             get_successors=get_successors,
                                             heuristic=heuristic,
                                             verbose=self._verbose,
                                             timeout=CFG.pnad_search_timeout)

        # Extract the best PNADs set.
        final_pnads = path[-1]
        sorted_final_pnads = sorted(final_pnads)

        # Fix naming.
        pnad_map: Dict[ParameterizedOption, List[PNAD]] = {
            p.option_spec[0]: []
            for p in sorted_final_pnads
        }
        for p in sorted_final_pnads:
            p.op = p.op.copy_with(name=p.option_spec[0].name)
            pnad_map[p.option_spec[0]].append(p)
        ret_pnads = self._get_uniquely_named_nec_pnads(pnad_map)
        return ret_pnads

    def _create_search_operators(self) -> List[_PNADSearchOperator]:
        op_classes = [
            _BackChainingPNADSearchOperator, _PruningPNADSearchOperator
        ]
        if CFG.pnad_search_without_del:
            op_classes.remove(_PruningPNADSearchOperator)
        ops: List[_PNADSearchOperator] = [
            cls(self._trajectories, self._train_tasks,
                self._predicates, self._segmented_trajs, self,
                self._create_heuristic()) for cls in op_classes
        ]
        return ops

    def _create_heuristic(self) -> _PNADSearchHeuristic:
        backchaining_heur = _BackChainingHeuristic(self._trajectories,
                                                   self._train_tasks,
                                                   self._predicates,
                                                   self._segmented_trajs, self)
        return backchaining_heur

    def backchain(self, segmented_traj: List[Segment], pnads: List[PNAD],
                  traj_goal: Set[GroundAtom]) -> List[_GroundSTRIPSOperator]:
        """Returns chain of ground operators in REVERSE order."""
        operator_chain: List[_GroundSTRIPSOperator] = []
        atoms_seq = utils.segment_trajectory_to_atoms_sequence(segmented_traj)
        objects = set(segmented_traj[0].states[0])
        assert traj_goal.issubset(atoms_seq[-1])
        necessary_image = set(traj_goal)
        for t in range(len(atoms_seq) - 2, -1, -1):
            segment = segmented_traj[t]
            segment.necessary_add_effects = necessary_image - atoms_seq[t]
            candidate_pnads = pnads.copy()
            while len(candidate_pnads) >= 0:
                pnad, var_to_obj = self._find_best_matching_pnad_and_sub(
                    segment, objects, candidate_pnads)
                # If no match found, terminate.
                if pnad is None:
                    return operator_chain
                assert var_to_obj is not None
                obj_to_var = {v: k for k, v in var_to_obj.items()}
                assert len(var_to_obj) == len(obj_to_var)
                ground_op = pnad.op.ground(
                    tuple(var_to_obj[var] for var in pnad.op.parameters))
                next_atoms = utils.apply_operator(ground_op,
                                                  segment.init_atoms)
                # Update the PNAD's seg_to_keep_effs_sub dict.
                self._update_pnad_seg_to_keep_effs(pnad, necessary_image,
                                                   ground_op, obj_to_var,
                                                   segment)
                # If we are not missing anything in the necessary image, we can
                # extend the chain.
                if necessary_image.issubset(next_atoms):
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
                    break

                # See if a different pnad matches.
                candidate_pnads.remove(pnad)

        return operator_chain
