"""Methods for learning to sideline predicates in NSRTs."""

import abc
from typing import Any, FrozenSet, Iterator, List, Set, Tuple, Dict

from predicators.src import utils
from predicators.src.planning import task_plan_grounding
from predicators.src.predicate_search_score_functions import \
    _PredictionErrorScoreFunction
from predicators.src.settings import CFG
from predicators.src.structs import GroundAtom, LowLevelTrajectory, \
    OptionSpec, PartialNSRTAndDatastore, Predicate, Segment, STRIPSOperator, \
    Task, _GroundNSRT, ParameterizedOption, Variable
from predicators.src.nsrt_learning.segmentation import segment_trajectory


class SidePredicateLearner(abc.ABC):
    """Base class for a side predicate learning strategy."""

    def __init__(self, initial_pnads: List[PartialNSRTAndDatastore],
                 trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task], predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]]) -> None:
        self._initial_pnads = initial_pnads
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs

    def sideline(self) -> List[PartialNSRTAndDatastore]:
        """The public method for a side predicate learning strategy.

        A simple wrapper around self._sideline() that remembers to call
        self._recompute_datastores_from_segments() afterward.
        """
        sidelined_pnads = self._sideline()
        # Recompute the datastores in the PNADs. We need to do this
        # because when we have side predicates, each transition may be
        # assigned to *multiple* datastores.
        self._recompute_datastores_from_segments(sidelined_pnads)
        return sidelined_pnads

    @abc.abstractmethod
    def _sideline(self) -> List[PartialNSRTAndDatastore]:
        """The key method that a side predicate learning strategy must
        implement.

        Returns a new list of PNADs with sidelining applied as desired.
        Note that self._initial_pnads is not modified.
        """
        raise NotImplementedError("Override me!")

    def _recompute_datastores_from_segments(
            self, sidelined_pnads: List[PartialNSRTAndDatastore]) -> None:
        for pnad in sidelined_pnads:
            pnad.datastore = []  # reset all PNAD datastores
        for seg_traj in self._segmented_trajs:
            objects = set(seg_traj[0].states[0])
            for segment in seg_traj:
                assert segment.has_option()
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
                # Get ground operators given these objects and option objs.
                for pnad in sidelined_pnads:
                    param_opt, opt_vars = pnad.option_spec
                    if param_opt != segment_param_option:
                        continue
                    isub = dict(zip(opt_vars, segment_option_objs))
                    # Consider adding this segment to each datastore.
                    for ground_op in utils.all_ground_operators_given_partial(
                            pnad.op, objects, isub):
                        # Check if preconditions hold.
                        if not ground_op.preconditions.issubset(
                                segment.init_atoms):
                            continue
                        # Check if effects match. Note that we're using the side
                        # predicates semantics here!
                        atoms = utils.apply_operator(ground_op,
                                                     segment.init_atoms)
                        if not atoms.issubset(segment.final_atoms):
                            continue
                        # Skip over segments that have multiple possible
                        # bindings.
                        if (len(set(ground_op.objects)) != len(
                                ground_op.objects)):
                            continue
                        # This segment belongs in this datastore, so add it.
                        sub = dict(zip(pnad.op.parameters, ground_op.objects))
                        pnad.add_to_datastore((segment, sub),
                                              check_effect_equality=False)


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
    def _evaluate(self, state: Tuple[PartialNSRTAndDatastore, ...]) -> float:
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

    def _evaluate(self, state: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        strips_ops = [pnad.op for pnad in state]
        option_specs = [pnad.option_spec for pnad in state]
        score = self._score_func.evaluate_with_operators(
            frozenset(), self._trajectories, self._segmented_trajs, strips_ops,
            option_specs)
        return score


class PreserveSkeletonsHillClimbingSidePredicateLearner(
        HillClimbingSidePredicateLearner):
    """A side predicate learning strategy that does hill climbing with a
    skeleton preservation (harmlessness) score function."""

    def _evaluate(self, state: Tuple[PartialNSRTAndDatastore, ...]) -> float:
        preserves_harmlessness = self._check_harmlessness(state)
        if preserves_harmlessness:
            # If harmlessness is preserved, the score is the number of
            # operators in the state, minus the number of side predicates.
            # This means we prefer fewer operators and more side predicates.
            score = 2 * len(state)
            for pnad in state:
                score -= len(pnad.op.side_predicates)
        else:
            # If harmlessness is not preserved, the score is an arbitrary
            # constant bigger than the total number of operators at the
            # start of the search. This is guaranteed to be worse (higher)
            # than any score that occurs if harmlessness is preserved.
            score = 10 * len(self._initial_pnads)
        return score

    def _check_harmlessness(
            self, state: Tuple[PartialNSRTAndDatastore, ...]) -> bool:
        """Function to check whether the given state in the search preserves
        harmlessness over demonstrations on the training tasks.

        Preserving harmlessness roughly means that the set of operators
        and predicates supports the agent's ability to plan to achieve
        all of the training tasks in the same way as was demonstrated
        (i.e., the predicates and operators don't render any
        demonstrated trajectory impossible).
        """
        strips_ops = [pnad.op for pnad in state]
        option_specs = [pnad.option_spec for pnad in state]
        assert len(self._trajectories) == len(self._segmented_trajs)
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            atoms_seq = utils.segment_trajectory_to_atoms_sequence(seg_traj)
            traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
            demo_preserved = self._check_single_demo_preservation(
                ll_traj, atoms_seq, traj_goal, strips_ops, option_specs)
            if not demo_preserved:
                return False
        return True

    def _check_single_demo_preservation(
            self, ll_traj: LowLevelTrajectory,
            atoms_seq: List[Set[GroundAtom]], traj_goal: Set[GroundAtom],
            strips_ops: List[STRIPSOperator],
            option_specs: List[OptionSpec]) -> bool:
        """Function to check whether a given set of operators and predicates
        preserves a single training trajectory."""
        init_atoms = utils.abstract(ll_traj.states[0], self._predicates)
        objects = set(ll_traj.states[0])
        ground_nsrts, _ = task_plan_grounding(init_atoms, objects, strips_ops,
                                              option_specs)
        heuristic = utils.create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic, init_atoms, traj_goal,
            ground_nsrts, self._predicates, objects)

        def _check_goal(state: Tuple[FrozenSet[GroundAtom], int]) -> bool:
            return traj_goal.issubset(state[0])

        def _get_successor_with_correct_option(
            searchnode_state: Tuple[FrozenSet[GroundAtom], int]
        ) -> Iterator[Tuple[_GroundNSRT, Tuple[FrozenSet[GroundAtom], int],
                            float]]:
            state = searchnode_state[0]
            idx_into_traj = searchnode_state[1]

            if idx_into_traj > len(ll_traj.actions) - 1:
                return

            gt_option = ll_traj.actions[idx_into_traj].get_option()
            expected_next_hl_state = atoms_seq[idx_into_traj + 1]

            for applicable_nsrt in utils.get_applicable_operators(
                    ground_nsrts, state):
                # NOTE: we check that the ParameterizedOptions are equal before
                # attempting to ground because otherwise, we might
                # get a parameter mismatch and trigger an AssertionError
                # during grounding.
                if applicable_nsrt.option != gt_option.parent:
                    continue
                if applicable_nsrt.option_objs != gt_option.objects:
                    continue
                next_hl_state = utils.apply_operator(applicable_nsrt,
                                                     set(state))
                if next_hl_state.issubset(expected_next_hl_state):
                    # The returned cost is uniform because we don't
                    # actually care about finding the shortest path;
                    # just one that matches!
                    yield (applicable_nsrt, (frozenset(next_hl_state),
                                             idx_into_traj + 1), 1.0)

        init_atoms_frozen = frozenset(init_atoms)
        init_searchnode_state = (init_atoms_frozen, 0)
        # NOTE: each state in the below GBFS is a tuple of
        # (current_atoms, idx_into_traj). The idx_into_traj is necessary because
        # we need to check whether the atoms that are true at this particular
        # index into the trajectory is what we would expect given the demo
        # trajectory.
        state_seq, _ = utils.run_gbfs(
            init_searchnode_state, _check_goal,
            _get_successor_with_correct_option,
            lambda searchnode_state: heuristic(searchnode_state[0]))

        return _check_goal(state_seq[-1])


class GeneralToSpecificSidePredicateLearner(SidePredicateLearner):
    def _induce_pnads_from_spec_and_effects(self,
        option_spec_effects_dict: Dict[ParameterizedOption, Dict[str, Any]]
        ) -> List[PartialNSRTAndDatastore]:
        # This is quite hacky and should be refactored.
        # First, set up PNADs where all of the preconditions are True.
        pnads = []
        for param_option, spec_and_effects in option_spec_effects_dict.items():
            name = f"{param_option.name}LearnedOp"
            parameter_set = set(spec_and_effects["option_vars"])
            for atom in spec_and_effects["add"] | spec_and_effects["delete"]:
                parameter_set.update(atom.variables)
            parameters = sorted(parameter_set)
            op = STRIPSOperator(
                name,
                parameters,
                set(),  # preconditions always True
                spec_and_effects["add"],
                spec_and_effects["delete"],
                spec_and_effects["side"])
            option_spec = (param_option, spec_and_effects["option_vars"])
            datastore = []  # will get filled in later
            pnad = PartialNSRTAndDatastore(op, datastore, option_spec)
            pnads.append(pnad)
        # Sort the data into the PNADs.
        self._recompute_datastores_from_segments(pnads)
        # Update the preconditions based on the sorted data.
        # Also: remove any side predicates that are no longer needed.
        # TODO: this is copy and pasted fom strips_learning.py. Refactor.
        for pnad in pnads:
            new_side_predicates = set()
            for i, (segment, var_to_obj) in enumerate(pnad.datastore):
                objects = set(var_to_obj.values())
                obj_to_var = {o: v for v, o in var_to_obj.items()}
                atoms = {
                    atom
                    for atom in segment.init_atoms
                    if all(o in objects for o in atom.objects)
                }
                lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}
                if i == 0:
                    variables = sorted(set(var_to_obj.keys()))
                else:
                    assert variables == sorted(set(var_to_obj.keys()))
                if i == 0:
                    preconditions = lifted_atoms
                else:
                    preconditions &= lifted_atoms
                assert len(set(var_to_obj.values())) == len(var_to_obj)
                # Update the side predicates.
                pnad_ground_add_effects = {a.ground(var_to_obj)
                                           for a in pnad.op.add_effects}
                pnad_ground_delete_effects = {a.ground(var_to_obj)
                                              for a in pnad.op.delete_effects}
                for atom in segment.add_effects - pnad_ground_add_effects:
                    new_side_predicates.add(atom.predicate)
                for atom in segment.delete_effects - pnad_ground_delete_effects:
                    new_side_predicates.add(atom.predicate)
            assert new_side_predicates.issubset(pnad.op.side_predicates)
            pnad.op = STRIPSOperator(
                pnad.op.name,
                pnad.op.parameters,
                preconditions,
                pnad.op.add_effects,
                pnad.op.delete_effects,
                new_side_predicates)
        return pnads


class BackchainingSidePredicateLearner(GeneralToSpecificSidePredicateLearner):
    """Sideline by backchaining, making operators increasingly specific."""

    def _sideline(self) -> List[PartialNSRTAndDatastore]:
        # Start with the most general PNADs, which correspond to all effects
        # sidelined per parameterized option. Note that we will track option
        # vars and effects only, because given these, we can induce the rest.
        param_option_to_spec_and_effects = {}
        for pnad in self._initial_pnads:
            param_option, option_vars = pnad.option_spec
            # Treat all effects in the pnad as side predicates.
            side_predicates = {a.predicate for a in \
                               pnad.op.add_effects | pnad.op.delete_effects}
            # If this is the first pnad with this option spec, init a new one.
            if param_option not in param_option_to_spec_and_effects:
                param_option_to_spec_and_effects[param_option] = {
                    "option_vars": option_vars,
                    "add": set(),
                    # TODO: is this the right way to handle delete effects?
                    # Probably not, because this will never lead to any
                    # delete effects... Not sure what to do.
                    "delete": set(),
                    "side": side_predicates
                }
            # Otherwise, update the effects.
            else:
                param_option_to_spec_and_effects[param_option]["side"] |= \
                    side_predicates
        # Induce the PNADS from the current spec and effects.
        current_pnads = self._induce_pnads_from_spec_and_effects(
            param_option_to_spec_and_effects)
        # Go through the demonstrations from backward to forward, making the
        # PNADs more specific whenever needed.
        assert len(self._trajectories) == len(self._segmented_trajs)
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            atoms_seq = utils.segment_trajectory_to_atoms_sequence(seg_traj)
            # TODO: avoid this assumption.
            assert len(ll_traj.states) == len(atoms_seq)
            options_seq = [a.get_option() for a in ll_traj.actions]
            traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
            assert traj_goal.issubset(atoms_seq[-1])
            # These are the ground atoms that need to be "known".
            current_image = set(traj_goal)
            for t in range(len(atoms_seq)-2, -1, -1):
                option = options_seq[t]
                segment = seg_traj[t]
                # Get the (unique!) PNAD for this option.
                matching_pnads = [p for p in current_pnads
                                  if p.option_spec[0] == option.parent]
                assert len(matching_pnads) == 1
                pnad = matching_pnads[0]
                # By construction, this segment should exist in this PNAD.
                var_to_obj = None
                for pnad_seg, sub in pnad.datastore:
                    if pnad_seg is segment:
                        var_to_obj = sub
                        break
                assert var_to_obj is not None
                # Check if the current image is predicted by the pnad. If not,
                # then we will need to make the pnad effects more specific.
                current_add_effects = current_image - atoms_seq[t]
                pnad_ground_add_effects = {a.ground(var_to_obj)
                                           for a in pnad.op.add_effects}
                if not current_add_effects.issubset(pnad_ground_add_effects):
                    # Need to make the effects more specific.
                    assert len(set(var_to_obj.values())) == len(var_to_obj)
                    obj_to_var = {o: v for v, o in var_to_obj.items()}
                    missing_effects = current_add_effects - \
                        pnad_ground_add_effects
                    # Unclear if the below is necessary in general, maybe.
                    # Check if we need to introduce new parameters.
                    # all_objects = {o for a in missing_effects
                    #                for o in a.objects}
                    # unbound_objects = all_objects - set(obj_to_var)
                    # new_var_idx = 0
                    # for v in var_to_obj:
                    #     assert v.name.startswith("?x")
                    #     new_var_idx = max(new_var_idx, int(v.name[2:])+1)
                    # for obj in sorted(unbound_objects):
                    #     new_var = Variable(f"?x{new_var_idx}", obj.type)
                    #     new_var_idx += 1
                    #     obj_to_var[obj] = new_var
                    #     var_to_obj[new_var] = obj
                    new_add_effects = {a.lift(obj_to_var)
                                       for a in missing_effects}
                    param_option_to_spec_and_effects[option.parent]["add"] |= \
                        new_add_effects
                    # Update the PNAD based on the new effects.
                    # TODO: refactor to only update the PNAD that just changed.
                    # This is horrible.
                    current_pnads = self._induce_pnads_from_spec_and_effects(
                        param_option_to_spec_and_effects)
                    matching_pnads = [p for p in current_pnads
                                      if p.option_spec[0] == option.parent]
                    assert len(matching_pnads) == 1
                    pnad = matching_pnads[0]
                    pnad_ground_add_effects = {a.ground(var_to_obj)
                                               for a in pnad.op.add_effects}
                # Update the current image based on the PNAD preconditions.
                pnad_ground_preconditions = {a.ground(var_to_obj)
                                             for a in pnad.op.preconditions}
                current_image -= pnad_ground_add_effects
                current_image |= pnad_ground_preconditions
        return current_pnads


class IntersectionSidePredicateLearner(GeneralToSpecificSidePredicateLearner):
    """Sideline by simply (1) clustering all transitions by option and then
    (2) taking the intersection over preconditions and effects."""
    
    def _sideline(self) -> List[PartialNSRTAndDatastore]:
        # Step 0. Remove all non-goal atoms from the final state in every 
        # segmented trajectory. This will be leveraged during Step 2.
        # This requires re-assigning self._segmented_trajs (for now)
        # TODO: Re-assigning self._segmented_trajs is a dirty hack...
        # Fix to pass in new trajs without the final action or
        # something like that if/when we actually get serious about this
        # approach.
        goal_only_segmented_trajs: List[List[Segment]] = []
        assert len(self._trajectories) == len(self._segmented_trajs)
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            atoms_seq = utils.segment_trajectory_to_atoms_sequence(seg_traj)
            # TODO: avoid this assumption.
            assert len(ll_traj.states) == len(atoms_seq)
            traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
            assert traj_goal.issubset(atoms_seq[-1])
            atoms_seq[-1] = traj_goal
            seg_traj_only_goal = segment_trajectory((ll_traj, atoms_seq))
            goal_only_segmented_trajs.append(seg_traj_only_goal)
        self._segmented_trajs = goal_only_segmented_trajs
                
        # Step 1. Cluster all transitions by option. To do this, we 
        # create operators whose precondition is just 'True' and all
        # possible effects are sidelined. We can then leverage the
        # _induce_pnads_from_spec_and_effects method, which will
        # compute exactly 1 PNAD per option!

        # TODO: This code is literally copied from Tom's Backchaining
        # implementation above. If/when we actually decide to get
        # serious about these approaches, be sure to clean this up!
        param_option_to_spec_and_effects = {}
        for pnad in self._initial_pnads:
            param_option, option_vars = pnad.option_spec
            # Treat all effects in the pnad as side predicates.
            side_predicates = {a.predicate for a in \
                               pnad.op.add_effects | pnad.op.delete_effects}
            # If this is the first pnad with this option spec, init a new one.
            if param_option not in param_option_to_spec_and_effects:
                param_option_to_spec_and_effects[param_option] = {
                    "option_vars": option_vars,
                    "add": set(),
                    # TODO: is this the right way to handle delete effects?
                    # Probably not, because this will never lead to any
                    # delete effects... Not sure what to do.
                    "delete": set(),
                    "side": side_predicates
                }
            # Otherwise, update the effects.
            else:
                param_option_to_spec_and_effects[param_option]["side"] |= \
                    side_predicates
        # Induce the PNADS from the current spec and effects.
        current_pnads = self._induce_pnads_from_spec_and_effects(
            param_option_to_spec_and_effects)

        # Step 2. Learn both preconditions and effects via intersection
        # Learn the preconditions of the operators in the PNADs via intersection.
        for pnad in current_pnads:
            new_op_add_effects = set()
            new_op_delete_effects = set()
            for i, (segment, var_to_obj) in enumerate(pnad.datastore):
                objects = set(var_to_obj.values())
                obj_to_var = {o: v for v, o in var_to_obj.items()}
                init_atoms = {
                    atom
                    for atom in segment.init_atoms
                    if all(o in objects for o in atom.objects)
                }
                lifted_init_atoms = {atom.lift(obj_to_var) for atom in init_atoms}
                
                # Code to lift up add effects 
                curr_ground_add_effects = {a.ground(var_to_obj) for a in new_op_add_effects}
                if len(segment.add_effects) > 0:
                    missing_add_effects = segment.add_effects - curr_ground_add_effects
                    all_objects = {o for a in missing_add_effects
                                for o in a.objects}
                    unbound_objects = all_objects - set(obj_to_var)
                    new_var_idx = 0
                    for v in var_to_obj:
                        assert v.name.startswith("?x")
                        new_var_idx = max(new_var_idx, int(v.name[2:])+1)
                    for obj in sorted(unbound_objects):
                        new_var = Variable(f"?x{new_var_idx}", obj.type)
                        new_var_idx += 1
                        obj_to_var[obj] = new_var
                        var_to_obj[new_var] = obj
                    lifted_seg_add_effects = {
                        atom.lift(obj_to_var)
                        for atom in segment.add_effects
                    }
                else:
                    # If the add effects are empty, then just set the
                    # lifted_seg_add_effects to be what the current
                    # add effects are so our estimate of the current
                    # add effects doesn't change when we do the intersection.
                    lifted_seg_add_effects = new_op_add_effects

                # Code to lift up delete effects
                # TODO: This is the same as the add_effects code above
                # write a function or otherwise simplify to avoid code duplication!
                curr_ground_delete_effects = {a.ground(var_to_obj) for a in new_op_delete_effects}
                if len(segment.delete_effects) > 0:
                    missing_delete_effects = segment.delete_effects - curr_ground_delete_effects
                    all_objects = {o for a in missing_delete_effects
                                for o in a.objects}
                    unbound_objects = all_objects - set(obj_to_var)
                    new_var_idx = 0
                    for v in var_to_obj:
                        assert v.name.startswith("?x")
                        new_var_idx = max(new_var_idx, int(v.name[2:])+1)
                    for obj in sorted(unbound_objects):
                        new_var = Variable(f"?x{new_var_idx}", obj.type)
                        new_var_idx += 1
                        obj_to_var[obj] = new_var
                        var_to_obj[new_var] = obj
                    lifted_seg_delete_effects = {
                        atom.lift(obj_to_var)
                        for atom in segment.delete_effects
                    }
                else:
                    # If the delete effects are empty, then just set the
                    # lifted_seg_delete_effects to be what the current
                    # delete effects are so our estimate of the current
                    # delete effects doesn't change when we do the
                    # intersection.
                    lifted_seg_delete_effects = new_op_delete_effects

                # if i == 0:
                #     variables = sorted(set(var_to_obj.keys()))
                # else:
                #     assert variables == sorted(set(var_to_obj.keys()))
                
                if i == 0:
                    new_op_preconditions = lifted_init_atoms
                    new_op_add_effects = lifted_seg_add_effects
                    new_op_delete_effects = lifted_seg_delete_effects
                else:
                    new_op_preconditions &= lifted_init_atoms
                    new_op_add_effects &= lifted_seg_add_effects
                    new_op_delete_effects &= lifted_seg_delete_effects

                # if len(new_op_add_effects) < 1:
                #     import ipdb; ipdb.set_trace()
            
            # Replace the operator with one that contains the newly learned
            # preconditions, add effects and delete effects. 
            # We do this because STRIPSOperator objects are frozen, so 
            # their fields cannot be modified. We keep all the side predicates
            # as they were because 
            # TODO: May need to change parameters depending on what the final
            # add and delete effects end up being?
            pnad.op = STRIPSOperator(pnad.op.name, pnad.op.parameters,
                                    new_op_preconditions, new_op_add_effects,
                                    new_op_delete_effects,
                                    pnad.op.side_predicates)
        
        import ipdb; ipdb.set_trace()

