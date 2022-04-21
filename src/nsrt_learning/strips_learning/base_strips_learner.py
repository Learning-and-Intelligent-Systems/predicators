"""Base class for a STRIPS operator learning algorithm."""

import abc
from typing import FrozenSet, Iterator, List, Set, Tuple

from predicators.src import utils
from predicators.src.planning import task_plan_grounding
from predicators.src.settings import CFG
from predicators.src.structs import DummyOption, GroundAtom, LiftedAtom, \
    LowLevelTrajectory, OptionSpec, PartialNSRTAndDatastore, Predicate, \
    Segment, State, STRIPSOperator, Task, _GroundNSRT


class BaseSTRIPSLearner(abc.ABC):
    """Base class definition."""

    def __init__(
        self,
        trajectories: List[LowLevelTrajectory],
        train_tasks: List[Task],
        predicates: Set[Predicate],
        segmented_trajs: List[List[Segment]],
        verify_harmlessness: bool = False,
        verbose: bool = True,
    ) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._verify_harmlessness = verify_harmlessness
        self._verbose = verbose
        assert len(self._trajectories) == len(self._segmented_trajs)

    def learn(self) -> List[PartialNSRTAndDatastore]:
        """The public method for a STRIPS operator learning strategy.

        A wrapper around self._learn() to sanity check that harmlessness
        holds on the training data, and then filter out operators
        without enough data. We check harmlessness first because
        filtering may break it.
        """
        learned_pnads = self._learn()
        if self._verify_harmlessness and not CFG.disable_harmlessness_check:
            assert self._check_harmlessness(learned_pnads)
        learned_pnads = [
            pnad for pnad in learned_pnads
            if len(pnad.datastore) >= CFG.min_data_for_nsrt
        ]
        return learned_pnads

    @abc.abstractmethod
    def _learn(self) -> List[PartialNSRTAndDatastore]:
        """The key method that a STRIPS operator learning strategy must
        implement.

        Returns a new list of PNADs learned from the data, with op
        (STRIPSOperator), datastore, and option_spec fields filled in
        (but not sampler).
        """
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this STRIPS learner, used as the
        strips_learner setting in settings.py."""
        raise NotImplementedError("Override me!")

    def _check_harmlessness(self,
                            pnads: List[PartialNSRTAndDatastore]) -> bool:
        """Function to check whether the given PNADs holistically preserve
        harmlessness over demonstrations on the training tasks.

        Preserving harmlessness roughly means that the set of operators
        and predicates supports the agent's ability to plan to achieve
        all of the training tasks in the same way as was demonstrated
        (i.e., the predicates and operators don't render any
        demonstrated trajectory impossible).
        """
        strips_ops = [pnad.op for pnad in pnads]
        option_specs = [pnad.option_spec for pnad in pnads]
        for ll_traj, seg_traj in zip(self._trajectories,
                                     self._segmented_trajs):
            if not ll_traj.is_demo:
                continue
            atoms_seq = utils.segment_trajectory_to_atoms_sequence(seg_traj)
            task = self._train_tasks[ll_traj.train_task_idx]
            traj_goal = task.goal
            if not traj_goal.issubset(atoms_seq[-1]):
                # In this case, the goal predicates are not correct (e.g.,
                # we are learning them), so we skip this demonstration.
                continue
            demo_preserved = self._check_single_demo_preservation(
                seg_traj, ll_traj.states[0], atoms_seq, traj_goal, strips_ops,
                option_specs)
            if not demo_preserved:
                # Useful debug point.
                # print("Harmlessness not preserved for demo!")
                # print("Initial atoms:", atoms_seq[0])
                # for t in range(1, len(atoms_seq)):
                #     print(f"Timestep {t} add effects:",
                #           atoms_seq[t] - atoms_seq[t-1])
                #     print(f"Timestep {t} del effects:",
                #           atoms_seq[t-1] - atoms_seq[t])
                return False
        return True

    def _check_single_demo_preservation(
            self, seg_traj: List[Segment], init_state: State,
            atoms_seq: List[Set[GroundAtom]], traj_goal: Set[GroundAtom],
            strips_ops: List[STRIPSOperator],
            option_specs: List[OptionSpec]) -> bool:
        """Function to check whether a given set of operators preserves a
        single training trajectory."""
        init_atoms = utils.abstract(init_state, self._predicates)
        objects = set(init_state)
        options = []
        for seg in seg_traj:
            if seg.has_option():
                options.append(seg.get_option())
            else:
                options.append(DummyOption)
        ground_nsrts, _ = task_plan_grounding(init_atoms,
                                              objects,
                                              strips_ops,
                                              option_specs,
                                              allow_noops=True)
        heuristic = utils.create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic, init_atoms, traj_goal,
            ground_nsrts, self._predicates, objects)

        def _check_goal(
                searchnode_state: Tuple[FrozenSet[GroundAtom], int]) -> bool:
            return traj_goal.issubset(searchnode_state[0])

        def _get_successor_with_correct_option(
            searchnode_state: Tuple[FrozenSet[GroundAtom], int]
        ) -> Iterator[Tuple[_GroundNSRT, Tuple[FrozenSet[GroundAtom], int],
                            float]]:
            atoms = searchnode_state[0]
            idx_into_traj = searchnode_state[1]

            if idx_into_traj > len(options) - 1:
                return

            gt_option = options[idx_into_traj]
            expected_next_atoms = atoms_seq[idx_into_traj + 1]

            for applicable_nsrt in utils.get_applicable_operators(
                    ground_nsrts, atoms):
                # NOTE: we check that the ParameterizedOptions are equal before
                # attempting to ground because otherwise, we might
                # get a parameter mismatch and trigger an AssertionError
                # during grounding.
                if applicable_nsrt.option != gt_option.parent:
                    continue
                if applicable_nsrt.option_objs != gt_option.objects:
                    continue
                next_atoms = utils.apply_operator(applicable_nsrt, set(atoms))
                if next_atoms.issubset(expected_next_atoms):
                    # The returned cost is uniform because we don't
                    # actually care about finding the shortest path;
                    # just one that matches!
                    yield (applicable_nsrt, (frozenset(next_atoms),
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

    def _recompute_datastores_from_segments(
            self,
            pnads: List[PartialNSRTAndDatastore],
            semantics: str = "apply_operator") -> None:
        """For the given PNADs, wipe and recompute the datastores.

        If semantics is "apply_operator", then a segment is included in
        a datastore if, for some ground PNAD, the preconditions are
        satisfied and apply_operator() results in an abstract next state
        that is a subset of the segment's final_atoms. If semantics is
        "add_effects", then rather than using apply_operator(), we
        simply check that the add effects are a subset of the segment's
        add effects.
        """
        assert semantics in ("apply_operator", "add_effects")
        for pnad in pnads:
            pnad.datastore = []  # reset all PNAD datastores
        # Note: we want to loop over all segments, NOT just the ones
        # associated with demonstrations.
        for seg_traj in self._segmented_trajs:
            objects = set(seg_traj[0].states[0])
            for segment in seg_traj:
                if segment.has_option():
                    segment_option = segment.get_option()
                else:
                    segment_option = DummyOption
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
                # Get ground operators given these objects and option objs.
                for pnad in pnads:
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
                        # Check if effects match.
                        if semantics == "apply_operator":
                            atoms = utils.apply_operator(
                                ground_op, segment.init_atoms)
                            if not atoms.issubset(segment.final_atoms):
                                continue
                        elif semantics == "add_effects":
                            if not ground_op.add_effects.issubset(
                                    segment.add_effects):
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

    @staticmethod
    def _induce_preconditions_via_intersection(
            pnad: PartialNSRTAndDatastore) -> Set[LiftedAtom]:
        """Given a PNAD with a nonempty datastore, compute the preconditions
        for the PNAD's operator by intersecting all lifted preimages."""
        assert len(pnad.datastore) > 0
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
                preconditions = lifted_atoms
            else:
                preconditions &= lifted_atoms
        return preconditions
