"""Base class for a STRIPS operator learning algorithm."""

import abc
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from predicators import utils
from predicators.planning import task_plan_with_option_plan_constraint
from predicators.settings import CFG
from predicators.structs import PNAD, DummyOption, GroundAtom, LiftedAtom, \
    LowLevelTrajectory, Object, OptionSpec, ParameterizedOption, Predicate, \
    Segment, State, STRIPSOperator, Task, Variable, _GroundSTRIPSOperator


class BaseSTRIPSLearner(abc.ABC):
    """Base class definition."""

    def __init__(self,
                 trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task],
                 predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]],
                 verify_harmlessness: bool,
                 annotations: Optional[List[Any]],
                 clusters: Optional[Dict[str, List[Any]]],
                 stuff_needed: Any,
                 verbose: bool = True) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._verify_harmlessness = verify_harmlessness
        self._verbose = verbose
        self._num_segments = sum(len(t) for t in segmented_trajs)
        self._annotations = annotations
        self._clusters = clusters
        self._stuff_needed = stuff_needed
        assert len(self._trajectories) == len(self._segmented_trajs)

    def learn(self) -> List[PNAD]:
        """The public method for a STRIPS operator learning strategy.

        A wrapper around self._learn() to sanity check that harmlessness
        holds on the training data, and then filter out operators
        without enough data. We check harmlessness first because
        filtering may break it.
        """
        learned_pnads = self._learn()
        if self._verify_harmlessness and not CFG.disable_harmlessness_check:
            logging.info("\nRunning harmlessness check...")
            assert self._check_harmlessness(learned_pnads)
        # Remove pnads by increasing min_data_perc until harmlessness breaks.
        if CFG.enable_harmless_op_pruning:
            assert self._verify_harmlessness
            assert not CFG.disable_harmlessness_check
            # Keeps track of latest set of harmless pnads.
            min_harmless_pnads = learned_pnads
            # Find the percentage of data in each PNAD uses from lowest
            # to highest.
            pnad_perc_data_low_to_high = [
                len(pnad.datastore) / float(self._num_segments)
                for pnad in learned_pnads
            ]
            pnad_perc_data_low_to_high.sort()
        else:
            # If we are not doing harmless operator pruning, return
            # PNADs at current min_perc_data_for_nsrts.
            pnad_perc_data_low_to_high = [CFG.min_perc_data_for_nsrt / 100.0]
        # Iterates over each PNAD in the learned PNADs removing the
        # PNAD that uses the least amount of data.
        for min_perc_data_for_nsrt in pnad_perc_data_low_to_high:
            min_data = max(CFG.min_data_for_nsrt,
                           self._num_segments * min_perc_data_for_nsrt)
            learned_pnads = [
                pnad for pnad in learned_pnads
                if len(pnad.datastore) >= min_data
            ]
            if not CFG.enable_harmless_op_pruning:
                # If we are not doing harmless operator pruning, return
                # PNADs at current min_perc_data_for_nsrts.
                return learned_pnads
            # Runs harmlessness check after we have pruned operators.
            logging.info("\nRunning harmlessness check...")
            if not self._check_harmlessness(learned_pnads):
                break
            # We successfully verified harmlessness than we save this set of
            # pnads and continue reducing min_perc_data_for_nsrts.
            min_harmless_pnads = learned_pnads
        learned_pnads = min_harmless_pnads
        return learned_pnads

    @abc.abstractmethod
    def _learn(self) -> List[PNAD]:
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

    def _check_harmlessness(self, pnads: List[PNAD]) -> bool:
        """Function to check whether the given PNADs holistically preserve
        harmlessness over demonstrations on the training tasks.

        Preserving harmlessness roughly means that the set of operators
        and predicates supports the agent's ability to plan to achieve
        all of the training tasks in the same way as was demonstrated
        (i.e., the predicates and operators don't render any
        demonstrated trajectory impossible).
        """
        ####
        # testing score of operators with predicate score function

        from predicators.predicate_search_score_functions import _ExpectedNodesScoreFunction
        from predicators.nsrt_learning.segmentation import segment_trajectory
        z = self._stuff_needed
        # score_function = _ExpectedNodesScoreFunction(initial_predicates, atom_dataset, candidates, train_tasks, metric_name)
        score_function = _ExpectedNodesScoreFunction(z[0], z[1], z[2], z[3], z[4])

        # pruned_atom_data = utils.prune_ground_atom_dataset(
        #     self._atom_dataset,
        #     candidate_predicates | self._initial_predicates)
        pruned_atom_data = utils.prune_ground_atom_dataset(z[1], z[5] | z[0])
        segmented_trajs = [segment_trajectory(traj) for traj in pruned_atom_data]
        low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
        strips_ops = [pnad.op for pnad in pnads]
        option_specs = [pnad.option_spec for pnad in pnads]
        op_score = score_function.evaluate_with_operators(
            z[5],
            low_level_trajs,
            segmented_trajs,
            strips_ops,
            option_specs
        )
        import pdb; pdb.set_trace()
        ####

        strips_ops = [pnad.op for pnad in pnads]
        option_specs = [pnad.option_spec for pnad in pnads]
        counter = 0
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
                # logging.debug("Harmlessness not preserved for demo!")
                # logging.debug(f"Initial atoms: {atoms_seq[0]}")
                # for t in range(1, len(atoms_seq)):
                #     logging.debug(f"Timestep {t} add effects: "
                #                   f"{atoms_seq[t] - atoms_seq[t-1]}")
                #     logging.debug(f"Timestep {t} del effects: "
                #                   f"{atoms_seq[t-1] - atoms_seq[t]}")
                print("Harmlessness not preserved for demo!")
                print("Number of demo: ", counter)
                print(f"Initial atoms: {atoms_seq[0]}")
                for t in range(1, len(atoms_seq)):
                    print(f"Timestep {t} add effects: "
                                  f"{atoms_seq[t] - atoms_seq[t-1]}")
                    print(f"Timestep {t} del effects: "
                                  f"{atoms_seq[t-1] - atoms_seq[t]}")
                import pdb; pdb.set_trace()
                return False
            counter += 1
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
        option_plan = []
        for seg in seg_traj:
            if seg.has_option():
                option = seg.get_option()
            else:
                option = DummyOption
            option_plan.append((option.parent, option.objects))
        ground_nsrt_plan = task_plan_with_option_plan_constraint(
            objects, self._predicates, strips_ops, option_specs, init_atoms,
            traj_goal, option_plan, atoms_seq)


        print("doing a test")
        # import pdb; pdb.set_trace()
        if ground_nsrt_plan is None:
            import pdb; pdb.set_trace()

            from predicators.planning import task_plan_with_option_plan_constraint2
            ground_nsrt_plan = task_plan_with_option_plan_constraint2(objects, self._predicates, strips_ops, option_specs, init_atoms, traj_goal, option_plan, atoms_seq)

            for k, v in init_state.data.items(): print(f"{k}: {v.tolist()}")

            def check_plan(plan, init_atoms):
                abstract_states = [init_atoms]
                for p in plan:
                    try:
                        p.preconditions.issubset(abstract_states[-1])
                    except:
                        import pdb; pdb.set_trace()
                    new_abstract_state = (abstract_states[-1] | p.add_effects) - p.delete_effects
                    # new_abstract_state = utils.apply_operator(p, abstract_states[-1]) # this does not work properly!
                    abstract_states.append(new_abstract_state)

                return abstract_states

            import pdb; pdb.set_trace()



            Op0_FastenNailWithHammer = [op for op in strips_ops if op.name=="Op0-FastenNailWithHammer"][0]
            Op1_PickWrench = [op for op in strips_ops if op.name=="Op1-PickWrench"][0]
            Op2_PickScrew = [op for op in strips_ops if op.name=="Op2-PickScrew"][0]
            Op3_FastenScrewByHand = [op for op in strips_ops if op.name=="Op3-FastenScrewByHand"][0]
            Op4_PickBolt = [op for op in strips_ops if op.name=="Op4-PickBolt"][0]
            Op5_PickScrewdriver = [op for op in strips_ops if op.name=="Op5-PickScrewdriver"][0]
            Op6_Place = [op for op in strips_ops if op.name=="Op6-Place"][0]
            Op7_Place = [op for op in strips_ops if op.name=="Op7-Place"][0]
            Op8_Place = [op for op in strips_ops if op.name=="Op8-Place"][0]
            Op9_PickNail = [op for op in strips_ops if op.name=="Op9-PickNail"][0]
            Op10_FastenBoltWithWrench = [op for op in strips_ops if op.name=="Op10-FastenBoltWithWrench"][0]
            Op11_Place = [op for op in strips_ops if op.name=="Op11-Place"][0]
            Op12_Place = [op for op in strips_ops if op.name=="Op12-Place"][0]
            Op13_Place = [op for op in strips_ops if op.name=="Op13-Place"][0]
            Op14_PickHammer = [op for op in strips_ops if op.name=="Op14-PickHammer"][0]
            Op15_FastenScrewWithScrewdriver = [op for op in strips_ops if op.name=="Op15-FastenScrewWithScrewdriver"][0]

            objs = list(init_state.data.keys())
            robot = [o for o in objs if o.name == "robby"][0]
            screw0 = [o for o in objs if o.name == "screw0"][0]
            bolt0 = [o for o in objs if o.name == "bolt0"][0]
            contraption0 = [o for o in objs if o.name == "contraption0"][0]
            contraption1 = [o for o in objs if o.name == "contraption1"][0]
            hammer0 = [o for o in objs if o.name == "hammer0"][0]
            hammer1 = [o for o in objs if o.name == "hammer1"][0]
            screwdriver0 = [o for o in objs if o.name == "screwdriver0"][0]
            screwdriver1 = [o for o in objs if o.name == "screwdriver1"][0]
            screwdriver2 = [o for o in objs if o.name == "screwdriver2"][0]
            wrench0 = [o for o in objs if o.name == "wrench0"][0]

            first = Op2_PickScrew.ground((robot, screw0))
            second = Op12_Place.ground((contraption0, robot, screw0))
            third = Op3_FastenScrewByHand.ground((contraption0, robot, screw0))

            first.preconditions == first.preconditions.intersection(init_atoms)
            after_first = (init_atoms | first.add_effects) - first.delete_effects
            assert second.preconditions.issubset(after_first)
            after_second = (after_first | second.add_effects) - second.delete_effects

            assert third.preconditions.issubset(after_second)
            after_third = (after_second | third.add_effects) - third.delete_effects

            # (Pdb) third.preconditions - third.preconditions.intersection(after_second)
            # {Forall[0:screw].[ScrewPlaced(0,1)](contraption0:contraption), NOT-Forall[0:screw].[NOT-ScrewPlaced(0,1)](contraption0:contraption)}
            # why are we learnring this predicate, if it comes out of the intersection?
            # I should look into why discard isn't working as expected -- if so, there could be errosr in other places where that is used.

            plan = [Op2_PickScrew.ground((robot, screw0)), Op12_Place.ground((contraption0, robot, screw0)), Op3_FastenScrewByHand.ground((contraption0, robot, screw0)), Op4_PickBolt.ground((bolt0, robot)), Op13_Place.ground((bolt0, contraption1, robot)), Op1_PickWrench.ground((robot, wrench0)), Op10_FastenBoltWithWrench.ground((bolt0, contraption1, robot, wrench0))]

            # Op9_OpenLid = [op for op in strips_ops if op.name=="Op9-OpenLid"][0]
            # Op1_Pick = [op for op in strips_ops if op.name=="Op1-Pick"][0]
            # Op8_Dry = [op for op in strips_ops if op.name=="Op8-Dry"][0]
            # Op6_Paint = [op for op in strips_ops if op.name=="Op6-Paint"][0]
            # Op7_Place = [op for op in strips_ops if op.name=="Op7-Place"][0]
            # Op2_Pick = [op for op in strips_ops if op.name=="Op2-Pick"][0]
            # Op5_Wash = [op for op in strips_ops if op.name=="Op5-Wash"][0]
            # Op10_Paint = [op for op in strips_ops if op.name=="Op10-Paint"][0]
            # Op0_Place = [op for op in strips_ops if op.name=="Op0-Place"][0]
            # Op3_Place = [op for op in strips_ops if op.name=="Op3-Place"][0]
            # Op4_Place = [op for op in strips_ops if op.name=="Op4-Place"][0]
            #
            # objs = list(init_state.data.keys())
            # robot = [o for o in objs if o.name == "robby"][0]
            # box_lid = [o for o in objs if o.name == "box_lid"][0]
            # obj0 = [o for o in objs if o.name == "obj0"][0]
            # obj1 = [o for o in objs if o.name == "obj1"][0]
            # receptacle_box = [o for o in objs if o.name == "receptacle_box"][0]
            # receptacle_shelf = [o for o in objs if o.name == "receptacle_shelf"][0]
            #
            # first = Op9_OpenLid.ground((box_lid, robot))
            # second = Op1_Pick.ground((obj1, robot))
            # third = Op8_Dry.ground((obj1, robot))
            # fourth = Op6_Paint.ground((obj1, receptacle_box, robot))
            # fifth = Op7_Place.ground((obj1, receptacle_box, robot))
            # sixth = Op2_Pick.ground((obj0, robot))
            # seventh = Op5_Wash.ground((obj0, robot))
            # eighth = Op8_Dry.ground((obj0, robot))
            # ninth = Op10_Paint.ground((obj0, receptacle_shelf, robot))
            # tenth = Op0_Place.ground((obj0, receptacle_shelf, robot))
            #
            # import pdb; pdb.set_trace()
            #
            # first.preconditions == first.preconditions.intersection(init_atoms)
            # after_first = (init_atoms | first.add_effects) - first.delete_effects
            # assert second.preconditions.issubset(after_first)
            # after_second = (after_first | second.add_effects) - second.delete_effects
            # assert third.preconditions.issubset(after_second)
            # after_third = (after_second | third.add_effects) - third.delete_effects
            # assert fourth.preconditions.issubset(after_third)
            # after_fourth = (after_third | fourth.add_effects) - fourth.delete_effects
            # assert fifth.preconditions.issubset(after_fourth)
            # after_fifth = (after_fourth | fifth.add_effects) - fifth.delete_effects
            # assert sixth.preconditions.issubset(after_fifth)
            # after_sixth = (after_fifth | sixth.add_effects) - sixth.delete_effects
            # assert seventh.preconditions.issubset(after_sixth)
            # after_seventh = (after_sixth | seventh.add_effects) - seventh.delete_effects
            # assert eighth.preconditions.issubset(after_seventh)
            # after_eighth = (after_seventh | eighth.add_effects) - eighth.delete_effects
            # assert ninth.preconditions.issubset(after_eighth)
            # after_ninth = (after_eigth | ninth.add_effects) - ninth.delete_effects
            # assert tenth.preconditions.issubset(after_ninth)
            # after_tenth = (after_ninth | ninth.add_effects) - ninth.delete_effects
            # assert traj_goal.issubset(after_tenth)
            #
            # import pdb; pdb.set_trace()
            #
            #
            # ##########################################
            # Op0Pick = [op for op in strips_ops if op.name=="Op0-Pick"][0]
            # Op1PutOnTable = [op for op in strips_ops if op.name=="Op1-PutOnTable"][0]
            # Op2Stack = [op for op in strips_ops if op.name=="Op2-Stack"][0]
            # Op3Pick = [op for op in strips_ops if op.name=="Op3-Pick"][0]
            #
            # objs = list(init_state.data.keys())
            # robot = [o for o in objs if o.name == "robby"][0]
            # block0 = [b for b in objs if b.name == "block0"][0]
            # block1 = [b for b in objs if b.name == "block1"][0]
            # block2 = [b for b in objs if b.name == "block2"][0]
            # # block3 = [b for b in objs if b.name == "block3"][0]
            # # block4 = [b for b in objs if b.name == "block4"][0]
            # # block5 = [b for b in objs if b.name == "block5"][0]
            #
            # first = Op3Pick.ground((block1, block2, robot))
            # first.preconditions == first.preconditions.intersection(init_atoms)
            # after_first = (init_atoms | first.add_effects) - first.delete_effects
            #
            # second = Op1PutOnTable.ground((block2, robot))
            # second.preconditions == second.preconditions.intersection(after_first)
            # after_second = (after_first | second.add_effects) - second.delete_effects
            #
            # third = Op3Pick.ground((block0, block1, robot))
            # third.preconditions == third.preconditions.intersection(after_second)

            # zero = utils.abstract(task.init, preds)
            # nsrt_one = nsrt_list[3].ground((block2, block3, robot))
            # nsrt_one.preconditions.issubset(zero)
            # one = (zero | nsrt_one.add_effects) - nsrt_one.delete_effects
            # nsrt_two = nsrt_list[2].ground((block5, block3, robot))
            # nsrt_two.preconditions.issubset(one)
            # two = (one | nsrt_two.add_effects) - nsrt_two.delete_effects
            # nsrt_three = nsrt_list[3].ground((block1, block2, robot))
            # nsrt_three.preconditions.issubset(two)
            # three = (two | nsrt_three.add_effects) - nsrt_three.delete_effects
            # nsrt_four = nsrt_list[2].ground((block3, block2, robot))
            # nsrt_four.preconditions.issubset(three)
            # four = (three | nsrt_four.add_effects) - nsrt_four.delete_effects
            # nsrt_five = nsrt_list[1].ground((block1, robot))
            # nsrt_five.preconditions.issubset(four)
            # five = (four | nsrt_five.add_effects) - nsrt_five.delete_effects
            # nsrt_six = nsrt_list[2].ground((block0, block1, robot))
            # nsrt_six.preconditions.issubset(five)
            # six = (five | nsrt_six.add_effects) - nsrt_six.delete_effects
            # nsrt_seven = nsrt_list[3].ground((block3, block2, robot))
            # nsrt_seven.preconditions.issubset(six)
            # seven = (six | nsrt_seven.add_effects) - nsrt_seven.delete_effects
            # nsrt_eight = nsrt_list[2].ground((block1, block2, robot))
            # nsrt_eight.preconditions.issubset(seven)
            # eight = (seven | nsrt_eight.add_effects) - nsrt_eight.delete_effects
            # nsrt_nine = nsrt_list[3].ground((block5, block3, robot))
            # nsrt_nine.preconditions.issubset(eight)
            # nine = (eight | nsrt_nine.add_effects) - nsrt_nine.delete_effects
            # nsrt_ten = nsrt_list[2].ground((block2, block3, robot))
            # nsrt_ten.preconditions.issubset(nine)
            # ten = (nine | nsrt_ten.add_effects) - nsrt_ten.delete_effects


            # first = Op3Pick.ground((block1, block2, robot))
            # second = Op1PutOnTable.ground((block2, robot))
            # third  = Op0Pick.ground((block1, robot))
            # fourth = Op2Stack.ground((block0, block1, robot))
            # fifth = Op0Pick.ground((block2, robot))
            # sixth = Op2Stack.ground((block1, block2, robot))

            # first = Op3Pick.ground((block2, block3, robot))
            # second = Op1PutOnTable.ground((block3, robot))
            # third = Op3Pick.ground((block1, block2, robot))
            # fourth = Op2Stack.ground((block3, block2, robot))
            #
            # import pdb; pdb.set_trace()
            #
            # after_first = (init_atoms | first.add_effects) - first.delete_effects
            # # assert second.preconditions == second.preconditions.intersection(after_first)
            # after_second = (after_first | second.add_effects) - second.delete_effects
            # # assert third.preconditions == third.preconditions.intersection(after_second)
            # after_third = (after_second | third.add_effects) - third.delete_effects
            # # assert fourth.preconditions == fourth.preconditions.intersection(after_third)
            # after_fourth = (after_third | fourth.add_effects) - fourth.delete_effects
            #
            # after_fifth = (after_fourth | fifth.add_effects) - fifth.delete_effects
            # after_sixth = (after_fifth | sixth.add_effects) - sixth.delete_effects



        # robot = [o for o in objs if o.name == "robby"][0]
        # block1 = [b for b in objs if b.name == "block1"][0]
        # block2 = [b for b in objs if b.name == "block2"][0]
        # block3 = [b for b in objs if b.name == "block3"][0]
        # import pdb; pdb.set_trace()
        # first = Op0Pick.ground((block2, robot))
        # # second = Op2Stack.ground((block2, block1, robot))
        # second = Op2Stack.ground((block1, block2, robot))
        # third = Op0Pick.ground((block3, robot))
        # fourth = Op2Stack.ground((block2, block3, robot))
        #
        # after_first = (init_atoms | first.add_effects) - first.delete_effects
        # assert second.preconditions == second.preconditions.intersection(after_first)
        # after_second = (after_first | second.add_effects) - second.delete_effects
        # assert third.preconditions == third.preconditions.intersection(after_second)
        # after_third = (after_second | third.add_effects) - third.delete_effects
        # assert fourth.preconditions == fourth.preconditions.intersection(after_third)
        # after_fourth = (after_third | fourth.add_effects) - fourth.delete_effects
        #
        #
        return ground_nsrt_plan is not None

    def _recompute_datastores_from_segments(self, pnads: List[PNAD]) -> None:
        """For the given PNADs, wipe and recompute the datastores.

        Uses a "rationality" heuristic, where for each segment, we
        select, among the ground PNADs covering it, the one whose add
        and delete effects match the segment's most closely (breaking
        ties arbitrarily). At the end of this procedure, each segment is
        guaranteed to be in at most one PNAD's datastore.
        """

        # print("At beginning of _recompute_datastores_from_segments... len pnad datastores: ")
        # for i, p in enumerate(pnads):
        #     print(f"length of datastore in pnad {i}: {len(p.datastore)}")

        for pnad in pnads:
            pnad.datastore = []  # reset all PNAD datastores
        # Note: we want to loop over all segments, NOT just the ones
        # associated with demonstrations.
        for seg_traj in self._segmented_trajs:
            objects = set(seg_traj[0].states[0])
            for segment in seg_traj:
                best_pnad, best_sub = self._find_best_matching_pnad_and_sub(
                    segment, objects, pnads)
                if best_pnad is not None:
                    assert best_sub is not None
                    best_pnad.add_to_datastore((segment, best_sub),
                                               check_effect_equality=False)

        # print("At end of _recompute_datastores_from_segments... len pnad datastores: ")
        # for i, p in enumerate(pnads):
        #     print(f"length of datastore in pnad {i}: {len(p.datastore)}")

    def _find_best_matching_pnad_and_sub(
        self,
        segment: Segment,
        objects: Set[Object],
        pnads: List[PNAD],
        check_only_preconditions: bool = False
    ) -> Tuple[Optional[PNAD], Optional[Dict[Variable, Object]]]:
        """Find the best matching PNAD (if any) given our rationality-based
        score function, and return the PNAD and substitution necessary to
        ground it. If no PNAD from the input list matches the segment, then
        return Nones.

        If check_only_preconditions is True, we must be calling this function
        during spawning of a new PNAD during backchaining. In this case,
        we want to find a grounding whose preconditions are satisfied in
        the segment.init_atoms. Otherwise, we want to find a grounding that
        not only satisfies the above check, but also is such that calling
        utils.apply_operator() from the segment.init_atoms results in a subset
        of the segment's final atoms, and - if the
        segment.necessary_add_effects are not None - that these are satisfied
        by calling utils.apply_operator() from the segment.init_atoms. This
        effectively checks that the grounding can be applied to this segment
        in a harmless way.
        """
        if segment.has_option():
            segment_option = segment.get_option()
        else:
            segment_option = DummyOption
        segment_param_option = segment_option.parent
        segment_option_objs = tuple(segment_option.objects)
        # Loop over all ground operators, looking for the most
        # rational match for this segment.
        best_score = float("inf")
        best_pnad = None
        best_sub = None
        for pnad in pnads:
            param_opt, opt_vars = pnad.option_spec
            if param_opt != segment_param_option:
                continue
            isub = dict(zip(opt_vars, segment_option_objs))
            if segment in pnad.seg_to_keep_effects_sub:
                # If there are any variables only in the keep effects,
                # their mappings should be put into isub, since their
                # grounding is underconstrained by the segment itself.
                keep_eff_sub = pnad.seg_to_keep_effects_sub[segment]
                for var in pnad.op.parameters:
                    if var in keep_eff_sub:
                        assert var not in isub
                        isub[var] = keep_eff_sub[var]
            for ground_op in utils.all_ground_operators_given_partial(
                    pnad.op, objects, isub):
                if len(ground_op.objects) != len(set(ground_op.objects)):
                    continue
                # If the preconditions don't hold in the segment's
                # initial atoms, skip.
                if not ground_op.preconditions.issubset(segment.init_atoms):
                    continue
                next_atoms = utils.apply_operator(ground_op,
                                                  segment.init_atoms)
                if not check_only_preconditions:
                    # If the atoms resulting from apply_operator() don't
                    # all hold in the segment's final atoms, skip.
                    if not next_atoms.issubset(segment.final_atoms):
                        continue
                    # If the segment has a non-None necessary_add_effects,
                    # and the ground operator's add effects don't fit this,
                    # skip.
                    if segment.necessary_add_effects is not None and \
                       not segment.necessary_add_effects.issubset(
                           ground_op.add_effects):
                        continue
                else:
                    # If check_only_preconditions is True, we must be
                    # calling this from spawning during backchaining
                    # with a most-general PNAD that has no add effects
                    # and all other predicates sidelined, and thus this
                    # assertion must hold.
                    assert next_atoms.issubset(segment.final_atoms)
                # This ground PNAD covers this segment. Score it!
                score = self._score_segment_ground_op_match(segment, ground_op)
                if score < best_score:  # we want a closer match
                    best_score = score
                    best_pnad = pnad
                    best_sub = dict(zip(pnad.op.parameters, ground_op.objects))
        return best_pnad, best_sub

    @staticmethod
    def _score_segment_ground_op_match(
            segment: Segment, ground_op: _GroundSTRIPSOperator) -> float:
        """Return a score for how well the given segment matches the given
        ground operator, used in recompute_datastores_from_segments().

        A lower score is a CLOSER match. We use a heuristic to estimate
        the quality of the match, where we check how many ground atoms
        are different between the segment's add/delete effects and the
        operator's add/delete effects. However, we must be careful to
        treat keep effects specially, since they will not appear in
        segment.add_effects. In general, we favor more keep effects
        (hence we subtract len(keep_effects)), since we can only ever
        call this function on ground operators whose preconditions are
        satisfied in segment.init_atoms.
        """
        keep_effects = ground_op.preconditions & ground_op.add_effects
        nonkeep_add_effects = ground_op.add_effects - keep_effects
        return len(segment.add_effects - nonkeep_add_effects) + \
            len(nonkeep_add_effects - segment.add_effects) + \
            len(segment.delete_effects - ground_op.delete_effects) + \
            len(ground_op.delete_effects - segment.delete_effects) - \
            len(keep_effects)

    @staticmethod
    def _induce_preconditions_via_intersection(pnad: PNAD) -> Set[LiftedAtom]:
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

    @staticmethod
    def _compute_pnad_delete_effects(pnad: PNAD) -> None:
        """Update the given PNAD to change the delete effects to ones obtained
        by unioning all lifted images in the datastore.

        IMPORTANT NOTE: We want to do a union here because the most
        general delete effects are the ones that capture _any possible_
        deletion that occurred in a training transition. (This is
        contrast to preconditions, where we want to take an intersection
        over our training transitions.) However, we do not allow
        creating new variables when we create these delete effects.
        Instead, we filter out delete effects that include new
        variables. Therefore, even though it may seem on the surface
        like this procedure will cause all delete effects in the data to
        be modeled accurately, this is not actually true.
        """
        delete_effects = set()
        for segment, var_to_obj in pnad.datastore:
            obj_to_var = {o: v for v, o in var_to_obj.items()}
            atoms = {
                atom
                for atom in segment.delete_effects
                if all(o in obj_to_var for o in atom.objects)
            }
            lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}
            delete_effects |= lifted_atoms
        pnad.op = pnad.op.copy_with(delete_effects=delete_effects)

    @staticmethod
    def _compute_pnad_ignore_effects(pnad: PNAD) -> None:
        """Update the given PNAD to change the ignore effects to ones that
        include every unmodeled add or delete effect seen in the data."""
        # First, strip out any existing ignore effects so that the call
        # to apply_operator() cannot use them, which would defeat the purpose.
        pnad.op = pnad.op.copy_with(ignore_effects=set())
        ignore_effects = set()
        for (segment, var_to_obj) in pnad.datastore:
            objs = tuple(var_to_obj[param] for param in pnad.op.parameters)
            ground_op = pnad.op.ground(objs)
            next_atoms = utils.apply_operator(ground_op, segment.init_atoms)
            # Note that we only induce ignore effects for atoms that are
            # predicted to be in the next_atoms but are not actually there
            # (since the converse doesn't change the soundness of our
            # planning strategy).
            for atom in next_atoms - segment.final_atoms:
                ignore_effects.add(atom.predicate)
        pnad.op = pnad.op.copy_with(ignore_effects=ignore_effects)

    @staticmethod
    def _get_uniquely_named_nec_pnads(
        param_opt_to_nec_pnads: Dict[ParameterizedOption, List[PNAD]]
    ) -> List[PNAD]:
        """Given a dictionary mapping parameterized options to PNADs, return a
        list of PNADs that have unique names and can be used for planning."""
        uniquely_named_nec_pnads: List[PNAD] = []
        for pnad_list in sorted(param_opt_to_nec_pnads.values(), key=str):
            for i, pnad in enumerate(pnad_list):
                new_op = pnad.op.copy_with(name=(pnad.op.name + str(i)))
                new_pnad = PNAD(new_op, list(pnad.datastore), pnad.option_spec)
                new_pnad.poss_keep_effects = pnad.poss_keep_effects
                new_pnad.seg_to_keep_effects_sub = pnad.seg_to_keep_effects_sub
                uniquely_named_nec_pnads.append(new_pnad)
        return uniquely_named_nec_pnads
