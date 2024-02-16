"""An abstract approach that does planning to solve tasks.

Uses the SeSamE bilevel planning strategy: SEarch-and-SAMple planning,
then Execution.
"""

import abc
import os
import sys
import time
from typing import Any, Callable, List, Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach
from predicators.option_model import _OptionModelBase, create_option_model
from predicators.planning import PlanningFailure, PlanningTimeout, \
    fd_plan_from_sas_file, generate_sas_file_for_fd, sesame_plan, task_plan, \
    task_plan_grounding
from predicators.settings import CFG
from predicators.structs import NSRT, Action, GroundAtom, Metrics, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, _Option


class BilevelPlanningApproach(BaseApproach):
    """Bilevel planning approach."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        if task_planning_heuristic == "default":
            task_planning_heuristic = CFG.sesame_task_planning_heuristic
        if max_skeletons_optimized == -1:
            max_skeletons_optimized = CFG.sesame_max_skeletons_optimized
        self._task_planning_heuristic = task_planning_heuristic
        self._max_skeletons_optimized = max_skeletons_optimized
        self._plan_without_sim = CFG.bilevel_plan_without_sim
        self._option_model = create_option_model(CFG.option_model_name)
        self._num_calls = 0
        self._last_plan: List[_Option] = []  # used if plan WITH sim
        self._last_nsrt_plan: List[_GroundNSRT] = []  # plan WITHOUT sim

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        # ensure random over successive calls
        seed = self._seed + self._num_calls
        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()


        # print("DEMO NUMBER: ", self._num_calls-1)
        # import pdb; pdb.set_trace()
        # def print_state(state):
        # for k, v in task.init.data.items(): print(f"{k}: {v.tolist()}")

        if self._num_calls in [14]:
            import pdb; pdb.set_trace()

            # objs = list(task.init.data.keys())
            # bolt0 = [o for o in objs if o.name == "bolt0"][0]
            # contraption0 = [o for o in objs if o.name == "contraption0"][0]
            # contraption1 = [o for o in objs if o.name == "contraption1"][0]
            # contraption2 = [o for o in objs if o.name == "contraption2"][0]
            # nail0 = [o for o in objs if o.name == "nail0"][0]
            # hammer0 = [o for o in objs if o.name == "hammer0"][0]
            # hammer1 = [o for o in objs if o.name == "hammer1"][0]
            # robot = [o for o in objs if o.name == "robby"][0]
            # screw0 = [o for o in objs if o.name == "screw0"][0]
            # screwdriver0 = [o for o in objs if o.name == "screwdriver0"][0]
            # screwdriver1 = [o for o in objs if o.name == "screwdriver1"][0]
            # screwdriver2 = [o for o in objs if o.name == "screwdriver2"][0]
            # wrench0 = [o for o in objs if o.name == "wrench0"][0]
            #
            # Op0_FastenNailWithHammer = [op for op in nsrts if op.name=="Op0-FastenNailWithHammer"][0]
            # Op1_PickWrench = [op for op in nsrts if op.name=="Op1-PickWrench"][0]
            # Op2_PickScrew = [op for op in nsrts if op.name=="Op2-PickScrew"][0]
            # Op3_FastenScrewByHand = [op for op in nsrts if op.name=="Op3-FastenScrewByHand"][0]
            # Op4_PickBolt = [op for op in nsrts if op.name=="Op4-PickBolt"][0]
            # Op5_PickScrewdriver = [op for op in nsrts if op.name=="Op5-PickScrewdriver"][0]
            # Op6_Place = [op for op in nsrts if op.name=="Op6-Place"][0]
            # Op7_Place = [op for op in nsrts if op.name=="Op7-Place"][0]
            # Op8_Place = [op for op in nsrts if op.name=="Op8-Place"][0]
            # Op9_PickNail = [op for op in nsrts if op.name=="Op9-PickNail"][0]
            # Op10_FastenBoltWithWrench = [op for op in nsrts if op.name=="Op10-FastenBoltWithWrench"][0]
            # Op11_Place = [op for op in nsrts if op.name=="Op11-Place"][0]
            # Op12_Place = [op for op in nsrts if op.name=="Op12-Place"][0]
            # Op13_Place = [op for op in nsrts if op.name=="Op13-Place"][0]
            # Op14_PickHammer = [op for op in nsrts if op.name=="Op14-PickHammer"][0]
            # Op15_FastenScrewWithScrewdriver = [op for op in nsrts if op.name=="Op15-FastenScrewWithScrewdriver"][0]
            #
            # init_atoms = utils.abstract(task.init, preds)
            #
            # plan = [
            #     Op2_PickScrew.ground((robot, screw0)),
            #     Op12_Place.ground((contraption0, robot, screw0)),
            #     Op4_PickBolt.ground((bolt0, robot)),
            #     Op13_Place.ground((bolt0, contraption2, robot)),
            #     Op1_PickWrench.ground((robot, wrench0)),
            #     Op10_FastenBoltWithWrench.ground((bolt0, contraption2, robot, wrench0)),
            #     Op11_Place.ground((robot, wrench0)),
            #     Op3_FastenScrewByHand.ground((contraption0, robot, screw0)),
            #     Op9_PickNail.ground((nail0, robot)),
            #     Op8_Place.ground((contraption2, nail0, robot)),
            #     Op14_PickHammer.ground((hammer0, robot)),
            #     Op0_FastenNailWithHammer.ground((contraption2, hammer0, nail0, robot))
            # ]
            #
            # plan = [Op2_PickScrew.ground((robot, screw0)), Op12_Place.ground((contraption0, robot, screw0)), Op4_PickBolt.ground((bolt0, robot)), Op13_Place.ground((bolt0, contraption2, robot)), Op1_PickWrench.ground((robot, wrench0)), Op10_FastenBoltWithWrench.ground((bolt0, contraption2, robot, wrench0)), Op11_Place.ground((robot, wrench0)), Op3_FastenScrewByHand.ground((contraption0, robot, screw0)), Op9_PickNail.ground((nail0, robot)), Op8_Place.ground((contraption2, nail0, robot)), Op14_PickHammer.ground((hammer0, robot)), Op0_FastenNailWithHammer.ground((contraption2, hammer0, nail0, robot))]
            #
            # def check_plan(plan, init_atoms):
            #     abstract_states = [init_atoms]
            #     for x, p in enumerate(plan):
            #         try:
            #             p.preconditions.issubset(abstract_states[-1])
            #         except:
            #             import pdb; pdb.set_trace()
            #         new_abstract_state = (abstract_states[-1] | p.add_effects) - p.delete_effects
            #         # new_abstract_state = utils.apply_operator(p, abstract_states[-1]) # this does not work properly!
            #         abstract_states.append(new_abstract_state)
            #
            #     return abstract_states
            #
            # import pdb; pdb.set_trace()
            #
            # check_plan(plan, init_atoms)
            ###########################################################

            # first = Op2_PickScrew.ground((robot, screw0))
            # second = Op12_Place.ground((contraption0, robot, screw0))
            # third = Op4_PickBolt.ground((bolt0, robot))
            #
            # assert first.preconditions.issubset(init_atoms)
            # after_first = after_first = (init_atoms | first.add_effects) - first.delete_effects
            # assert second.preconditions.issubset(after_first)
            # after_second = (after_first | second.add_effects) - second.delete_effects
            # assert third.preconditions.issubset(after_second)


        #
        #     objects = sorted(task.init.data.keys())
        #     robot = [o for o in objects if o.name == "robby"][0]
        #     obj0 = [o for o in objects if o.name == "obj0"][0]
        #     obj1 = [o for o in objects if o.name == "obj1"][0]
        #     obj2 = [o for o in objects if o.name == "obj2"][0]
        #     obj3 = [o for o in objects if o.name == "obj3"][0]
        #     box_lid = [o for o in objects if o.name == "box_lid"][0]
        #     box = [o for o in objects if o.name == "receptacle_box"][0]
        #     shelf = [o for o in objects if o.name == "receptacle_shelf"][0]
        #     nsrts_names = [n.name for n in sorted(nsrts)]
        #     nsrts_list = [n for n in sorted(nsrts)]
        #
        #     # do obj3
        #
        #     first = nsrts_list[5].ground((obj3, robot)) # place on table
        #     assert first.preconditions.issubset(utils.abstract(task.init, preds))
        #     after_first = (utils.abstract(task.init, preds) | first.add_effects) - first.delete_effects
        #
        #     second = nsrts_list[10].ground((box_lid, robot)) # open lid
        #     assert second.preconditions.issubset(after_first)
        #     after_second = (after_first | second.add_effects) - second.delete_effects
        #
        #     third = nsrts_list[1].ground((obj3, robot)) # pick from top
        #     assert third.preconditions.issubset(after_second)
        #     after_third = (after_second | third.add_effects) - third.delete_effects
        #
        #     fourth = nsrts_list[7].ground((obj3, box, robot)) # paint to box
        #     assert fourth.preconditions.issubset(after_third)
        #     after_fourth = (after_third | fourth.add_effects) - fourth.delete_effects
        #
        #     fifth = nsrts_list[8].ground((obj3, box, robot)) # place in box
        #     assert fifth.preconditions.issubset(after_fourth)
        #     after_fifth = (after_fourth | fifth.add_effects) - fifth.delete_effects
        #
        #     # do obj0
        #
        #     sixth = nsrts_list[3].ground((obj0, robot)) # pick from side
        #     assert sixth.preconditions.issubset(after_fifth)
        #     after_sixth = (after_fifth | sixth.add_effects) - sixth.delete_effects
        #
        #     seventh = nsrts_list[9].ground((obj0, robot)) # dry
        #     assert seventh.preconditions.issubset(after_sixth)
        #     after_seventh = (after_sixth | seventh.add_effects) - seventh.delete_effects
        #
        #     eighth = nsrts_list[2].ground((obj0, shelf, robot)) # paint to shelf
        #     assert eighth.preconditions.issubset(after_seventh)
        #     after_eighth = (after_seventh | eighth.add_effects) - eighth.delete_effects
        #
        #     ninth = nsrts_list[0].ground((obj0, shelf, robot)) # place in shelf
        #     assert ninth.preconditions.issubset(after_eighth)
        #     after_ninth = (after_eighth | ninth.add_effects) - ninth.delete_effects
        #
        #     # do obj1
        #
        #     tenth = nsrts_list[3].ground((obj1, robot)) # pick from side
        #     assert tenth.preconditions.issubset(after_ninth)
        #     after_tenth = (after_ninth | tenth.add_effects) - tenth.delete_effects
        #
        #     eleventh = nsrts_list[6].ground((obj1, robot)) # wash
        #     assert eleventh.preconditions.issubset(after_tenth)
        #     after_eleventh = (after_tenth | eleventh.add_effects) - eleventh.delete_effects
        #
        #     twelvth = nsrts_list[9].ground((obj1, robot)) # dry
        #     assert twelvth.preconditions.issubset(after_eleventh)
        #     after_twelvth = (after_eleventh | twelvth.add_effects) - twelvth.delete_effects
        #
        #     thirteenth = nsrts_list[2].ground((obj1, shelf, robot)) # paint to shelf
        #     assert thirteenth.preconditions.issubset(after_twelvth)
        #     after_thirteenth = (after_twelvth | thirteenth.add_effects) - thirteenth.delete_effects
        #
        #     fourteenth = nsrts_list[0].ground((obj1, shelf, robot)) # place in shelf
        #     assert fourteenth.preconditions.issubset(after_thirteenth)
        #     after_fourteenth = (after_thirteenth | fourteenth.add_effects) - fourteenth.delete_effects
        #
        #     # do obj2
        #
        #     fifteenth = nsrts_list[3].ground((obj2, robot)) # pick from side
        #     assert fifteenth.preconditions.issubset(after_fourteenth)
        #     after_fifteenth = (after_fourteenth | fifteenth.add_effects) - fifteenth.delete_effects
        #
        #     sixteenth = nsrts_list[6].ground((obj2, robot)) # wash
        #     assert sixteenth.preconditions.issubset(after_fifteenth)
        #     after_sixteenth = (after_fifteenth | sixteenth.add_effects) - sixteenth.delete_effects
        #
        #     seventeenth = nsrts_list[9].ground((obj2, robot)) # dry
        #     assert seventeenth.preconditions.issubset(after_sixteenth)
        #     after_seventeenth = (after_sixteenth | seventeenth.add_effects) - seventeenth.delete_effects
        #
        #     eighteenth = nsrts_list[2].ground((obj2, shelf, robot)) # paint to shelf
        #     assert eighteenth.preconditions.issubset(after_seventeenth)
        #     after_eighteenth = (after_seventeenth | eighteenth.add_effects) - eighteenth.delete_effects
        #
        #     nineteenth = nsrts_list[0].ground((obj2, shelf, robot)) # place in shelf
        #     assert nineteenth.preconditions.issubset(after_eighteenth)
        #     after_nineteenth = (after_eighteenth | nineteenth.add_effects) - nineteenth.delete_effects

            # first = nsrts_list[10].ground((box_lid, robot))
            #
            #
            #
            #
            # # first = nsrts_list[1].ground((obj0, robot)) # Op1-Pick
            # first = nsrts_list[7].ground((obj0, box, robot))
            # # satisfied = first.preconditions.intersection(utils.abstract(task.init, preds))
            # assert first.preconditions.issubset(utils.abstract(task.init, preds))
            # after_first = (utils.abstract(task.init, preds) | first.add_effects) - first.delete_effects
            # second = nsrts_list[10].ground((box_lid, robot))
            # assert second.preconditions.issubset(after_first)
            #
            #
            # third = nsrts_list[8].ground((obj0, box, robot))
            # assert third.preconditions.issubset(after_second)


        #
        #     nsrt_list = list(nsrts)

        # Run task planning only and then greedily sample and execute in the
        # policy.
        if self._plan_without_sim:
            nsrt_plan, _, metrics = self._run_task_plan(task, nsrts, preds, timeout, seed)
            # nsrt_plan, _, metrics = self._run_task_plan(
            #     task, nsrts, preds, timeout, seed)
            self._last_nsrt_plan = nsrt_plan
            policy = utils.nsrt_plan_to_greedy_policy(nsrt_plan, task.goal,
                                                      self._rng)

        # Run full bilevel planning.
        else:
            option_plan, nsrt_plan, metrics = self._run_sesame_plan(task, nsrts, preds, timeout, seed)
            # for n in nsrt_plan:
            #     if "Back" in n.name:
            #         print("Plan with placeback: ", [no.name for no in nsrt_plan])
            # option_plan, nsrt_plan, metrics = self._run_sesame_plan(
            #     task, nsrts, preds, timeout, seed)
            self._last_plan = option_plan
            self._last_nsrt_plan = nsrt_plan
            policy = utils.option_plan_to_policy(option_plan)

        self._save_metrics(metrics, nsrts, preds)

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _run_sesame_plan(
            self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
            timeout: float, seed: int,
            **kwargs: Any) -> Tuple[List[_Option], List[_GroundNSRT], Metrics]:
        """Subclasses may override.

        For example, PG4 inserts an abstract policy into kwargs.
        """
        try:
            option_plan, nsrt_plan, metrics = sesame_plan(
                task,
                self._option_model,
                nsrts,
                preds,
                self._types,
                timeout,
                seed,
                self._task_planning_heuristic,
                self._max_skeletons_optimized,
                max_horizon=CFG.horizon,
                allow_noops=CFG.sesame_allow_noops,
                use_visited_state_set=CFG.sesame_use_visited_state_set,
                **kwargs)
        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        except PlanningTimeout as e:
            raise ApproachTimeout(e.args[0], e.info)

        return option_plan, nsrt_plan, metrics

    def _run_task_plan(
        self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
        timeout: float, seed: int, **kwargs: Any
    ) -> Tuple[List[_GroundNSRT], List[Set[GroundAtom]], Metrics]:

        init_atoms = utils.abstract(task.init, preds)
        goal = task.goal
        objects = set(task.init)

        try:
            start_time = time.perf_counter()

            if CFG.sesame_task_planner == "astar":
                ground_nsrts, reachable_atoms = task_plan_grounding(init_atoms, objects, nsrts)
                ground_nsrts, reachable_atoms = task_plan_grounding(
                    init_atoms, objects, nsrts)
                heuristic = utils.create_task_planning_heuristic(
                    self._task_planning_heuristic, init_atoms, goal,
                    ground_nsrts, preds, objects)
                duration = time.perf_counter() - start_time
                timeout -= duration
                plan, atoms_seq, metrics = next(
                    task_plan(init_atoms,
                              goal,
                              ground_nsrts,
                              reachable_atoms,
                              heuristic,
                              seed,
                              timeout,
                              max_skeletons_optimized=1,
                              use_visited_state_set=True,
                              **kwargs))
            elif "fd" in CFG.sesame_task_planner:  # pragma: no cover
                fd_exec_path = os.environ["FD_EXEC_PATH"]
                exec_str = os.path.join(fd_exec_path, "fast-downward.py")
                timeout_cmd = "gtimeout" if sys.platform == "darwin" \
                    else "timeout"
                # Run Fast Downward followed by cleanup. Capture the output.
                assert "FD_EXEC_PATH" in os.environ, \
                    "Please follow instructions in the docstring of the" +\
                    "_sesame_plan_with_fast_downward method in planning.py"
                if CFG.sesame_task_planner == "fdopt":
                    alias_flag = "--alias seq-opt-lmcut"
                elif CFG.sesame_task_planner == "fdsat":
                    alias_flag = "--alias lama-first"
                else:
                    raise ValueError("Unrecognized sesame_task_planner: "
                                     f"{CFG.sesame_task_planner}")

                sas_file = generate_sas_file_for_fd(task, nsrts, preds,
                                                    self._types, timeout,
                                                    timeout_cmd,
                                                    alias_flag, exec_str,
                                                    list(objects), init_atoms)
                plan, atoms_seq, metrics = fd_plan_from_sas_file(
                    sas_file, timeout_cmd, timeout, exec_str, alias_flag,
                    start_time, list(objects), init_atoms, nsrts, CFG.horizon)
            else:
                raise ValueError("Unrecognized sesame_task_planner: "
                                 f"{CFG.sesame_task_planner}")

        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        except PlanningTimeout as e:
            raise ApproachTimeout(e.args[0], e.info)

        return plan, atoms_seq, metrics

    def reset_metrics(self) -> None:
        super().reset_metrics()
        # Initialize min to inf (max gets initialized to 0 by default).
        self._metrics["min_num_samples"] = float("inf")
        self._metrics["min_num_skeletons_optimized"] = float("inf")

    def _save_metrics(self, metrics: Metrics, nsrts: Set[NSRT],
                      predicates: Set[Predicate]) -> None:
        for metric in [
                "num_samples", "num_skeletons_optimized",
                "num_failures_discovered", "num_nodes_expanded",
                "num_nodes_created", "plan_length"
        ]:
            self._metrics[f"total_{metric}"] += metrics[metric]
        self._metrics["total_num_nsrts"] += len(nsrts)
        self._metrics["total_num_preds"] += len(predicates)
        for metric in [
                "num_samples",
                "num_skeletons_optimized",
        ]:
            self._metrics[f"min_{metric}"] = min(
                metrics[metric], self._metrics[f"min_{metric}"])
            self._metrics[f"max_{metric}"] = max(
                metrics[metric], self._metrics[f"max_{metric}"])

    @abc.abstractmethod
    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        raise NotImplementedError("Override me!")

    def _get_current_predicates(self) -> Set[Predicate]:
        """Get the current set of predicates.

        Defaults to initial predicates.
        """
        return self._initial_predicates

    def get_option_model(self) -> _OptionModelBase:
        """For ONLY an oracle approach, we allow the user to get the current
        option model."""
        assert self.get_name() == "oracle"
        return self._option_model

    def get_last_plan(self) -> List[_Option]:
        """Note that this doesn't fit into the standard API for an Approach,
        since solve() returns a policy, which abstracts away the details of
        whether that policy is actually a plan under the hood."""
        assert self.get_name() == "oracle"
        assert not self._plan_without_sim
        return self._last_plan

    def get_last_nsrt_plan(self) -> List[_GroundNSRT]:
        """Similar to get_last_plan() in that only oracle should use this.

        And this will only be used when bilevel_plan_without_sim is
        True.
        """
        assert self.get_name() == "oracle"
        return self._last_nsrt_plan
