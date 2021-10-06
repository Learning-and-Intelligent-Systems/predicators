"""An abstract approach that does TAMP to solve tasks. Uses the SeSamE
planning strategy: SEarch-and-SAMple planning, then Execution.
"""

from __future__ import annotations
import abc
import heapq as hq
import time
from typing import Collection, Callable, List, Set, Optional, Tuple, Dict, \
    Any
from dataclasses import dataclass, field
import numpy as np
from predicators.src.approaches import BaseApproach, ApproachFailure, \
    ApproachTimeout
from predicators.src.structs import State, Task, Operator, Predicate, \
    GroundAtom, _GroundOperator, DefaultOption, DefaultState, _Option, Action, \
    PyperplanFacts
from predicators.src import utils
from predicators.src.settings import CFG


class TAMPApproach(BaseApproach):
    """TAMP approach.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._num_calls = 0

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        seed = self._seed+self._num_calls  # ensure random over successive calls
        plan = TAMPApproach.sesame_plan(task, self._simulator,
                                        self._get_current_operators(),
                                        self._initial_predicates, timeout, seed)
        def _policy(_: State) -> Action:
            if not plan:
                raise ApproachFailure("Finished executing plan!")
            return plan.pop(0)
        return _policy

    @abc.abstractmethod
    def _get_current_operators(self) -> Set[Operator]:
        """Get the current set of operators.
        """
        raise NotImplementedError("Override me!")

    @staticmethod
    def sesame_plan(task: Task,
                    simulator: Callable[[State, Action], State],
                    current_operators: Set[Operator],
                    initial_predicates: Set[Predicate],
                    timeout: int, seed: int,
                    check_dr_reachable: bool = True) -> List[Action]:
        """Run TAMP. Return a sequence of low-level actions.
        Uses the SeSamE strategy: SEarch-and-SAMple planning, then Execution.
        """
        op_preds, _ = utils.extract_preds_and_types(current_operators)
        # Ensure that initial predicates are always included.
        predicates = initial_predicates | set(op_preds.values())
        atoms = utils.abstract(task.init, predicates)
        objects = list(task.init)
        ground_operators = []
        for op in current_operators:
            for ground_op in utils.all_ground_operators(op, objects):
                ground_operators.append(ground_op)
        ground_operators = utils.filter_static_operators(
            ground_operators, atoms)
        if check_dr_reachable and \
           not utils.is_dr_reachable(ground_operators, atoms, task.goal):
            raise ApproachFailure(f"Goal {task.goal} not dr-reachable")
        plan = TAMPApproach._run_search(
            task, simulator, ground_operators, atoms, predicates, timeout, seed)
        return plan

    @staticmethod
    def _run_search(task: Task,
                    simulator: Callable[[State, Action], State],
                    ground_operators: List[_GroundOperator],
                    init_atoms: Collection[GroundAtom],
                    predicates: Set[Predicate],
                    timeout: int, seed: int) -> List[Action]:
        """A* search over skeletons (sequences of ground operators).
        """
        start_time = time.time()
        queue: List[Tuple[float, float, Node]] = []
        root_node = Node(atoms=init_atoms, skeleton=[],
                         atoms_sequence=[init_atoms], parent=None)
        rng_prio = np.random.default_rng(seed)
        rng_sampler = np.random.default_rng(seed)
        # Set up stuff for pyperplan heuristic.
        relaxed_operators = frozenset({utils.RelaxedOperator(
            op.name, utils.atoms_to_tuples(op.preconditions),
            utils.atoms_to_tuples(op.add_effects)) for op in ground_operators})
        heuristic_cache: Dict[PyperplanFacts, float] = {}
        heuristic: Callable[[PyperplanFacts], float] = utils.HAddHeuristic(
            utils.atoms_to_tuples(init_atoms),
            utils.atoms_to_tuples(task.goal), relaxed_operators)
        heuristic_cache[root_node.pyperplan_facts] = heuristic(
            root_node.pyperplan_facts)
        hq.heappush(queue, (heuristic_cache[root_node.pyperplan_facts],
                            rng_prio.uniform(),
                            root_node))
        # Start search.
        num_skeletons = 0
        while queue and (time.time()-start_time < timeout):
            if num_skeletons == CFG.max_skeletons:
                raise ApproachFailure("Planning reached max_skeletons!")
            _, _, node = hq.heappop(queue)
            # Good debug point #1: print node.skeleton here to see what
            # the high-level search is doing.
            if task.goal.issubset(node.atoms):
                # If this skeleton satisfies the goal, run low-level search.
                assert node.atoms == node.atoms_sequence[-1]
                plan = TAMPApproach._run_low_level_search(
                    task, simulator, node.skeleton, node.atoms_sequence,
                    rng_sampler, predicates, start_time, timeout)
                if plan is not None:
                    print("Planning succeeded! Found plan of length "
                          f"{len(plan)}")
                    return plan
                num_skeletons += 1
            else:
                # Generate successors.
                for operator in utils.get_applicable_operators(
                        ground_operators, node.atoms):
                    child_atoms = utils.apply_operator(
                        operator, set(node.atoms))
                    child_node = Node(
                        atoms=child_atoms,
                        skeleton=node.skeleton+[operator],
                        atoms_sequence=node.atoms_sequence+[child_atoms],
                        parent=node)
                    if child_node.pyperplan_facts not in heuristic_cache:
                        heuristic_cache[child_node.pyperplan_facts] = heuristic(
                            child_node.pyperplan_facts)
                    # priority is g [plan length] plus h [heuristic]
                    priority = (len(child_node.skeleton)+
                                heuristic_cache[child_node.pyperplan_facts])
                    hq.heappush(queue, (priority,
                                        rng_prio.uniform(),
                                        child_node))
        if not queue:
            raise ApproachFailure("Planning ran out of skeletons!")
        assert time.time()-start_time > timeout
        raise ApproachTimeout("Planning timed out in skeleton search!")

    @staticmethod
    def _run_low_level_search(
            task: Task,
            simulator: Callable[[State, Action], State],
            skeleton: List[_GroundOperator],
            atoms_sequence: List[Collection[GroundAtom]],
            rng_sampler: np.random.Generator,
            predicates: Set[Predicate],
            start_time: float,
            timeout: int) -> Optional[List[Action]]:
        """Backtracking search over continuous values.
        """
        cur_idx = 0
        num_tries = [0 for _ in skeleton]
        options: List[_Option] = [DefaultOption for _ in skeleton]
        plan: List[List[Action]] = [[] for _ in skeleton]  # unflattened
        traj: List[State] = [task.init]+[DefaultState for _ in skeleton]
        while cur_idx < len(skeleton):
            if time.time()-start_time > timeout:
                raise ApproachTimeout("Planning timed out in backtracking!")
            assert num_tries[cur_idx] < CFG.max_samples_per_step
            # Good debug point #2: if you have a skeleton that you think is
            # reasonable, but sampling isn't working, print num_tries here to
            # see at what step the backtracking search is getting stuck.
            num_tries[cur_idx] += 1
            state = traj[cur_idx]
            operator = skeleton[cur_idx]
            # Ground the operator's ParameterizedOption using the
            # operator's sampler on this state.
            option = operator.option.ground(
                operator.sampler(state, rng_sampler))
            options[cur_idx] = option
            option_traj_states, option_traj_acts = utils.option_to_trajectory(
                state, simulator, option,
                max_num_steps=CFG.max_num_steps_option_rollout)
            traj[cur_idx+1] = option_traj_states[-1]  # ignore previous states
            plan[cur_idx] = option_traj_acts
            cur_idx += 1
            # Check atoms_sequence constraint. Backtrack if failed.
            assert len(traj) == len(atoms_sequence)
            atoms = utils.abstract(traj[cur_idx], predicates)
            if atoms == atoms_sequence[cur_idx]:
                if cur_idx == len(skeleton):  # success!
                    return [act for acts in plan for act in acts]  # flatten
            else:
                # Do backtracking.
                cur_idx -= 1
                while num_tries[cur_idx] == CFG.max_samples_per_step:
                    num_tries[cur_idx] = 0
                    options[cur_idx] = DefaultOption
                    plan[cur_idx] = []
                    traj[cur_idx+1] = DefaultState
                    cur_idx -= 1
                    if cur_idx < 0:
                        return None  # backtracking exhausted
        # Should only get here if the skeleton was empty
        assert not skeleton
        return []


@dataclass(repr=False, eq=False)
class Node:
    """A node for the search over skeletons.
    """
    atoms: Collection[GroundAtom]
    skeleton: List[_GroundOperator]
    atoms_sequence: List[Collection[GroundAtom]]  # expected state sequence
    parent: Optional[Node]
    pyperplan_facts: PyperplanFacts = field(
        init=False, default_factory=frozenset)

    def __post_init__(self) -> None:
        self.pyperplan_facts = utils.atoms_to_tuples(self.atoms)
