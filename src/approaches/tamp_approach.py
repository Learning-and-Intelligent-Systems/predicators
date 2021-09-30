"""An abstract approach that does TAMP to solve tasks.
"""

from __future__ import annotations
import abc
import heapq as hq
import time
from typing import Collection, Callable, List, Sequence, Set, Optional
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from predicators.src.approaches import BaseApproach, ApproachFailure, \
    ApproachTimeout
from predicators.src.structs import State, Task, Operator, Predicate, \
    ParameterizedOption, GroundAtom, _GroundOperator
from predicators.src import utils

Array = NDArray[np.float32]


class TAMPApproach(BaseApproach):
    """TAMP approach.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_calls = 0

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Array]:
        self._num_calls += 1
        seed = self._seed+self._num_calls  # ensure random over successive calls
        plan = TAMPApproach._plan(task, self._simulator,
                                  self._get_current_operators(),
                                  self._initial_predicates, timeout, seed)
        policy = lambda _: plan.pop(0)
        return policy

    @abc.abstractmethod
    def _get_current_operators(self) -> Set[Operator]:
        """Get the current set of operators.
        """
        raise NotImplementedError("Override me!")

    @staticmethod
    def _plan(task: Task,
              simulator: Callable[[State, Array], State],
              current_operators: Set[Operator],
              initial_predicates: Set[Predicate],
              timeout: int, seed: int) -> List[Array]:
        """Run TAMP. Return a sequence of low-level actions.
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
        print("num before filter:", len(ground_operators))
        ground_operators = utils.filter_static_operators(
            ground_operators, atoms)
        print("num after filter:", len(ground_operators))
        if not utils.is_dr_reachable(ground_operators, atoms, task.goal):
            raise ApproachFailure(f"Goal {task.goal} not dr-reachable")
        # TODO: pyperplan heuristic
        heuristic = lambda n: 0
        plan = TAMPApproach._run_search(
            task, simulator, ground_operators, atoms, heuristic, timeout, seed)
        return plan

    @staticmethod
    def _run_search(task: Task,
                    simulator: Callable[[State, Array], State],
                    ground_operators: List[_GroundOperator],
                    init_atoms: Collection[GroundAtom],
                    heuristic: Callable[[Node], float],
                    timeout: int, seed: int) -> List[Array]:
        # Do search over skeletons.
        start_time = time.time()
        queue: List[Node] = []
        root_node = Node(atoms=init_atoms, skeleton=[],
                         atoms_sequence=[init_atoms], parent=None)
        rng_prio = np.random.RandomState(seed)
        rng_sampler = np.random.RandomState(seed)
        hq.heappush(queue, (heuristic(root_node),
                            rng_prio.uniform(),
                            root_node))
        while queue and (time.time()-start_time < timeout):
            node: Node = hq.heappop(queue)[2]
            # Good debug point #1: print node.skeleton here to see what
            # the high-level search is doing.
            if task.goal.issubset(node.atoms):
                # If this skeleton satisfies the goal, run low-level search.
                assert node.atoms == node.atoms_sequence[-1]
                plan = TAMPApproach._sample_continuous_values(
                    task, simulator, node.skeleton, node.atoms_sequence,
                    rng_sampler, start_time)
                if plan is not None:
                    print(f"Success! Found plan of length {len(plan)}: {plan}")
                    return plan
            else:
                # Generate successors.
                for operator in utils.get_applicable_operators(
                        ground_operators, node.atoms):
                    child_atoms = utils.apply_operator(operator, node.atoms)
                    child_node = Node(
                        atoms=child_atoms,
                        skeleton=node.skeleton+[operator],
                        atoms_sequence=node.atoms_sequence+[child_atoms],
                        parent=node)
                    # priority = g [plan length] + h [heuristic]
                    priority = len(child_node.skeleton)+heuristic(child_node)
                    hq.heappush(queue, (priority,
                                        rng_prio.uniform(),
                                        child_node))
        if not queue:
            raise ApproachFailure("Planning ran out of skeletons!")
        assert time.time()-start_time > timeout
        raise ApproachTimeout("Planning timed out!")


@dataclass(frozen=True, repr=False, eq=False)
class Node:
    """A node for the search over skeletons.
    """
    atoms: Collection[GroundAtom]
    skeleton: Sequence[_GroundOperator]
    atoms_sequence: Sequence[Collection[GroundAtom]]  # expected state sequence
    parent: Optional[Node]
