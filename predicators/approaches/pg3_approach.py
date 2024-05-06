"""Policy-guided planning for generalized policy generation (PG3).

PG3 requires known STRIPS operators. The command below uses oracle operators,
but it is also possible to use this approach with operators learned from
demonstrations.

Example command line:
    python predicators/main.py --approach pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 10

Example to load from a policy text file:
    python predicators/main.py --approach pg3 --seed 0 \
        --env pddl_gripper_procedural_tasks \
        --strips_learner oracle --num_train_tasks 10 \
        --pg3_init_policy gripper_ldl_policy.txt
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Set

import dill as pkl
from pg3.policy_search import learn_policy

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.planning import PlanningFailure, run_low_level_search
from predicators.settings import CFG
from predicators.structs import Action, Box, Dataset, GroundAtom, \
    LiftedDecisionList, Object, ParameterizedOption, Predicate, State, Task, \
    Type, _GroundNSRT


class PG3Approach(NSRTLearningApproach):
    """Policy-guided planning for generalized policy generation (PG3)."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._current_ldl = LiftedDecisionList([])

    @classmethod
    def get_name(cls) -> str:
        return "pg3"

    def _predict_ground_nsrt(self, atoms: Set[GroundAtom],
                             objects: Set[Object], goal: Set[GroundAtom],
                             static_predicates: Set[Predicate],
                             init_atoms: Set[GroundAtom]) -> _GroundNSRT:
        """Predicts next GroundNSRT to be deployed based on the PG3 generated
        policy."""
        ground_nsrt = utils.query_ldl(self._current_ldl,
                                      atoms,
                                      objects,
                                      goal,
                                      static_predicates=static_predicates,
                                      init_atoms=init_atoms)
        if ground_nsrt is None:
            raise ApproachFailure("PG3 policy was not applicable!")
        return ground_nsrt

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Searches for a low level policy that satisfies PG3's abstract
        policy."""
        skeleton = []
        atoms_sequence = []
        init_atoms = utils.abstract(task.init, self._initial_predicates)
        atoms = init_atoms
        atoms_sequence.append(atoms)
        current_objects = set(task.init)
        # Compute static predicates on-the-fly in case the NSRTs change.
        static_predicates = utils.get_static_preds(
            self._get_current_nsrts(), self._get_current_predicates())
        start_time = time.perf_counter()

        while not task.goal.issubset(atoms):
            if (time.perf_counter() - start_time) >= timeout:
                raise ApproachFailure("Timeout exceeded")
            ground_nsrt = self._predict_ground_nsrt(atoms, current_objects,
                                                    task.goal,
                                                    static_predicates,
                                                    init_atoms)
            atoms = utils.apply_operator(ground_nsrt, atoms)
            skeleton.append(ground_nsrt)
            atoms_sequence.append(atoms)
        try:
            option_list, succeeded = run_low_level_search(
                task, self._option_model, skeleton, atoms_sequence, self._seed,
                timeout - (time.perf_counter() - start_time), self._metrics,
                CFG.horizon)
        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        if not succeeded:
            raise ApproachFailure("Low-level search failed")
        policy = utils.option_plan_to_policy(option_list)
        return policy

    def _learn_ldl(self, online_learning_cycle: Optional[int]) -> None:
        """Learn a lifted decision list policy."""
        # Create the domain str.
        nsrts = self._get_current_nsrts()
        predicates = self._get_current_predicates()
        types = self._types
        domain_name = CFG.env
        domain_str = utils.create_pddl_domain(nsrts, predicates, types,
                                              domain_name)

        # Create the problem strs.
        problem_strs = []

        for i, train_task in enumerate(self._train_tasks):
            problem_name = f"problem{i}"
            goal = train_task.goal
            objects = set(train_task.init)
            init_atoms = utils.abstract(train_task.init, predicates)
            problem_str = utils.create_pddl_problem(objects, init_atoms, goal,
                                                    domain_name, problem_name)
            problem_strs.append(problem_str)

        # Get the initial policies to start the search.
        initial_ldls = self._get_policy_search_initial_ldls()
        initial_ldl_strs = [str(ldl) for ldl in initial_ldls]

        learned_ldl_str, num_pg3_calls = learn_policy(
            domain_str,
            problem_strs,
            horizon=CFG.horizon,
            heuristic_name=CFG.pg3_heuristic,
            search_method=CFG.pg3_search_method,
            max_policy_guided_rollout=CFG.pg3_max_policy_guided_rollout,
            gbfs_max_expansions=CFG.pg3_gbfs_max_expansions,
            hc_enforced_depth=CFG.pg3_hc_enforced_depth,
            allow_new_vars=CFG.pg3_add_condition_allow_new_vars,
            initial_policy_strs=initial_ldl_strs)
        learned_ldl = utils.parse_ldl_from_str(learned_ldl_str, types,
                                               predicates, nsrts)

        self._current_ldl = learned_ldl
        logging.info(f"Keeping best policy:\n{self._current_ldl}")
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.ldl", "wb") as f:
            pkl.dump(self._current_ldl, f)
        with open(f"{save_path}_{online_learning_cycle}_num_calls.pkl", "wb") as f:
            pkl.dump(num_pg3_calls, f)
        self._metrics['pg3_num_calls'] = num_pg3_calls

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First, learn NSRTs.
        annotations = None
        if dataset.has_annotations:
            annotations = dataset.annotations  # pragma: no cover
        self._learn_nsrts(dataset.trajectories,
                          online_learning_cycle=None,
                          annotations=annotations)
        # Now, learn the LDL policy.
        self._learn_ldl(online_learning_cycle=None)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        # Load the NSRTs.
        super().load(online_learning_cycle)
        # Load the LDL policy.
        load_path = utils.get_approach_load_path_str()
        with open(f"{load_path}_{online_learning_cycle}.ldl", "rb") as f:
            self._current_ldl = pkl.load(f)

    def _get_policy_search_initial_ldls(self) -> List[LiftedDecisionList]:
        # Initialize with an empty list by default, but subclasses may
        # override.
        if CFG.pg3_init_policy is not None:
            with open(CFG.pg3_init_policy, "r", encoding="utf-8") as f:
                policy_str = f.read()
            predicates = self._get_current_predicates()
            nsrts = self._get_current_nsrts()
            init_policy = utils.parse_ldl_from_str(policy_str, self._types,
                                                   predicates, nsrts)
            return [init_policy]
        return [LiftedDecisionList([])]
