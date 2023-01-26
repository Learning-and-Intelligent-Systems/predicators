"""Use Popper (ILP system) to learn an abstract policy.

Requires known STRIPS operators. The command below uses oracle operators,
but it is also possible to use this approach with operators learned from
demonstrations.

Example command line:
    python predicators/main.py --approach popper_policy --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 10
"""
from __future__ import annotations

import abc
import functools
import logging
import time
from typing import Callable, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set, Tuple
from typing import Type as TypingType

import dill as pkl
from popper_policies.learn import learn_policy
from popper_policies.structs import LiftedDecisionList as PopperLDL
from popper_policies.utils import apply_substitution as popper_apply_sub
from typing_extensions import TypeAlias

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.planning import PlanningFailure, run_low_level_search, \
    task_plan_with_option_plan_constraint
from predicators.settings import CFG
from predicators.structs import NSRT, Action, Box, Dataset, GroundAtom, \
    LDLRule, LiftedAtom, LiftedDecisionList, Object, ParameterizedOption, \
    Predicate, State, Task, Type, Variable, _GroundNSRT


class PopperPolicyApproach(NSRTLearningApproach):
    """Use Popper (ILP system) to learn an abstract policy."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._current_ldl = LiftedDecisionList([])

    @classmethod
    def get_name(cls) -> str:
        return "popper_policy"

    def _predict_ground_nsrt(self, atoms: Set[GroundAtom],
                             objects: Set[Object],
                             goal: Set[GroundAtom]) -> _GroundNSRT:
        """Predicts next GroundNSRT to be deployed based on the PG3 generated
        policy."""
        ground_nsrt = utils.query_ldl(self._current_ldl, atoms, objects, goal)
        if ground_nsrt is None:
            raise ApproachFailure("PG3 policy was not applicable!")
        return ground_nsrt

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Searches for a low level policy that satisfies the abstract
        policy."""
        # TODO: factor out common code between this and PG3.
        skeleton = []
        atoms_sequence = []
        atoms = utils.abstract(task.init, self._initial_predicates)
        atoms_sequence.append(atoms)
        current_objects = set(task.init)
        start_time = time.perf_counter()

        while not task.goal.issubset(atoms):
            if (time.perf_counter() - start_time) >= timeout:
                raise ApproachFailure("Timeout exceeded")
            ground_nsrt = self._predict_ground_nsrt(atoms, current_objects,
                                                    task.goal)
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

    def _learn_ldl(self, trajectories: List[LowLevelTrajectory],
                   online_learning_cycle: Optional[int]) -> None:
        """Learn a lifted decision list policy."""

        nsrts = self._get_current_nsrts()
        predicates = self._get_current_predicates()
        strips_ops = [n.op for n in nsrts]
        option_specs = [(n.option, list(n.option_vars)) for n in nsrts]
        types = self._types
        domain_name = "mydomain"
        domain_str = utils.create_pddl_domain(nsrts, predicates, types,
                                              domain_name)

        ground_atom_dataset = utils.create_ground_atom_dataset(
            trajectories, predicates)

        problem_strs = []
        plan_strs = []

        for i, (ll_traj, atom_traj) in enumerate(ground_atom_dataset):
            problem_name = f"problem{i}"
            train_task = self._train_tasks[ll_traj.train_task_idx]
            goal = train_task.goal
            objects = set(train_task.init)
            init_atoms = atom_traj[0]
            problem_str = utils.create_pddl_problem(objects, init_atoms, goal,
                                                    domain_name, problem_name)
            problem_strs.append(problem_str)
            # Need to find a matching ground NSRT trajectory for the atom traj.
            segment_traj = self._segmented_trajs[i]
            seg_options = []
            for segment in segment_traj:
                assert segment.has_option()
                seg_options.append(segment.get_option())
            option_plan = [(o.parent, o.objects) for o in seg_options]
            nsrt_plan = task_plan_with_option_plan_constraint(
                objects, predicates, strips_ops, option_specs, init_atoms,
                goal, option_plan)
            plan_str = []
            for ground_nsrt in nsrt_plan:
                if not ground_nsrt.objects:
                    pddl_str = f"({ground_nsrt.name})"
                else:
                    obj_str = " ".join([o.name for o in ground_nsrt.objects])
                    pddl_str = f"({ground_nsrt.name} {obj_str})"
                plan_str.append(pddl_str)
            plan_strs.append(plan_str)

        popper_policy = learn_policy(domain_str,
                                     problem_strs,
                                     plan_strs,
                                     planner_name="fastdownward")

        # In this repo, we make an annoying assumption about the parameters
        # of the LDL matching those of the NSRT.
        nsrt_name_to_nsrt = {n.name: n for n in nsrts}
        new_rules = []
        for rule in popper_policy.rules:
            rule_op = rule.operator
            nsrt = nsrt_name_to_nsrt[rule_op.name]
            sub = {
                old: new.name
                for (old, _), new in zip(rule_op.signature, nsrt.parameters)
            }
            for (v, _) in rule.parameters:
                if v not in sub:
                    sub[v] = v
            new_rule = popper_apply_sub(rule, sub)
            new_rules.append(new_rule)
        popper_policy = PopperLDL(new_rules)

        policy_str = str(popper_policy)
        ldl = utils.parse_ldl_from_str(policy_str, types, predicates, nsrts)
        self._current_ldl = ldl
        logging.info(f"Keeping best policy:\n{self._current_ldl}")
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.ldl", "wb") as f:
            pkl.dump(self._current_ldl, f)

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First, learn NSRTs.
        self._learn_nsrts(dataset.trajectories, online_learning_cycle=None)
        # Now, learn the LDL policy.
        self._learn_ldl(dataset.trajectories, online_learning_cycle=None)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        # Load the NSRTs.
        super().load(online_learning_cycle)
        # Load the LDL policy.
        load_path = utils.get_approach_load_path_str()
        with open(f"{load_path}_{online_learning_cycle}.ldl", "rb") as f:
            self._current_ldl = pkl.load(f)
