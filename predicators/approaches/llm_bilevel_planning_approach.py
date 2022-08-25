"""Large language model (LLM) based policy-guided bilevel planning approach.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python predicators/main.py --approach llm_bilevel_planning --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug

Easier setting:
    python predicators/main.py --approach llm_bilevel_planning --seed 0 \
        --strips_learner oracle \
        --env pddl_easy_delivery_procedural_tasks \
        --pddl_easy_delivery_procedural_train_min_num_locs 2 \
        --pddl_easy_delivery_procedural_train_max_num_locs 2 \
        --pddl_easy_delivery_procedural_train_min_want_locs 1 \
        --pddl_easy_delivery_procedural_train_max_want_locs 1 \
        --pddl_easy_delivery_procedural_train_min_extra_newspapers 0 \
        --pddl_easy_delivery_procedural_train_max_extra_newspapers 0 \
        --pddl_easy_delivery_procedural_test_min_num_locs 2 \
        --pddl_easy_delivery_procedural_test_max_num_locs 2 \
        --pddl_easy_delivery_procedural_test_min_want_locs 1 \
        --pddl_easy_delivery_procedural_test_max_want_locs 1 \
        --pddl_easy_delivery_procedural_test_min_extra_newspapers 0 \
        --pddl_easy_delivery_procedural_test_max_extra_newspapers 0 \
        --num_train_tasks 5 \
        --num_test_tasks 10 \
        --debug
"""
from __future__ import annotations

import time
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from predicators import utils
from predicators.approaches.llm_open_loop_approach import LLMOpenLoopApproach
from predicators.settings import CFG
from predicators.structs import GroundAtom, Object, ParameterizedOption, \
    Sequence, State, Task, _GroundNSRT


class LLMBilevelPlanningApproach(LLMOpenLoopApproach):
    """LLMBilevelPlanningApproach definition."""

    @classmethod
    def get_name(cls) -> str:
        return "llm_bilevel_planning"

    def _get_llm_based_plan(
            self, state: State, atoms: Set[GroundAtom],
            goal: Set[GroundAtom]) -> Optional[List[_GroundNSRT]]:
        """Query the LLM and use the outputs to initialize the queue in A*."""
        # To initialize the queue in A*, we will build up a partial policy
        # and then run policy-guided A*.
        start_time = time.time()
        objects = set(state)
        partial_policy_dict: Dict[FrozenSet[GroundAtom], _GroundNSRT] = {}
        # All option plans are merged into a single partial policy.
        for option_plan in self._get_llm_based_option_plans(
                atoms, objects, goal):
            # Note that this calls the overridden method, not the parent's
            # implementation.
            ground_nsrt_plan = self._option_plan_to_nsrt_plan(
                option_plan, atoms, objects, goal)
            assert ground_nsrt_plan is not None
            # Simulate the ground NSRT plan and record the (atoms, ground NSRT)
            # pairs encountered along the way to build up a partial policy.
            single_partial_policy_dict = {}
            cur_atoms = atoms
            for ground_nsrt in ground_nsrt_plan:
                # If we run into an invalid ground NSRT, the rest of the plan
                # is going to be garbage, so stop here.
                if not ground_nsrt.preconditions.issubset(cur_atoms):
                    break
                single_partial_policy_dict[frozenset(cur_atoms)] = ground_nsrt
                cur_atoms = utils.apply_operator(ground_nsrt, cur_atoms)
            # Update the overall partial policy. Note that there may be
            # entries that get overridden. This is okay, and there's no
            # reason to prefer one option plan over another, so we just let
            # the overriding happen.
            partial_policy_dict.update(single_partial_policy_dict)
        # Run policy-guided A*. Note that we importantly want to use
        # sesame_plan because we will want to report the metrics collected in
        # sesame_plan.
        abstract_policy = lambda a, o, g: partial_policy_dict.get(
            frozenset(a), None)
        # Could equivalently make this the length of the longest option plan,
        # but when the option plan runs out, the abstract policy will return
        # None, so we just make this an upper bound.
        max_policy_guided_rollout = CFG.horizon
        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()
        task = Task(state, goal)
        options, metrics, _ = self._run_sesame_plan(
            task,
            nsrts,
            preds,
            CFG.timeout - (time.time() - start_time),
            CFG.seed,
            abstract_policy=abstract_policy,
            max_policy_guided_rollout=max_policy_guided_rollout)
        self._save_metrics(metrics, nsrts, preds)
        # Now convert the options back into ground NSRTs (:facepalm:). This
        # is very circuitous and we should refactor it later. There are two
        # original sins that are converging and making this hard: (1) the
        # fact that the skeleton generator implements its own search, because
        # otherwise we could just run the search utility functions here; and
        # (2) the fact that this and the parent class are implemented as
        # metacontrollers, which do not use sesame plan.
        option_plan = [(o.parent, o.objects) for o in options]
        ground_nsrt_plan = self._option_plan_to_nsrt_plan(
            option_plan, atoms, objects, goal)
        assert ground_nsrt_plan is not None
        assert len(ground_nsrt_plan) == len(option_plan)
        return ground_nsrt_plan

    def _option_plan_to_nsrt_plan(
            self, option_plan: List[Tuple[ParameterizedOption,
                                          Sequence[Object]]],
            atoms: Set[GroundAtom], objects: Set[Object],
            goal: Set[GroundAtom]) -> Optional[List[_GroundNSRT]]:
        """In the parent class, we are assuming that each option plan
        corresponds to a full ground NSRT plan that achieves the goal.

        Here, we do not need to make that assumption, because we are
        using the outputs of the LLM to guide planning, rather than to
        replace planning. Since each option plan may not correspond to a
        full ground NSRT plan, we cannot use
        task_plan_with_option_plan_constraint. Furthermore, it is
        nontrivial to identify a sequence of ground NSRTs that matches
        the partial option plan when there are multiple possible NSRTs
        for each option. For now, we make the simplifying assumption
        that NSRTs and options are one-to-one, which holds in PDDL-only
        environments.
        """
        nsrts = self._get_current_nsrts()
        options = self._initial_options
        assert all(sum(n.option == o for n in nsrts) == 1 for o in options)
        option_to_nsrt = {n.option: n for n in nsrts}
        return [option_to_nsrt[o].ground(objs) for (o, objs) in option_plan]
