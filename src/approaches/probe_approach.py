"""Open-loop large language model (LLM) meta-controller approach.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python src/main.py --approach probe --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug

Easier setting:
    python src/main.py --approach probe --seed 0 \
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

import logging
from typing import Collection, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set, Tuple

from predicators.src import utils
from predicators.src.approaches.open_loop_llm_approach import \
    OpenLoopLLMApproach
from predicators.src.llm_interface import OpenAILLM
from predicators.src.planning import task_plan_with_option_plan_constraint
from predicators.src.settings import CFG
from predicators.src.structs import Box, Dataset, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, _Option
from predicators.src.utils import apply_operator, \
    create_task_planning_heuristic, get_applicable_operators, \
    run_policy_guided_astar


class Probe(OpenLoopLLMApproach):
    """Probe definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Set up the LLM.
        self._llm = OpenAILLM(CFG.open_loop_llm_model_name)
        # Set after learning.
        self._prompt_prefix = ""
        self.last_char = ")"

    @classmethod
    def get_name(cls) -> str:
        return "probe"

    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom], memory: Dict) -> _GroundNSRT:
        # If we already have an abstract plan, execute the next step.
        if "abstract_plan" in memory and memory["abstract_plan"]:
            return memory["abstract_plan"].pop(0)
        # Otherwise, we need to make a new abstract plan.
        init = atoms
        initial_atoms: FrozenSet[GroundAtom] = frozenset(atoms)
        objects = set(state)
        states, actions = self._generate_states(atoms, goal, state)
        final_states = []
        final_actions = []
        final_states.append(initial_atoms)
        if not actions and not actions[0]:
            final_actions.append(
                actions[0][0])  #dealing with initial that repeats
        else:
            dictionary = None
        if not states:
            for t_states in states:
                t_states.pop()
                for counter in range(0, len(t_states)):
                    if counter == 0:
                        continue
                    final_states.append(t_states[counter])
                    final_actions.append(
                        actions[states.index(t_states)][counter])

            new_states = []
            for s in final_states:
                new_states.append(frozenset(s))

            dictionary = dict(zip(new_states, final_actions))

        else:
            dictionary = None

        ground_nsrts = [
            ground_nsrt for nsrt in self._get_current_nsrts()
            for ground_nsrt in utils.all_ground_nsrts(nsrt, objects)
        ]

        def policy(
            sete: FrozenSet[GroundAtom]
        ) -> Optional[Dict[FrozenSet[GroundAtom], _GroundNSRT]]:
            if dictionary and sete in dictionary:
                return dictionary[sete]
            return None

        def check_goal(atoms: FrozenSet[GroundAtom]) -> bool:
            return goal.issubset(atoms)

        def get_valid_actions(
            atoms: FrozenSet[GroundAtom]
        ) -> Iterator[Tuple[_GroundNSRT, float]]:
            for op in get_applicable_operators(ground_nsrts, atoms):
                yield (op, 1.0)

        def get_next_state(atoms: FrozenSet[GroundAtom],
                           ground_nsrt: _GroundNSRT) -> FrozenSet[GroundAtom]:
            return frozenset(apply_operator(ground_nsrt, set(atoms)))

        heuristic = create_task_planning_heuristic(
            heuristic_name=CFG.pg3_task_planning_heuristic,
            init_atoms=init,
            goal=goal,
            ground_ops=ground_nsrts,
            predicates=self._initial_predicates,
            objects=set(state),
        )

        _, action_seq = run_policy_guided_astar(
            initial_state=initial_atoms,
            check_goal=check_goal,
            get_valid_actions=get_valid_actions,
            get_next_state=get_next_state,
            heuristic=heuristic,
            policy=policy,
            num_rollout_steps=CFG.pg3_max_policy_guided_rollout,
            rollout_step_cost=0)

        if action_seq is not None:
            # If valid plan, add plan to memory so it can be refined!
            memory["abstract_plan"] = action_seq
            return memory["abstract_plan"].pop(0)
        raise "Approach does not work"

    def _generate_states(self, atoms: Set[GroundAtom], goal: Set[GroundAtom],
                         state: State) -> Tuple[List, List]:
        states = []
        actions = []

        initial_atoms = atoms

        new_prompt = self._create_prompt(atoms, goal, [])
        prompt = self._prompt_prefix + new_prompt
        llm_predictions = self._llm.sample_completions(
            prompt=prompt,
            temperature=CFG.open_loop_llm_temperature,
            seed=CFG.seed,
            num_completions=5)

        for llm_prediction in llm_predictions:
            temp_states = []
            temp_actions = []
            temp_states.append(initial_atoms)
            trimmed_prediction = ""
            if len(llm_prediction) != 0 and llm_prediction[-1] != ')':
                _, _, a = llm_prediction[::-1].partition(')')
                trimmed_prediction = a[::-1] + ')'
            else:
                trimmed_prediction = llm_prediction

            objects = set(state)
            option_plan = self._llm_prediction_to_option_plan(
                trimmed_prediction, objects)
            if len(option_plan) == 0:
                continue
            nsrts = self._get_current_nsrts()
            ground_nsrt_plan = []
            for option in option_plan:
                for n in nsrts:
                    if option[0].name == n.option.name:
                        ground_nsrt_plan.append(n.ground(option[1]))
                        break

            for nsrt in ground_nsrt_plan:
                atoms = apply_operator(nsrt, atoms)
                temp_actions.append(nsrt)
                temp_states.append(atoms)
            states.append(temp_states)
            actions.append(temp_actions)
        return states, actions
