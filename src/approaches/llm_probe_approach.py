"""Large language model (LLM) based policy guided bilevel planning approach.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python src/main.py --approach llm_probe --seed 0 \
        --strips_learner oracle \
        --env pddl_blocks_procedural_tasks \
        --num_train_tasks 3 \
        --num_test_tasks 1 \
        --debug

Easier setting:
    python src/main.py --approach llm_probe --seed 0 \
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

from typing import Collection, Dict, FrozenSet, Iterator, List, Optional, \
    Set, Tuple

from predicators.src import utils
from predicators.src.approaches.open_loop_llm_approach import \
    OpenLoopLLMApproach
from predicators.src.settings import CFG
from predicators.src.structs import GroundAtom, Object, ParameterizedOption, \
    Sequence, State, _GroundNSRT


class LLMProbeApproach(OpenLoopLLMApproach):
    """LLMProbeApproach definition."""

    @classmethod
    def get_name(cls) -> str:
        return "llm_probe"

    def _get_llm_based_plan(self, state: State, atoms: Set[GroundAtom],
                            goal: Set[GroundAtom]) -> List[_GroundNSRT]:
        """In this function, we take predictions the llm outputs regarding the
        solution to our PDDL problem and use them to make a dictionary of
        corresponding states and actions.

        This dictionary is used to create a policy which guides the A*
        search algorithm in searching for a solution to the PDDL
        problem.
        """
        initial_atoms: FrozenSet[GroundAtom] = frozenset(atoms)
        environment_objects: Collection[Object] = set(state)
        # Get trajectories from the LLM.
        trajectories = self._get_llm_based_state_action_predictions(
            atoms, goal, environment_objects)
        # Dictionary used to create policy for search function.
        abstract_policy_dict = self._get_dictionary(trajectories)
        action_seq = self._run_policy_guided_planning(abstract_policy_dict,
                                                      goal,
                                                      environment_objects,
                                                      atoms, initial_atoms)
        return action_seq

    def _get_llm_based_state_action_predictions(
        self, atoms: Set[GroundAtom], goal: Set[GroundAtom],
        environment_objects: Collection[Object]
    ) -> List[Tuple[List[Set[GroundAtom]], List[_GroundNSRT]]]:
        trajectories = []
        initial_atoms = atoms

        new_prompt = self._create_prompt(atoms, goal, [])
        prompt = self._prompt_prefix + new_prompt
        llm_predictions = self._llm.sample_completions(
            prompt=prompt,
            temperature=CFG.open_loop_llm_temperature,
            seed=CFG.seed,
            num_completions=CFG.open_loop_llm_num_completions)

        for llm_prediction in llm_predictions:

            option_plan = self._llm_prediction_to_option_plan(
                llm_prediction, environment_objects)
            if len(option_plan) == 0:
                continue
            states, actions = self._get_ground_nsrt_plan(
                initial_atoms, option_plan)

            trajectories.append((states, actions))
        return trajectories

    def _get_ground_nsrt_plan(
        self, initial_atoms: Set[GroundAtom],
        option_plan: List[Tuple[ParameterizedOption, Sequence]]
    ) -> Tuple[List[Set[GroundAtom]], List[_GroundNSRT]]:

        states = []
        actions = []
        states.append(initial_atoms)
        nsrts = self._get_current_nsrts()
        # We assume that operators and NSRTs are the same
        # for our use case, so this function would break down
        # for environments not written in PDDL.
        ground_nsrt_plan = []
        for (option, objects) in option_plan:
            for n in nsrts:
                if option.name == n.option.name:
                    ground_nsrt_plan.append(n.ground(objects))
                    break
        atoms = initial_atoms
        for nsrt in ground_nsrt_plan:
            atoms = utils.apply_operator(nsrt, atoms)
            actions.append(nsrt)
            states.append(atoms)
        return states, actions

    def _get_dictionary(
        self, trajectories: List[Tuple[List[Set[GroundAtom]],
                                       List[_GroundNSRT]]]
    ) -> Optional[Dict]:

        abstract_policy_dict = {}
        for (states, actions) in trajectories:
            assert len(states) == len(actions) + 1
            for (s, a) in zip(states, actions):
                abstract_policy_dict[frozenset(s)] = a
        return abstract_policy_dict

    def _run_policy_guided_planning(
            self, abstract_policy_dict: Optional[Dict], goal: Set[GroundAtom],
            environment_objects: Collection[Object], init: Set[GroundAtom],
            initial_atoms: FrozenSet[GroundAtom]) -> List[_GroundNSRT]:
        ground_nsrts = [
            ground_nsrt for nsrt in self._get_current_nsrts() for ground_nsrt
            in utils.all_ground_nsrts(nsrt, environment_objects)
        ]

        def policy(
            sete: FrozenSet[GroundAtom]
        ) -> Optional[Dict[FrozenSet[GroundAtom], _GroundNSRT]]:
            if abstract_policy_dict and sete in abstract_policy_dict:
                return abstract_policy_dict[sete]
            return None

        def check_goal(atoms: FrozenSet[GroundAtom]) -> bool:
            return goal.issubset(atoms)

        def get_valid_actions(
            atoms: FrozenSet[GroundAtom]
        ) -> Iterator[Tuple[_GroundNSRT, float]]:
            for op in utils.get_applicable_operators(ground_nsrts, atoms):
                yield (op, 1.0)

        def get_next_state(atoms: FrozenSet[GroundAtom],
                           ground_nsrt: _GroundNSRT) -> FrozenSet[GroundAtom]:
            return frozenset(utils.apply_operator(ground_nsrt, set(atoms)))

        heuristic = utils.create_task_planning_heuristic(
            heuristic_name=CFG.pg3_task_planning_heuristic,
            init_atoms=init,
            goal=goal,
            ground_ops=ground_nsrts,
            predicates=self._initial_predicates,
            objects=environment_objects,
        )
        _, action_seq = utils.run_policy_guided_astar(
            initial_state=initial_atoms,
            check_goal=check_goal,
            get_valid_actions=get_valid_actions,
            get_next_state=get_next_state,
            heuristic=heuristic,
            policy=policy,
            num_rollout_steps=CFG.pg3_max_policy_guided_rollout,
            rollout_step_cost=0)
        return action_seq
