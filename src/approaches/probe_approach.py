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
from predicators.src.approaches.nsrt_metacontroller_approach import \
    NSRTMetacontrollerApproach
from predicators.src.llm_interface import OpenAILLM
from predicators.src.planning import task_plan_with_option_plan_constraint
from predicators.src.settings import CFG
from predicators.src.structs import Box, Dataset, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, _Option
from predicators.src.utils import apply_operator, \
    create_task_planning_heuristic, get_applicable_operators, \
    run_policy_guided_astar


class Probe(NSRTMetacontrollerApproach):
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

    def _process_single_prediction(
            self, llm_prediction: str, state: State, atoms: Set[GroundAtom],
            goal: Set[GroundAtom]) -> Optional[List[_GroundNSRT]]:
        objects = set(state)

        option_plan = self._llm_prediction_to_option_plan(
            llm_prediction, objects)
        # If we failed to find a nontrivial plan with this prediction,
        # continue on to next prediction.
        #import pdb; pdb.set_trace()
        if len(option_plan) == 0:
            return None
        # Attempt to turn the plan into a sequence of ground NSRTs.
        nsrts = self._get_current_nsrts()
        predicates = self._initial_predicates
        strips_ops = [n.op for n in nsrts]
        option_specs = [(n.option, list(n.option_vars)) for n in nsrts]
        ground_nsrt_plan = task_plan_with_option_plan_constraint(
            objects, predicates, strips_ops, option_specs, atoms, goal,
            option_plan)
        # If we can't find an NSRT plan that achieves the goal,
        # continue on to next prediction.
        if not ground_nsrt_plan:
            return None

        return ground_nsrt_plan

    def _llm_prediction_to_option_plan(
        self, llm_prediction: str, objects: Collection[Object]
    ) -> List[Tuple[ParameterizedOption, Sequence[Object]]]:
        """Convert the output of the LLM into a sequence of
        ParameterizedOptions coupled with a list of objects that will be used
        to ground the ParameterizedOption."""
        option_plan: List[Tuple[ParameterizedOption, Sequence[Object]]] = []
        # Setup dictionaries enabling us to easily map names to specific
        # Python objects during parsing.
        option_name_to_option = {op.name: op for op in self._initial_options}
        obj_name_to_obj = {o.name: o for o in objects}
        # We assume the LLM's output is such that each line contains a
        # option_name(obj0:type0, obj1:type1,...).

        options_str_list = llm_prediction.split('\n')
        for option_str in options_str_list:
            option_str_stripped = option_str.strip()
            option_name = option_str_stripped.split('(')[0]
            # Skip empty option strs.
            if not option_str:
                continue
            if option_name not in option_name_to_option.keys():
                logging.info(
                    f"Line {option_str} output by LLM doesn't "
                    "contain a valid option name. Terminating option plan "
                    "parsing.")
                break
            option = option_name_to_option[option_name]
            # Now that we have the option, we need to parse out the objects
            # along with specified types.
            typed_objects_str_list = option_str_stripped.split('(')[1].strip(
                ')').split(',')
            objs_list = []
            malformed = False
            for i, type_object_string in enumerate(typed_objects_str_list):
                object_type_str_list = type_object_string.strip().split(':')
                # We expect this list to be [object_name, type_name].
                if len(object_type_str_list) != 2:
                    logging.info(f"Line {option_str} output by LLM has a "
                                 "malformed object-type list.")
                    malformed = True
                    break
                object_name = object_type_str_list[0]
                type_name = object_type_str_list[1]
                if object_name not in obj_name_to_obj.keys():
                    logging.info(f"Line {option_str} output by LLM has an "
                                 "invalid object name.")
                    malformed = True
                    break
                obj = obj_name_to_obj[object_name]
                # Check that the type of this object agrees
                # with what's expected given the ParameterizedOption.
                if type_name != option.types[i].name:
                    logging.info(f"Line {option_str} output by LLM has an "
                                 "invalid type name.")
                    malformed = True
                    break
                objs_list.append(obj)
            if not malformed:
                option_plan.append((option, objs_list))
        return option_plan

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First, learn NSRTs.
        super().learn_from_offline_dataset(dataset)
        # Then, parse the data into the prompting format expected by the LLM.
        self._prompt_prefix = self._data_to_prompt_prefix(dataset)

    def _data_to_prompt_prefix(self, dataset: Dataset) -> str:
        # In this approach, we learned NSRTs, so we just use the segmented
        # trajectories that NSRT learning returned to us.
        prompts = []
        assert len(self._segmented_trajs) == len(dataset.trajectories)
        for segment_traj, ll_traj in zip(self._segmented_trajs,
                                         dataset.trajectories):
            if not ll_traj.is_demo:
                continue
            init = segment_traj[0].init_atoms
            goal = self._train_tasks[ll_traj.train_task_idx].goal
            seg_options = []
            for segment in segment_traj:
                assert segment.has_option()
                seg_options.append(segment.get_option())
            prompt = self._create_prompt(init, goal, seg_options)
            prompts.append(prompt)
        return "\n\n".join(prompts) + "\n\n"

    def _create_prompt(self, init: Set[GroundAtom], goal: Set[GroundAtom],
                       options: Sequence[_Option]) -> str:
        init_str = "\n  ".join(map(str, sorted(init)))
        goal_str = "\n  ".join(map(str, sorted(goal)))
        options_str = "\n  ".join(map(self._option_to_str, options))
        prompt = f"""
(:init
  {init_str}
)
(:goal
  {goal_str}
)
Solution:
  {options_str}"""
        return prompt

    @staticmethod
    def _option_to_str(option: _Option) -> str:
        objects_str = ", ".join(map(str, option.objects))
        return f"{option.name}({objects_str})"
