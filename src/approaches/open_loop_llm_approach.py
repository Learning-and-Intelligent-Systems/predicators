"""Open-loop large language model (LLM) meta-controller approach.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python src/main.py --approach open_loop_llm --seed 0 --env pddl_blocks_procedural_tasks --strips_learner oracle --num_train_tasks 3

Easier setting:
    python src/main.py --approach open_loop_llm --seed 0 --strips_learner oracle \
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

import abc
import functools
import logging
import os
from typing import Collection, Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple, Union

import dill as pkl
import openai

from predicators.src import utils
from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.nsrt_metacontroller_approach import \
    NSRTMetacontrollerApproach
from predicators.src.settings import CFG
from predicators.src.planning import task_plan_grounding
from predicators.src.structs import NSRT, Box, Dataset, \
    GroundAtom, GroundAtomTrajectory, LDLRule, LiftedAtom, \
    LiftedDecisionList, LowLevelTrajectory, Object, ParameterizedOption, \
    Predicate, State, Task, Type, Variable, _GroundNSRT, _Option, \
    STRIPSOperator, OptionSpec


class OpenLoopLLMApproach(NSRTMetacontrollerApproach):
    """OpenLoopLLMApproach definition."""

    @classmethod
    def get_name(cls) -> str:
        return "open_loop_llm"

    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom], memory: Dict) -> _GroundNSRT:
        # If we already have an abstract plan, execute the next step.
        if "abstract_plan" in memory and memory["abstract_plan"]:
            return memory["abstract_plan"].pop(0)
        # Otherwise, we need to make a new abstract plan.
        new_prompt = self._create_prompt(atoms, goal, [])
        prompt = self._prompt_prefix + new_prompt
        # Query the LLM.
        llm_prediction = self._predict_llm(prompt)
        # Try to convert the output into an abstract plan.
        objects = set(state)
        option_plan = self._llm_prediction_to_option_plan(llm_prediction,
            objects)
        # If we failed to find a nontrivial plan, give up.
        if len(option_plan) == 0:
            raise ApproachFailure("LLM did not predict an abstract plan.")
        # Otherwise, we succeeded, so attempt to turn the plan into a
        # sequence of ground NSRT's.
        nsrts = self._get_current_nsrts()
        strips_ops = [nsrt.op for nsrt in nsrts]
        option_specs = [(nsrt.option, nsrt.option_vars) for nsrt in nsrts]
        best_matching_plan = self._find_best_match_plan(objects, atoms, strips_ops, option_specs, goal, option_plan)
        # If we can't find a best matching plan that achieves the goal,
        # give up.
        if not best_matching_plan:
            raise ApproachFailure("LLM predicted plan does not achieve goal.")
        # Else, add this plan to memory so it can be refined!
        memory["abstract_plan"] = best_matching_plan
        return memory["abstract_plan"].pop(0)

    def _predict_llm(self, prompt: str) -> str:
        """Query the LLM."""

        # TODO: Uncomment this for the real deal.
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(model=CFG.llm_model_name,
            prompt=prompt, temperature=0, max_tokens=CFG.llm_max_tokens)
        assert len(response["choices"]) == 1
        text_response = response["choices"][0]["text"]

        # This is an example incorrect response for debugging. TODO remove.
        # text_response = ' unstack(b2:block, b1:block)\n  unstack(b1:block, b4:block)\n  stack(b1:block, b4:block)\n  pick-up(b4:block)\n  unstack(b3:block, b2:block)\n  stack(b3:block, b1:block)\n  unstack(b1:block, b2:block)\n  stack(b1:block, b0:block)\n  pick-up(b0:block)\n  unstack(b4:block, b3:block)\n  stack(b4:block, b0:block)\n  unstack(b3:block, b2:block)\n  unstack(b2:block, b1:block)\n  unstack(b1:block, b0:block)\n  unstack(b0:block, b4:block)\n  handempty()\n  on(b0:block, b3:block)\n  on(b4:block, b3:block)\n  ontable(b'
        # This is an example correct response for debugging. TODO remove.
        # text_response = ' unstack(b4:block, b3:block)\n  put-down(b4:block)\n  unstack(b1:block, b0:block)\n  put-down(b1:block)\n  pick-up(b2:block)\n  stack(b2:block, b0:block)\n  handempty()\n  on(b0:block, b3:block)\n  on(b4:block, b3:block)\n  ontable(b'

        logging.debug("\nPROMPT:")
        logging.debug(prompt)
        logging.debug("\nPREDICTION:")
        logging.debug(text_response)

        return text_response

    def _llm_prediction_to_option_plan(self, llm_prediction: str, objects: Collection[Object]) -> List[Tuple[ParameterizedOption, List[Object]]]:
        """Convert the output of the LLM into a sequence of
        ParameterizedOptions coupled with a list of objects that will be used
        to ground the ParameterizedOption."""
        option_plan: List[Tuple[ParameterizedOption, List[Object]]] = []
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
                logging.info(f"Line {option_str} output by LLM doesn't " +
                "contain a valid option name. Terminating option plan " +
                "parsing.")
                import ipdb; ipdb.set_trace()
                break
            option = option_name_to_option[option_name]
            # Now that we have the option, we need to parse out the objects
            # along with specified types.
            typed_objects_str_list = option_str_stripped.split('(')[1].strip(')').split(',')
            objs_list = []
            malformed = False
            for i, type_object_string in enumerate(typed_objects_str_list):
                object_type_str_list = type_object_string.strip().split(':')
                # We expect this list to be [object_name, type_name]
                if len(object_type_str_list) != 2:
                    logging.info(f"Line {option_str} output by LLM has a " +
                    "malformed object-type list.")
                    malformed = True
                    break
                object_name = object_type_str_list[0]
                type_name = object_type_str_list[1]
                if object_name not in obj_name_to_obj.keys():
                    logging.info(f"Line {option_str} output by LLM has an " +
                    "invalid object name.")
                    malformed = True
                    break
                obj = obj_name_to_obj[object_name]
                # Check that the type of this object agrees
                # with what's expected given the ParameterizedOption.
                if type_name != option.types[i].name:
                    logging.info(f"Line {option_str} output by LLM has an " +
                    "invalid type name.")
                    malformed = True
                    break
                objs_list.append(obj)
            if not malformed:
                option_plan.append((option, objs_list))
        return option_plan

    def _find_best_match_plan(
            self, objects: Set[Object],
            init_atoms: Set[GroundAtom],
            strips_ops: List[STRIPSOperator],
            option_specs: List[OptionSpec],
            goal: Set[GroundAtom],
            option_plan: List[Tuple[ParameterizedOption, List[Object]]]) -> Optional[List[_GroundNSRT]]:
        """Function to turn an option plan generated from LLM output
        into a best-fitting plan of ground NSRT's that achieve the goal.
        
        If no goal-achieving sequence of ground NSRT's corresponds to
        the option plan, return None.
        """        
        ground_nsrts, _ = task_plan_grounding(init_atoms,
                                              objects,
                                              strips_ops,
                                              option_specs,
                                              allow_noops=True)
        heuristic = utils.create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic, init_atoms, goal,
            ground_nsrts, self._initial_predicates, objects)

        def _check_goal(
                searchnode_state: Tuple[FrozenSet[GroundAtom], int]) -> bool:
            return goal.issubset(searchnode_state[0])

        def _get_successor_with_correct_option(
            searchnode_state: Tuple[FrozenSet[GroundAtom], int]
        ) -> Iterator[Tuple[_GroundNSRT, Tuple[FrozenSet[GroundAtom], int],
                            float]]:
            atoms = searchnode_state[0]
            idx_into_traj = searchnode_state[1]

            if idx_into_traj > len(option_plan) - 1:
                return

            gt_param_option = option_plan[idx_into_traj][0]
            gt_objects = option_plan[idx_into_traj][1]

            for applicable_nsrt in utils.get_applicable_operators(
                    ground_nsrts, atoms):
                # NOTE: we check that the ParameterizedOptions are equal before
                # attempting to ground because otherwise, we might
                # get a parameter mismatch and trigger an AssertionError
                # during grounding.
                if applicable_nsrt.option != gt_param_option:
                    continue
                if applicable_nsrt.option_objs != gt_objects:
                    continue
                next_atoms = utils.apply_operator(applicable_nsrt, set(atoms))
                # The returned cost is uniform because we don't
                # actually care about finding the shortest path;
                # just one that matches!
                yield (applicable_nsrt, (frozenset(next_atoms),
                                            idx_into_traj + 1), 1.0)

        init_atoms_frozen = frozenset(init_atoms)
        init_searchnode_state = (init_atoms_frozen, 0)
        # NOTE: each state in the below GBFS is a tuple of
        # (current_atoms, idx_into_traj). The idx_into_traj is necessary because
        # we need to check whether the atoms that are true at this particular
        # index into the trajectory is what we would expect given the demo
        # trajectory.
        state_seq, action_seq = utils.run_gbfs(
            init_searchnode_state, _check_goal,
            _get_successor_with_correct_option,
            lambda searchnode_state: heuristic(searchnode_state[0]))

        ret_nsrt_plan = None
        if _check_goal(state_seq[-1]):
            ret_nsrt_plan = action_seq
        return ret_nsrt_plan

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
        options_str = "\n  ".join(map(self._option_or_nsrt_to_str, options))
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
    def _option_or_nsrt_to_str(option: Union[_Option, _GroundNSRT]) -> str:
        objects_str = ", ".join(map(str, option.objects))
        return f"{option.name}({objects_str})"
