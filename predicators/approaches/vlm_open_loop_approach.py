"""Open-loop vision-language model (VLM) planner approach based on the 'Look
Before You Leap' paper (https://arxiv.org/abs/2311.17842). Depending on command
line options, the VLM can be made to use the training trajectories as few-shot
examples for solving the task, or directly solve the task with no few-shot
prompting.

python predicators/main.py --env burger --approach vlm_open_loop --seed
0 --num_train_tasks 0 --num_test_tasks 1 --bilevel_plan_without_sim True
--make_failure_videos --sesame_task_planner fdopt --vlm_model_name
gpt-4o
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Set

import numpy as np
import PIL

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.settings import CFG
from predicators.structs import Action, Box, Dataset, ParameterizedOption, \
    Predicate, State, Task, Type, _Option


class VLMOpenLoopApproach(BilevelPlanningApproach):
    """VLMOpenLoopApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Set up the vlm and base prompt.
        self._vlm = utils.create_vlm_by_name(CFG.vlm_model_name)
        self._base_prompt_imgs = []
        filepath_to_vlm_prompt = utils.get_path_to_predicators_root() + \
        "/predicators/approaches/vlm_planning_prompts/no_few_shot.txt"
        with open(filepath_to_vlm_prompt, "r", encoding="utf-8") as f:
            self.base_prompt = f.read()

    @classmethod
    def get_name(cls) -> str:
        return "vlm_open_loop"

    @property
    def is_learning_based(self) -> bool:
        return True

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Adds the images and plans from the training dataset to the base
        prompt for use at test time!"""
        pass

    def _get_current_nsrts(self) -> Set[utils.NSRT]:
        """This method doesn't explicitly learn NSRTs, so we simply return the
        empty set."""
        return set()

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        option_plan = self._query_vlm_for_option_plan(task)
        policy = utils.option_plan_to_policy(option_plan)

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _query_vlm_for_option_plan(self, task: Task) -> Sequence[_Option]:
        init_state = task.init
        assert init_state.simulator_state is not None
        assert isinstance(init_state.simulator_state["images"], List)
        curr_options = sorted(self._initial_options)
        imgs = init_state.simulator_state["images"]
        imgs_for_vlm = [PIL.Image.fromarray(img_arr)
                        for img_arr in imgs]  # type: ignore
        options_str = "\n".join(str(opt) for opt in curr_options)
        objects_list = sorted(set(task.init))
        objects_str = "\n".join(str(obj) for obj in objects_list)
        goal_expr_list = sorted(set(task.goal))
        type_hierarchy_str = utils.create_pddl_types_str(self._types)
        goal_str = "\n".join(str(obj) for obj in goal_expr_list)
        prompt = self.base_prompt.format(options=options_str,
                                         typed_objects=objects_str,
                                         type_hierarchy=type_hierarchy_str,
                                         goal_str=goal_str)
        vlm_output = self._vlm.sample_completions(
            prompt,
            imgs_for_vlm,
            temperature=CFG.vlm_temperature,
            seed=CFG.seed,
            num_completions=1)
        plan_prediction_txt = vlm_output[0]
        option_plan: List[_Option] = []
        try:
            start_index = plan_prediction_txt.index("Plan:\n") + len("Plan:\n")
            parsable_plan_prediction = plan_prediction_txt[start_index:]
        except ValueError:
            return "Marker not found in the input string."
        parsed_option_plan = utils.parse_model_output_into_option_plan(
            parsable_plan_prediction, objects_list, self._types,
            self._initial_options, True)
        for option_tuple in parsed_option_plan:
            option_plan.append(option_tuple[0].ground(
                option_tuple[1], np.array(option_tuple[2])))
        return option_plan
