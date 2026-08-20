"""Open-loop vision-language model (VLM) planner approach based on the 'Look
Before You Leap' paper (https://arxiv.org/abs/2311.17842). Depending on command
line options (specifically --vlm_open_loop_use_training_demos), the VLM can be
made to use the training trajectories as few-shot examples for solving the
task, or directly solve the task with no few-shot prompting.

Example command in burger that doesn't use few-shot examples:
python predicators/main.py --env burger --approach vlm_open_loop --seed 0 \
--num_train_tasks 0 --num_test_tasks 1 --bilevel_plan_without_sim True \
--make_failure_videos --sesame_task_planner fdopt --vlm_model_name gpt-4o

Example command that does have few-shot examples:
python predicators/main.py --env burger --approach vlm_open_loop --seed 0 \
--num_train_tasks 1 --num_test_tasks 1 --bilevel_plan_without_sim True \
--make_failure_videos --sesame_task_planner fdopt --debug \
--vlm_model_name gemini-1.5-pro-latest --vlm_open_loop_use_training_demos True
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np
import PIL
from PIL import ImageDraw

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.pretrained_model_interface import PretrainedLargeModel
from predicators.settings import CFG
from predicators.structs import Action, Box, Dataset, LowLevelTrajectory, \
    ParameterizedOption, Predicate, Segment, State, Task, Type, _Option


class VLMOpenLoopApproach(BilevelPlanningApproach):  # pragma: no cover
    """VLMOpenLoopApproach definition.

    NOTE: we don't test this approach because it requires an environment
    that has rendering, and our current envs that have this are burger
    and kitchen, which are slow/untestable on GitHub remote...
    """

    _vlm: Optional[PretrainedLargeModel]

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Set up the vlm and base prompt.
        if CFG.vlm_open_loop_no_image:
            self._vlm = utils.create_llm_by_name(CFG.llm_model_name)
        else:
            self._vlm = utils.create_vlm_by_name(CFG.vlm_model_name)
        prompt_suffix = ""
        if CFG.vlm_open_loop_no_image:
            prompt_suffix = "_oc"
        filepath_to_vlm_prompt = utils.get_path_to_predicators_root() + \
        "/predicators/approaches/vlm_planning_prompts/no_few_shot" +\
        f"{prompt_suffix}.txt"
        if CFG.vlm_open_loop_use_training_demos:
            filepath_to_vlm_prompt = utils.get_path_to_predicators_root() + \
        "/predicators/approaches/vlm_planning_prompts/few_shot" +\
        f"{prompt_suffix}.txt"
        with open(filepath_to_vlm_prompt, "r", encoding="utf-8") as f:
            self.base_prompt = f.read()
        self._prompt_state_imgs_list: List[PIL.Image.Image] = []
        self._prompt_demos_str = ""

    @classmethod
    def get_name(cls) -> str:
        return "vlm_open_loop"

    @property
    def is_learning_based(self) -> bool:
        return True

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Adds the images and plans from the training dataset to the base
        prompt for use at test time!"""

        def _append_to_prompt_state_imgs_list(state: State) -> None:
            assert state.simulator_state is not None
            assert len(state.simulator_state["images"]) == num_imgs_per_state
            for img_num, img in enumerate(state.simulator_state["images"]):
                pil_img = PIL.Image.fromarray(img)  # type: ignore
                width, height = pil_img.size
                font_size = 15
                text = f"Demonstration {traj_num}, " + \
                    f"State {state_num}, Image {img_num}"
                draw = ImageDraw.Draw(pil_img)
                font = utils.get_scaled_default_font(draw, font_size)
                text_width, text_height = draw.textbbox((0, 0),
                                                        text,
                                                        font=font)[2:]
                # Create a new image with additional space for text!
                new_image = PIL.Image.new(
                    "RGB", (width, height + int(text_height) + 10), "white")
                new_image.paste(pil_img, (0, 0))
                draw = ImageDraw.Draw(new_image)
                text_x = (width - text_width) / 2
                text_y = height + 5
                draw.text((text_x, text_y), text, font=font, fill="black")
                # pylint:disable=protected-access
                self._prompt_state_imgs_list.append(
                    draw._image)  # type: ignore[attr-defined]
                # pylint: enable=protected-access

        def _generate_string_demonstration(segmented_traj: List[Segment],
                                           ll_traj: LowLevelTrajectory,
                                           traj_num: int) -> str:
            """Generate string-based demonstration similar to
            _get_transition_str in
            pp_online_predicate_invention_approach.py."""
            demo_str = ""
            traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
            goal_str = str(sorted(traj_goal))
            demo_str += (f"Demonstration {traj_num}, "
                         f"Goal: {goal_str}\n")

            state_hash_to_id: Dict[int, int] = {}

            for state_num, seg in enumerate(segmented_traj):
                # Get initial state info
                init_state = seg.states[0]
                init_state_hash = hash(init_state)
                if init_state_hash not in state_hash_to_id:
                    state_hash_to_id[init_state_hash] = len(state_hash_to_id)
                init_state_id = state_hash_to_id[init_state_hash]
                state_name = f"state_{init_state_id}"

                # Add state description
                if state_num == 0:
                    demo_str += f"Starting at {state_name} with state info:\n"
                else:
                    demo_str += f"Now at {state_name} with state info:\n"

                state_str = init_state.dict_str(
                    indent=2, use_object_id=CFG.rgb_observation)
                demo_str += f"{state_str}\n"

                # Add action
                action = seg.get_option()
                action_str = action.simple_str(
                    use_object_id=CFG.rgb_observation)
                demo_str += f"Action {state_num}: {action_str} was executed\n"

                # Add resulting state
                end_state = seg.states[-1]
                end_state_hash = hash(end_state)
                if end_state_hash not in state_hash_to_id:
                    state_hash_to_id[end_state_hash] = len(state_hash_to_id)
                end_state_id = state_hash_to_id[end_state_hash]
                end_state_name = f"state_{end_state_id}"
                demo_str += (f"This resulted in {end_state_name}"
                             " with state info:\n")
                end_state_str = end_state.dict_str(
                    indent=2, use_object_id=CFG.rgb_observation)
                demo_str += f"{end_state_str}\n\n"

            return demo_str

        if not CFG.vlm_open_loop_use_training_demos:
            return None

        # Handle both image and string-based demonstrations
        if not CFG.vlm_open_loop_no_image:
            # Original image-based demonstration logic
            assert dataset.trajectories[0].states[
                0].simulator_state is not None
            assert isinstance(
                dataset.trajectories[0].states[0].simulator_state["images"],
                List)
            num_imgs_per_state = len(
                dataset.trajectories[0].states[0].simulator_state["images"])

        segmented_trajs = [
            segment_trajectory(traj, self._initial_predicates)
            for traj in dataset.trajectories
        ]
        self._prompt_demos_str = ""

        for traj_num, seg_traj in enumerate(
                zip(segmented_trajs, dataset.trajectories)):
            segment_traj, ll_traj = seg_traj
            if not ll_traj.is_demo:
                continue

            if CFG.vlm_open_loop_no_image:
                # Use string-based demonstrations with detailed
                # state information
                self._prompt_demos_str += _generate_string_demonstration(
                    segment_traj, ll_traj, traj_num)
            else:
                # Both image and string-based demonstrations
                traj_goal = self._train_tasks[ll_traj.train_task_idx].goal
                self._prompt_demos_str += f"Demonstration {traj_num}, " + \
                    f"Goal: {str(sorted(traj_goal))}\n"
                self._prompt_demos_str += _generate_string_demonstration(
                    segment_traj, ll_traj, traj_num)
                assert len(segment_traj) > 0
                for state_num, seg in enumerate(segment_traj):
                    state = seg.states[0]
                    _append_to_prompt_state_imgs_list(state)
                    action = seg.get_option()
                    self._prompt_demos_str += f"Action {state_num}, from " + \
                        f"state {state_num} is {action}\n"
                # Make sure to append the final state of the final segment!
                state = seg.states[-1]  # pylint:disable=undefined-loop-variable
                _append_to_prompt_state_imgs_list(state)
        return None

    def _get_current_nsrts(self) -> Set[utils.NSRT]:
        """This method doesn't explicitly learn NSRTs, so we simply return the
        empty set."""
        return set()

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        try:
            option_plan = self._query_vlm_for_option_plan(task)
        except Exception as e:
            raise ApproachFailure(
                f"VLM failed to produce coherent option plan. Reason: {e}")

        policy = utils.option_plan_to_policy(
            option_plan,
            abstract_function=lambda s: utils.abstract(
                s, self._get_current_predicates()))

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _query_vlm_for_option_plan(self, task: Task) -> Sequence[_Option]:
        assert self._vlm is not None
        init_state = task.init
        assert init_state.simulator_state is not None
        if not CFG.vlm_open_loop_no_image:
            assert isinstance(init_state.simulator_state["images"], List)
            imgs = init_state.simulator_state["images"]
            pil_imgs = [
                PIL.Image.fromarray(img_arr)  # type: ignore
                for img_arr in imgs
            ]
            imgs_for_vlm = []
            for img_num, pil_img in enumerate(pil_imgs):
                draw = ImageDraw.Draw(pil_img)
                img_font = utils.get_scaled_default_font(draw, 10)
                img_with_txt = utils.add_text_to_draw_img(
                    draw, (50, 50),
                    f"Initial state to plan from, Image {img_num}", img_font)
                # pylint:disable=protected-access
                imgs_for_vlm.append(
                    img_with_txt._image)  # type: ignore[attr-defined]
                # pylint: enable=protected-access
        curr_options = sorted(self._initial_options)
        options_str = "\n".join(
            str(opt) + ", params_space=" + str(opt.params_space)
            for opt in curr_options)
        objects_list = sorted(set(task.init))
        objects_str = "\n".join(str(obj) for obj in objects_list)
        goal_expr_list = sorted(set(task.goal))
        type_hierarchy_str = utils.create_pddl_types_str(self._types)
        goal_str = "\n".join(str(obj) for obj in goal_expr_list)
        if not CFG.vlm_open_loop_use_training_demos:
            if CFG.vlm_open_loop_no_image:
                prompt = self.base_prompt.format(
                    init_state_str=init_state.dict_str(
                        indent=2, use_object_id=CFG.rgb_observation),
                    options=options_str,
                    typed_objects=objects_str,
                    type_hierarchy=type_hierarchy_str,
                    goal_str=goal_str)
            else:
                prompt = self.base_prompt.format(
                    options=options_str,
                    typed_objects=objects_str,
                    type_hierarchy=type_hierarchy_str,
                    goal_str=goal_str)
            vlm_output = self._vlm.sample_completions(
                prompt,
                imgs_for_vlm if not CFG.vlm_open_loop_no_image else None,
                temperature=CFG.vlm_temperature,
                seed=CFG.seed,
                num_completions=1)
        else:
            prompt = self.base_prompt.format(
                options=options_str,
                demonstration_trajs=self._prompt_demos_str,
                typed_objects=objects_str,
                type_hierarchy=type_hierarchy_str,
                goal_str=goal_str)
            vlm_output = self._vlm.sample_completions(
                prompt,
                self._prompt_state_imgs_list + imgs_for_vlm if \
                    not CFG.vlm_open_loop_no_image else None,
                temperature=CFG.vlm_temperature,
                seed=CFG.seed,
                num_completions=1)
        plan_prediction_txt = vlm_output[0]
        option_plan: List[_Option] = []
        try:
            start_index = plan_prediction_txt.index("Plan:\n") + len("Plan:\n")
            parsable_plan_prediction = plan_prediction_txt[start_index:]
        except ValueError:
            raise ValueError("VLM output is badly formatted; cannot "
                             "parse plan!")
        parsed_option_plan = utils.parse_model_output_into_option_plan(
            parsable_plan_prediction, objects_list, self._types,
            self._initial_options, True)
        for option_tuple in parsed_option_plan:
            params = np.array(option_tuple[2], dtype=np.float32)
            option_plan.append(option_tuple[0].ground(option_tuple[1], params))
        return option_plan
