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

Example command for pybullet cover with no examples:
python predicators/main.py --env pybullet_cover_typed_options \
--approach vlm_open_loop --seed 0 \
--num_train_tasks 0 --num_test_tasks 1\
--make_failure_videos --render_init_state True \
--pybullet_camera_width 900 --pybullet_camera_height 900 \
--label_objs_with_id_name True \
--vlm_model_name gemini-1.5-flash --debug
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Set

import numpy as np
import PIL
from PIL import ImageDraw

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.settings import CFG
from predicators.structs import Action, Box, Dataset, ParameterizedOption, \
    Predicate, State, Task, Type, _Option


class VLMOpenLoopApproach(BilevelPlanningApproach):  # pragma: no cover
    """VLMOpenLoopApproach definition.

    NOTE: we don't test this approach because it requires an environment
    that has rendering, and our current envs that have this are burger
    and kitchen, which are slow/untestable on GitHub remote...
    """

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Set up the vlm and base prompt.
        self._vlm = utils.create_vlm_by_name(CFG.vlm_model_name)
        filepath_to_vlm_prompt = utils.get_path_to_predicators_root() + \
        "/predicators/approaches/vlm_planning_prompts/no_few_shot.txt"
        if CFG.vlm_open_loop_use_training_demos:
            filepath_to_vlm_prompt = utils.get_path_to_predicators_root() + \
        "/predicators/approaches/vlm_planning_prompts/few_shot.txt"
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
        if not CFG.vlm_open_loop_use_training_demos:
            return None
        # Crawl thru the dataset and pull out all the images.
        # For each image, add text to it in the bototm left indicating the
        # trajectory and timestep it's from.
        assert dataset.trajectories[0].states[0].simulator_state is not None
        assert isinstance(
            dataset.trajectories[0].states[0].simulator_state["images"], List)
        num_imgs_per_state = len(
            dataset.trajectories[0].states[0].simulator_state["images"])
        self._prompt_demos_str = ""
        for traj_num, traj in enumerate(dataset.trajectories):
            traj_goal = self._train_tasks[traj.train_task_idx].goal
            self._prompt_demos_str += f"Demonstration {traj_num}, " + \
                f"Goal: {str(sorted(traj_goal))}\n"
            for state_num, state in enumerate(traj.states):
                assert state.simulator_state is not None
                assert len(
                    state.simulator_state["images"]) == num_imgs_per_state
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
                        "RGB", (width, height + text_height + 10), "white")
                    new_image.paste(pil_img, (0, 0))
                    draw = ImageDraw.Draw(new_image)
                    text_x = (width - text_width) / 2
                    text_y = height + 5
                    draw.text((text_x, text_y), text, font=font, fill="black")
                    self._prompt_state_imgs_list.append(draw._image)  # pylint:disable=protected-access
            for action_num, action in enumerate(traj.actions):
                self._prompt_demos_str += f"Action {action_num}, from " + \
                    f"state {action_num} is {action.get_option()}\n"
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
        if type(init_state) is utils.PyBulletState:
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
                imgs_for_vlm.append(img_with_txt._image)  # pylint:disable=protected-access
        elif type(init_state) is utils.RawState:
            assert isinstance(init_state.state_image, PIL.Image.Image)
            imgs_for_vlm = [init_state.labeled_image]
        else:
            raise NotImplementedError

        curr_options = sorted(self._initial_options)
        options_str = "\n".join(
            str(opt) + ", params_space=" + str(opt.params_space)
            for opt in curr_options)
        objects_list = sorted(set(task.init))
        goal_expr_list = sorted(set(task.goal))
        if type(init_state) is utils.RawState:
            objects_str = "\n".join(obj.id_name + ":" + obj.type.name
                                    for obj in objects_list)
            goal_str = "\n".join(atom._id_name_str for atom in goal_expr_list)
            breakpoint()
        else:
            objects_str = "\n".join(str(obj) for obj in objects_list)
            goal_str = "\n".join(str(obj) for obj in goal_expr_list)
        type_hierarchy_str = utils.create_pddl_types_str(self._types)
        if not CFG.vlm_open_loop_use_training_demos:
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
        else:
            prompt = self.base_prompt.format(
                options=options_str,
                demonstration_trajs=self._prompt_demos_str,
                typed_objects=objects_str,
                type_hierarchy=type_hierarchy_str,
                goal_str=goal_str)
            vlm_output = self._vlm.sample_completions(
                prompt,
                self._prompt_state_imgs_list + imgs_for_vlm,
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
            option_plan.append(option_tuple[0].ground(
                option_tuple[1], np.array(option_tuple[2])))
        return option_plan
