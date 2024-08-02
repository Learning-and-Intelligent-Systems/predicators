"""Open-loop vision-language model (VLM) meta-controller approach.
The idea here is that the VLM is given a set of training trajectories
consisting of the initial state image as well as the low-level
object-oriented state, and then asked to output a full plan.
TODO: we probably want to ensure the model is prompted with entire
trajectories (so all intermediate states), and not just the
initial state and then the plan. we also probably want to not
have to do sampler learning somehow?

TODO: example command.
"""

from __future__ import annotations

from typing import Collection, Dict, Iterator, List, Optional, Sequence, Set, \
    Tuple, Callable

from predicators.approaches import ApproachFailure
from predicators.approaches.bilevel_planning_approach import BilevelPlanningApproach
from predicators.planning import task_plan_with_option_plan_constraint
from predicators.settings import CFG
from predicators.structs import Box, Dataset, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, _Option, \
    Action
from predicators import utils


class VLMOpenLoopApproach(BilevelPlanningApproach):
    """VLMOpenLoopApproach definition."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Set up the vlm and base prompt.
        self._vlm = utils.create_vlm_by_name(CFG.vlm_model_name)
        self._base_prompt_txt = ""
        self._base_prompt_imgs = []
        # TODO: probably want to parse the base prompt here, and add
        # available options into the prompt!

    @classmethod
    def get_name(cls) -> str:
        return "vlm_open_loop"
    
    @property
    def is_learning_based(self) -> bool:
        return True

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Adds the images and plans from the training dataset
        to the base prompt for use at test time!"""
        pass

    def _get_current_nsrts(self) -> Set[utils.NSRT]:
        """This method doesn't explicitly learn NSRTs, so we simply
        return the empty set."""
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
        filepath_to_vlm_prompt = utils.get_path_to_predicators_root() + \
        "/predicators/approaches/vlm_planning_prompts/no_few_shot.txt"
        with open(filepath_to_vlm_prompt, "r", encoding="utf-8") as f:
            base_prompt = f.read()
        options_str = "\n".join(str(opt) for opt in curr_options)
        objects_list = sorted(set(task.init))
        objects_str = "\n".join(str(obj) for obj in objects_list)
        goal_expr_list = sorted(set(task.goal))
        goal_str = "\n".join(str(obj) for obj in goal_expr_list)
        # TODO: generate the type hierarchy (which will require adjusting the prompt), then call the VLM
        # TODO: parse the VLM's responses into an option plan!
