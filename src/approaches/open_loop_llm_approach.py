"""Open-loop large language model (LLM) meta-controller approach.

Example command line:
    export OPENAI_API_KEY=<your API key>
    python src/main.py --approach open_loop_llm --seed 0 \
        --env pddl_blocks_procedural_tasks \
        --strips_learner oracle --num_train_tasks 3
"""
from __future__ import annotations

import abc
import functools
import logging
import os
from typing import Collection, Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple

import dill as pkl
import openai

from predicators.src import utils
from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.nsrt_metacontroller_approach import \
    NSRTMetacontrollerApproach
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Box, Dataset, \
    GroundAtom, GroundAtomTrajectory, LDLRule, LiftedAtom, \
    LiftedDecisionList, LowLevelTrajectory, Object, ParameterizedOption, \
    Predicate, State, Task, Type, Variable, _GroundNSRT


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
        abstract_plan = self._llm_prediction_to_abstract_plan(llm_prediction,
            objects)
        # If we failed to find a nontrivial plan, give up.
        if not abstract_plan:
            raise ApproachFailure("LLM did not predict an abstract plan.")
        # Otherwise, we succeeded, so save the plan and take the first step.
        memory["abstract_plan"] = abstract_plan
        return memory["abstract_plan"].pop(0)

    def _predict_llm(self, prompt: str) -> str:
        """Query the LLM."""

        # TODO: Uncomment this for the real deal.
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        # response = openai.Completion.create(model=CFG.llm_model_name,
        #     prompt=prompt, temperature=0, max_tokens=CFG.llm_max_tokens)
        # assert len(response["choices"]) == 1
        # text_response = response["choices"][0]["text"]

        # This is an example response for debugging. TODO remove.
        text_response = ' unstack(b2:block, b1:block)\n  unstack(b1:block, b4:block)\n  stack(b1:block, b4:block)\n  pick-up(b4:block)\n  unstack(b3:block, b2:block)\n  stack(b3:block, b1:block)\n  unstack(b1:block, b2:block)\n  stack(b1:block, b0:block)\n  pick-up(b0:block)\n  unstack(b4:block, b3:block)\n  stack(b4:block, b0:block)\n  unstack(b3:block, b2:block)\n  unstack(b2:block, b1:block)\n  unstack(b1:block, b0:block)\n  unstack(b0:block, b4:block)\n  handempty()\n  on(b0:block, b3:block)\n  on(b4:block, b3:block)\n  ontable(b'

        return text_response

    def _llm_prediction_to_abstract_plan(self, llm_prediction: str, objects: Collection[Object]) -> List[_GroundNSRT]:
        """Convert the output of the LLM into an abstract plan."""
        nsrt_name_to_nsrt = {n.name: n for n in self._get_current_nsrts()}
        obj_name_to_obj = {o.name: o for o in objects}
        import ipdb; ipdb.set_trace()

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
