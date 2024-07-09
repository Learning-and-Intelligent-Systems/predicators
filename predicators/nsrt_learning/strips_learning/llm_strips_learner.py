"""Approaches that use an LLM to learn STRIPS operators instead of
performing symbolic learning of any kind."""

import abc
import functools
import logging
from collections import defaultdict
from typing import List, Set, cast, Optional, Any

from predicators import utils
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import PNAD, Datastore, DummyOption, LiftedAtom, \
    ParameterizedOption, Predicate, STRIPSOperator, VarToObjSub, LowLevelTrajectory, Task, Segment, Type


class LLMStripsLearner(BaseSTRIPSLearner):
    """Base class for all LLM-based learners."""

    def __init__(self,
                 trajectories: List[LowLevelTrajectory],
                 train_tasks: List[Task],
                 predicates: Set[Predicate],
                 segmented_trajs: List[List[Segment]],
                 verify_harmlessness: bool,
                 annotations: Optional[List[Any]],
                 verbose: bool = True) -> None:
        super().__init__(trajectories, train_tasks, predicates, segmented_trajs, verify_harmlessness, annotations, verbose)
        self._llm = utils.create_llm_by_name(CFG.llm_model_name)
        prompt_file = utils.get_path_to_predicators_root() + \
        "/predicators/nsrt_learning/strips_learning/llm_op_learning_prompts/naive_no_examples.txt"
        with open(prompt_file,
                  "r",
                  encoding="utf-8") as f:
            self._base_prompt = f.read()

    def _get_all_types_from_preds(self) -> Set[Type]:
        all_types: Set[Type] = set()
        for pred in self._predicates:
            all_types |= set(pred.types)
        return all_types

    def _get_all_options_from_segs(self) -> Set[ParameterizedOption]:
        # NOTE: this assumes all segments in self._segmented_trajs
        # have options associated with them.
        all_options: Set[ParameterizedOption] = set()
        for seg_traj in self._segmented_trajs:
            for seg in seg_traj:
                all_options.add(seg.get_option().parent)
        return all_options

    def _learn(self) -> List[PNAD]:
        # TODO: Steps
        # (1) prompt llm
            # (a) collect types
            # (b) collect predicates
            # (c) collect skills
        # (2) parse out ridiculous and malformed ops
        # (3) create PNADs
        # (4) add datastores based on matching?
        prompt = self._base_prompt
        all_types = self._get_all_types_from_preds()
        prompt += "Types:\n"
        for t in sorted(all_types):
            prompt += f"- {t.name}\n"
        prompt += "\nPredicates:\n"
        for pred in sorted(self._predicates):
            prompt += f"- {pred.pddl_str()}\n"
        prompt += "\nActions:\n"
        for act in sorted(self._get_all_options_from_segs()):
            prompt += act.pddl_str() + "\n"
        prompt += "\nTrajectory data:\n"
        for i, seg_traj in enumerate(self._segmented_trajs):
            curr_goal = self._train_tasks[i].goal
            prompt += f"Trajectory {i} (Goal: {str(curr_goal)}):\n"
            for t, seg in enumerate(seg_traj):
                # TODO: get segment init state and action and add to prompt
                # Remember to get final state at the very end

                import ipdb; ipdb.set_trace()

    @classmethod
    def get_name(cls) -> str:
        return "llm"
