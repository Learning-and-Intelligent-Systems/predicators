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
    ParameterizedOption, Predicate, STRIPSOperator, VarToObjSub, LowLevelTrajectory, Task, Segment


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
        "/predicators/approaches/nsrt_learning/strips_learning/llm_op_learning_prompts/naive_no_examples.txt"
        with open(prompt_file,
                  "r",
                  encoding="utf-8") as f:
            self._base_prompt = f.read()

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
        # Add all the types in the env to the prompt.
        all_types: Set[Type] = set()
        for pred in self._predicates:
            all_types |= set(pred.types)
        prompt += "Types:\n"
        for t in sorted(all_types):
            prompt += f"- {t.name}\n"
        prompt += "Predicates:\n"
        for pred in self._predicates:
            prompt += f"- {pred.pddl_str()}"
        import ipdb; ipdb.set_trace()

    @classmethod
    def get_name(cls) -> str:
        return "llm"
