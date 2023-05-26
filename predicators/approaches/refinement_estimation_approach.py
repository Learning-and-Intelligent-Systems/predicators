"""A bilevel planning approach that uses a refinement cost estimator.

Generates N proposed skeletons and then ranks them based on a given
refinement cost estimation function (e.g. a heuristic, learned model),
attempting to refine them in this order.
"""

import logging
from pathlib import Path
from typing import Any, List, Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.approaches.oracle_approach import OracleApproach
from predicators.refinement_estimators import BaseRefinementEstimator, \
    create_refinement_estimator
from predicators.settings import CFG
from predicators.structs import NSRT, Metrics, ParameterizedOption, \
    Predicate, Task, Type, _GroundNSRT, _Option


class RefinementEstimationApproach(OracleApproach):
    """A bilevel planning approach that uses a refinement cost estimator."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, task_planning_heuristic,
                         max_skeletons_optimized)
        assert (CFG.refinement_estimation_num_skeletons_generated <=
                CFG.sesame_max_skeletons_optimized), \
               "refinement_estimation_num_skeletons_generated should not be" \
               "greater than sesame_max_skeletons_optimized"
        estimator_name = CFG.refinement_estimator
        self._refinement_estimator = create_refinement_estimator(
            estimator_name)
        # If the refinement estimator is learning based, try to load
        # trained model if the file exists.
        if self._refinement_estimator.is_learning_based:
            config_path_str = utils.get_config_path_str()
            model_file = f"{estimator_name}_{config_path_str}.estimator"
            model_file_path = Path(CFG.approach_dir) / model_file
            try:
                self._refinement_estimator.load_model(model_file_path)
                logging.info(f"Loaded trained estimator model "
                             f"from {model_file_path}")
            except FileNotFoundError:
                logging.info(f"Could not find estimator model file "
                             f"at {model_file_path}")

    @classmethod
    def get_name(cls) -> str:
        return "refinement_estimation"

    def _run_sesame_plan(
            self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
            timeout: float, seed: int,
            **kwargs: Any) -> Tuple[List[_Option], List[_GroundNSRT], Metrics]:
        """Generates a plan choosing the best skeletons based on a given
        refinement cost estimator."""
        result = super()._run_sesame_plan(
            task,
            nsrts,
            preds,
            timeout,
            seed,
            refinement_estimator=self._refinement_estimator,
            **kwargs)
        return result

    @property
    def refinement_estimator(self) -> BaseRefinementEstimator:
        """Getter for _refinement_estimator."""
        return self._refinement_estimator
