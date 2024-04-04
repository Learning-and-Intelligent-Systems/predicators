"""This directory contains algorithms for STRIPS operator learning."""

import importlib
import pkgutil
from typing import TYPE_CHECKING, List, Set

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning.base_strips_learner import \
    BaseSTRIPSLearner
from predicators.src.settings import CFG
from predicators.src.structs import LowLevelTrajectory, \
    PartialNSRTAndDatastore, Predicate, Segment, Task

__all__ = ["BaseSTRIPSLearner"]

if not TYPE_CHECKING:
    # Load all modules so that utils.get_all_subclasses() works.
    for _, module_name, _ in pkgutil.walk_packages(__path__):
        if "__init__" not in module_name:
            # Important! We use an absolute import here to avoid issues
            # with isinstance checking when using relative imports.
            importlib.import_module(f"{__name__}.{module_name}")


def learn_strips_operators(
    trajectories: List[LowLevelTrajectory],
    train_tasks: List[Task],
    predicates: Set[Predicate],
    segmented_trajs: List[List[Segment]],
    verify_harmlessness: bool,
    verbose: bool = True,
) -> List[PartialNSRTAndDatastore]:
    """Learn strips operators on the given data segments.

    Return a list of PNADs with op (STRIPSOperator), datastore, and
    option_spec fields filled in (but not sampler).
    """
    for cls in utils.get_all_subclasses(BaseSTRIPSLearner):
        if not cls.__abstractmethods__ and \
           cls.get_name() == CFG.strips_learner:
            learner = cls(trajectories, train_tasks, predicates,
                          segmented_trajs, verify_harmlessness, verbose)
            break
    else:
        raise ValueError(f"Unrecognized STRIPS learner: {CFG.strips_learner}")
    return learner.learn()
