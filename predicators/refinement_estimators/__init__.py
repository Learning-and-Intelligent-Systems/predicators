"""Handle creation of refinement cost estimators."""

import importlib
import pkgutil
from typing import TYPE_CHECKING

from predicators import utils
from predicators.refinement_estimators.base_refinement_estimator import \
    BaseRefinementEstimator

__all__ = ["BaseRefinementEstimator", "create_refinement_estimator"]

if not TYPE_CHECKING:
    # Load all modules so that utils.get_all_subclasses() works.
    for _, module_name, _ in pkgutil.walk_packages(__path__):
        if "__init__" not in module_name:
            # Important! We use an absolute import here to avoid issues
            # with isinstance checking when using relative imports.
            importlib.import_module(f"{__name__}.{module_name}")


def create_refinement_estimator(name: str) -> BaseRefinementEstimator:
    """Create an approach given its name."""
    for cls in utils.get_all_subclasses(BaseRefinementEstimator):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            estimator = cls()
            break
    else:
        raise NotImplementedError(f"Unknown refinement cost estimator: {name}")
    return estimator
