"""Abstract base class for task generators."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from predicators.structs import EnvironmentTask


class TaskGenerator(ABC):
    """Abstract base class for generating environment tasks."""

    @abstractmethod
    def generate_tasks(self,
                       num_tasks: int,
                       rng: np.random.Generator,
                       log_debug: bool = False) -> List[EnvironmentTask]:
        """Generate a list of environment tasks.

        Args:
            num_tasks: Number of tasks to generate.
            rng: Random number generator.
            log_debug: Whether to print debug information.

        Returns:
            List of EnvironmentTask instances.
        """
        raise NotImplementedError
