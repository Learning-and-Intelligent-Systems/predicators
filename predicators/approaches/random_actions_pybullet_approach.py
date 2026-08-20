"""An approach that takes small random perturbations from current joint
positions, suitable for PyBullet environments."""

from typing import Callable

import numpy as np

from predicators.approaches import BaseApproach
from predicators.structs import Action, State, Task
from predicators.utils import PyBulletState


class RandomActionsPyBulletApproach(BaseApproach):
    """Samples small random perturbations from current joint positions."""

    # Maximum perturbation per joint per step (radians).
    _PERTURBATION_SCALE = 0.05

    @classmethod
    def get_name(cls) -> str:
        return "random_actions_pybullet"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        low = self._action_space.low
        high = self._action_space.high

        def _policy(state: State) -> Action:
            assert isinstance(state, PyBulletState)
            current = np.array(state.joint_positions, dtype=np.float32)
            delta = self._rng.uniform(
                -self._PERTURBATION_SCALE,
                self._PERTURBATION_SCALE,
                size=current.shape,
            ).astype(np.float32)
            new_joints = np.clip(current + delta, low, high)
            return Action(new_joints)

        return _policy
