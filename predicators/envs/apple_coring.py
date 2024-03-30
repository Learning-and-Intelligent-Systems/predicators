"""Dummy apple-coring environment to test predicate invention.
"""

from typing import ClassVar, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class AppleCoringEnv(BaseEnv):
    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._gripper_type = Type("gripper",
                              ["open_percentage"])
        self._apple_type = Type("apple", [])
        self._apple_type = Type("slicing_tool", [])
        self._apple_type = Type("plate", [])

        # Predicates


    @classmethod
    def get_name(cls) -> str:
        return "apple_coring"

    def simulate(self, state: State, action: Action) -> State:
        raise ValueError("simulate shouldn't be getting called!")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_sat_lst=CFG.satellites_num_sat_train,
                               num_obj_lst=CFG.satellites_num_obj_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               num_sat_lst=CFG.satellites_num_sat_test,
                               num_obj_lst=CFG.satellites_num_obj_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Sees, self._CalibrationTarget, self._IsCalibrated,
            self._HasCamera, self._HasInfrared, self._HasGeiger,
            self._ShootsChemX, self._ShootsChemY, self._HasChemX,
            self._HasChemY, self._CameraReadingTaken,
            self._InfraredReadingTaken, self._GeigerReadingTaken
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {
            self._CameraReadingTaken, self._InfraredReadingTaken,
            self._GeigerReadingTaken
        }

    @property
    def types(self) -> Set[Type]:
        return {self._sat_type, self._obj_type}

    @property
    def action_space(self) -> Box:
        # [cur sat x, cur sat y, obj x, obj y, target sat x, target sat y,
        # calibrate, shoot Chemical X, shoot Chemical Y, use instrument]
        return Box(low=0.0, high=1.0, shape=(10, ), dtype=np.float32)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise ValueError("shouldn't be trying to render env at any point!")

    def _get_tasks(self, num: int, num_sat_lst: List[int],
                   num_obj_lst: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        return []
