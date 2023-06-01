"""An execution monitor that leverages knowledge of the high-level plan to only
suggest replanning when the expected atoms check is not met."""

import logging
from typing import Set

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.settings import CFG
from predicators.structs import State


class ExpectedAtomsExecutionMonitor(BaseExecutionMonitor):
    """An execution monitor that only suggests replanning when we're doing
    bilevel planning and the expected atoms check fails."""

    @classmethod
    def get_name(cls) -> str:
        return "expected_atoms"

    def step(self, state: State) -> bool:
        # This monitor only makes sense to use with an oracle
        # bilevel planning approach.
        assert CFG.approach == "oracle"
        assert len(self._approach_info) > self._curr_plan_timestep
        assert isinstance(self._approach_info[0], Set)
        curr_env = get_or_create_env(CFG.env)
        predicates = curr_env.predicates
        curr_atoms = utils.abstract(state, predicates)
        # If the expected atoms are a subset of the current atoms, then
        # we don't have to replan.
        if self._approach_info[self._curr_plan_timestep].issubset(curr_atoms):
            self._curr_plan_timestep += 1
            return False
        logging.info(
            "Expected Atoms Check Execution Failure.")  # pragma: no cover
        logging.info(
            curr_atoms -
            self._approach_info[self._curr_plan_timestep])  # pragma: no cover
        self._curr_plan_timestep += 1  # pragma: no cover
        return True  # pragma: no cover
