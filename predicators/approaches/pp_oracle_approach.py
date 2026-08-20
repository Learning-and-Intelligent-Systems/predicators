"""Oracle bilevel process planning approach."""
from typing import List, Optional, Set

from gym.spaces import Box

from predicators.approaches.process_planning_approach import \
    BilevelProcessPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, CausalProcess, ParameterizedOption, \
    Predicate, Task, Type


class OracleBilevelProcessPlanningApproach(BilevelProcessPlanningApproach):
    """A bilevel planning approach that uses hand-specified processes."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 processes: Optional[Set[CausalProcess]] = None,
                 option_model: Optional[_OptionModelBase] = None) -> None:
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)
        # The optional helper types/predicates (e.g. the domino grid) are
        # added by the base class because _use_gt_helpers() returns True here.

        if processes is None:
            # use only_endogenous for the no_invent baseline
            processes = get_gt_processes(
                CFG.env,
                self._initial_predicates,
                self._initial_options,
                only_endogenous=CFG.running_no_invent_baseline)

        # Set all processes' strength parameters to 1 if flag is enabled
        if CFG.process_planning_set_parameters_one:
            import torch  # pylint: disable=import-outside-toplevel
            modified_processes = set()
            for process in processes:
                # Create a copy with strength set to 1
                strength_params = torch.tensor([1.0])
                delay_params = torch.ones(
                    len(process.delay_distribution.get_parameters()))
                process._set_parameters(
                    torch.cat([strength_params, delay_params]).tolist())
                modified_processes.add(process)
            processes = modified_processes

        self._processes = processes

    @classmethod
    def get_name(cls) -> str:
        return "oracle_process_planning"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _use_gt_helpers(self) -> bool:
        # The oracle always uses the ground-truth helper types/predicates/
        # objects (e.g. the domino grid), independent of the CFG flag.
        return True

    def _get_current_processes(self) -> Set[CausalProcess]:
        return self._processes

    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        return set()
