"""Process learning and planning approach."""
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import dill as pkl
from gym.spaces import Box

from predicators import utils
from predicators.approaches.pp_param_learning_approach import \
    ParamLearningBilevelProcessPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.nsrt_learning.process_learning_main import \
    learn_processes_from_data
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import CausalProcess, Dataset, GroundAtomTrajectory, \
    LiftedAtom, LowLevelTrajectory, ParameterizedOption, Predicate, Task, \
    Type


class ProcessLearningAndPlanningApproach(
        ParamLearningBilevelProcessPlanningApproach):
    """A bilevel planning approach that learns processes."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 option_model: Optional[_OptionModelBase] = None):
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)
        if CFG.only_learn_exogenous_processes:
            self._processes = get_gt_processes(CFG.env,
                                               self._initial_predicates,
                                               self._initial_options,
                                               only_endogenous=True)
        else:
            # Learn all
            self._processes: Set[CausalProcess] = set()
        self._proc_name_to_results: Dict[str, List[Tuple[float,
                                                         FrozenSet[LiftedAtom],
                                                         Tuple,
                                                         CausalProcess]]] = {}

    @classmethod
    def get_name(cls) -> str:
        return "process_learning_and_planning"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Learn models from the offline datasets."""
        self._learn_processes(dataset.trajectories,
                              online_learning_cycle=None,
                              annotations=(dataset.annotations if
                                           dataset.has_annotations else None))
        # Optional: learn process parameters
        if CFG.learn_process_parameters:
            self._learn_process_parameters(dataset.trajectories)

    def _learn_processes(self,
                         trajectories: List[LowLevelTrajectory],
                         online_learning_cycle: Optional[int],
                         annotations: Optional[List[Any]] = None) -> None:
        """Learn processes from the offline datasets."""
        dataset_fname, _ = utils.create_dataset_filename_str(
            saving_ground_atoms=True,
            online_learning_cycle=online_learning_cycle)
        ground_atom_dataset: Optional[List[GroundAtomTrajectory]] = None
        if CFG.load_atoms:
            ground_atom_dataset = utils.load_ground_atom_dataset(
                dataset_fname, trajectories)
        elif CFG.save_atoms:
            ground_atom_dataset = utils.create_ground_atom_dataset(
                trajectories, self._get_current_predicates())
        self._processes, self._proc_name_to_results = \
            learn_processes_from_data(trajectories,
                                self._train_tasks,
                                self._get_current_predicates(),
                                self._initial_options,
                                self._action_space,
                                ground_atom_dataset,
                                sampler_learner=CFG.sampler_learner,
                                annotations=annotations,
                                current_processes=self._get_current_processes(),
                                online_learning_cycle=online_learning_cycle,)

        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.PROCes", "wb") as f:
            pkl.dump(self._processes, f)
