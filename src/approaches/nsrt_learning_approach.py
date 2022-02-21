"""A TAMP approach that learns NSRTs.

In contrast to other approaches, this approach does not attempt to learn
new predicates or options.
"""

from typing import Set, List, Sequence, Optional
import dill as pkl
from gym.spaces import Box
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import Dataset, NSRT, ParameterizedOption, \
    Predicate, Type, Task, LowLevelTrajectory
from predicators.src.nsrt_learning.nsrt_learning_main import \
    learn_nsrts_from_data
from predicators.src.settings import CFG
from predicators.src import utils


class NSRTLearningApproach(TAMPApproach):
    """A TAMP approach that learns NSRTs."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._nsrts: Set[NSRT] = set()

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # The only thing we need to do here is learn NSRTs,
        # which we split off into a different function in case
        # subclasses want to make use of it.
        self._learn_nsrts(dataset.trajectories, online_learning_cycle=None)

    def _learn_nsrts(self, trajectories: Sequence[LowLevelTrajectory],
                     online_learning_cycle: Optional[int]) -> None:
        self._nsrts = learn_nsrts_from_data(
            trajectories,
            self._train_tasks,
            self._get_current_predicates(),
            sampler_learner=CFG.sampler_learner)
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.NSRTs", "wb") as f:
            pkl.dump(self._nsrts, f)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.NSRTs", "rb") as f:
            self._nsrts = pkl.load(f)
        if CFG.pretty_print_when_loading:
            preds, _ = utils.extract_preds_and_types(self._nsrts)
            name_map = {}
            print("Invented predicates:")
            for idx, pred in enumerate(
                    sorted(set(preds.values()) - self._initial_predicates)):
                vars_str, body_str = pred.pretty_str()
                print("\t", f"P{idx+1}({vars_str}) ≜ {body_str}")
                name_map[body_str] = f"P{idx+1}"
        print("\n\nLoaded NSRTs:")
        for nsrt in sorted(self._nsrts):
            if CFG.pretty_print_when_loading:
                print(nsrt.pretty_str(name_map))
            else:
                print(nsrt)
        print()
        # Seed the option parameter spaces after loading.
        for nsrt in self._nsrts:
            nsrt.option.params_space.seed(CFG.seed)
