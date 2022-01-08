"""A TAMP approach that learns NSRTs.

In contrast to other approaches, this approach does not attempt to learn
new predicates or options.
"""

import pickle as pkl
from typing import Callable, Set, List
from gym.spaces import Box
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import Dataset, NSRT, ParameterizedOption, \
    State, Action, Predicate, Type, Task
from predicators.src.nsrt_learning import learn_nsrts_from_data
from predicators.src.settings import CFG
from predicators.src import utils


class NSRTLearningApproach(TAMPApproach):
    """A TAMP approach that learns NSRTs."""

    def __init__(self, simulator: Callable[[State, Action], State],
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box) -> None:
        super().__init__(simulator, initial_predicates, initial_options, types,
                         action_space)
        self._nsrts: Set[NSRT] = set()
        self._dataset: Dataset = []

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
        assert self._nsrts, "NSRTs not learned"
        return self._nsrts

    def learn_from_offline_dataset(self, dataset: Dataset,
                                   train_tasks: List[Task]) -> None:
        # The only thing we need to do here is learn NSRTs,
        # which we split off into a different function in case
        # subclasses want to make use of it.
        self._dataset.extend(dataset)
        del dataset
        self._learn_nsrts()

    def _learn_nsrts(self) -> None:
        self._nsrts = learn_nsrts_from_data(
            self._dataset,
            self._get_current_predicates(),
            sampler_learner=CFG.sampler_learner)
        # save_path = utils.get_save_path_str()
        # with open(f"{save_path}.NSRTs", "wb") as f:
        #     pkl.dump(self._nsrts, f)

    def load(self) -> None:
        save_path = utils.get_save_path_str()
        with open(f"{save_path}.NSRTs", "rb") as f:
            self._nsrts = pkl.load(f)
        print("\n\nLoaded NSRTs:")
        for nsrt in sorted(self._nsrts):
            print(nsrt)
        print()
        # Seed the option parameter spaces after loading.
        for nsrt in self._nsrts:
            nsrt.option.params_space.seed(CFG.seed)
