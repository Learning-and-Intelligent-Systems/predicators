"""A learning-based refinement cost estimator that trains and stores a separate
cost predictor for each unique (skeleton, atoms_sequence) key."""

import abc
from pathlib import Path
from typing import Dict, FrozenSet, Generic, List, Optional, Set, Tuple, \
    TypeVar

import dill as pkl

from predicators.refinement_estimators import BaseRefinementEstimator
from predicators.structs import GroundAtom, Task, _GroundNSRT

# Type of the (skeleton, atoms_sequence) key for model dictionary
# which converts both of them to be immutable
ModelDictKey = Tuple[Tuple[_GroundNSRT, ...],  # skeleton converted to tuple
                     Tuple[FrozenSet[GroundAtom], ...]  # atoms_sequence
                     ]
Model = TypeVar('Model')


class PerSkeletonRefinementEstimator(BaseRefinementEstimator, Generic[Model]):
    """A refinement cost estimator that trains a separate cost predictor per
    skeleton, which is given as a (skeleton, atoms_sequence) pair."""

    def __init__(self) -> None:
        super().__init__()
        # _model_dict maps immutable skeleton atoms_sequence pair to model
        self._model_dict: Optional[Dict[ModelDictKey, Model]] = None

    @property
    def is_learning_based(self) -> bool:
        return True

    def get_cost(self, initial_task: Task, skeleton: List[_GroundNSRT],
                 atoms_sequence: List[Set[GroundAtom]]) -> float:
        assert self._model_dict is not None, "Need to train"
        key = self._immutable_model_dict_key(skeleton, atoms_sequence)
        # If key isn't in dictionary (no data for skeleton), cost is infinite
        if key not in self._model_dict:
            return float('inf')
        model = self._model_dict[key]
        cost = self._model_predict(model, initial_task)
        return cost

    @abc.abstractmethod
    def _model_predict(self, model: Model, initial_task: Task) -> float:
        """Get the cost prediction from a model using the initial task as
        input."""

    @staticmethod
    def _immutable_model_dict_key(
            skeleton: List[_GroundNSRT],
            atoms_sequence: List[Set[GroundAtom]]) -> ModelDictKey:
        """Converts a skeleton and atoms_sequence into immutable types to use
        as a key for the model dictionary."""
        return (tuple(skeleton),
                tuple(frozenset(atoms) for atoms in atoms_sequence))

    def save_model(self, filepath: Path) -> None:
        with open(filepath, "wb") as f:
            pkl.dump(self._model_dict, f)

    def load_model(self, filepath: Path) -> None:
        with open(filepath, "rb") as f:
            self._model_dict = pkl.load(f)
        # Run every model once to avoid weird delay issue
        if self._model_dict is not None:
            for v in self._model_dict.values():
                self._model_predict(v, self._env.get_train_tasks()[0].task)
