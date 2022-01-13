"""TODO"""

from typing import Sequence, Tuple
from dataclasses import dataclass
from sklearn.gaussian_process import GaussianProcessClassifier
from predicators.src.structs import Array, Object, State


@dataclass(frozen=True, eq=False, repr=False)
class GaussianProcessPredicateClassifier:
    """A convenience class for holding the GP model underlying a learned
    predicate."""
    _model: GaussianProcessClassifier

    def classifier(self, state: State, objects: Sequence[Object]) -> bool:
        """The classifier corresponding to the given model.

        May be used as the _classifier field in a Predicate.
        """
        v = state.vec(objects).reshape(1, -1)
        prediction = self._model.predict(v)
        assert prediction.shape == (1,)
        assert prediction[0] in [False, True]
        # prob = self._model.predict_proba(v)
        # assert prob.shape == (1, 2)
        return prediction[0]
