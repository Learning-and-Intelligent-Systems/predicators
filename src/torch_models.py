"""Models useful for classification/regression.

Note: to promote modularity, this file should NOT import CFG.
"""

import abc
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

from predicators.src.structs import Array, Object, State

torch.use_deterministic_algorithms(mode=True)  # type: ignore

################################ Base Classes #################################


class Regressor(abc.ABC):
    """ABC for regressor classes.

    All regressors normalize the input and output data.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        # Set in fit().
        self._x_dim = 0
        self._y_dim = 0
        self._input_shift = np.zeros(1, dtype=np.float32)
        self._input_scale = np.zeros(1, dtype=np.float32)
        self._output_shift = np.zeros(1, dtype=np.float32)
        self._output_scale = np.zeros(1, dtype=np.float32)

    def fit(self, X: Array, Y: Array) -> None:
        """Train the regressor on the given data.

        X and Y are both two-dimensional.
        """
        num_data, self._x_dim = X.shape
        _, self._y_dim = Y.shape
        assert Y.shape[0] == num_data
        logging.info(f"Training {self.__class__.__name__} on {num_data} "
                     "datapoints")
        X, self._input_shift, self._input_scale = _normalize_data(X)
        Y, self._output_shift, self._output_scale = _normalize_data(Y)
        self._fit(X, Y)

    def predict(self, x: Array) -> Array:
        """Return a prediction for the given datapoint.

        x is single-dimensional.
        """
        assert x.shape == (self._x_dim, )
        # Normalize.
        x = (x - self._input_shift) / self._input_scale
        # Make prediction.
        y = self._predict(x)
        assert y.shape == (self._y_dim, )
        # Denormalize.
        y = (y * self._output_scale) + self._output_shift
        return y

    @abc.abstractmethod
    def _fit(self, X: Array, Y: Array) -> None:
        """Train the regressor on normalized data."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _predict(self, x: Array) -> Array:
        """Return a normalized prediction for the normalized input."""
        raise NotImplementedError("Override me!")


class PyTorchRegressor(Regressor, nn.Module):
    """ABC for PyTorch regression models."""

    def __init__(self, seed: int, max_train_iters: int, clip_gradients: bool,
                 clip_value: float, learning_rate: float) -> None:
        torch.manual_seed(seed)
        Regressor.__init__(self, seed)
        nn.Module.__init__(self)  # type: ignore
        self._max_train_iters = max_train_iters
        self._clip_gradients = clip_gradients
        self._clip_value = clip_value
        self._learning_rate = learning_rate

    @abc.abstractmethod
    def forward(self, tensor_X: Tensor) -> Tensor:
        """Pytorch forward method."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _initialize_net(self) -> None:
        """Initialize the network once the data dimensions are known."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Create the loss function used for optimization."""
        raise NotImplementedError("Override me!")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create an optimizer after the model is initialized."""
        return optim.Adam(self.parameters(), lr=self._learning_rate)

    def _fit(self, X: Array, Y: Array) -> None:
        # Initialize the network.
        self._initialize_net()
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Create the optimizer.
        optimizer = self._create_optimizer()
        # Convert data to tensors.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32))
        tensor_Y = torch.from_numpy(np.array(Y, dtype=np.float32))
        batch_generator = _single_batch_generator(tensor_X, tensor_Y)
        # Run training.
        _train_predictive_pytorch_model(self,
                                        loss_fn,
                                        optimizer,
                                        batch_generator,
                                        max_iters=self._max_train_iters,
                                        clip_gradients=self._clip_gradients,
                                        clip_value=self._clip_value)

    def _predict(self, x: Array) -> Array:
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32))
        tensor_X = tensor_x.unsqueeze(dim=0)
        tensor_Y = self(tensor_X)
        tensor_y = tensor_Y.squeeze(dim=0)
        y = tensor_y.detach().numpy()  # type: ignore
        return y


class BinaryClassifier(abc.ABC):
    """ABC for binary classifier classes.

    All binary classifiers normalize the input data.
    """

    def __init__(self, seed: int, balance_data: bool) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._balance_data = balance_data
        # Set in fit().
        self._x_dim = 0
        self._input_shift = np.zeros(1, dtype=np.float32)
        self._input_scale = np.zeros(1, dtype=np.float32)
        self._do_single_class_prediction = False
        self._predicted_single_class = False

    def fit(self, X: Array, y: Array) -> None:
        """Train the classifier on the given data.

        X is two-dimensional, y is one-dimensional.
        """
        num_data, self._x_dim = X.shape
        assert y.shape == (num_data, )
        logging.info(f"Training {self.__class__.__name__} on {num_data} "
                     "datapoints")
        # If there is only one class in the data, then there's no point in
        # learning, since any predictions other than that one class could
        # only be strange generalization issues.
        if np.all(y == 0):
            self._do_single_class_prediction = True
            self._predicted_single_class = False
            return
        if np.all(y == 1):
            self._do_single_class_prediction = True
            self._predicted_single_class = True
            return
        # Balance the classes.
        if self._balance_data and len(y) // 2 > sum(y):
            old_len = len(y)
            X, y = _balance_binary_classification_data(X, y, self._rng)
            logging.info(f"Reduced dataset size from {old_len} to {len(y)}")
        X, self._input_shift, self._input_scale = _normalize_data(X)
        self._fit(X, y)

    def classify(self, x: Array) -> bool:
        """Return a predicted class for the given datapoint.

        x is single-dimensional.
        """
        assert x.shape == (self._x_dim, )
        if self._do_single_class_prediction:
            return self._predicted_single_class
        # Normalize.
        x = (x - self._input_shift) / self._input_scale
        # Make prediction.
        return self._classify(x)

    @abc.abstractmethod
    def _fit(self, X: Array, y: Array) -> None:
        """Train the regressor on normalized data."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _classify(self, x: Array) -> bool:
        """Return a predicted class for the normalized input."""
        raise NotImplementedError("Override me!")


class PyTorchClassifier(BinaryClassifier, nn.Module):
    """ABC for PyTorch binary classification models."""

    def __init__(self, seed: int, balance_data: bool, max_train_iters: int,
                 learning_rate: float, n_iter_no_change: int) -> None:
        torch.manual_seed(seed)
        BinaryClassifier.__init__(self, seed, balance_data)
        nn.Module.__init__(self)  # type: ignore
        self._max_train_iters = max_train_iters
        self._learning_rate = learning_rate
        self._n_iter_no_change = n_iter_no_change

    @abc.abstractmethod
    def forward(self, tensor_X: Tensor) -> Tensor:
        """Pytorch forward method."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _initialize_net(self) -> None:
        """Initialize the network once the data dimensions are known."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Create the loss function used for optimization."""
        raise NotImplementedError("Override me!")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create an optimizer after the model is initialized."""
        return optim.Adam(self.parameters(), lr=self._learning_rate)

    def _fit(self, X: Array, y: Array) -> None:
        # Initialize the network.
        self._initialize_net()
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Create the optimizer.
        optimizer = self._create_optimizer()
        # Convert data to tensors.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32))
        tensor_y = torch.from_numpy(np.array(y, dtype=np.float32))
        batch_generator = _single_batch_generator(tensor_X, tensor_y)
        # Run training.
        _train_predictive_pytorch_model(
            self,
            loss_fn,
            optimizer,
            batch_generator,
            max_iters=self._max_train_iters,
            n_iter_no_change=self._n_iter_no_change)

    def _forward_single_input_np(self, x: Array) -> float:
        """Helper for _classify() and predict_proba()."""
        assert x.shape == (self._x_dim, )
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32))
        tensor_X = tensor_x.unsqueeze(dim=0)
        tensor_Y = self(tensor_X)
        tensor_y = tensor_Y.squeeze(dim=0)
        y = tensor_y.detach().numpy()  # type: ignore
        proba = y.item()
        assert 0 <= proba <= 1
        return proba

    def _classify(self, x: Array) -> bool:
        return self._forward_single_input_np(x) > 0.5

    def predict_proba(self, denorm_x: Array) -> float:
        """Get the predicted probability that the input classifies to 1.

        The input is unnormalized.
        """
        x = (denorm_x - self._input_shift) / self._input_scale
        return self._forward_single_input_np(x)


################################# Regressors ##################################


class MLPRegressor(PyTorchRegressor):
    """A basic multilayer perceptron regressor."""

    def __init__(self, seed: int, hid_sizes: List[int], max_train_iters: int,
                 clip_gradients: bool, clip_value: float,
                 learning_rate: float) -> None:
        super().__init__(seed, max_train_iters, clip_gradients, clip_value,
                         learning_rate)
        self._hid_sizes = hid_sizes
        # Set in fit().
        self._linears = nn.ModuleList()

    def forward(self, tensor_X: Tensor) -> Tensor:
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        return tensor_X

    def _initialize_net(self) -> None:
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(self._x_dim, self._hid_sizes[0]))
        for i in range(len(self._hid_sizes) - 1):
            self._linears.append(
                nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]))
        self._linears.append(nn.Linear(self._hid_sizes[-1], self._y_dim))

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.MSELoss()


class ImplicitMLPRegressor(PyTorchRegressor):
    """A regressor implemented via an "energy function".

    Currently the energy function is treated as a binary classifier, which is
    not consistent with how energy functions are usually treated/trained.
    This will change soon.

    Negative examples are generated within the class.

    Inference is currently performed by sampling a fixed number of possible
    inputs and returning the sample that has the highest probability of
    classifying to 1, under the learned classifier. Other inference methods are
    coming soon.
    """

    def __init__(self, seed: int, hid_sizes: List[int], max_train_iters: int,
                 clip_gradients: bool, clip_value: float, learning_rate: float,
                 num_samples_per_inference: int,
                 num_negative_data_per_input: int) -> None:
        super().__init__(seed, max_train_iters, clip_gradients, clip_value,
                         learning_rate)
        self._hid_sizes = hid_sizes
        self._num_samples_per_inference = num_samples_per_inference
        self._num_negative_data_per_input = num_negative_data_per_input
        # Set in fit().
        self._linears = nn.ModuleList()

    def forward(self, tensor_X: Tensor) -> Tensor:
        # The input here is the concatenation of the regressor's input and a
        # candidate output. A better name would be tensor_XY, but we leave it
        # as tensor_X for consistency with the parent class.
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        return tensor_X.squeeze(dim=-1)

    def _initialize_net(self) -> None:
        self._linears = nn.ModuleList()
        self._linears.append(
            nn.Linear(self._x_dim + self._y_dim, self._hid_sizes[0]))
        for i in range(len(self._hid_sizes) - 1):
            self._linears.append(
                nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]))
        self._linears.append(nn.Linear(self._hid_sizes[-1], 1))

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.BCEWithLogitsLoss()

    def _create_batch_generator(self, X: Array,
                                Y: Array) -> Iterator[Tuple[Tensor, Tensor]]:
        # Resample negative examples on each iteration.
        pos_concat_inputs = np.hstack([X, Y])
        num_pos_inputs = len(pos_concat_inputs)
        num_neg_inputs = num_pos_inputs * self._num_negative_data_per_input
        targets = np.array([1] * num_pos_inputs + [0] * num_neg_inputs,
                           dtype=np.float32)
        tensor_Y = torch.from_numpy(np.array(targets, dtype=np.float32))
        while True:
            neg_X, neg_Y = self._create_negative_data(X, Y)
            # Set up the data for the classifier.
            neg_concat_inputs = np.hstack([neg_X, neg_Y])
            concat_inputs = np.vstack([pos_concat_inputs, neg_concat_inputs])
            # Convert data to tensors.
            tensor_X = torch.from_numpy(
                np.array(concat_inputs, dtype=np.float32))
            yield (tensor_X, tensor_Y)

    def _fit(self, X: Array, Y: Array) -> None:
        # Initialize the network.
        self._initialize_net()
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Create the optimizer.
        optimizer = self._create_optimizer()
        # Create the negative data.
        batch_generator = self._create_batch_generator(X, Y)
        # Run training.
        _train_predictive_pytorch_model(self,
                                        loss_fn,
                                        optimizer,
                                        batch_generator,
                                        max_iters=self._max_train_iters,
                                        clip_gradients=self._clip_gradients,
                                        clip_value=self._clip_value)

    def _predict(self, x: Array) -> Array:
        # This sampling-based inference method is okay in 1 dimension, but
        # won't work well with higher dimensions.
        assert x.shape == (self._x_dim, )
        num_samples = self._num_samples_per_inference
        sample_ys = self._rng.uniform(size=(num_samples, self._y_dim))
        # Concatenate the x and ys.
        concat_xy = np.array([np.hstack([x, y]) for y in sample_ys],
                             dtype=np.float32)
        assert concat_xy.shape == (num_samples, self._x_dim + self._y_dim)
        # Pass through network.
        scores = self(torch.from_numpy(concat_xy))
        # Find the highest probability sample.
        sample_idx = torch.argmax(scores)
        return sample_ys[sample_idx]

    def _create_negative_data(self, X: Array, Y: Array) -> Tuple[Array, Array]:
        """This makes the assumption that negative data are far, far more
        common than positive data.

        Under this assumption, the negative y data are simply randomly
        sampled from a uniform distribution bounded by the min/max seen
        in the data. There may be false negatives in general, but they
        should be rare, under the assumption. The x values are taken
        directly from the X array, not sampled.
        """
        # Note that the data has already been normalized.
        del Y  # unused for now, but may be used in the future
        num_samples = self._num_negative_data_per_input
        neg_input_lst = []
        neg_output_lst = []
        for pos_input in X:
            samples = self._rng.uniform(size=(num_samples, self._y_dim))
            for neg_output in samples:
                neg_input_lst.append(pos_input)
                neg_output_lst.append(neg_output)
        neg_inputs = np.array(neg_input_lst, dtype=np.float32)
        neg_outputs = np.array(neg_output_lst, dtype=np.float32)
        return neg_inputs, neg_outputs


class NeuralGaussianRegressor(PyTorchRegressor):
    """NeuralGaussianRegressor definition."""

    def __init__(self, seed: int, hid_sizes: List[int], max_train_iters: int,
                 clip_gradients: bool, clip_value: float,
                 learning_rate: float) -> None:
        super().__init__(seed, max_train_iters, clip_gradients, clip_value,
                         learning_rate)
        self._hid_sizes = hid_sizes
        # Set in fit().
        self._linears = nn.ModuleList()

    def forward(self, tensor_X: Tensor) -> Tensor:
        """Pytorch forward method."""
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        # Force pred var positive.
        # Note: use of elu here is very important. Tried several other things
        # and none worked. Use of elu recommended here:
        # https://engineering.taboola.com/predicting-probability-distributions/
        mean, variance = self._split_prediction(tensor_X)
        variance = F.elu(variance) + 1
        return torch.cat([mean, variance], dim=-1)

    def _initialize_net(self) -> None:
        # Versus MLPRegressor, the only difference here is that the output
        # size is 2 * self._y_dim, rather than self._y_dim, because we are
        # predicting both mean and diagonal variance.
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(self._x_dim, self._hid_sizes[0]))
        for i in range(len(self._hid_sizes) - 1):
            self._linears.append(
                nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]))
        self._linears.append(nn.Linear(self._hid_sizes[-1], 2 * self._y_dim))

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        _nll_loss = nn.GaussianNLLLoss()

        def _loss_fn(Y_hat: Tensor, Y: Tensor) -> Tensor:
            pred_mean, pred_var = self._split_prediction(Y_hat)
            return _nll_loss(pred_mean, Y, pred_var)

        return _loss_fn

    def predict_mean(self, x: Array) -> Array:
        """Return a mean prediction on the given datapoint.

        x is single-dimensional.
        """
        assert x.ndim == 1
        mean, _ = self._predict_mean_var(x)
        return mean

    def predict_sample(self, x: Array, rng: np.random.Generator) -> Array:
        """Return a sampled prediction on the given datapoint.

        x is single-dimensional.
        """
        assert x.ndim == 1
        mean, variance = self._predict_mean_var(x)
        y = []
        for mu, sigma_sq in zip(mean, variance):
            y_i = rng.normal(loc=mu, scale=np.sqrt(sigma_sq))
            y.append(y_i)
        return np.array(y)

    def _predict_mean_var(self, x: Array) -> Tuple[Array, Array]:
        # Note: we need to use _predict(), rather than predict(), because
        # we need to apply normalization separately to the mean and variance
        # components of the prediction (see below).
        assert x.shape == (self._x_dim, )
        # Normalize.
        norm_x = (x - self._input_shift) / self._input_scale
        norm_y = self._predict(norm_x)
        assert norm_y.shape == (2 * self._y_dim, )
        norm_mean = norm_y[:self._y_dim]
        norm_variance = norm_y[self._y_dim:]
        # Denormalize output.
        mean = (norm_mean * self._output_scale) + self._output_shift
        variance = norm_variance * (np.square(self._output_scale))
        return mean, variance

    @staticmethod
    def _split_prediction(Y: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.split(Y, Y.shape[-1] // 2, dim=-1)  # type: ignore


################################ Classifiers ##################################


class MLPClassifier(PyTorchClassifier):
    """MLPClassifier definition."""

    def __init__(self, seed: int, balance_data: bool, max_train_iters: int,
                 learning_rate: float, n_iter_no_change: int,
                 hid_sizes: List[int]) -> None:
        super().__init__(seed, balance_data, max_train_iters, learning_rate,
                         n_iter_no_change)
        self._hid_sizes = hid_sizes
        # Set in fit().
        self._linears = nn.ModuleList()

    def _initialize_net(self) -> None:
        self._linears.append(nn.Linear(self._x_dim, self._hid_sizes[0]))
        for i in range(len(self._hid_sizes) - 1):
            self._linears.append(
                nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]))
        self._linears.append(nn.Linear(self._hid_sizes[-1], 1))

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.BCELoss()

    def forward(self, tensor_X: Tensor) -> Tensor:
        assert not self._do_single_class_prediction
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        return torch.sigmoid(tensor_X.squeeze(dim=-1))


class MLPClassifierEnsemble(BinaryClassifier):
    """MLPClassifierEnsemble definition."""

    def __init__(self, seed: int, balance_data: bool, max_train_iters: int,
                 learning_rate: float, n_iter_no_change: int,
                 hid_sizes: List[int], ensemble_size: int) -> None:
        super().__init__(seed, balance_data)
        self._members = [
            MLPClassifier(seed + i, balance_data, max_train_iters,
                          learning_rate, n_iter_no_change, hid_sizes)
            for i in range(ensemble_size)
        ]

    def fit(self, X: Array, y: Array) -> None:
        # Each member maintains its own normalizers.
        for i, member in enumerate(self._members):
            logging.info(f"Fitting member {i} of ensemble...")
            member.fit(X, y)

    def classify(self, x: Array) -> bool:
        # Each member maintains its own normalizers.
        avg = np.mean(self.predict_member_probas(x))
        classification = avg > 0.5
        assert classification in [True, False]
        return classification

    def _fit(self, X: Array, y: Array) -> None:
        raise NotImplementedError("Not used.")

    def _classify(self, x: Array) -> bool:
        raise NotImplementedError("Not used.")

    def predict_member_probas(self, x: Array) -> Array:
        """Return class probabilities predicted by each member."""
        return np.array([m.predict_proba(x) for m in self._members])


@dataclass(frozen=True, eq=False, repr=False)
class LearnedPredicateClassifier:
    """A convenience class for holding the model underlying a learned
    predicate."""
    _model: BinaryClassifier

    def classifier(self, state: State, objects: Sequence[Object]) -> bool:
        """The classifier corresponding to the given model.

        May be used as the _classifier field in a Predicate.
        """
        v = state.vec(objects)
        return self._model.classify(v)


################################## Utilities ##################################


def _normalize_data(data: Array,
                    scale_clip: float = 1) -> Tuple[Array, Array, Array]:
    shift = np.min(data, axis=0)  # type: ignore
    scale = np.max(data - shift, axis=0)  # type: ignore
    scale = np.clip(scale, scale_clip, None)
    return (data - shift) / scale, shift, scale


def _balance_binary_classification_data(
        X: Array, y: Array, rng: np.random.Generator) -> Tuple[Array, Array]:
    pos_idxs_np = np.argwhere(np.array(y) == 1).squeeze()
    neg_idxs_np = np.argwhere(np.array(y) == 0).squeeze()
    pos_idxs = ([pos_idxs_np.item()]
                if not pos_idxs_np.shape else list(pos_idxs_np))
    neg_idxs = ([neg_idxs_np.item()]
                if not neg_idxs_np.shape else list(neg_idxs_np))
    assert len(pos_idxs) + len(neg_idxs) == len(y) == len(X)
    keep_neg_idxs = list(
        rng.choice(neg_idxs, replace=False, size=len(pos_idxs)))
    keep_idxs = pos_idxs + keep_neg_idxs
    X_lst = [X[i] for i in keep_idxs]
    y_lst = [y[i] for i in keep_idxs]
    X = np.array(X_lst)
    y = np.array(y_lst)
    return (X, y)


def _single_batch_generator(
        tensor_X: Tensor, tensor_Y: Tensor) -> Iterator[Tuple[Tensor, Tensor]]:
    """Infinitely generate all of the data in one batch."""
    while True:
        yield (tensor_X, tensor_Y)


def _train_predictive_pytorch_model(model: nn.Module,
                                    loss_fn: Callable[[Tensor, Tensor],
                                                      Tensor],
                                    optimizer: optim.Optimizer,
                                    batch_generator: Iterator[Tuple[Tensor,
                                                                    Tensor]],
                                    max_iters: int,
                                    print_every: int = 1000,
                                    clip_gradients: bool = False,
                                    clip_value: float = 5,
                                    n_iter_no_change: int = 10000000) -> None:
    """Note that this currently does not use minibatches.

    In the future, with very large datasets, we would want to switch to
    minibatches.
    """
    model.train()
    itr = 0
    best_loss = float("inf")
    best_itr = 0
    model_name = tempfile.NamedTemporaryFile(delete=False).name
    for tensor_X, tensor_Y in batch_generator:
        Y_hat = model(tensor_X)
        loss = loss_fn(Y_hat, tensor_Y)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_itr = itr
            # Save this best model.
            torch.save(model.state_dict(), model_name)
        if itr % print_every == 0:
            logging.info(f"Loss: {loss:.5f}, iter: {itr}/{max_iters}")
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        if clip_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        if itr - best_itr > n_iter_no_change:
            logging.info(f"Loss did not improve after {n_iter_no_change} "
                         f"itrs, terminating at itr {itr}.")
            break
        if itr == max_iters:
            break
        itr += 1
    # Load best model.
    model.load_state_dict(torch.load(model_name))  # type: ignore
    os.remove(model_name)
    model.eval()
    logging.info(f"Loaded best model with loss: {best_loss:.5f}")
