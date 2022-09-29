"""Machine learning models useful for classification/regression.

Note: to promote modularity, this file should NOT import CFG.
"""

import abc
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple
from typing import Type as TypingType

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.neighbors import \
    KNeighborsClassifier as _SKLearnKNeighborsClassifier
from sklearn.neighbors import \
    KNeighborsRegressor as _SKLearnKNeighborsRegressor
from torch import Tensor, nn, optim
from torch.distributions.categorical import Categorical

from predicators.structs import Array, MaxTrainIters, Object, State

torch.use_deterministic_algorithms(mode=True)  # type: ignore
torch.set_num_threads(1)  # fixes libglomp error on supercloud

################################ Base Classes #################################


class Regressor(abc.ABC):
    """ABC for regressor classes."""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    @abc.abstractmethod
    def fit(self, X: Array, Y: Array) -> None:
        """Train the regressor on the given data.

        X and Y are both two-dimensional.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def predict(self, x: Array) -> Array:
        """Return a prediction for the given datapoint.

        x is single-dimensional.
        """
        raise NotImplementedError("Override me!")


class _ScikitLearnRegressor(Regressor):
    """A regressor that lightly wraps a scikit-learn regression model."""

    def __init__(self, seed: int, **kwargs: Any) -> None:
        super().__init__(seed)
        self._model = self._initialize_model(**kwargs)

    @abc.abstractmethod
    def _initialize_model(self, **kwargs: Any) -> BaseEstimator:
        raise NotImplementedError("Override me!")

    def fit(self, X: Array, Y: Array) -> None:
        return self._model.fit(X, Y)

    def predict(self, x: Array) -> Array:
        return self._model.predict([x])[0]


class _NormalizingRegressor(Regressor):
    """A regressor that normalizes the data.

    Also infers the dimensionality of the inputs and outputs from fit().
    """

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        # Set in fit().
        self._x_dim = -1
        self._y_dim = -1
        self._input_shift = np.zeros(1, dtype=np.float32)
        self._input_scale = np.zeros(1, dtype=np.float32)
        self._output_shift = np.zeros(1, dtype=np.float32)
        self._output_scale = np.zeros(1, dtype=np.float32)

    def fit(self, X: Array, Y: Array) -> None:
        num_data, self._x_dim = X.shape
        _, self._y_dim = Y.shape
        assert Y.shape[0] == num_data
        logging.info(f"Training {self.__class__.__name__} on {num_data} "
                     "datapoints")
        X, self._input_shift, self._input_scale = _normalize_data(X)
        Y, self._output_shift, self._output_scale = _normalize_data(Y)
        self._fit(X, Y)

    def predict(self, x: Array) -> Array:
        assert self._x_dim > -1, "Fit must be called before predict."
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


class PyTorchRegressor(_NormalizingRegressor, nn.Module):
    """ABC for PyTorch regression models."""

    def __init__(self, seed: int, max_train_iters: MaxTrainIters,
                 clip_gradients: bool, clip_value: float,
                 learning_rate: float) -> None:
        torch.manual_seed(seed)
        _NormalizingRegressor.__init__(self, seed)
        nn.Module.__init__(self)  # type: ignore
        self._max_train_iters = max_train_iters
        self._clip_gradients = clip_gradients
        self._clip_value = clip_value
        self._learning_rate = learning_rate

    @abc.abstractmethod
    def forward(self, tensor_X: Tensor) -> Tensor:
        """PyTorch forward method."""
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
        _train_pytorch_model(self,
                             loss_fn,
                             optimizer,
                             batch_generator,
                             max_train_iters=self._max_train_iters,
                             dataset_size=X.shape[0],
                             clip_gradients=self._clip_gradients,
                             clip_value=self._clip_value)

    def _predict(self, x: Array) -> Array:
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32))
        tensor_X = tensor_x.unsqueeze(dim=0)
        tensor_Y = self(tensor_X)
        tensor_y = tensor_Y.squeeze(dim=0)
        y = tensor_y.detach().numpy()
        return y


class DistributionRegressor(abc.ABC):
    """ABC for classes that learn a continuous conditional sampler."""

    @abc.abstractmethod
    def fit(self, X: Array, y: Array) -> None:
        """Train the model on the given data.

        X is two-dimensional, y is one-dimensional.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def predict_sample(self, x: Array, rng: np.random.Generator) -> Array:
        """Return a sampled prediction on the given datapoint.

        x is single-dimensional.
        """
        raise NotImplementedError("Override me!")


class BinaryClassifier(abc.ABC):
    """ABC for binary classifier classes."""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def fit(self, X: Array, y: Array) -> None:
        """Train the classifier on the given data.

        X is two-dimensional, y is one-dimensional.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def classify(self, x: Array) -> bool:
        """Return a predicted class for the given datapoint.

        x is single-dimensional.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def predict_proba(self, x: Array) -> float:
        """Get the predicted probability that the input classifies to 1.

        x is single-dimensional.
        """
        raise NotImplementedError("Override me!")


class _ScikitLearnBinaryClassifier(BinaryClassifier):
    """A regressor that lightly wraps a scikit-learn classification model."""

    def __init__(self, seed: int, **kwargs: Any) -> None:
        super().__init__(seed)
        self._model = self._initialize_model(**kwargs)

    @abc.abstractmethod
    def _initialize_model(self, **kwargs: Any) -> BaseEstimator:
        raise NotImplementedError("Override me!")

    def fit(self, X: Array, y: Array) -> None:
        return self._model.fit(X, y)

    def classify(self, x: Array) -> bool:
        class_prediction = self._model.predict([x])[0]
        assert class_prediction in [0, 1]
        return bool(class_prediction)

    def predict_proba(self, x: Array) -> float:
        probs = self._model.predict_proba([x])[0]
        assert probs.shape == (2, )  # [P(x is class 0), P(x is class 1)]
        return probs[1]  # return the second element of probs


class _NormalizingBinaryClassifier(BinaryClassifier):
    """A binary classifier that normalizes the data.

    Also infers the dimensionality of the inputs and outputs from fit().

    Also implements data balancing (optionally) and single-class prediction.
    """

    def __init__(self, seed: int, balance_data: bool) -> None:
        super().__init__(seed)
        self._balance_data = balance_data
        # Set in fit().
        self._x_dim = -1
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
        # only be generalization issues.
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
        assert self._x_dim > -1, "Fit must be called before classify."
        assert x.shape == (self._x_dim, )
        if self._do_single_class_prediction:
            return self._predicted_single_class
        # Normalize.
        x = (x - self._input_shift) / self._input_scale
        # Make prediction.
        return self._classify(x)

    @abc.abstractmethod
    def _fit(self, X: Array, y: Array) -> None:
        """Train the classifier on normalized data."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _classify(self, x: Array) -> bool:
        """Return a predicted class for the normalized input."""
        raise NotImplementedError("Override me!")


class PyTorchBinaryClassifier(_NormalizingBinaryClassifier, nn.Module):
    """ABC for PyTorch binary classification models."""

    def __init__(self, seed: int, balance_data: bool,
                 max_train_iters: MaxTrainIters, learning_rate: float,
                 n_iter_no_change: int, n_reinitialize_tries: int,
                 weight_init: str) -> None:
        torch.manual_seed(seed)
        _NormalizingBinaryClassifier.__init__(self, seed, balance_data)
        nn.Module.__init__(self)  # type: ignore
        self._max_train_iters = max_train_iters
        self._learning_rate = learning_rate
        self._n_iter_no_change = n_iter_no_change
        self._n_reinitialize_tries = n_reinitialize_tries
        self._weight_init = weight_init

    @abc.abstractmethod
    def forward(self, tensor_X: Tensor) -> Tensor:
        """PyTorch forward method."""
        raise NotImplementedError("Override me!")

    def predict_proba(self, x: Array) -> float:
        """Get the predicted probability that the input classifies to 1.

        The input is NOT normalized.
        """
        norm_x = (x - self._input_shift) / self._input_scale
        return self._forward_single_input_np(norm_x)

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

    def _reset_weights(self) -> None:
        """(Re-)initialize the network weights."""
        self.apply(lambda m: self._weight_reset(m, self._weight_init))

    def _weight_reset(self, m: torch.nn.Module, weight_init: str) -> None:
        if isinstance(m, nn.Linear):
            if weight_init == "default":
                m.reset_parameters()
            elif weight_init == "normal":
                torch.nn.init.normal_(m.weight)
            else:
                raise NotImplementedError(
                    f"{weight_init} weight initialization unknown")
        else:
            # To make sure all the weights are being reset
            assert m is self or isinstance(m, nn.ModuleList)

    def _fit(self, X: Array, y: Array) -> None:
        # Initialize the network.
        self._initialize_net()
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Convert data to tensors.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32))
        tensor_y = torch.from_numpy(np.array(y, dtype=np.float32))
        batch_generator = _single_batch_generator(tensor_X, tensor_y)
        # Run training.
        for _ in range(self._n_reinitialize_tries):
            # (Re-)initialize weights.
            self._reset_weights()
            # Create the optimizer.
            optimizer = self._create_optimizer()
            # Run training.
            best_loss = _train_pytorch_model(
                self,
                loss_fn,
                optimizer,
                batch_generator,
                max_train_iters=self._max_train_iters,
                dataset_size=X.shape[0],
                n_iter_no_change=self._n_iter_no_change)
            # Weights may not have converged during training.
            if best_loss < 1:
                break  # success!
        else:
            raise RuntimeError(f"Failed to converge within "
                               f"{self._n_reinitialize_tries} tries")

    def _forward_single_input_np(self, x: Array) -> float:
        """Helper for _classify() and predict_proba()."""
        assert x.shape == (self._x_dim, )
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32))
        tensor_X = tensor_x.unsqueeze(dim=0)
        tensor_Y = self(tensor_X)
        tensor_y = tensor_Y.squeeze(dim=0)
        y = tensor_y.detach().numpy()
        proba = y.item()
        assert 0 <= proba <= 1
        return proba

    def _classify(self, x: Array) -> bool:
        return self._forward_single_input_np(x) > 0.5


################################# Regressors ##################################


class MLPRegressor(PyTorchRegressor):
    """A basic multilayer perceptron regressor."""

    def __init__(self, seed: int, hid_sizes: List[int],
                 max_train_iters: MaxTrainIters, clip_gradients: bool,
                 clip_value: float, learning_rate: float) -> None:
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
    """A regressor implemented via an energy function.

    For each positive (x, y) pair, a number of "negative" (x, y') pairs are
    generated. The model is then trained to distinguish positive from negative
    conditioned on x using a contrastive loss.

    The implementation idea is the following. We want to use a contrastive
    loss that looks like this:

        L = E[-log(p(y | x, {y'}))]

        p(y | x, {y'})) = exp(-f(x, y)) / [
            (exp(-f(x, y)) + sum_{y'} exp(-f(x, y')))
        ]

    where (x, y) is an example "positive" input/output from (X, Y), f is
    the energy function that we are learning in this class, and {y'} is a set
    of "negative" output examples for input x. The size of that set is
    self._num_negatives_per_input.

    One way to interpret the expression is that the numerator exp(-f(x, y))
    represents an unnormalized probability that this (x, y) belongs to
    a certain ground truth "class". Each of the exp(-f(x, y')) in the
    denominator then corresponds to an artificial incorrect "class".
    So the entire expression is just a softmax over (num_negatives + 1)
    classes.

    Inference with the "sample_once" method samples a fixed number of possible
    inputs and returns the sample that has the highest probability of
    classifying to 1, under the learned classifier.

    Inference with the "derivative_free" method follows Algorithm 1 from the
    implicit BC paper (https://arxiv.org/pdf/2109.00137.pdf). It is very
    similar to CEM.

    Inference with the "grid" method is similar to "sample_once", except that
    the samples are evenly distributed over the Y space. Note that this method
    ignores the num_samples_per_inference keyword argument and instead uses the
    grid_num_ticks_per_dim.
    """

    def __init__(self,
                 seed: int,
                 hid_sizes: List[int],
                 max_train_iters: MaxTrainIters,
                 clip_gradients: bool,
                 clip_value: float,
                 learning_rate: float,
                 num_samples_per_inference: int,
                 num_negative_data_per_input: int,
                 temperature: float,
                 inference_method: str,
                 derivative_free_num_iters: Optional[int] = None,
                 derivative_free_sigma_init: Optional[float] = None,
                 derivative_free_shrink_scale: Optional[float] = None,
                 grid_num_ticks_per_dim: Optional[int] = None) -> None:
        super().__init__(seed, max_train_iters, clip_gradients, clip_value,
                         learning_rate)
        self._inference_method = inference_method
        self._derivative_free_num_iters = derivative_free_num_iters
        self._derivative_free_sigma_init = derivative_free_sigma_init
        self._derivative_free_shrink_scale = derivative_free_shrink_scale
        self._grid_num_ticks_per_dim = grid_num_ticks_per_dim
        self._hid_sizes = hid_sizes
        self._num_samples_per_inference = num_samples_per_inference
        self._num_negatives_per_input = num_negative_data_per_input
        self._temperature = temperature
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

        # See the class docstring for context.
        def _loss_fn(Y_hat: Tensor, Y: Tensor) -> Tensor:
            # The shape of Y_hat is (num_samples * (num_negatives + 1), ).
            # The shape of Y is (num_samples, (num_negatives + 1)).
            # Each row of Y is a one-hot vector with the first entry 1. We
            # could reconstruct that here, but we stick with this to conform
            # to the _train_pytorch_model API, where target outputs are always
            # passed into the loss function.
            pred = Y_hat.reshape(Y.shape)
            log_probs = F.log_softmax(pred / self._temperature, dim=-1)
            # Note: batchmean is recommended in the PyTorch documentation
            # and will become the default in a future version.
            loss = F.kl_div(log_probs, Y, reduction='batchmean')
            return loss

        return _loss_fn

    def _create_batch_generator(self, X: Array,
                                Y: Array) -> Iterator[Tuple[Tensor, Tensor]]:
        num_samples = X.shape[0]
        num_negatives = self._num_negatives_per_input
        # Cast to torch first.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32))
        tensor_Y = torch.from_numpy(np.array(Y, dtype=np.float32))
        assert tensor_X.shape == (num_samples, self._x_dim)
        assert tensor_Y.shape == (num_samples, self._y_dim)
        # Expand tensor_Y in preparation for concat in the loop below.
        tensor_Y = tensor_Y[:, None, :]
        assert tensor_Y.shape == (num_samples, 1, self._y_dim)
        # For each of the negative outputs, we need a corresponding input.
        # So we repeat each x value num_negatives + 1 times so that each of
        # the num_negatives outputs, and the 1 positive output, have a
        # corresponding input.
        tiled_X = tensor_X.unsqueeze(1).repeat(1, num_negatives + 1, 1)
        assert tiled_X.shape == (num_samples, num_negatives + 1, self._x_dim)
        extended_X = tiled_X.reshape([-1, tensor_X.shape[-1]])
        assert extended_X.shape == (num_samples * (num_negatives + 1),
                                    self._x_dim)
        while True:
            # Resample negative examples on each iteration.
            neg_Y = torch.rand(size=(num_samples, num_negatives, self._y_dim),
                               dtype=tensor_Y.dtype)
            # Create a multiclass classification-style target vector.
            combined_Y = torch.cat([tensor_Y, neg_Y], axis=1)  # type: ignore
            combined_Y = combined_Y.reshape([-1, tensor_Y.shape[-1]])
            # Concatenate to create the final input to the network.
            XY = torch.cat([extended_X, combined_Y], axis=1)  # type: ignore
            assert XY.shape == (num_samples * (num_negatives + 1),
                                self._x_dim + self._y_dim)
            # Create labels for multiclass loss. Note that the true inputs
            # are first, so the target labels are all zeros (see docstring).
            idxs = torch.zeros([num_samples], dtype=torch.int64)
            labels = F.one_hot(idxs, num_classes=(num_negatives + 1)).float()
            assert labels.shape == (num_samples, num_negatives + 1)
            # Note that XY is flattened and labels is not. XY is flattened
            # because we need to feed each entry through the network during
            # training. Labels is unflattened because we will want to use
            # F.kl_div in the loss function.
            yield (XY, labels)

    def _fit(self, X: Array, Y: Array) -> None:
        # Note: we need to override _fit() because we are not just training
        # a network that maps X to Y, but rather, training a network that
        # maps concatenated X and Y vectors to floats (energies).
        # Initialize the network.
        self._initialize_net()
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Create the optimizer.
        optimizer = self._create_optimizer()
        # Create the batch generator, which creates negative data.
        batch_generator = self._create_batch_generator(X, Y)
        # Run training.
        _train_pytorch_model(self,
                             loss_fn,
                             optimizer,
                             batch_generator,
                             max_train_iters=self._max_train_iters,
                             dataset_size=X.shape[0],
                             clip_gradients=self._clip_gradients,
                             clip_value=self._clip_value)

    def _predict(self, x: Array) -> Array:
        assert x.shape == (self._x_dim, )
        if self._inference_method == "sample_once":
            return self._predict_sample_once(x)
        if self._inference_method == "derivative_free":
            return self._predict_derivative_free(x)
        if self._inference_method == "grid":
            return self._predict_grid(x)
        raise NotImplementedError("Unrecognized inference method: "
                                  f"{self._inference_method}.")

    def _predict_sample_once(self, x: Array) -> Array:
        # This sampling-based inference method is okay in 1 dimension, but
        # won't work well with higher dimensions.
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

    def _predict_derivative_free(self, x: Array) -> Array:
        # Reference: https://arxiv.org/pdf/2109.00137.pdf (Algorithm 1).
        # This method reportedly works well in up to 5 dimensions.
        # Since we are using torch for random sampling, and since we want
        # to ensure deterministic predictions, we need to reseed torch.
        # Also note that we need to set the seed here because we need calls
        # on the same input to deterministically return the same output,
        # both when saved models are loaded, but also when the same model
        # is called multiple times in the same process. The latter case
        # happens when an option is called by the default option model and
        # then later called at execution time.
        torch.manual_seed(self._seed)
        num_samples = self._num_samples_per_inference
        num_iters = self._derivative_free_num_iters
        sigma = self._derivative_free_sigma_init
        K = self._derivative_free_shrink_scale
        assert num_samples is not None and num_samples > 0
        assert num_iters is not None and num_iters > 0
        assert sigma is not None and sigma > 0
        assert K is not None and 0 < K < 1
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32))
        repeated_x = tensor_x.repeat(num_samples, 1)
        # Initialize candidate outputs.
        Y = torch.rand(size=(num_samples, self._y_dim), dtype=tensor_x.dtype)
        for it in range(num_iters):
            # Compute candidate scores.
            concat_xy = torch.cat([repeated_x, Y], axis=1)  # type: ignore
            scores = self(concat_xy)
            if it < num_iters - 1:
                # Multinomial resampling with replacement.
                dist = Categorical(logits=scores)  # type: ignore
                indices = dist.sample((num_samples, ))  # type: ignore
                Y = Y[indices]
                # Add noise.
                noise = torch.randn(Y.shape) * sigma
                Y = Y + noise
                # Recall that Y is normalized to stay within [0, 1].
                Y = torch.clip(Y, 0.0, 1.0)
                sigma = K * sigma
        # Make a final selection.
        selected_idx = torch.argmax(scores)
        return Y[selected_idx].detach().numpy()  # type: ignore

    def _predict_grid(self, x: Array) -> Array:
        assert self._grid_num_ticks_per_dim is not None
        assert self._grid_num_ticks_per_dim > 0
        dy = 1.0 / self._grid_num_ticks_per_dim
        ticks = [np.arange(0.0, 1.0, dy)] * self._y_dim
        grid = np.meshgrid(*ticks)
        candidate_ys = np.transpose(grid).reshape((-1, self._y_dim))
        num_samples = candidate_ys.shape[0]
        assert num_samples == self._grid_num_ticks_per_dim**self._y_dim
        # Concatenate the x and ys.
        concat_xy = np.array([np.hstack([x, y]) for y in candidate_ys],
                             dtype=np.float32)
        assert concat_xy.shape == (num_samples, self._x_dim + self._y_dim)
        # Pass through network.
        scores = self(torch.from_numpy(concat_xy))
        # Find the highest probability sample.
        sample_idx = torch.argmax(scores)
        return candidate_ys[sample_idx]


class NeuralGaussianRegressor(PyTorchRegressor, DistributionRegressor):
    """NeuralGaussianRegressor definition."""

    def __init__(self, seed: int, hid_sizes: List[int],
                 max_train_iters: MaxTrainIters, clip_gradients: bool,
                 clip_value: float, learning_rate: float) -> None:
        super().__init__(seed, max_train_iters, clip_gradients, clip_value,
                         learning_rate)
        self._hid_sizes = hid_sizes
        # Set in fit().
        self._linears = nn.ModuleList()

    def forward(self, tensor_X: Tensor) -> Tensor:
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


class DegenerateMLPDistributionRegressor(MLPRegressor, DistributionRegressor):
    """A model that can be used as a DistributionRegressor, but that always
    returns the same output given the same input.

    Implemented as an MLPRegressor().
    """

    def predict_sample(self, x: Array, rng: np.random.Generator) -> Array:
        del rng  # unused
        return self.predict(x)


class KNeighborsRegressor(_ScikitLearnRegressor):
    """K nearest neighbors from scikit-learn."""

    def _initialize_model(self, **kwargs: Any) -> BaseEstimator:
        return _SKLearnKNeighborsRegressor(**kwargs)


################################ Classifiers ##################################


class MLPBinaryClassifier(PyTorchBinaryClassifier):
    """MLPBinaryClassifier definition."""

    def __init__(self, seed: int, balance_data: bool,
                 max_train_iters: MaxTrainIters, learning_rate: float,
                 n_iter_no_change: int, hid_sizes: List[int],
                 n_reinitialize_tries: int, weight_init: str) -> None:
        super().__init__(seed, balance_data, max_train_iters, learning_rate,
                         n_iter_no_change, n_reinitialize_tries, weight_init)
        self._hid_sizes = hid_sizes
        # Set in fit().
        self._linears = nn.ModuleList()

    def _initialize_net(self) -> None:
        self._linears.append(nn.Linear(self._x_dim, self._hid_sizes[0]))
        for i in range(len(self._hid_sizes) - 1):
            self._linears.append(
                nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]))
        self._linears.append(nn.Linear(self._hid_sizes[-1], 1))
        self._reset_weights()

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.BCELoss()

    def forward(self, tensor_X: Tensor) -> Tensor:
        assert not self._do_single_class_prediction
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        return torch.sigmoid(tensor_X.squeeze(dim=-1))


class KNeighborsClassifier(_ScikitLearnBinaryClassifier):
    """K nearest neighbors from scikit-learn."""

    def _initialize_model(self, **kwargs: Any) -> BaseEstimator:
        return _SKLearnKNeighborsClassifier(**kwargs)


class BinaryClassifierEnsemble(BinaryClassifier):
    """BinaryClassifierEnsemble definition."""

    def __init__(self, seed: int, ensemble_size: int,
                 member_cls: TypingType[BinaryClassifier],
                 **kwargs: Any) -> None:
        super().__init__(seed)
        self._members = [
            member_cls(seed + i, **kwargs) for i in range(ensemble_size)
        ]

    def fit(self, X: Array, y: Array) -> None:
        for i, member in enumerate(self._members):
            logging.info(f"Fitting member {i} of ensemble...")
            member.fit(X, y)

    def classify(self, x: Array) -> bool:
        avg = np.mean(self.predict_member_probas(x))
        classification = bool(avg > 0.5)
        return classification

    def predict_proba(self, x: Array) -> float:
        raise Exception("Can't call predict_proba() on an ensemble. Use "
                        "predict_member_probas() instead.")

    def predict_member_probas(self, x: Array) -> Array:
        """Return class probabilities predicted by each member."""
        return np.array([m.predict_proba(x) for m in self._members])


################################## Utilities ##################################


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


def _normalize_data(data: Array,
                    scale_clip: float = 1) -> Tuple[Array, Array, Array]:
    shift = np.min(data, axis=0)
    scale = np.max(data - shift, axis=0)
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


def _train_pytorch_model(model: nn.Module,
                         loss_fn: Callable[[Tensor, Tensor], Tensor],
                         optimizer: optim.Optimizer,
                         batch_generator: Iterator[Tuple[Tensor, Tensor]],
                         max_train_iters: MaxTrainIters,
                         dataset_size: int,
                         print_every: int = 1000,
                         clip_gradients: bool = False,
                         clip_value: float = 5,
                         n_iter_no_change: int = 10000000) -> float:
    """Note that this currently does not use minibatches.

    In the future, with very large datasets, we would want to switch to
    minibatches. Returns the best loss seen during training.
    """
    model.train()
    itr = 0
    best_loss = float("inf")
    best_itr = 0
    model_name = tempfile.NamedTemporaryFile(delete=False).name
    if isinstance(max_train_iters, int):
        max_iters = max_train_iters
    else:  # assume that it's a function from dataset size to max iters
        max_iters = max_train_iters(dataset_size)
    assert isinstance(max_iters, int)
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
    return best_loss
