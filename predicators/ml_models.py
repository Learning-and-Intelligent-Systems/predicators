"""Machine learning models useful for classification/regression.

Note: to promote modularity, this file should NOT import CFG.
"""

import abc
import logging
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Collection, Deque, Dict, FrozenSet, \
    Iterator, List, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta as BetaRV
from sklearn.base import BaseEstimator
from sklearn.neighbors import \
    KNeighborsClassifier as _SKLearnKNeighborsClassifier
from sklearn.neighbors import \
    KNeighborsRegressor as _SKLearnKNeighborsRegressor
from torch import Tensor, nn, optim
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, TensorDataset

from predicators import utils
from predicators.settings import CFG
from predicators.structs import Array, GroundAtom, MaxTrainIters, Object, \
    State, _GroundNSRT, _Option

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

    def __init__(self, seed: int, disable_normalization: bool = False) -> None:
        super().__init__(seed)
        # Set in fit().
        self._x_dims: Tuple[int, ...] = tuple()
        self._y_dim = -1
        self._disable_normalization = disable_normalization
        self._input_shift = np.zeros(1, dtype=np.float32)
        self._input_scale = np.zeros(1, dtype=np.float32)
        self._output_shift = np.zeros(1, dtype=np.float32)
        self._output_scale = np.zeros(1, dtype=np.float32)

    def fit(self, X: Array, Y: Array) -> None:
        num_data = X.shape[0]
        self._x_dims = tuple(X.shape[1:])
        _, self._y_dim = Y.shape
        assert Y.shape[0] == num_data
        logging.info(f"Training {self.__class__.__name__} on {num_data} "
                     "datapoints")
        if not self._disable_normalization:
            X, self._input_shift, self._input_scale = _normalize_data(X)
            Y, self._output_shift, self._output_scale = _normalize_data(Y)
        self._fit(X, Y)

    def predict(self, x: Array) -> Array:
        assert len(self._x_dims), "Fit must be called before predict."
        assert x.shape == self._x_dims
        # Normalize.
        if not self._disable_normalization:
            x = (x - self._input_shift) / self._input_scale
        # Make prediction.
        y = self._predict(x)
        assert y.shape == (self._y_dim, )
        # Denormalize.
        if not self._disable_normalization:
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

    def __init__(self,
                 seed: int,
                 max_train_iters: MaxTrainIters,
                 clip_gradients: bool,
                 clip_value: float,
                 learning_rate: float,
                 weight_decay: float = 0,
                 n_iter_no_change: int = 10000000,
                 use_torch_gpu: bool = False,
                 train_print_every: int = 1000,
                 disable_normalization: bool = False) -> None:
        torch.manual_seed(seed)
        _NormalizingRegressor.__init__(
            self, seed, disable_normalization=disable_normalization)
        nn.Module.__init__(self)  # type: ignore
        self._max_train_iters = max_train_iters
        self._clip_gradients = clip_gradients
        self._clip_value = clip_value
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._n_iter_no_change = n_iter_no_change
        self._device = _get_torch_device(use_torch_gpu)
        self._train_print_every = train_print_every

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
        return optim.Adam(self.parameters(),
                          lr=self._learning_rate,
                          weight_decay=self._weight_decay)

    def _fit(self, X: Array, Y: Array) -> None:
        # Initialize the network.
        self._initialize_net()
        self.to(self._device)
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Create the optimizer.
        optimizer = self._create_optimizer()
        # Convert data to tensors.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(
            self._device)
        tensor_Y = torch.from_numpy(np.array(Y, dtype=np.float32)).to(
            self._device)
        batch_generator = _single_batch_generator(tensor_X, tensor_Y)
        # Run training.
        _train_pytorch_model(self,
                             loss_fn,
                             optimizer,
                             batch_generator,
                             device=self._device,
                             print_every=self._train_print_every,
                             max_train_iters=self._max_train_iters,
                             dataset_size=X.shape[0],
                             clip_gradients=self._clip_gradients,
                             clip_value=self._clip_value,
                             n_iter_no_change=self._n_iter_no_change)

    def _predict(self, x: Array) -> Array:
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32)).to(
            self._device)
        tensor_X = tensor_x.unsqueeze(dim=0)
        tensor_Y = self(tensor_X)
        tensor_y = tensor_Y.squeeze(dim=0)
        y = tensor_y.detach().cpu().numpy()
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
        # Special case: only one class.
        if probs.shape == (1, ):
            return float(self.classify(x))
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
        self._x_dims: Tuple[int, ...] = tuple()
        self._input_shift = np.zeros(1, dtype=np.float32)
        self._input_scale = np.zeros(1, dtype=np.float32)
        self._do_single_class_prediction = False
        self._predicted_single_class = False

    def fit(self, X: Array, y: Array) -> None:
        """Train the classifier on the given data.

        X is two-dimensional, y is one-dimensional.
        """
        num_data = X.shape[0]
        self._x_dims = tuple(X.shape[1:])
        assert y.shape == (num_data, )
        logging.info(f"Training {self.__class__.__name__} on {num_data} "
                     f"datapoints ({sum(y)} positive)")
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
        assert len(self._x_dims), "Fit must be called before classify."
        assert x.shape == self._x_dims
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

    def __init__(self,
                 seed: int,
                 balance_data: bool,
                 max_train_iters: MaxTrainIters,
                 learning_rate: float,
                 n_iter_no_change: int,
                 n_reinitialize_tries: int,
                 weight_init: str,
                 weight_decay: float = 0,
                 use_torch_gpu: bool = False,
                 train_print_every: int = 1000) -> None:
        torch.manual_seed(seed)
        _NormalizingBinaryClassifier.__init__(self, seed, balance_data)
        nn.Module.__init__(self)  # type: ignore
        self._max_train_iters = max_train_iters
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._n_iter_no_change = n_iter_no_change
        self._n_reinitialize_tries = n_reinitialize_tries
        self._weight_init = weight_init
        self._device = _get_torch_device(use_torch_gpu)
        self._train_print_every = train_print_every

    @abc.abstractmethod
    def forward(self, tensor_X: Tensor) -> Tensor:
        """PyTorch forward method."""
        raise NotImplementedError("Override me!")

    def predict_proba(self, x: Array) -> float:
        """Get the predicted probability that the input classifies to 1.

        The input is NOT normalized.
        """
        if self._do_single_class_prediction:
            return float(self._predicted_single_class)
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
        return optim.Adam(self.parameters(),
                          lr=self._learning_rate,
                          weight_decay=self._weight_decay)

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
        self.to(self._device)
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Convert data to tensors.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(
            self._device)
        tensor_y = torch.from_numpy(np.array(y, dtype=np.float32)).to(
            self._device)
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
                device=self._device,
                print_every=self._train_print_every,
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
        assert x.shape == self._x_dims
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32)).to(
            self._device)
        tensor_X = tensor_x.unsqueeze(dim=0)
        tensor_Y = self(tensor_X)
        tensor_y = tensor_Y.squeeze(dim=0)
        y = tensor_y.detach().cpu().numpy()
        proba = y.item()
        assert 0 <= proba <= 1
        return proba

    def _classify(self, x: Array) -> bool:
        return self._forward_single_input_np(x) > 0.5


################################# Regressors ##################################


class MLPRegressor(PyTorchRegressor):
    """A basic multilayer perceptron regressor."""

    def __init__(self,
                 seed: int,
                 hid_sizes: List[int],
                 max_train_iters: MaxTrainIters,
                 clip_gradients: bool,
                 clip_value: float,
                 learning_rate: float,
                 weight_decay: float = 0,
                 use_torch_gpu: bool = False,
                 train_print_every: int = 1000,
                 n_iter_no_change: int = 10000000) -> None:
        super().__init__(seed,
                         max_train_iters,
                         clip_gradients,
                         clip_value,
                         learning_rate,
                         weight_decay=weight_decay,
                         n_iter_no_change=n_iter_no_change,
                         use_torch_gpu=use_torch_gpu,
                         train_print_every=train_print_every)
        self._hid_sizes = hid_sizes
        # Set in fit().
        self._linears = nn.ModuleList()

    def forward(self, tensor_X: Tensor) -> Tensor:
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        return tensor_X

    def _initialize_net(self) -> None:
        assert len(self._x_dims) == 1, "X should be two-dimensional"
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(self._x_dims[0], self._hid_sizes[0]))
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
                 weight_decay: float = 0,
                 use_torch_gpu: bool = False,
                 train_print_every: int = 1000,
                 derivative_free_num_iters: Optional[int] = None,
                 derivative_free_sigma_init: Optional[float] = None,
                 derivative_free_shrink_scale: Optional[float] = None,
                 grid_num_ticks_per_dim: Optional[int] = None) -> None:
        super().__init__(seed,
                         max_train_iters,
                         clip_gradients,
                         clip_value,
                         learning_rate,
                         weight_decay=weight_decay,
                         use_torch_gpu=use_torch_gpu,
                         train_print_every=train_print_every)
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
        assert len(self._x_dims) == 1, "X must be two-dimensional"
        self._linears = nn.ModuleList()
        self._linears.append(
            nn.Linear(self._x_dims[0] + self._y_dim, self._hid_sizes[0]))
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
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(
            self._device)
        tensor_Y = torch.from_numpy(np.array(Y, dtype=np.float32)).to(
            self._device)
        assert tensor_X.shape == (num_samples, *self._x_dims)
        assert tensor_Y.shape == (num_samples, self._y_dim)
        # Expand tensor_Y in preparation for concat in the loop below.
        tensor_Y = tensor_Y[:, None, :]
        assert tensor_Y.shape == (num_samples, 1, self._y_dim)
        # For each of the negative outputs, we need a corresponding input.
        # So we repeat each x value num_negatives + 1 times so that each of
        # the num_negatives outputs, and the 1 positive output, have a
        # corresponding input.
        tiled_X = tensor_X.unsqueeze(1).repeat(1, num_negatives + 1, 1)
        assert tiled_X.shape == (num_samples, num_negatives + 1, *self._x_dims)
        extended_X = tiled_X.reshape([-1, tensor_X.shape[-1]])
        assert extended_X.shape == (num_samples * (num_negatives + 1),
                                    *self._x_dims)
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
                                self._x_dims[0] + self._y_dim)
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
        self.to(self._device)
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
                             device=self._device,
                             max_train_iters=self._max_train_iters,
                             dataset_size=X.shape[0],
                             clip_gradients=self._clip_gradients,
                             clip_value=self._clip_value)

    def _predict(self, x: Array) -> Array:
        assert x.shape == self._x_dims
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
        assert concat_xy.shape == (num_samples, self._x_dims[0] + self._y_dim)
        # Pass through network.
        scores = self(torch.from_numpy(concat_xy).to(self._device))
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
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32)).to(
            self._device)
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
        return Y[selected_idx].detach().cpu().numpy()  # type: ignore

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
        assert concat_xy.shape == (num_samples, self._x_dims[0] + self._y_dim)
        # Pass through network.
        scores = self(torch.from_numpy(concat_xy).to(self._device))
        # Find the highest probability sample.
        sample_idx = torch.argmax(scores)
        return candidate_ys[sample_idx]


class CNNRegressor(PyTorchRegressor):
    """A basic CNN regressor operating on 2D images with multiple channels."""

    def __init__(self,
                 seed: int,
                 conv_channel_nums: List[int],
                 conv_kernel_sizes: List[int],
                 linear_hid_sizes: List[int],
                 max_train_iters: MaxTrainIters,
                 clip_gradients: bool,
                 clip_value: float,
                 learning_rate: float,
                 weight_decay: float = 0,
                 use_torch_gpu: bool = False,
                 train_print_every: int = 1000) -> None:
        """Create a CNNRegressor.

        conv_channel_nums and conv_kernel_sizes define the sizes of the
        output channels and square kernels for the Conv2d layers.
        linear_hid_sizes is the same as hid_sizes for MLPRegressor.
        """
        super().__init__(seed,
                         max_train_iters,
                         clip_gradients,
                         clip_value,
                         learning_rate,
                         weight_decay=weight_decay,
                         use_torch_gpu=use_torch_gpu,
                         train_print_every=train_print_every)
        assert len(conv_channel_nums) == len(conv_kernel_sizes)
        self._conv_channel_nums = conv_channel_nums
        self._conv_kernel_sizes = conv_kernel_sizes
        self._linear_hid_sizes = linear_hid_sizes

        self._max_pool = nn.MaxPool2d(2, 2)
        # Set in fit().
        self._convs = nn.ModuleList()
        self._linears = nn.ModuleList()

    def forward(self, tensor_X: Tensor) -> Tensor:
        for _, conv in enumerate(self._convs):
            tensor_X = self._max_pool(F.relu(conv(tensor_X)))
        tensor_X = torch.flatten(tensor_X, 1)
        for _, linear in enumerate(self._linears[:-1]):
            tensor_X = F.relu(linear(tensor_X))
        tensor_X = self._linears[-1](tensor_X)
        return tensor_X

    def _initialize_net(self) -> None:
        self._convs = nn.ModuleList()

        # We need to calculate the size of the tensor outputted from the Conv2d
        # layers to use as the input dim for the linear layers post-flatten.
        assert len(self._x_dims) == 3, "X should be 4-dimensional (N, C, H, W)"
        c_dim, h_dim, w_dim = self._x_dims
        for i in range(len(self._conv_channel_nums)):
            kernel_size = self._conv_kernel_sizes[i]
            self._convs.append(
                nn.Conv2d(c_dim, self._conv_channel_nums[i], kernel_size))
            # Calculate size after Conv2d + MaxPool2d
            c_dim = self._conv_channel_nums[i]
            h_dim = (h_dim - kernel_size + 1) // 2
            w_dim = (w_dim - kernel_size + 1) // 2

        flattened_size = c_dim * h_dim * w_dim
        self._linears = nn.ModuleList()
        self._linears.append(
            nn.Linear(flattened_size, self._linear_hid_sizes[0]))
        for i in range(len(self._linear_hid_sizes) - 1):
            self._linears.append(
                nn.Linear(self._linear_hid_sizes[i],
                          self._linear_hid_sizes[i + 1]))
        self._linears.append(nn.Linear(self._linear_hid_sizes[-1],
                                       self._y_dim))

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.MSELoss()


class NeuralGaussianRegressor(PyTorchRegressor, DistributionRegressor):
    """NeuralGaussianRegressor definition."""

    def __init__(self,
                 seed: int,
                 hid_sizes: List[int],
                 max_train_iters: MaxTrainIters,
                 clip_gradients: bool,
                 clip_value: float,
                 learning_rate: float,
                 weight_decay: float = 0,
                 use_torch_gpu: bool = False,
                 train_print_every: int = 1000) -> None:
        super().__init__(seed,
                         max_train_iters,
                         clip_gradients,
                         clip_value,
                         learning_rate,
                         weight_decay=weight_decay,
                         use_torch_gpu=use_torch_gpu,
                         train_print_every=train_print_every)
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
        assert len(self._x_dims) == 1, "X should be two-dimensional"
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(self._x_dims[0], self._hid_sizes[0]))
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
        assert x.shape == self._x_dims
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


class MonotonicBetaRegressor(PyTorchRegressor, DistributionRegressor):
    """A model that learns conditional beta distributions with the requirement
    that the mean of the distribution increases with the (assumed 1d) input.

    This regressor is used primarily for competence modeling.
    """

    def __init__(self,
                 seed: int,
                 max_train_iters: MaxTrainIters,
                 clip_gradients: bool,
                 clip_value: float,
                 learning_rate: float,
                 weight_decay: float = 0,
                 use_torch_gpu: bool = False,
                 train_print_every: int = 1000,
                 n_iter_no_change: int = 10000000,
                 constant_variance: float = 1e-2) -> None:

        super().__init__(seed,
                         max_train_iters,
                         clip_gradients,
                         clip_value,
                         learning_rate,
                         weight_decay=weight_decay,
                         n_iter_no_change=n_iter_no_change,
                         use_torch_gpu=use_torch_gpu,
                         disable_normalization=True,
                         train_print_every=train_print_every)

        # This model has three learnable parameters.
        self.theta = torch.nn.Parameter(torch.randn(3), requires_grad=True)
        # We use a constant variance.
        assert 0 < constant_variance < 0.25
        self.variance = constant_variance

    def _transform_theta(self) -> List[Tensor]:
        # Map unbounded parameters to constrained parameters with the following
        # guarantees: (1) 0 <= theta0 <= 1; (2) theta0 <= theta1 <= 1; and
        # (3) theta2 >= 0.
        theta0 = self.theta[0]
        theta1 = self.theta[1]
        theta2 = self.theta[2]
        ctheta0 = F.sigmoid(theta0)
        ctheta1 = F.sigmoid(theta0 + (F.elu(theta1) + 1))
        ctheta2 = F.elu(theta2) + 1
        return [ctheta0, ctheta1, ctheta2]

    def forward(self, tensor_X: Tensor) -> Tensor:
        # Transform weights to obey constraints.
        c0, c1, c2 = self._transform_theta()
        # Exponential saturation function.
        mean = c0 + (c1 - c0) * (1 - torch.exp(-c2 * tensor_X))  # type: ignore
        # Clip mean to avoid numerical issues.
        mean = torch.clip(mean, 1e-3, 1.0 - 1e-3)
        return mean

    def _initialize_net(self) -> None:
        # Reset the learnable parameters.
        self.theta = torch.nn.Parameter(torch.randn(3), requires_grad=True)

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        # Just regress the mean for stability.
        return nn.MSELoss()

    def predict_beta(self, x: float) -> BetaRV:
        """Predict a beta distribution given the input."""
        mean = self._predict(np.array([x], dtype=np.float32))[0]
        return utils.beta_from_mean_and_variance(mean, self.variance)

    def predict_sample(self, x: Array, rng: np.random.Generator) -> Array:
        assert len(x) == 1
        rv = self.predict_beta(x[0])
        return rv.rvs(random_state=rng).reshape(x.shape)

    def get_transformed_params(self) -> List[float]:
        """For interpretability."""
        return [v.item() for v in self._transform_theta()]


################################ Classifiers ##################################


class MLPBinaryClassifier(PyTorchBinaryClassifier):
    """MLPBinaryClassifier definition."""

    def __init__(self,
                 seed: int,
                 balance_data: bool,
                 max_train_iters: MaxTrainIters,
                 learning_rate: float,
                 n_iter_no_change: int,
                 hid_sizes: List[int],
                 n_reinitialize_tries: int,
                 weight_init: str,
                 weight_decay: float = 0,
                 use_torch_gpu: bool = False,
                 train_print_every: int = 1000) -> None:
        super().__init__(seed,
                         balance_data,
                         max_train_iters,
                         learning_rate,
                         n_iter_no_change,
                         n_reinitialize_tries,
                         weight_init,
                         weight_decay=weight_decay,
                         use_torch_gpu=use_torch_gpu,
                         train_print_every=train_print_every)
        self._hid_sizes = hid_sizes
        # Set in fit().
        self._linears = nn.ModuleList()

    def _initialize_net(self) -> None:
        assert len(self._x_dims) == 1, "X should be two-dimensional"
        self._linears.append(nn.Linear(self._x_dims[0], self._hid_sizes[0]))
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


def _get_torch_device(use_torch_gpu: bool) -> torch.device:
    return torch.device(
        "cuda:0" if use_torch_gpu and torch.cuda.is_available() else "cpu")


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
                         device: torch.device,
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
    model.load_state_dict(torch.load(model_name,
                                     map_location='cpu'))  # type: ignore
    model.to(device)
    os.remove(model_name)
    model.eval()
    logging.info(f"Loaded best model with loss: {best_loss:.5f}")
    return best_loss


# Low-level state, current high-level (predicate) state, option taken,
# next low-level state, reward, done.
MapleQData = Tuple[State, Set[GroundAtom], _Option, State, float, bool]


class MapleQFunction(MLPRegressor):
    """A Q function inspired by MAPLE (https://ut-austin-rpl.github.io/maple/)
    that has access to ground NSRTs.

    The ground NSRTs are used to approximately argmax the learned Q.

    Assumes a fixed set of objects and ground NSRTs.
    """

    def __init__(self,
                 seed: int,
                 hid_sizes: List[int],
                 max_train_iters: MaxTrainIters,
                 clip_gradients: bool,
                 clip_value: float,
                 learning_rate: float,
                 weight_decay: float = 0,
                 use_torch_gpu: bool = False,
                 train_print_every: int = 1000,
                 n_iter_no_change: int = 10000000,
                 discount: float = 0.99,
                 num_lookahead_samples: int = 5,
                 replay_buffer_max_size: int = 1000000,
                 replay_buffer_sample_with_replacement: bool = True) -> None:
        super().__init__(seed, hid_sizes, max_train_iters, clip_gradients,
                         clip_value, learning_rate, weight_decay,
                         use_torch_gpu, train_print_every, n_iter_no_change)
        self._rng = np.random.default_rng(seed)
        self._discount = discount
        self._num_lookahead_samples = num_lookahead_samples
        self._replay_buffer_max_size = replay_buffer_max_size
        self._replay_buffer_sample_with_replacement = \
            replay_buffer_sample_with_replacement

        # Updated once, after the first round of learning.
        self._ordered_objects: List[Object] = []
        self._ordered_frozen_goals: List[FrozenSet[GroundAtom]] = []
        self._ordered_ground_nsrts: List[_GroundNSRT] = []
        self._ground_nsrt_to_idx: Dict[_GroundNSRT, int] = {}
        self._max_num_params = 0
        self._num_ground_nsrts = 0
        self._replay_buffer: Deque[MapleQData] = deque(
            maxlen=self._replay_buffer_max_size)

    def set_grounding(self, objects: Set[Object],
                      goals: Collection[Set[GroundAtom]],
                      ground_nsrts: Collection[_GroundNSRT]) -> None:
        """After initialization because NSRTs not learned at first."""
        for ground_nsrt in ground_nsrts:
            num_params = ground_nsrt.option.params_space.shape[0]
            self._max_num_params = max(self._max_num_params, num_params)
        self._ordered_objects = sorted(objects)
        self._ordered_frozen_goals = sorted({frozenset(g) for g in goals})
        self._num_ground_nsrts = len(ground_nsrts)
        self._ordered_ground_nsrts = sorted(ground_nsrts)
        self._ground_nsrt_to_idx = {
            n: i
            for i, n in enumerate(self._ordered_ground_nsrts)
        }

    def get_option(self,
                   state: State,
                   goal: Set[GroundAtom],
                   num_samples_per_ground_nsrt: int,
                   epsilon: float = 0.0) -> _Option:
        """Get the best option under Q, epsilon-greedy."""
        # Return a random option.
        if self._rng.uniform() < epsilon:
            options = self._sample_applicable_options_from_state(
                state, num_samples_per_applicable_nsrt=1)
            return options[0]
        # Return the best option (approx argmax.)
        options = self._sample_applicable_options_from_state(
            state, num_samples_per_applicable_nsrt=num_samples_per_ground_nsrt)
        scores = [
            self.predict_q_value(state, goal, option) for option in options
        ]
        idx = np.argmax(scores)
        return options[idx]

    def add_datum_to_replay_buffer(self, datum: MapleQData) -> None:
        """Add one datapoint to the replay buffer.

        If the buffer is full, data is appended in a FIFO manner.
        """
        self._replay_buffer.append(datum)

    def train_q_function(self) -> None:
        """Fit the model."""
        # First, precompute the size of the input and output from the
        # Q-network.
        X_size = sum(o.type.dim for o in self._ordered_objects) + len(
            self._ordered_frozen_goals
        ) + self._num_ground_nsrts + self._max_num_params
        Y_size = 1
        # If there's no data in the replay buffer, we can't train.
        if len(self._replay_buffer) == 0:
            return
        # Otherwise, start by vectorizing all data in the replay buffer.
        X_arr = np.zeros((len(self._replay_buffer), X_size), dtype=np.float32)
        Y_arr = np.zeros((len(self._replay_buffer), Y_size), dtype=np.float32)
        for i, (state, goal, option, next_state, reward,
                terminal) in enumerate(self._replay_buffer):
            # Compute the input to the Q-function.
            vectorized_state = self._vectorize_state(state)
            vectorized_goal = self._vectorize_goal(goal)
            vectorized_action = self._vectorize_option(option)
            X_arr[i] = np.concatenate(
                [vectorized_state, vectorized_goal, vectorized_action])
            # Next, compute the target for Q-learning by sampling next actions.
            vectorized_next_state = self._vectorize_state(next_state)
            if not terminal and self._y_dim != -1:
                best_next_value = -np.inf
                next_option_vecs: List[Array] = []
                # We want to pick a total of num_lookahead_samples samples.
                while len(next_option_vecs) < self._num_lookahead_samples:
                    # Sample 1 per NSRT until we reach the target number.
                    for option in self._sample_applicable_options_from_state(
                            next_state):
                        next_option_vecs.append(self._vectorize_option(option))
                for next_action_vec in next_option_vecs:
                    x_hat = np.concatenate([
                        vectorized_next_state, vectorized_goal, next_action_vec
                    ])
                    q_x_hat = self.predict(x_hat)[0]
                    best_next_value = max(best_next_value, q_x_hat)
            else:
                best_next_value = 0.0
            Y_arr[i] = reward + self._discount * best_next_value

        # Finally, pass all this vectorized data to the training function.
        # This will implicitly sample mini batches and train for a certain
        # number of iterations. It will also normalize all the data.
        self.fit(X_arr, Y_arr)

    def minibatch_generator(
            self, tensor_X: Tensor, tensor_Y: Tensor,
            batch_size: int) -> Iterator[Tuple[Tensor, Tensor]]:
        """Assuming both tensor_X and tensor_Y are 2D with the batch dimension
        first, sample a minibatch of size batch_size to train on."""
        train_dataset = TensorDataset(tensor_X, tensor_Y)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        iterable_loader = iter(train_dataloader)
        while True:
            try:
                X_batch, Y_batch = next(iterable_loader)
            # pylint:disable=stop-iteration-return
            except StopIteration:
                iterable_loader = iter(train_dataloader)
                X_batch, Y_batch = next(iterable_loader)
            yield X_batch, Y_batch

    def _fit(self, X: Array, Y: Array) -> None:
        # Initialize the network.
        self._initialize_net()
        self.to(self._device)
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Create the optimizer.
        optimizer = self._create_optimizer()
        # Convert data to tensors.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(
            self._device)
        tensor_Y = torch.from_numpy(np.array(Y, dtype=np.float32)).to(
            self._device)
        batch_generator = self.minibatch_generator(
            tensor_X, tensor_Y, CFG.active_sampler_learning_batch_size)
        # Run training.
        _train_pytorch_model(self,
                             loss_fn,
                             optimizer,
                             batch_generator,
                             device=self._device,
                             print_every=self._train_print_every,
                             max_train_iters=self._max_train_iters,
                             dataset_size=X.shape[0],
                             clip_gradients=self._clip_gradients,
                             clip_value=self._clip_value,
                             n_iter_no_change=self._n_iter_no_change)

    def _vectorize_state(self, state: State) -> Array:
        # Cannot just call state.vec() directly because some objects may not
        # appear in this state.
        vecs: List[Array] = []
        for o in self._ordered_objects:
            try:
                vec = state[o]
            except KeyError:
                vec = np.zeros(o.type.dim, dtype=np.float32)
            vecs.append(vec)
        return np.concatenate(vecs)

    def _vectorize_goal(self, goal: Set[GroundAtom]) -> Array:
        frozen_goal = frozenset(goal)
        idx = self._ordered_frozen_goals.index(frozen_goal)
        vec = np.zeros(len(self._ordered_frozen_goals), dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def _vectorize_option(self, option: _Option) -> Array:
        matches = [
            i for (n, i) in self._ground_nsrt_to_idx.items()
            if n.option == option.parent
            and tuple(n.objects) == tuple(option.objects)
        ]
        assert len(matches) == 1
        # Create discrete part.
        discrete_vec = np.zeros(self._num_ground_nsrts)
        discrete_vec[matches[0]] = 1.0
        # Create continuous part.
        continuous_vec = np.zeros(self._max_num_params)
        continuous_vec[:len(option.params)] = option.params
        # Concatenate.
        vec = np.concatenate([discrete_vec, continuous_vec]).astype(np.float32)
        return vec

    def predict_q_value(self, state: State, goal: Set[GroundAtom],
                        option: _Option) -> float:
        """Predict the Q value."""
        # Default value if not yet fit.
        if self._y_dim == -1:
            return 0.0
        x = np.concatenate([
            self._vectorize_state(state),
            self._vectorize_goal(goal),
            self._vectorize_option(option)
        ])
        y = self.predict(x)[0]
        return y

    def _sample_applicable_options_from_state(
            self,
            state: State,
            num_samples_per_applicable_nsrt: int = 1) -> List[_Option]:
        """Use NSRTs to sample options in the current state."""
        # Create all applicable ground NSRTs.
        state_objs = set(state)
        applicable_nsrts = [
            o for o in self._ordered_ground_nsrts if \
                set(o.objects).issubset(state_objs) and all(
                a.holds(state) for a in o.preconditions)
        ]
        sampled_options: List[_Option] = []
        for app_nsrt in applicable_nsrts:
            for _ in range(num_samples_per_applicable_nsrt):
                # Sample an option.
                option = app_nsrt.sample_option(
                    state,
                    goal=set(),  # goal not used
                    rng=self._rng)
                assert option.initiable(state)
                sampled_options.append(option)
        return sampled_options
