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
import math
import torch
import torch.nn.functional as F
import torchvision
from sklearn.base import BaseEstimator
from sklearn.neighbors import \
    KNeighborsClassifier as _SKLearnKNeighborsClassifier
from sklearn.neighbors import \
    KNeighborsRegressor as _SKLearnKNeighborsRegressor
from torch import Tensor, nn, optim
from torch.distributions.categorical import Categorical

from predicators.structs import Array, MaxTrainIters, Object, State
from predicators.settings import CFG

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
        # assert x.shape == (self._x_dim, )
        # Normalize.
        x = (x - self._input_shift) / self._input_scale
        # Make prediction.
        y = self._predict(x)
        # assert y.shape == (self._y_dim, )
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
        self._device = 'cuda' if CFG.use_cuda else 'cpu'

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
        self.to(self._device)
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Create the optimizer.
        optimizer = self._create_optimizer()
        # Convert data to tensors.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self._device)
        tensor_Y = torch.from_numpy(np.array(Y, dtype=np.float32)).to(self._device)
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
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32)).to(self._device)
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
    def fit(self, X: Array, y: Array, X_val: Array = None, y_val: Array = None) -> None:
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

    def fit(self, X: Array, y: Array, X_val: Array = None, y_val: Array = None) -> None:
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
        if not (hasattr(self, '_skip_norm') and self._skip_norm):
            X, self._input_shift, self._input_scale = _normalize_data(X)
        else:
            self._input_shift = 0
            self._input_scale = 1
        if X_val is not None:
            X_val = (X_val - self._input_shift) / self._input_scale
        self._fit(X, y, X_val, y_val)

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
    def _fit(self, X: Array, y: Array, X_val: Array = None, y_val: Array = None) -> None:
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
        assert n_reinitialize_tries == 1, "Changed code to ignore n_reinitialize_tries"
        self._n_reinitialize_tries = n_reinitialize_tries
        self._weight_init = weight_init
        self._device = 'cuda' if CFG.use_cuda else 'cpu'
        self._optimizer = None

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
        if self._optimizer is None:
            print('Creating optimizer afresh')
            self._optimizer = optim.Adam(self.parameters(), lr=self._learning_rate)
        return self._optimizer

    def _reset_weights(self) -> None:
        """(Re-)initialize the network weights."""
        print("Resetting weights")
        self.apply(lambda m: self._weight_reset(m, self._weight_init))

    def _weight_reset(self, m: torch.nn.Module, weight_init: str) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if weight_init == "default":
                m.reset_parameters()
            elif weight_init == "normal":
                torch.nn.init.normal_(m.weight)
            else:
                raise NotImplementedError(
                    f"{weight_init} weight initialization unknown")
        # else:
        #     # To make sure all the weights are being reset
        #     assert m is self or isinstance(m, nn.ModuleList)

    def _fit(self, X: Array, y: Array, X_val: Array = None, y_val: Array = None) -> None:
        # Initialize the network.
        self._initialize_net()
        # Create the optimizer.
        optimizer = self._create_optimizer()
        self.to(self._device)
        # Create the loss function.
        loss_fn = self._create_loss_fn()
        # Convert data to tensors.
        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self._device)
        tensor_y = torch.from_numpy(np.array(y, dtype=np.float32)).to(self._device)
        batch_generator = _single_batch_generator(tensor_X, tensor_y)
        if X_val is not None:
            tensor_X_val = torch.from_numpy(np.array(X_val, dtype=np.float32)).to(self._device)
            tensor_y_val = torch.from_numpy(np.array(y_val, dtype=np.float32)).to(self._device)
            batch_generator_val = _single_batch_generator(tensor_X_val, tensor_y_val)
        else:
            batch_generator_val = None
        # Run training.
        loss = _train_pytorch_model(
            self,
            loss_fn,
            optimizer,
            batch_generator,
            max_train_iters=self._max_train_iters,
            dataset_size=X.shape[0],
            n_iter_no_change=self._n_iter_no_change,
            batch_generator_val=batch_generator_val)
        # Weights may not have converged during training.
        if loss >= 1:
            raise RuntimeError(f"Failed to converge within "
                               f"{self._n_reinitialize_tries} tries")

    def _forward_single_input_np(self, x: Array) -> float:
        """Helper for _classify() and predict_proba()."""
        assert x.shape == (self._x_dim, )
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32))
        tensor_X = tensor_x.unsqueeze(dim=0).to(self._device)
        tensor_Y = self(tensor_X)
        tensor_y = tensor_Y.squeeze(dim=0)
        y = tensor_y.detach().to('cpu').numpy()
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
        self._is_initialized = False

    def _initialize_net(self) -> None:
        if len(self._linears) == 0:
            self._linears.append(nn.Sequential(
                nn.Linear(self._x_dim, self._hid_sizes[0]),
                nn.Dropout(0.5),
                nn.ReLU(),
                # nn.BatchNorm1d(self._hid_sizes[0])
            ))
            for i in range(len(self._hid_sizes) - 1):
                self._linears.append(nn.Sequential(
                    nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    # nn.BatchNorm1d(self._hid_sizes[i + 1])
                ))
            self._linears.append(nn.Linear(self._hid_sizes[-1], 1))
            self._reset_weights()
        self._is_initialized = True

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.BCELoss()

    def forward(self, tensor_X: Tensor) -> Tensor:
        assert not self._do_single_class_prediction
        for _, linear in enumerate(self._linears[:-1]):
            # tensor_X = F.relu(linear(tensor_X))
            tensor_X = linear(tensor_X)
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

############################# Energy-based models #############################
class BinaryEBM(MLPBinaryClassifier, DistributionRegressor):
    """A wrapper around a binary classifier that uses Langevin dynamics
    to generate samples using the classifier as an energy function."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_norm = True
        self._parallel_samples = 1

    @property
    def is_trained(self):
        return self._is_initialized

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.BCEWithLogitsLoss()

    def forward(self, tensor_X: Tensor) -> Tensor:
        assert not self._do_single_class_prediction
        for _, linear in enumerate(self._linears[:-1]):
            # tensor_X = F.relu(linear(tensor_X))
            tensor_X = linear(tensor_X)
        tensor_X = self._linears[-1](tensor_X)
        return tensor_X.squeeze(dim=-1)
    
    def predict_probas(self, X: Array) -> float:
        """Get the predicted probability that the input classifies to 1.

        The input is NOT normalized.
        """
        norm_X = (X - self._input_shift) / self._input_scale

        assert X.shape[-1] == self._x_dim
        tensor_X = torch.from_numpy(np.array(norm_X, dtype=np.float32)).to(self._device)
        tensor_Y = torch.sigmoid(self(tensor_X))
        Y = tensor_Y.detach().to('cpu').numpy()
        assert (0 <= Y).all() and  (Y <= 1).all()
        return Y


    def predict_sample(self, x: Array, rng: np.random.Generator) -> Array:
        """Assume that x contains the conditioning variables and that these
        correspond to the first x.shape[1] inputs to the model."""

        cond_dim = x.shape[0]
        out_dim = self._x_dim - cond_dim
        x = (x - self._input_shift[:cond_dim]) / self._input_scale[:cond_dim]
        tensor_x = torch.from_numpy(np.repeat(np.array(x, dtype=np.float32).reshape(1, -1), self._parallel_samples, axis=0)).to(self._device)
        tensor_X = tensor_x#.unsqueeze(dim=0)
        stepsize = 1e-4 # TODO: pass to CFG
        n_steps = 10 # TODO: pass to CFG
        noise_scale = np.sqrt(stepsize * 2)
        noise_step = 0#noise_scale / n_steps
        samples = torch.from_numpy(rng.uniform(size=(self._parallel_samples, out_dim)).astype(np.float32)).to(self._device)
        samples.requires_grad = True
        for _ in range(n_steps):
            noise = torch.from_numpy(rng.normal(size=(self._parallel_samples, out_dim)).astype(np.float32)).to(self._device) * noise_scale
            out = self.forward(torch.cat((tensor_X, samples), dim=1))
            grad = torch.autograd.grad(out.sum(), samples)[0]
            dynamics = stepsize * grad + noise
            samples = samples + dynamics

            noise_scale -= noise_step
        # sample = samples[out.argmax()].detach().to('cpu').numpy()
        if (out > 0).any():
            samples = samples[out > 0]
        sample = samples[0].detach().to('cpu').numpy()
        sample = sample * self._input_scale[cond_dim:] + self._input_shift[cond_dim:]
        return sample

class BinaryCNNEBM(BinaryEBM):

    def __init__(self, seed: int, balance_data: bool,
                 max_train_iters: MaxTrainIters, learning_rate: float,
                 n_iter_no_change: int, hid_sizes: List[int],
                 n_reinitialize_tries: int, weight_init: str) -> None:
        super().__init__(seed, balance_data, max_train_iters, learning_rate, 
                         n_iter_no_change, hid_sizes, n_reinitialize_tries,
                         weight_init)

        # Store information about CNN 
        self._conv_backbone = None


    def _initialize_net(self) -> None:
        if self._conv_backbone is None:
            # self._conv_backbone = nn.Sequential(
            #     # nn.Conv2d(1, 6, kernel_size=5, padding=2),
            #     nn.Conv2d(3, 6, kernel_size=3),
            #     nn.MaxPool2d(2, stride=2),
            #     nn.Dropout(0.5),
            #     nn.ReLU(),
            #     # nn.BatchNorm2d(6),
            #     # nn.Conv2d(6, 16, kernel_size=5),
            #     nn.Conv2d(6, 16, kernel_size=3),
            #     nn.MaxPool2d(2, stride=2),
            #     nn.Dropout(0.5),
            #     nn.ReLU(),
            #     # nn.BatchNorm2d(16),
            #     # nn.Conv2d(16, 32, kernel_size=5),
            #     nn.Conv2d(16, 32, kernel_size=3),
            #     nn.Dropout(0.5),
            #     nn.ReLU(),
            #     # nn.BatchNorm2d(32),
            # )      
            # self._conv_backbone = nn.Sequential(
            #     nn.Conv2d(self._img_shape[2], 16, 5),
            #     nn.MaxPool2d(2, stride=2),
            #     nn.Dropout(0.5),
            #     nn.ReLU(),
            #     nn.Conv2d(16, 32, 5),
            #     nn.MaxPool2d(2, stride=2),
            #     nn.Dropout(0.5),
            #     nn.ReLU(),
            #     nn.Conv2d(32, 64, 5),
            #     nn.MaxPool2d(2, stride=2),
            #     nn.Dropout(0.5),
            #     nn.ReLU(),
            #     nn.Conv2d(64, 128, 5),
            #     nn.MaxPool2d(2, stride=2),
            #     nn.Dropout(0.5),
            #     nn.ReLU(),
            #     nn.Conv2d(128, 256, 4),
            #     nn.Dropout(0.5),
            #     nn.ReLU()
            # )
            logging.info("Using resnet18!")
            self._conv_backbone = nn.Sequential(*list(torchvision.models.resnet18(weights='DEFAULT').children())[:-1])

            if self._x_dim - math.prod(self._img_shape) > 0:
                self._mlp_backbone = nn.Sequential(
                    nn.Linear(self._x_dim - math.prod(self._img_shape), 256),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.Dropout(0.5),
                    nn.ReLU()
                )
            else:
                self._mlp_backbone = None
        # self._x_dim -= math.prod(self._img_shape)
        # self._x_dim += 256   # add CNN output dimensions
        orig_x_dim = self._x_dim
        self._x_dim = 2 * 512   # concatenate outputs of backbones
        super()._initialize_net()
        # self._x_dim += math.prod(self._img_shape)
        # self._x_dim -= 256 
        self._x_dim = orig_x_dim   
    
    def fit(self, X: Array, y: Array, X_val: Array = None, y_val: Array = None, img_shape: Tuple = None) -> None:
        self._img_shape = img_shape
        assert 124 <= img_shape[0] <= 139 and 124 <= img_shape[1] <= 139, f'124 <= {img_shape} <= 139'    # this makes sure the output of the CNN is 1x1xchannels
        super().fit(X, y, X_val, y_val)

    def forward(self, tensor_X: Tensor) -> Tensor:
        assert not self._do_single_class_prediction
        img_X = tensor_X[:, :math.prod(self._img_shape)].reshape(tensor_X.shape[0], *(self._img_shape)).permute((0, 3, 1, 2))
        tensor_X = tensor_X[:, math.prod(self._img_shape):]
        img_X = self._conv_backbone(img_X)
        if self._mlp_backbone is not None:
            tensor_X = self._mlp_backbone(tensor_X)
        #     tensor_X = tensor_X * img_X.reshape(img_X.shape[0], -1)
        # else:
        #     tensor_X = img_X.reshape(img_X.shape[0], -1)
        tensor_X = torch.cat((tensor_X, img_X.reshape(img_X.shape[0], -1)), dim=1)
        return super().forward(tensor_X)

class RegressionEBM(MLPRegressor, DistributionRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_x = None
        self._use_cache = True  # TODO: move to global CFG
        if self._use_cache:
            self._cache_size = CFG.sesame_max_samples_per_step
        else:
            self._cache_size = 1

    @property
    def is_trained(self):
        return self._x_dim != -1

    def predict_sample(self, x: Array, rng: np.random.Generator) -> Array:
        """Assume that x contains the conditioning variables and that these
        correspond to the first x.shape[1] inputs to the model."""

        cond_dim = x.shape[0]
        if self._cached_x is not None and self._cached_sample_idx < self._cached_samples.shape[0] and np.allclose(x, self._cached_x):
            sample = self._cached_samples[self._cached_sample_idx]
            self._cached_sample_idx += 1
            sample = sample * self._input_scale[cond_dim:] + self._input_shift[cond_dim:]
            return sample
        self._cached_x = x
        out_dim = self._x_dim - cond_dim
        # samples = torch.from_numpy(rng.normal(size=out_dim).astype(np.float32)).unsqueeze(0).to(self._device)
        # samples = torch.from_numpy(rng.uniform(size=out_dim).astype(np.float32)).unsqueeze(0).to(self._device)
        samples = torch.from_numpy(rng.uniform(size=(self._cache_size,out_dim)).astype(np.float32)).to(self._device)
        x = (x - self._input_shift[:cond_dim]) / self._input_scale[:cond_dim]
        tensor_x = torch.from_numpy(np.repeat(np.array(x, dtype=np.float32).reshape(1, -1), self._cache_size, axis=0)).to(self._device)
        tensor_X = tensor_x#.unsqueeze(dim=0)
        stepsize = 1e-4 # TODO: pass to CFG
        n_steps = 10 # TODO: pass to CFG
        noise_scale = np.sqrt(stepsize * 2)
        samples.requires_grad = True
        for _ in range(n_steps):
            noise = torch.from_numpy(rng.normal(size=(self._cache_size, out_dim)).astype(np.float32)).to(self._device) * noise_scale
            out = self.forward(torch.cat((tensor_X, samples), dim=1))
            grad = torch.autograd.grad(out.sum(), samples)[0]
            dynamics = stepsize * grad + noise
            samples = samples + dynamics
        self._cached_samples = samples.detach().to('cpu').numpy()
        self._cached_sample_idx = 0
        sample = self._cached_samples[self._cached_sample_idx]
        self._cached_sample_idx += 1
        sample = sample * self._input_scale[cond_dim:] + self._input_shift[cond_dim:]
        return sample

    def predict_samples(self, x: Array, rng: np.random.Generator) -> Array:
        """Assume that x contains the conditioning variables and that these
        correspond to the first x.shape[1] inputs to the model."""

        cond_dim = x.shape[1]
        out_dim = self._x_dim - cond_dim
        samples = torch.from_numpy(rng.uniform(size=(x.shape[0],out_dim)).astype(np.float32)).to(self._device)
        x = (x - self._input_shift[:cond_dim]) / self._input_scale[:cond_dim]
        tensor_x = torch.from_numpy(np.array(x, dtype=np.float32)).to(self._device)
        tensor_X = tensor_x#.unsqueeze(dim=0)
        stepsize = 1e-4 # TODO: pass to CFG
        n_steps = 10 # TODO: pass to CFG
        noise_scale = np.sqrt(stepsize * 2)
        samples.requires_grad = True
        output_scale = torch.from_numpy(np.array(self._output_scale, dtype=np.float32)).to(self._device)
        output_shift = torch.from_numpy(np.array(self._output_shift, dtype=np.float32)).to(self._device)
        for _ in range(n_steps):
            noise = torch.from_numpy(rng.normal(size=(x.shape[0], out_dim)).astype(np.float32)).to(self._device) * noise_scale
            out = self.forward(torch.cat((tensor_X, samples), dim=1))
            out = out * output_scale + output_shift
            grad = torch.autograd.grad(out.sum(), samples)[0]
            dynamics = stepsize * grad + noise
            samples = samples + dynamics
        samples = samples.detach().to('cpu').numpy()
        samples = samples * self._input_scale[cond_dim:] + self._input_shift[cond_dim:]
        return samples


    # def predict(self, x: Array) -> Array:
    #     tensor_x = torch.from_numpy(x).to(self._device)
    #     tensor_X = tensor_x.unsqueeze(dim=0)
    #     tensor_Y = self(tensor_X)
    #     tensor_y = tensor_Y.squeeze(dim=0)
    #     y = tensor_y.detach().cpu().numpy()
    #     return y


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


# def _single_batch_generator(
#         tensor_X: Tensor, tensor_Y: Tensor) -> Iterator[Tuple[Tensor, Tensor]]:
#     """Infinitely generate all of the data in one batch."""
#     while True:
#         yield (tensor_X, tensor_Y)

def _single_batch_generator(
        tensor_X: Tensor, tensor_Y: Tensor) -> Iterator[Tuple[Tensor, Tensor]]:
    """Infinitely generate all of the data in one batch."""
    data = torch.utils.data.TensorDataset(tensor_X, tensor_Y)
    return torch.utils.data.DataLoader(data, batch_size=512, shuffle=True)

    # while True:
    #     yield (tensor_X, tensor_Y)

def _train_pytorch_model(model: nn.Module,
                         loss_fn: Callable[[Tensor, Tensor], Tensor],
                         optimizer: optim.Optimizer,
                         batch_generator: Iterator[Tuple[Tensor, Tensor]],
                         max_train_iters: MaxTrainIters,
                         dataset_size: int,
                         print_every: int = 100,
                         clip_gradients: bool = False,
                         clip_value: float = 5,
                         n_iter_no_change: int = 10000000,
                         batch_generator_val: Optional[Iterator[Tuple[Tensor, Tensor]]] = None) -> float:
    """Note that this currently does not use minibatches.

    In the future, with very large datasets, we would want to switch to
    minibatches. Returns the best loss seen during training.
    """
    import time

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
    for itr in range(max_iters):
        cum_loss = 0
        n = 0
        for tensor_X, tensor_Y in batch_generator:
            Y_hat = model(tensor_X)
            loss = loss_fn(Y_hat, tensor_Y)
            # if torch.isnan(loss):
            #     print(tensor_Y)
            #     raise ValueError('nan')
            optimizer.zero_grad()
            loss.backward()  # type: ignore
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            cum_loss += loss.item() * tensor_X.shape[0]
            n += tensor_X.shape[0]
        cum_loss /= n
        if cum_loss < best_loss:
            best_loss = cum_loss
            best_itr = itr
            # Save this best model.
            torch.save(model.state_dict(), model_name)
        if itr % print_every == 0:
            logging.info(f"Loss: {cum_loss:.5f}, iter: {itr}/{max_iters}")
            if batch_generator_val is not None:
                model.eval()
                cum_loss = 0
                cum_acc = 0
                n = 0
                with torch.no_grad():
                    for tensor_X, tensor_Y in batch_generator_val:
                        Y_hat = model(tensor_X)
                        loss = loss_fn(Y_hat, tensor_Y)
                        cum_loss += loss.item() * tensor_X.shape[0]
                        cum_acc += ((Y_hat > 0) == tensor_Y).float().sum()
                        n += tensor_X.shape[0]
                cum_loss /= n
                cum_acc /= n
                model.train()
                logging.info(f"\tValidation loss: {cum_loss:.5f}, acc: {cum_acc:.3f}")
        if itr - best_itr > n_iter_no_change:
            logging.info(f"Loss did not improve after {n_iter_no_change} "
                         f"itrs, terminating at itr {itr}.")
            break

    # Load best model.
    model.load_state_dict(torch.load(model_name))  # type: ignore
    os.remove(model_name)
    model.eval()
    logging.info(f"Loaded best model with loss: {best_loss:.5f}")
    return best_loss


class DiffusionRegressor(nn.Module):
    def __init__(self, seed: int, hid_sizes: List[int],
                 max_train_iters: int, timesteps: int,
                 learning_rate: float) -> None:
        super().__init__()
        torch.set_num_threads(CFG.torch_num_threads)    # reset here to get the cmd line arg
        self._linears = nn.ModuleList()
        self._seed = seed
        self._hid_sizes = hid_sizes
        self._max_train_iters = max_train_iters
        self._timesteps = timesteps
        self._device = 'cuda' if CFG.use_cuda else 'cpu'
        self._optimizer = None
        self._learning_rate = learning_rate
        self.is_trained = False

        # define beta schedule
        self._betas = self._cosine_beta_schedule(timesteps=timesteps)
        # self._betas = self._linear_beta_schedule(timesteps=timesteps)

        # define alphas 
        alphas = 1. - self._betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self._sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self._sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self._posterior_variance = self._betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self._cache_num_samples = 10# CFG.sesame_max_samples_per_step #* CFG.ebm_aux_n_samples
        self._cache = {}

    def forward(self, X_cond, Y_out, t, return_aux=False):
        half_t_dim = self._t_dim // 2
        t_embeddings = math.log(10000) / (half_t_dim - 1)
        t_embeddings = torch.exp(torch.arange(half_t_dim, device=self._device) * -t_embeddings)
        t_embeddings = t[:, None] * t_embeddings[None, :]
        t_embeddings = torch.cat((t_embeddings.sin(), t_embeddings.cos()), dim=-1)
        X = torch.cat((X_cond, Y_out, t_embeddings), dim=1)
        for linear in self._linears[:-1]:
            # X = F.relu(F.dropout(linear(X), p=0.5, training=self.training))
            X = F.relu(linear(X))
        X = self._linears[-1](X)
        if return_aux:
            return X[:, self._y_dim:]
        return X[:, :self._y_dim]

    def fit(self, X_cond: Array, Y_out: Array, Y_aux: Optional[Array] = None) -> None:# X_neg: Array, Y_neg: Array) -> None:
        num_data, _ = Y_out.shape
        if not self.is_trained:
            self.is_trained = True
            self._x_cond_dim = X_cond.shape[1]
            self._t_dim = (X_cond.shape[1] // 2) * 2    # make sure it's even
            _, self._y_dim = Y_out.shape
            self._x_dim = self._x_cond_dim + self._t_dim + self._y_dim

            self._input_shift = np.min(X_cond, axis=0)
            self._input_scale = np.max(X_cond - self._input_shift, axis=0)
            self._input_scale = np.clip(self._input_scale, 1e-6, None)

            self._output_shift = np.min(Y_out, axis=0)
            self._output_scale = np.max(Y_out - self._output_shift, axis=0)
            self._output_scale = np.clip(self._output_scale, 1e-6, None)

            if Y_aux is not None:
                self._y_aux_dim = Y_aux.shape[1]
                self._output_aux_shift = np.min(Y_aux, axis=0)
                self._output_aux_scale = np.max(Y_aux - self._output_aux_shift, axis=0)
                self._output_aux_scale = np.clip(self._output_aux_scale, 1e-6, None)

        X_cond = ((X_cond - self._input_shift) / self._input_scale) * 2 - 1
        Y_out = ((Y_out - self._output_shift) / self._output_scale) * 2 - 1
        if Y_aux is not None:
            Y_aux = ((Y_aux - self._output_aux_shift) / self._output_aux_scale) * 2 - 1       

        logging.info(f"Training {self.__class__.__name__} on {num_data} "
                     "datapoints")

        self._initialize_net()
        self.to(self._device) 
        optimizer = self._create_optimizer()

        tensor_X_cond = torch.from_numpy(np.array(X_cond, dtype=np.float32)).to(self._device)
        tensor_Y_out = torch.from_numpy(np.array(Y_out, dtype=np.float32)).to(self._device)
        tensor_Y_out = torch.from_numpy(np.array(Y_out, dtype=np.float32)).to(self._device)
        # tensor_X_neg = torch.from_numpy(np.array(X_neg, dtype=np.float32)).to(self._device)
        # tensor_Y_neg = torch.from_numpy(np.array(Y_neg, dtype=np.float32)).to(self._device)
        if Y_aux is not None:
            tensor_Y_aux = torch.from_numpy(np.array(Y_aux, dtype=np.float32)).to(self._device)
            data = torch.utils.data.TensorDataset(tensor_X_cond, tensor_Y_out, tensor_Y_aux)
        else:
            data = torch.utils.data.TensorDataset(tensor_X_cond, tensor_Y_out)
        # data_neg = torch.utils.data.TensorDataset(tensor_X_neg, tensor_Y_neg)
        dataloader = torch.utils.data.DataLoader(data, batch_size=512, shuffle=True)
        # dataloader_neg = torch.utils.data.DataLoader(data_neg, batch_size=(512 * len(data_neg)) // len(data), shuffle=True)

        assert isinstance(self._max_train_iters, int)
        self.train()
        for itr in range(self._max_train_iters):
            cum_loss = 0
            n = 0
            # for (tensor_X, tensor_Y), (tensor_X_neg, tensor_Y_neg) in zip(dataloader, dataloader_neg):
            # for tensor_X, tensor_Y in dataloader:
            for tensors in dataloader:
                if len(tensors) == 3:
                    tensor_X, tensor_Y, tensor_Y_aux = tensors
                else:
                    tensor_X, tensor_Y = tensors
                    tensor_Y_aux = None
                t = torch.randint(0, self._timesteps, (tensor_X.shape[0],), device=self._device)
                # t_neg = torch.randint(0, self._timesteps, (tensor_X_neg.shape[0],), device=self._device)
                loss = self._p_losses(tensor_X, tensor_Y, t, tensor_Y_aux)#, tensor_X_neg, tensor_Y_neg, t_neg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cum_loss += loss.item() * tensor_X.shape[0]
                n += tensor_X.shape[0]
            cum_loss /= n
            if itr % 100 == 0:
                logging.info(f"Loss: {cum_loss:.5f}, iter: {itr}/{self._max_train_iters}")

        self.eval()
        logging.info(f"Trained model with loss: {cum_loss:.5f}")
        return cum_loss


    def distill_half_steps(self, X_cond: Array, Y_out: Array, Y_aux: Optional[Array] = None) -> None:
        '''
        This method starts from the trained model and trains student models of half the #steps, iteratively
        '''
        assert self.is_trained

        num_data, _ = Y_out.shape
        X_cond = ((X_cond - self._input_shift) / self._input_scale) * 2 - 1
        Y_out = ((Y_out - self._output_shift) / self._output_scale) * 2 - 1
        if Y_aux is not None:
            Y_aux = ((Y_aux - self._output_aux_shift) / self._output_aux_scale) * 2 - 1       


        tensor_X_cond = torch.from_numpy(np.array(X_cond, dtype=np.float32)).to(self._device)
        tensor_Y_out = torch.from_numpy(np.array(Y_out, dtype=np.float32)).to(self._device)
        tensor_Y_out = torch.from_numpy(np.array(Y_out, dtype=np.float32)).to(self._device)
        # tensor_X_neg = torch.from_numpy(np.array(X_neg, dtype=np.float32)).to(self._device)
        # tensor_Y_neg = torch.from_numpy(np.array(Y_neg, dtype=np.float32)).to(self._device)
        if Y_aux is not None:
            tensor_Y_aux = torch.from_numpy(np.array(Y_aux, dtype=np.float32)).to(self._device)
            data = torch.utils.data.TensorDataset(tensor_X_cond, tensor_Y_out, tensor_Y_aux)
        else:
            data = torch.utils.data.TensorDataset(tensor_X_cond, tensor_Y_out)
        dataloader = torch.utils.data.DataLoader(data, batch_size=512, shuffle=True)
        


        teacher = self
        num_steps = self._timesteps // 2
        all_students = {}
        while num_steps > 0:
            student = DiffusionRegressor(seed=teacher._seed, hid_sizes=teacher._hid_sizes,
                                         max_train_iters=teacher._max_train_iters,
                                         timesteps=num_steps,
                                         learning_rate=teacher._learning_rate)
            student.is_trained = True
            student._x_cond_dim = teacher._x_cond_dim
            student._t_dim = teacher._t_dim
            student._y_dim = teacher._y_dim
            student._x_dim = teacher._x_dim

            student._input_shift = teacher._input_shift
            student._input_scale = teacher._input_scale
            student._input_scale = teacher._input_scale

            student._output_shift = teacher._output_shift
            student._output_scale = teacher._output_scale
            student._output_scale = teacher._output_scale

            if Y_aux is not None:
                student._y_aux_dim = teacher._y_aux_dim
                student._output_aux_shift = teacher._output_aux_shift
                student._output_aux_scale = teacher._output_aux_scale
                student._output_aux_scale = teacher._output_aux_scale

            logging.info(f"Training {student.__class__.__name__} on {num_data} "
                         f"datapoints for {student._timesteps} diffusion steps")

            student._initialize_net()
            # Key! Initialize student to teacher's params
            student.load_state_dict(teacher.state_dict())
            student.to(student._device) 
            optimizer = student._create_optimizer()

            assert isinstance(student._max_train_iters, int)
            student.train()
            for itr in range(student._max_train_iters):
                cum_loss = 0
                n = 0
                for tensors in dataloader:
                    if len(tensors) == 3:
                        tensor_X, tensor_Y, tensor_Y_aux = tensors
                    else:
                        tensor_X, tensor_Y = tensors
                        tensor_Y_aux = None
                    t = torch.randint(0, student._timesteps, (tensor_X.shape[0],), device=student._device)
                    loss = student._distill_half_losses(teacher, tensor_X, tensor_Y, t, tensor_Y_aux)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    cum_loss += loss.item() * tensor_X.shape[0]
                    n += tensor_X.shape[0]
                cum_loss /= n
                if itr % 100 == 0:
                    logging.info(f"Loss: {cum_loss:.5f}, iter: {itr}/{student._max_train_iters}")

            student.eval()
            logging.info(f"Trained model with loss: {cum_loss:.5f}")


            all_students[num_steps] = student
            num_steps = (num_steps // 2)
            teacher = student
        return all_students

    def distill(self, model_data_1, model_data_2) -> None:
        assert self.is_trained 

        model_1, data_1 = model_data_1
        X_cond_1, Y_out_1, Y_aux_1 = data_1
        model_2, data_2 = model_data_2
        X_cond_2, Y_out_2, Y_aux_2 = data_2

        assert (model_1._input_scale == self._input_scale).all() and (model_2._input_scale == self._input_scale).all()
        assert (model_1._input_shift == self._input_shift).all() and (model_2._input_shift == self._input_shift).all()
        assert (model_1._output_scale == self._output_scale).all() and (model_2._output_scale == self._output_scale).all()
        assert (model_1._output_shift == self._output_shift).all() and (model_2._output_shift == self._output_shift).all()
        if hasattr(self, '_output_aux_scale'):
            assert (model_1._output_aux_scale == self._output_aux_scale).all() and (model_2._output_aux_scale == self._output_aux_scale).all()
            assert (model_1._output_aux_shift == self._output_aux_shift).all() and (model_2._output_aux_shift == self._output_aux_shift).all()

        X_cond_1 = ((X_cond_1 - self._input_shift) / self._input_scale) * 2 - 1
        Y_out_1 = ((Y_out_1 - self._output_shift) / self._output_scale) * 2 - 1
        if Y_aux_1 is not None:
            Y_aux_1 = ((Y_aux_1 - self._output_aux_shift) / self._output_aux_scale) * 2 - 1       
        X_cond_2 = ((X_cond_2 - self._input_shift) / self._input_scale) * 2 - 1
        Y_out_2 = ((Y_out_2 - self._output_shift) / self._output_scale) * 2 - 1
        if Y_aux_2 is not None:
            Y_aux_2 = ((Y_aux_2 - self._output_aux_shift) / self._output_aux_scale) * 2 - 1       


        size_1 = X_cond_1.shape[0]
        size_2 = X_cond_2.shape[0]
        # Make sure both dataloaders have the same # batches
        pow_2 = 512
        batch_size_2 = 0
        while batch_size_2 == 0:
            batch_size_1 = pow_2
            batch_size_2 = pow_2 * min(size_1, size_2) // max(size_1, size_2)
            pow_2 *= 2
        if size_2 > size_1:
            batch_size_2, batch_size_1 = batch_size_1, batch_size_2
        
        logging.info(f"Distilling {model_1.__class__.__name__} with {size_1} datapoints and "
                      f"{model_2.__class__.__name__} with {size_2} datapoints into "
                      f"{self.__class__.__name__}")

        tensor_X_cond_1 = torch.from_numpy(np.array(X_cond_1, dtype=np.float32)).to(self._device)
        tensor_Y_out_1 = torch.from_numpy(np.array(Y_out_1, dtype=np.float32)).to(self._device)
        if Y_aux_1 is not None:
            tensor_Y_aux_1 = torch.from_numpy(np.array(Y_aux_1, dtype=np.float32)).to(self._device)
            data_1 = torch.utils.data.TensorDataset(tensor_X_cond_1, tensor_Y_out_1, tensor_Y_aux_1)
        else:
            data_1 = torch.utils.data.TensorDataset(tensor_X_cond_1, tensor_Y_out_1)
        dataloader_1 = torch.utils.data.DataLoader(data_1, batch_size=batch_size_1, shuffle=True)

        tensor_X_cond_2 = torch.from_numpy(np.array(X_cond_2, dtype=np.float32)).to(self._device)
        tensor_Y_out_2 = torch.from_numpy(np.array(Y_out_2, dtype=np.float32)).to(self._device)
        if Y_aux_2 is not None:
            tensor_Y_aux_2 = torch.from_numpy(np.array(Y_aux_2, dtype=np.float32)).to(self._device)
            data_2 = torch.utils.data.TensorDataset(tensor_X_cond_2, tensor_Y_out_2, tensor_Y_aux_2)
        else:
            data_2 = torch.utils.data.TensorDataset(tensor_X_cond_2, tensor_Y_out_2)
        dataloader_2 = torch.utils.data.DataLoader(data_2, batch_size=batch_size_2, shuffle=True)

        optimizer = self._optimizer
        assert isinstance(self._max_train_iters, int)
        self.train()
        for itr in range(self._max_train_iters // 10):
            cum_loss = 0
            n = 0
            for tensors_1, tensors_2 in zip(dataloader_1, dataloader_2):
                if len(tensors_1) == 3:
                    tensor_X_1, tensor_Y_1, tensor_Y_aux_1 = tensors_1
                    tensor_X_2, tensor_Y_2, tensor_Y_aux_2 = tensors_2
                else:
                    tensor_X_1, tensor_Y_1 = tensors_1
                    tensor_X_2, tensor_Y_2 = tensors_2
                    tensor_Y_aux_1 = None
                    tensor_Y_aux_2 = None
                t_1 = torch.randint(0, self._timesteps, (tensor_X_1.shape[0],), device=self._device)
                t_2 = torch.randint(0, self._timesteps, (tensor_X_2.shape[0],), device=self._device)
                loss = self._distill_losses(model_1, tensor_X_1, tensor_Y_1, t_1, tensor_Y_aux_1, model_2, tensor_X_2, tensor_Y_2, t_2, tensor_Y_aux_2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cum_loss += loss.item() * (tensor_X_1.shape[0] + tensor_X_2.shape[0])
                n += tensor_X_1.shape[0] + tensor_X_2.shape[0]
            cum_loss /= n
            if itr % 10 == 0:
                logging.info(f"Loss: {cum_loss:.5f}, iter: {itr}/{self._max_train_iters}")

        self.eval()
        logging.info(f"Distilled model with loss: {cum_loss:.5f}")
        return cum_loss

    def fit_balanced(self, data_1, data_2) -> None:
        assert self.is_trained 

        X_cond_1, Y_out_1, Y_aux_1 = data_1
        X_cond_2, Y_out_2, Y_aux_2 = data_2

        X_cond_1 = ((X_cond_1 - self._input_shift) / self._input_scale) * 2 - 1
        Y_out_1 = ((Y_out_1 - self._output_shift) / self._output_scale) * 2 - 1
        if Y_aux_1 is not None:
            Y_aux_1 = ((Y_aux_1 - self._output_aux_shift) / self._output_aux_scale) * 2 - 1       
        X_cond_2 = ((X_cond_2 - self._input_shift) / self._input_scale) * 2 - 1
        Y_out_2 = ((Y_out_2 - self._output_shift) / self._output_scale) * 2 - 1
        if Y_aux_2 is not None:
            Y_aux_2 = ((Y_aux_2 - self._output_aux_shift) / self._output_aux_scale) * 2 - 1       


        size_1 = X_cond_1.shape[0]
        size_2 = X_cond_2.shape[0]
        # Make sure both dataloaders have the same # batches
        pow_2 = 512
        batch_size_2 = 0
        while batch_size_2 == 0:
            batch_size_1 = pow_2
            batch_size_2 = pow_2 * min(size_1, size_2) // max(size_1, size_2)
            pow_2 *= 2
        if size_2 > size_1:
            batch_size_2, batch_size_1 = batch_size_1, batch_size_2
        
        logging.info(f"Balanced training with {size_1} datapoints and "
                      f"{size_2} datapoints into {self.__class__.__name__}")

        tensor_X_cond_1 = torch.from_numpy(np.array(X_cond_1, dtype=np.float32)).to(self._device)
        tensor_Y_out_1 = torch.from_numpy(np.array(Y_out_1, dtype=np.float32)).to(self._device)
        if Y_aux_1 is not None:
            tensor_Y_aux_1 = torch.from_numpy(np.array(Y_aux_1, dtype=np.float32)).to(self._device)
            data_1 = torch.utils.data.TensorDataset(tensor_X_cond_1, tensor_Y_out_1, tensor_Y_aux_1)
        else:
            data_1 = torch.utils.data.TensorDataset(tensor_X_cond_1, tensor_Y_out_1)
        dataloader_1 = torch.utils.data.DataLoader(data_1, batch_size=batch_size_1, shuffle=True)

        tensor_X_cond_2 = torch.from_numpy(np.array(X_cond_2, dtype=np.float32)).to(self._device)
        tensor_Y_out_2 = torch.from_numpy(np.array(Y_out_2, dtype=np.float32)).to(self._device)
        if Y_aux_2 is not None:
            tensor_Y_aux_2 = torch.from_numpy(np.array(Y_aux_2, dtype=np.float32)).to(self._device)
            data_2 = torch.utils.data.TensorDataset(tensor_X_cond_2, tensor_Y_out_2, tensor_Y_aux_2)
        else:
            data_2 = torch.utils.data.TensorDataset(tensor_X_cond_2, tensor_Y_out_2)
        dataloader_2 = torch.utils.data.DataLoader(data_2, batch_size=batch_size_2, shuffle=True)

        optimizer = self._optimizer
        assert isinstance(self._max_train_iters, int)
        self.train()
        for itr in range(self._max_train_iters // 10):
            cum_loss = 0
            n = 0
            for tensors_1, tensors_2 in zip(dataloader_1, dataloader_2):
                if len(tensors_1) == 3:
                    tensor_X_1, tensor_Y_1, tensor_Y_aux_1 = tensors_1
                    tensor_X_2, tensor_Y_2, tensor_Y_aux_2 = tensors_2
                else:
                    tensor_X_1, tensor_Y_1 = tensors_1
                    tensor_X_2, tensor_Y_2 = tensors_2
                    tensor_Y_aux_1 = None
                    tensor_Y_aux_2 = None
                t_1 = torch.randint(0, self._timesteps, (tensor_X_1.shape[0],), device=self._device)
                t_2 = torch.randint(0, self._timesteps, (tensor_X_2.shape[0],), device=self._device)
                loss = self._p_losses(tensor_X_1, tensor_Y_1, t_1, tensor_Y_aux_1)
                loss += self._p_losses(tensor_X_2, tensor_Y_2, t_2, tensor_Y_aux_2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cum_loss += loss.item() * (tensor_X_1.shape[0] + tensor_X_2.shape[0])
                n += tensor_X_1.shape[0] + tensor_X_2.shape[0]
            cum_loss /= n
            if itr % 10 == 0:
                logging.info(f"Loss: {cum_loss:.5f}, iter: {itr}/{self._max_train_iters}")

        self.eval()
        logging.info(f"Distilled model with loss: {cum_loss:.5f}")
        return cum_loss


    @torch.no_grad()
    def _p_sample(self, x_cond, y_out, t, t_index):
        betas_t = self._extract(self._betas, t, y_out.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self._sqrt_one_minus_alphas_cumprod, t, y_out.shape
        )
        sqrt_recip_alphas_t = self._extract(self._sqrt_recip_alphas, t, y_out.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        if CFG.classifier_free_guidance:
            epsilon_empty = self(torch.zeros_like(x_cond), y_out, t)
            epsilon_cond = self(x_cond, y_out, t)
            epsilon = epsilon_empty + 2 * (epsilon_cond - epsilon_empty)
        else:
            epsilon_out = self(x_cond, y_out, t)
            epsilon = epsilon_out#[:, :-1]
        # out = epsilon_out[:, -1]
        # grad = torch.autograd.grad(out.sum(), y_out)[0]
        # epsilon -= (sqrt_one_minus_alphas_cumprod_t * grad)
        model_mean = sqrt_recip_alphas_t * (
            y_out - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self._posterior_variance, t, y_out.shape)
            noise = torch.randn_like(y_out)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 


    @torch.no_grad()
    def _p_sample_loop(self, x_cond):
        # start from pure noise (for each example in the batch)
        y_out = torch.randn(self._cache_num_samples, self._y_dim, device=self._device, requires_grad=True)
        y_outs = []

        for i in reversed(range(0, self._timesteps)):
            y_out = self._p_sample(x_cond, y_out, torch.full((self._cache_num_samples,), i, device=self._device, dtype=torch.long), i)
            y_outs.append(y_out.detach().cpu().numpy())
        return y_outs

    def predict_sample(self, x_cond: Array, rng: np.random.Generator) -> Array:
        if x_cond.round(decimals=4).data.tobytes() not in self._cache:
            x_cond = ((x_cond - self._input_shift) / self._input_scale) * 2 - 1
            x_cond_tensor = torch.from_numpy(np.array(x_cond, dtype=np.float32)).to(self._device)
            x_cond_tensor = x_cond_tensor.view(1, -1).expand(self._cache_num_samples, -1)
            samples = self._p_sample_loop(x_cond_tensor)[-1]
            self._cache[x_cond.round(decimals=4).data.tobytes()] = (samples, 0)   # cache, idx
        sample = self._next_sample_in_cache(x_cond.round(decimals=4))
        return ((sample + 1) / 2 * self._output_scale) + self._output_shift

    def _next_sample_in_cache(self, arr):
        samples, idx = self._cache[arr.data.tobytes()]
        if idx < samples.shape[0] - 1:
            self._cache[arr.data.tobytes()] = (samples, idx + 1)
        else:
            del self._cache[arr.data.tobytes()]
        return samples[idx]

    def _initialize_net(self):
        if len(self._linears) == 0:
            self._linears.append(nn.Linear(self._x_dim, self._hid_sizes[0]))
            for i in range(len(self._hid_sizes) - 1):
                self._linears.append(
                    nn.Linear(self._hid_sizes[i], self._hid_sizes[i + 1]))
            # self._linears.append(nn.Linear(self._hid_sizes[-1], self._y_dim + 1))   # +1 for classifier guidance
            if CFG.ebm_aux_training:
                self._linears.append(nn.Linear(self._hid_sizes[-1], self._y_dim + self._y_aux_dim))
            else:
                self._linears.append(nn.Linear(self._hid_sizes[-1], self._y_dim))

    def _create_optimizer(self) -> optim.Optimizer:
        """Create an optimizer after the model is initialized."""
        if self._optimizer is None:
            print('Creating optimizer afresh')
            self._optimizer = optim.Adam(self.parameters(), lr=self._learning_rate)
        return self._optimizer

    @classmethod
    def _cosine_beta_schedule(cls, timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    @classmethod
    def _linear_beta_schedule(cls, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    def _p_losses(self, X_cond, Y_start, t, Y_aux):#, X_neg, Y_neg_start, t_neg):
        noise = torch.randn_like(Y_start)
        Y_noisy = self._q_sample(Y_start, t, noise)
        if CFG.classifier_free_guidance:
            mask = torch.randint(2, size=(X_cond.shape[0], 1), dtype=torch.long, device=X_cond.device)
        else:
            mask = torch.ones((X_cond.shape[0], 1), dtype=torch.long, device=X_cond.device)
        X_cond = X_cond * mask
        predicted_noise_label = self(X_cond, Y_noisy, t)
        predicted_noise = predicted_noise_label#[:, :-1]
        # label_pos = predicted_noise_label[:, -1]
        loss = F.smooth_l1_loss(noise, predicted_noise)

        # noise_neg = torch.randn_like(Y_neg_start)
        # Y_neg_noisy = self._q_sample(Y_neg_start, t_neg, noise_neg)
        # predicted_noise_label = self(X_neg, Y_neg_noisy, t_neg)
        # label_neg = predicted_noise_label[:, -1]

        # loss += F.binary_cross_entropy_with_logits(torch.cat((label_pos, label_neg)),
        #                 torch.cat((torch.ones_like(label_pos), torch.zeros_like(label_neg))))

        if Y_aux is not None:
            # X_cond_hat = self(X_cond, torch.zeros_like(Y_noisy), torch.zeros_like(t), return_reconstruction=True)
            # loss += F.smooth_l1_loss(X_cond, X_cond_hat)
            Y_aux_hat = self(X_cond, Y_start, torch.zeros_like(t), return_aux=True)
            loss += F.smooth_l1_loss(Y_aux, Y_aux_hat)

        return loss

    def _distill_half_losses(self, teacher, X_cond, Y_start, t, Y_aux):
        # The implementation is a bit weird bc the distillation loop takes the first teacher as "self", but the
        # loss computation always takes the current student as "self"

        assert teacher._timesteps // 2 == self._timesteps

        noise = torch.randn_like(Y_start)
        # TODO: this is a possible debug point. I'm assuming that the implementation considers the timesteps
        # in order to make the max noise the same regardless of self._timesteps. I'm not 100% sure (more like
        # 40%) this is true.
        Y_noisy = self._q_sample(Y_start, t, noise)
        predicted_noise = self(X_cond, Y_noisy, t)
        teacher_predicted_noise = teacher(X_cond, Y_noisy, t * 2)
        loss = F.smooth_l1_loss(teacher_predicted_noise, predicted_noise)
        if Y_aux is not None:
            Y_aux_hat = self(X_cond, Y_start, torch.zeros_like(t), return_aux=True)
            loss += F.smooth_l1_loss(Y_aux, Y_aux_hat)
        return loss

    def _distill_losses(self, model_1, X_cond_1, Y_start_1, t_1, Y_aux_1, model_2, X_cond_2, Y_start_2, t_2, Y_aux_2):
        if CFG.lifelong_method == "2-distill":
            noise_1 = torch.randn_like(Y_start_1)
            Y_noisy_1 = self._q_sample(Y_start_1, t_1, noise_1)
            predicted_noise_1 = self(X_cond_1, Y_noisy_1, t_1)

            noise_label_1 = model_1(X_cond_1, Y_noisy_1, t_1)
            loss_1 = F.smooth_l1_loss(noise_label_1, predicted_noise_1)

            if Y_aux_1 is not None:
                Y_aux_hat_1 = self(X_cond_1, Y_start_1, torch.zeros_like(t_1), return_aux=True)
                Y_aux_label_1 = model_1(X_cond_1, Y_start_1, torch.zeros_like(t_1), return_aux=True)
                loss_1 += F.smooth_l1_loss(Y_aux_label_1, Y_aux_hat_1)
        else:
            loss_1 = self._p_losses(X_cond_1, Y_start_1, t_1, Y_aux_1)

        noise_2 = torch.randn_like(Y_start_2)
        Y_noisy_2 = self._q_sample(Y_start_2, t_2, noise_2)
        predicted_noise_2 = self(X_cond_2, Y_noisy_2, t_2)

        noise_label_2 = model_2(X_cond_2, Y_noisy_2, t_2)
        loss_2 = F.smooth_l1_loss(noise_label_2, predicted_noise_2)

        if Y_aux_2 is not None:
            Y_aux_hat_2 = self(X_cond_2, Y_start_2, torch.zeros_like(t_2), return_aux=True)
            Y_aux_label_2 = model_2(X_cond_2, Y_start_2, torch.zeros_like(t_2), return_aux=True)
            loss_2 += F.smooth_l1_loss(Y_aux_label_2, Y_aux_hat_2)

        return loss_1 + loss_2

    @torch.no_grad()
    def predict_aux(self, x_cond: Array, y_out: Array) -> Array:
        x_cond = ((x_cond - self._input_shift) / self._input_scale) * 2 - 1
        x_cond_tensor = torch.from_numpy(np.array(x_cond, dtype=np.float32)).to(self._device)
        
        y_out = ((y_out - self._output_shift) / self._output_scale) * 2 - 1
        y_out_tensor = torch.from_numpy(np.array(y_out, dtype=np.float32)).to(self._device)

        t = torch.zeros(x_cond_tensor.shape[0], device=self._device, dtype=torch.int64)
        aux = self(x_cond_tensor, y_out_tensor, t, return_aux=True).detach().cpu().numpy()
        return ((aux + 1) / 2 * self._output_aux_scale) + self._output_aux_shift

    def aux_square_error(self, x_cond: Array, y_out: Array, aux_label: Array) -> Array:
        aux = self.predict_aux(x_cond, y_out)
        return (aux - aux_label) ** 2

    def _q_sample(self, Y_start, t, noise):
        sqrt_alphas_cumprod_t = self._extract(self._sqrt_alphas_cumprod, t, Y_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self._sqrt_one_minus_alphas_cumprod, t, Y_start.shape
        )

        return sqrt_alphas_cumprod_t * Y_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(self, a, t, shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(t.device)

# class CNNDiffusionRegressor(DiffusionRegressor):

#     def __init__(self, seed: int, hid_sizes: List[int],
#                  max_train_iters: int, timesteps: int,
#                  learning_rate: float) -> None:
#         super().__init__(seed, hid_sizes, max_train_iters, timesteps, learning_rate)

#         # Store information about CNN 
#         self._conv_backbone = None

#     def _initialize_net(self) -> None:
#         if self._conv_backbone is None:
#             # self._conv_backbone = nn.Sequential(
#             #     # nn.Conv2d(1, 6, kernel_size=5, padding=2),
#             #     nn.Conv2d(3, 6, kernel_size=3),
#             #     nn.MaxPool2d(2, stride=2),
#             #     # nn.Dropout(0.5),
#             #     nn.ReLU(),
#             #     # nn.Conv2d(6, 16, kernel_size=5),
#             #     nn.Conv2d(6, 16, kernel_size=3),
#             #     nn.MaxPool2d(2, stride=2),
#             #     # nn.Dropout(0.5),
#             #     nn.ReLU(),
#             #     # nn.Conv2d(16, 32, kernel_size=5),
#             #     nn.Conv2d(16, 32, kernel_size=3),
#             #     # nn.Dropout(0.5),
#             #     nn.ReLU()
#             # ) 
#             self._conv_backbone = nn.Sequential(
#                 nn.Conv2d(3, 16, 5),
#                 nn.MaxPool2d(2, stride=2),
#                 nn.ReLU(),
#                 nn.Conv2d(16, 32, 5),
#                 nn.MaxPool2d(2, stride=2),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 64, 5),
#                 nn.MaxPool2d(2, stride=2),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 64, 5),
#                 nn.MaxPool2d(2, stride=2),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 64, 5),
#                 nn.ReLU()
#             )     

#         # self._x_cond_dim -= (20*20*3)
#         self._x_cond_dim -= (150*150*3)
#         self._x_cond_dim += 64
#         self._t_dim = (self._x_cond_dim // 2) * 2
#         self._x_dim = self._x_cond_dim + self._t_dim + self._y_dim
#         super()._initialize_net()
#         # self._x_cond_dim += (20*20*3)
#         # self._x_cond_dim -= 32
#         # self._t_dim = (self._x_cond_dim // 2)

#         # # self._x_dim += (900)
#         # self._x_dim += (20*20*3)
#         # # self._x_dim += (12*12*3)
#         # self._x_dim -= 32    

#     def forward(self, X_cond, Y_out, t) -> Tensor:
#         # img_X = X_cond[:, :900].reshape(X_cond.shape[0], 1, 30, 30)
#         # X_cond = X_cond[:, 900:]
#         # img_X = X_cond[:, :20*20*3].reshape(X_cond.shape[0], 20, 20, 3).permute((0, 3, 1, 2))
#         # X_cond = X_cond[:, 20*20*3:]
#         # img_X = X_cond[:, :12*12*3].reshape(X_cond.shape[0], 12, 12, 3).permute((0, 3, 1, 2))
#         # X_cond = X_cond[:, 12*12*3:]
#         img_X = X_cond[:, :150*150*3].reshape(X_cond.shape[0], 150, 150, 3).permute((0, 3, 1, 2))
#         X_cond = X_cond[:, 150*150*3:]
#         img_X = self._forward_cnn(img_X)
#         X_cond = torch.cat((X_cond, img_X.reshape(X_cond.shape[0], -1)), dim=1)
#         return self._forward_fcn(X_cond, Y_out, t)
    
#     def _forward_cnn(self, img_X) -> Tensor:
#         # print(img_X.shape)
#         # tmp_x = self._conv_backbone[:3](img_X)
#         # print(tmp_x.shape)
#         # tmp_x = self._conv_backbone[3:6](tmp_x)
#         # print(tmp_x.shape)
#         # tmp_x = self._conv_backbone[6:](tmp_x)
#         # print(tmp_x.shape)
#         # exit()
#         return self._conv_backbone(img_X)


#     def _forward_fcn(self, X_cond, Y_out, t) -> Tensor:
#         return super().forward(X_cond, Y_out, t)

#     @torch.no_grad()
#     def _p_sample(self, x_cond, y_out, t, t_index):
#         betas_t = self._extract(self._betas, t, y_out.shape)
#         sqrt_one_minus_alphas_cumprod_t = self._extract(
#             self._sqrt_one_minus_alphas_cumprod, t, y_out.shape
#         )
#         sqrt_recip_alphas_t = self._extract(self._sqrt_recip_alphas, t, y_out.shape)
        
#         # Equation 11 in the paper
#         # Use our model (noise predictor) to predict the mean
#         epsilon_out = self._forward_fcn(x_cond, y_out, t)
#         epsilon = epsilon_out#[:, :-1]
#         # out = epsilon_out[:, -1]
#         # grad = torch.autograd.grad(out.sum(), y_out)[0]
#         # epsilon -= (sqrt_one_minus_alphas_cumprod_t * grad)
#         model_mean = sqrt_recip_alphas_t * (
#             y_out - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
#         )

#         if t_index == 0:
#             return model_mean
#         else:
#             posterior_variance_t = self._extract(self._posterior_variance, t, y_out.shape)
#             noise = torch.randn_like(y_out)
#             # Algorithm 2 line 4:
#             return model_mean + torch.sqrt(posterior_variance_t) * noise 


#     @torch.no_grad()
#     def _p_sample_loop(self, x_cond):
#         # start from pure noise (for each example in the batch)
#         y_out = torch.randn(self._cache_num_samples, self._y_dim, device=self._device, requires_grad=True)
#         y_outs = []

#         # img_X = x_cond[:, :20*20*3].reshape(x_cond.shape[0], 20, 20, 3).permute((0, 3, 1, 2))
#         # X_cond = x_cond[:, 20*20*3:]
#         img_X = x_cond[:, :150*150*3].reshape(x_cond.shape[0], 150, 150, 3).permute((0, 3, 1, 2))
#         X_cond = x_cond[:, 150*150*3:]
#         # img_X = X_cond[:, :12*12*3].reshape(X_cond.shape[0], 12, 12, 3).permute((0, 3, 1, 2))
#         # X_cond = X_cond[:, 12*12*3:]

#         img_X = self._forward_cnn(img_X)
#         X_cond = torch.cat((X_cond, img_X.reshape(X_cond.shape[0], -1)), dim=1)
#         for i in reversed(range(0, self._timesteps)):
#             y_out = self._p_sample(X_cond, y_out, torch.full((self._cache_num_samples,), i, device=self._device, dtype=torch.long), i)
#             y_outs.append(y_out.detach().cpu().numpy())
#         return y_outs

# class CNNDiffusionRegressor(DiffusionRegressor):
#     # THIS IS THE IMAGE-ONLY VERSION

#     def __init__(self, seed: int, hid_sizes: List[int],
#                  max_train_iters: int, timesteps: int,
#                  learning_rate: float) -> None:
#         super().__init__(seed, hid_sizes, max_train_iters, timesteps, learning_rate)

#         # Store information about CNN 
#         self._conv_backbone = None

#     class Upsample(nn.Module):
#         def __init__(self, in_channels, out_channels, kernel_size, stride, t_dim):
#             super().__init__()
#             self._up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
#             self._conv = nn.Conv2d(in_channels, out_channels, 3, padding='same')
#             self._t_linear = nn.Linear(t_dim, out_channels * 2)
#             self._relu = nn.ReLU()

#         def forward(self, x1, x2, t_embeddings):
#             x = self._up(x1)
#             x = torch.cat((x, x2), dim=1)
#             x = self._conv(x)
#             scale_shift = self._t_linear(t_embeddings)
#             scale, shift = scale_shift.view(*(scale_shift.shape), 1, 1).chunk(2, dim=1)
#             x = x * (scale + 1) + shift
#             return self._relu(x)

#     class Downsample(nn.Module):
#         def __init__(self, in_channels, out_channels, kernel_size, maxpool, t_dim):
#             super().__init__()
#             self._conv = nn.Conv2d(in_channels, out_channels, kernel_size)
#             if maxpool:
#                 self._pool = nn.MaxPool2d(2, stride=2)
#             else:
#                 self._pool = None
#             self._t_linear = nn.Linear(t_dim, out_channels * 2)
#             self._relu = nn.ReLU()

#         def forward(self, x, t_embeddings):
#             x = self._conv(x)
#             if self._pool:
#                 x = self._pool(x)
#             scale_shift = self._t_linear(t_embeddings)
#             scale, shift = scale_shift.view(*(scale_shift.shape), 1, 1).chunk(2, dim=1)
#             x = x * (scale + 1) + shift
#             return self._relu(x)

#     class SinusoidalPositionalEmbeddings(nn.Module):
#         def __init__(self, dim):
#             super().__init__()
#             self.dim = dim

#         def forward(self, t):
#             device = t.device
#             half_dim = self.dim // 2
#             embeddings = math.log(10000) / (half_dim - 1)
#             embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#             embeddings = t[:, None] * embeddings[None, :]
#             embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#             return embeddings

#     def _initialize_net(self) -> None:
#         if self._conv_backbone is None:
#             # time embeddings
#             self._t_dim = self._img_shape[1]    # height and width
#             self._time_mlp = nn.Sequential(
#                 self.SinusoidalPositionalEmbeddings(self._t_dim),
#                 nn.Linear(self._t_dim, 4 * self._t_dim),
#                 nn.ReLU(),
#                 nn.Linear(4 * self._t_dim, 4 * self._t_dim),
#                 nn.ReLU()
#             )
#             # Encode
#             self._down1 = self.Downsample(self._img_shape[2], 16, 5, maxpool=True, t_dim=4 * self._t_dim)
#             self._down2 = self.Downsample(16, 32, 5, maxpool=True, t_dim=4 * self._t_dim)
#             self._down3 = self.Downsample(32, 64, 5, maxpool=True, t_dim=4 * self._t_dim)
#             self._down4 = self.Downsample(64, 128, 5, maxpool=True, t_dim=4 * self._t_dim)
#             self._down5 = self.Downsample(128, 256, 4, maxpool=False, t_dim=4 * self._t_dim)
#             # Decode
#             # (h - 1) stride + 2 padding + kernel
#             self._up1 = self.Upsample(256, 128, kernel_size=4, stride=1, t_dim=4 * self._t_dim)
#             self._up2 = self.Upsample(128, 64, kernel_size=4, stride=3, t_dim=4 * self._t_dim)
#             self._up3 = self.Upsample(64, 32, kernel_size=6, stride=2, t_dim=4 * self._t_dim)
#             self._up4 = self.Upsample(32, 16, kernel_size=6, stride=2, t_dim=4 * self._t_dim)
#             self._out = nn.Sequential(
#                 nn.ConvTranspose2d(16, 8, 6, stride=2),
#                 nn.Conv2d(8, 1, kernel_size=3, padding='same'))

#     def fit(self, X: Array, img_shape: Array) -> None:
#         self._img_shape = img_shape
#         self.is_trained = True
#         num_data = X.shape[0]
#         logging.info(f"Training {self.__class__.__name__} on {num_data} "
#                      "datapoints")

#         self._initialize_net()
#         self.to(self._device) 
#         optimizer = self._create_optimizer()

#         tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self._device)
#         tensor_X = tensor_X.view(tensor_X.shape[0], *(self._img_shape)).permute((0, 3, 1, 2))

#         data = torch.utils.data.TensorDataset(tensor_X)
#         dataloader = torch.utils.data.DataLoader(data, batch_size=512, shuffle=True)

#         assert isinstance(self._max_train_iters, int)
#         self.train()
#         for itr in range(self._max_train_iters):
#             cum_loss = 0
#             n = 0
#             for tensor_X, in dataloader:
#                 t = torch.randint(0, self._timesteps, (tensor_X.shape[0],), device=self._device)
#                 loss = self._p_losses(tensor_X, t)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 cum_loss += loss.item() * tensor_X.shape[0]
#                 n += tensor_X.shape[0]
#             cum_loss /= n
#             if itr % 100 == 0:
#                 logging.info(f"Loss: {cum_loss:.5f}, iter: {itr}/{self._max_train_iters}")

#         self.eval()
#         logging.info(f"Trained model with loss: {cum_loss:.5f}")
#         return cum_loss


#     def forward(self, X, t) -> Tensor:
#         t_embeddings = self._time_mlp(t)
#         x1 = self._down1(X, t_embeddings)
#         x2 = self._down2(x1, t_embeddings)
#         x3 = self._down3(x2, t_embeddings)
#         x4 = self._down4(x3, t_embeddings)
#         x5 = self._down5(x4, t_embeddings)
#         x = self._up1(x5, x4, t_embeddings)
#         x = self._up2(x, x3, t_embeddings)
#         x = self._up3(x, x2, t_embeddings)
#         x = self._up4(x, x1, t_embeddings)
#         x = self._out(x)
#         return x
    
#     def _p_losses(self, X_start, t):
#         noise = torch.randn_like(X_start[:, -1])
#         X_noisy = self._q_sample(X_start[:, -1], t, noise)
#         X_noisy = torch.cat((X_start[:, :-1], X_noisy[:, None]), dim=1)
#         predicted_noise = self(X_noisy, t).squeeze()
#         loss = F.smooth_l1_loss(noise, predicted_noise)
#         return loss

#     @torch.no_grad()
#     def _p_sample(self, x, t, t_index):
#         y_out = x[:, -1].unsqueeze(dim=1)
#         betas_t = self._extract(self._betas, t, y_out.shape)
#         sqrt_one_minus_alphas_cumprod_t = self._extract(
#             self._sqrt_one_minus_alphas_cumprod, t, y_out.shape
#         )
#         sqrt_recip_alphas_t = self._extract(self._sqrt_recip_alphas, t, y_out.shape)
        
#         # Equation 11 in the paper
#         # Use our model (noise predictor) to predict the mean
#         # epsilon_out = self._forward_fcn(x_cond, y_out, t)
#         epsilon_out = self(x, t)
#         epsilon = epsilon_out#[:, :-1]
#         # out = epsilon_out[:, -1]
#         # grad = torch.autograd.grad(out.sum(), y_out)[0]
#         # epsilon -= (sqrt_one_minus_alphas_cumprod_t * grad)
#         model_mean = sqrt_recip_alphas_t * (
#             y_out - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
#         )

#         if t_index == 0:
#             return model_mean
#         else:
#             posterior_variance_t = self._extract(self._posterior_variance, t, y_out.shape)
#             noise = torch.randn_like(y_out)
#             # Algorithm 2 line 4:
#             return model_mean + torch.sqrt(posterior_variance_t) * noise 


#     @torch.no_grad()
#     def _p_sample_loop(self, x):
#         # start from pure noise (for each example in the batch)
#         x = x.view(x.shape[0], self._img_shape[0], self._img_shape[1], self._img_shape[2] - 1).permute((0, 3, 1, 2))
#         y_out = torch.randn(1, 1, self._img_shape[0], self._img_shape[1], device=self._device, requires_grad=True)
#         y_outs = []
#         for i in reversed(range(0, self._timesteps)):
#             y_out = self._p_sample(torch.cat((x, y_out), dim=1), torch.full((1,), i, device=self._device, dtype=torch.long), i)
#             y_outs.append(y_out.detach().cpu().numpy())
#         return y_outs


#     def predict_sample(self, x: Array, rng: np.random.Generator) -> Array:
#         x_tensor = torch.from_numpy(np.array(x, dtype=np.float32)).to(self._device).view(1, -1)
#         sample = self._p_sample_loop(x_tensor)[-1].squeeze()

#         import matplotlib
#         matplotlib.use('TkAgg')
#         import matplotlib.pyplot as plt
#         full_img = x.reshape(self._img_shape[0], self._img_shape[1], self._img_shape[2] - 1)
#         full_img = (full_img - full_img.min(axis=(0, 1))) / (full_img.max(axis=(0,1)) - full_img.min(axis=(0,1)))
#         env_img = full_img[:, :, :3]
#         local_img = full_img[:, :, 3]
#         # act_img = F.softmax(torch.from_numpy(sample).to(self._device).view(-1)).view(sample.shape).detach().cpu().numpy()
#         # act_img = (sample - sample.min()) / (sample.max() - sample.min())
#         act_img = (sample - sample.min()) / (sample.max() - sample.min()) > 0.5
#         print(act_img)
#         print(np.sort(act_img, axis=None))
#         plt.imshow(env_img)
#         plt.figure()
#         plt.imshow(local_img)
#         plt.figure()
#         plt.imshow(act_img)
#         plt.show()
#         exit()

#         i,j = np.unravel_index(sample.argmax(), self._img_shape[:2])

#         pos_x_low = j * 20 / self._img_shape[2]
#         pos_x_high = (j + 1) * 20 / self._img_shape[2]
#         pos_y_low = (self._img_shape[1] - 1 - i) * 20 / self._img_shape[1]
#         pos_y_high = (self._img_shape[1] - 1 - (i - 1)) * 20 / self._img_shape[1]

#         pos_x = rng.uniform(pos_x_low, pos_x_high)
#         pos_y = rng.uniform(pos_y_low, pos_y_high)
#         return pos_x, pos_y

class CNNDiffusionRegressor(DiffusionRegressor):
    # THIS IS THE IMAGE-ONLY VERSION

    def __init__(self, seed: int, hid_sizes: List[int],
                 max_train_iters: int, timesteps: int,
                 learning_rate: float) -> None:
        super().__init__(seed, hid_sizes, max_train_iters, timesteps, learning_rate)

        # Store information about CNN 
        self._conv_backbone = None

    class Downsample(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, maxpool, t_dim):
            super().__init__()
            self._conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            if maxpool:
                self._pool = nn.MaxPool2d(2, stride=2)
            else:
                self._pool = None
            self._t_linear = nn.Linear(t_dim, out_channels * 2)
            self._relu = nn.ReLU()

        def forward(self, x, t_embeddings):
            x = self._conv(x)
            if self._pool:
                x = self._pool(x)
            scale_shift = self._t_linear(t_embeddings)
            scale, shift = scale_shift.view(*(scale_shift.shape), 1, 1).chunk(2, dim=1)
            x = x * (scale + 1) + shift
            return self._relu(x)

    class SinusoidalPositionalEmbeddings(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            device = t.device
            half_dim = self.dim // 2
            embeddings = math.log(10000) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
            embeddings = t[:, None] * embeddings[None, :]
            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
            return embeddings

    def _initialize_net(self) -> None:
        if self._conv_backbone is None:
            # time embeddings
            self._t_dim = self._img_shape[1]    # height and width
            self._time_mlp = nn.Sequential(
                self.SinusoidalPositionalEmbeddings(self._t_dim),
                nn.Linear(self._t_dim, 4 * self._t_dim),
                nn.ReLU(),
                nn.Linear(4 * self._t_dim, 4 * self._t_dim),
                nn.ReLU()
            )
            # Encode
            # image_shape contains only the conditioning variables, but we'll convert the outputs to images and add one channel
            self._down1 = self.Downsample(self._img_shape[2] + 1, 16, 5, maxpool=True, t_dim=4 * self._t_dim)
            self._down2 = self.Downsample(16, 32, 5, maxpool=True, t_dim=4 * self._t_dim)
            self._down3 = self.Downsample(32, 64, 5, maxpool=True, t_dim=4 * self._t_dim)
            self._down4 = self.Downsample(64, 128, 5, maxpool=True, t_dim=4 * self._t_dim)
            self._down5 = self.Downsample(128, 256, 4, maxpool=False, t_dim=4 * self._t_dim)
            self._out = nn.Linear(256, self._out_dim)


    def fit(self, X: Array, Y: Array, Y_to_img_xform: Callable, xform_inputs: Array, Y_mean: Array, Y_std: Array, img_shape: Array) -> None:
        self._img_shape = img_shape
        self.is_trained = True
        self._out_dim = Y.shape[1]

        num_data = X.shape[0]
        logging.info(f"Training {self.__class__.__name__} on {num_data} "
                     "datapoints")

        self._initialize_net()
        self.to(self._device) 
        optimizer = self._create_optimizer()

        tensor_X = torch.from_numpy(np.array(X, dtype=np.float32)).to(self._device)
        tensor_X = tensor_X.view(tensor_X.shape[0], *(self._img_shape)).permute((0, 3, 1, 2))

        tensor_Y = torch.from_numpy(np.array(Y, dtype=np.float32)).to(self._device)
        tensor_xform_inputs = torch.from_numpy(np.array(xform_inputs, dtype=np.float32)).to(self._device)
        self._Y_to_img_xform = Y_to_img_xform
        self._Y_mean = torch.from_numpy(np.array(Y_mean, dtype=np.float32)).to(self._device)
        self._Y_std = torch.from_numpy(np.array(Y_std, dtype=np.float32)).to(self._device)
        self._Y_mean_np = Y_mean
        self._Y_std_np = Y_std

        data = torch.utils.data.TensorDataset(tensor_X, tensor_Y, tensor_xform_inputs)
        dataloader = torch.utils.data.DataLoader(data, batch_size=512, shuffle=True)

        assert isinstance(self._max_train_iters, int)
        self.train()
        for itr in range(self._max_train_iters):
            cum_loss = 0
            n = 0
            for tensor_X, tensor_Y, tensor_xform_inputs in dataloader:
                t = torch.randint(0, self._timesteps, (tensor_X.shape[0],), device=self._device)
                loss = self._p_losses(tensor_X, tensor_Y, tensor_xform_inputs, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cum_loss += loss.item() * tensor_X.shape[0]
                n += tensor_X.shape[0]
            cum_loss /= n
            if itr % 100 == 0:
                logging.info(f"Loss: {cum_loss:.5f}, iter: {itr}/{self._max_train_iters}")

        self.eval()
        logging.info(f"Trained model with loss: {cum_loss:.5f}")
        return cum_loss


    def forward(self, X, t) -> Tensor:
        t_embeddings = self._time_mlp(t)
        X = self._down1(X, t_embeddings)
        X = self._down2(X, t_embeddings)
        X = self._down3(X, t_embeddings)
        X = self._down4(X, t_embeddings)
        X = self._down5(X, t_embeddings)
        X = X.view(X.shape[0], -1)
        X = self._out(X)
        return X
    
    def _p_losses(self, X, Y_start, xform_inputs, t):
        noise = torch.randn_like(Y_start)
        Y_noisy = self._q_sample(Y_start, t, noise)

        # convert noisy sample into image
        Y_noisy_unnormalized = (Y_noisy * self._Y_std) + self._Y_mean
        Y_noisy_img = self._Y_to_img_xform(Y_noisy_unnormalized, xform_inputs, self._img_shape)
        # raveled_idx = Y_noisy_img.view(Y_noisy_img.shape[0], -1).argmax(dim=1)
        # i, j = np.unravel_index(raveled_idx, Y_noisy_img.shape[1:])
        X_noisy = torch.cat((X, Y_noisy_img[:, None]), dim=1)
        predicted_noise = self(X_noisy, t).squeeze()
        loss = F.smooth_l1_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def _p_sample(self, x, y_out, xform_inputs, t, t_index):
        betas_t = self._extract(self._betas, t, y_out.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self._sqrt_one_minus_alphas_cumprod, t, y_out.shape
        )
        sqrt_recip_alphas_t = self._extract(self._sqrt_recip_alphas, t, y_out.shape)
        
        # Convert y_out to img
        y_unnormalized = (y_out * self._Y_std) + self._Y_mean
        y_img = self._Y_to_img_xform(y_unnormalized, xform_inputs, self._img_shape)
        raveled_idx = y_img.view(y_img.shape[0], -1).argmax(dim=1)
        i, j = np.unravel_index(raveled_idx, y_img.shape[1:])
        x = torch.cat((x, y_img[:, None]), dim=1)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        # epsilon_out = self._forward_fcn(x_cond, y_out, t)
        epsilon_out = self(x, t)
        epsilon = epsilon_out#[:, :-1]
        # out = epsilon_out[:, -1]
        # grad = torch.autograd.grad(out.sum(), y_out)[0]
        # epsilon -= (sqrt_one_minus_alphas_cumprod_t * grad)
        model_mean = sqrt_recip_alphas_t * (
            y_out - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self._posterior_variance, t, y_out.shape)
            noise = torch.randn_like(y_out)
            # Algorithm 2 line 4:
            print('\t', t, epsilon)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 


    @torch.no_grad()
    def _p_sample_loop(self, x, xform_inputs):
        # start from pure noise (for each example in the batch)
        x = x.view(x.shape[0], self._img_shape[0], self._img_shape[1], self._img_shape[2]).permute((0, 3, 1, 2))
        y_out = torch.randn(1, self._out_dim, device=self._device, requires_grad=True)
        y_outs = []
        for i in reversed(range(0, self._timesteps)):
            y_out = self._p_sample(x, y_out, xform_inputs, torch.full((1,), i, device=self._device, dtype=torch.long), i)
            y_outs.append(y_out.detach().cpu().numpy())
        return y_outs


    def predict_sample(self, x: Array, xform_inputs: Array, rng: np.random.Generator) -> Array:
        x = torch.from_numpy(np.array(x, dtype=np.float32)).to(self._device)
        xform_inputs = torch.from_numpy(np.array(xform_inputs, dtype=np.float32)).to(self._device)
        sample = self._p_sample_loop(x.view(1, -1), xform_inputs.view(1, -1))[-1]
        return (sample * self._Y_std_np) + self._Y_mean_np