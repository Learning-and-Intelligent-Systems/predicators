"""Models useful for classification/regression."""

import abc
import os
from dataclasses import dataclass
import tempfile
from typing import Sequence, List, Tuple, Optional
from scipy.stats import truncnorm
import torch
from torch import nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from predicators.src.structs import Array, Object, State
from predicators.src.settings import CFG

torch.use_deterministic_algorithms(mode=True)  # type: ignore


class MLPRegressor(nn.Module):
    """A basic multilayer perceptron regressor."""

    def __init__(self) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()  # type: ignore
        self._input_shift = torch.zeros(1)
        self._input_scale = torch.zeros(1)
        self._output_shift = torch.zeros(1)
        self._output_scale = torch.zeros(1)
        self._linears = nn.ModuleList()
        self._loss_fn = nn.MSELoss()

    def fit(self, X: Array, Y: Array) -> None:
        """Train regressor on the given data.

        Both X and Y are multi-dimensional.
        """
        assert X.ndim == 2
        assert Y.ndim == 2
        return self._fit(X, Y)

    def forward(self, inputs: Array) -> Tensor:
        """Pytorch forward method."""
        x = torch.from_numpy(np.array(inputs, dtype=np.float32))
        for _, linear in enumerate(self._linears[:-1]):
            x = F.relu(linear(x))
        x = self._linears[-1](x)
        return x

    def predict(self, inputs: Array) -> Array:
        """Normalize, predict, un-normalize, and convert to array."""
        x = torch.from_numpy(np.array(inputs, dtype=np.float32))
        x = x.unsqueeze(dim=0)
        # Normalize input
        x = (x - self._input_shift) / self._input_scale
        y = self(x)
        # Un-normalize output
        y = (y * self._output_scale) + self._output_shift
        y = y.squeeze(dim=0)
        y = y.detach()  # type: ignore
        return y.numpy()

    def _initialize_net(self, in_size: int, hid_sizes: List[int],
                        out_size: int) -> None:
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(in_size, hid_sizes[0]))
        for i in range(len(hid_sizes) - 1):
            self._linears.append(nn.Linear(hid_sizes[i], hid_sizes[i + 1]))
        self._linears.append(nn.Linear(hid_sizes[-1], out_size))
        self._optimizer = optim.Adam(  # pylint: disable=attribute-defined-outside-init
            self.parameters(),
            lr=CFG.learning_rate)

    @staticmethod
    def _normalize_data(data: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        shift = torch.min(data, dim=0, keepdim=True).values
        scale = torch.max(data - shift, dim=0, keepdim=True).values
        scale = torch.clip(scale, min=CFG.normalization_scale_clip)
        return (data - shift) / scale, shift, scale

    def _fit(self, inputs: Array, outputs: Array) -> None:
        torch.manual_seed(CFG.seed)
        # Infer input and output sizes from data
        num_data, input_size = inputs.shape
        _, output_size = outputs.shape
        # Initialize net
        hid_sizes = CFG.mlp_regressor_hid_sizes
        self._initialize_net(input_size, hid_sizes, output_size)
        # Convert data to torch
        X = torch.from_numpy(np.array(inputs, dtype=np.float32))
        Y = torch.from_numpy(np.array(outputs, dtype=np.float32))
        # Normalize data
        X, self._input_shift, self._input_scale = self._normalize_data(X)
        Y, self._output_shift, self._output_scale = self._normalize_data(Y)
        # Train
        print(f"Training MLPRegressor on {num_data} datapoints")
        self.train()  # switch to train mode
        itr = 0
        max_itrs = CFG.mlp_regressor_max_itr
        best_loss = float("inf")
        model_name = tempfile.NamedTemporaryFile(delete=False).name
        while True:
            yhat = self(X)
            loss = self._loss_fn(yhat, Y)
            if loss.item() < best_loss:
                best_loss = loss.item()
                # Save this best model
                torch.save(self.state_dict(), model_name)
            if itr % 100 == 0:
                print(f"Loss: {loss:.5f}, iter: {itr}/{max_itrs}",
                      end="\r",
                      flush=True)
            self._optimizer.zero_grad()
            loss.backward()
            if CFG.mlp_regressor_clip_gradients:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), CFG.mlp_regressor_gradient_clip_value)
            self._optimizer.step()
            if itr == max_itrs:
                print()
                break
            itr += 1
        # Load best model
        self.load_state_dict(torch.load(model_name))  # type: ignore
        os.remove(model_name)
        self.eval()  # switch to eval mode
        yhat = self(X)
        loss = self._loss_fn(yhat, Y)
        print(f"Loaded best model with loss: {loss:.5f}")


class NeuralGaussianRegressor(nn.Module):
    """NeuralGaussianRegressor definition."""

    def __init__(self) -> None:  # pylint: disable=useless-super-delegation
        super().__init__()  # type: ignore
        self._input_shift = torch.zeros(1)
        self._input_scale = torch.zeros(1)
        self._output_shift = torch.zeros(1)
        self._output_scale = torch.zeros(1)
        self._linears = nn.ModuleList()
        self._loss_fn = nn.GaussianNLLLoss()

    def fit(self, X: Array, Y: Array) -> None:
        """Train regressor on the given data.

        Both X and Y are multi-dimensional.
        """
        assert X.ndim == 2
        assert Y.ndim == 2
        return self._fit(X, Y)

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
            y_i = truncnorm.rvs(-1.0 * CFG.neural_gaus_regressor_sample_clip,
                                CFG.neural_gaus_regressor_sample_clip,
                                loc=mu,
                                scale=np.sqrt(sigma_sq),
                                random_state=rng)
            y.append(y_i)
        return np.array(y)

    def forward(self, inputs: Array) -> Tensor:
        """Pytorch forward method."""
        x = torch.from_numpy(np.array(inputs, dtype=np.float32))
        for _, linear in enumerate(self._linears[:-1]):
            x = F.relu(linear(x))
        x = self._linears[-1](x)
        # Force pred var positive.
        # Note: use of elu here is very important. Tried several other things
        # and none worked. Use of elu recommended here:
        # https://engineering.taboola.com/predicting-probability-distributions/
        mean, variance = self._split_prediction(x)
        variance = F.elu(variance) + 1
        x = torch.cat([mean, variance], dim=-1)
        return x

    def _fit(self, inputs: Array, outputs: Array) -> None:
        torch.manual_seed(CFG.seed)
        # Infer input and output sizes from data
        num_data, input_size = inputs.shape
        _, output_size = outputs.shape
        # Initialize net
        hid_sizes = CFG.neural_gaus_regressor_hid_sizes
        self._initialize_net(input_size, hid_sizes, output_size)
        # Convert data to torch
        X = torch.from_numpy(np.array(inputs, dtype=np.float32))
        Y = torch.from_numpy(np.array(outputs, dtype=np.float32))
        # Normalize data
        X, self._input_shift, self._input_scale = self._normalize_data(X)
        Y, self._output_shift, self._output_scale = self._normalize_data(Y)
        # Train
        print(f"Training {self.__class__.__name__} on {num_data} datapoints")
        self.train()  # switch to train mode
        itr = 0
        max_itrs = CFG.neural_gaus_regressor_max_itr
        best_loss = float("inf")
        model_name = tempfile.NamedTemporaryFile(delete=False).name
        while True:
            pred_mean, pred_var = self._split_prediction(self(X))
            loss = self._loss_fn(pred_mean, Y, pred_var)
            if loss.item() < best_loss:
                best_loss = loss.item()
                # Save this best model
                torch.save(self.state_dict(), model_name)
            if itr % 100 == 0:
                print(f"Loss: {loss:.5f}, iter: {itr}/{max_itrs}",
                      end="\r",
                      flush=True)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            if itr == max_itrs:
                print()
                break
            itr += 1
        # Load best model
        self.load_state_dict(torch.load(model_name))  # type: ignore
        os.remove(model_name)
        self.eval()  # switch to eval mode
        pred_mean, pred_var = self._split_prediction(self(X))
        loss = self._loss_fn(pred_mean, Y, pred_var)
        print(f"Loaded best model with loss: {loss:.5f}")

    def _initialize_net(self, in_size: int, hid_sizes: List[int],
                        out_size: int) -> None:
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(in_size, hid_sizes[0]))
        for i in range(len(hid_sizes) - 1):
            self._linears.append(nn.Linear(hid_sizes[i], hid_sizes[i + 1]))
        # The 2 here is for mean and variance
        self._linears.append(nn.Linear(hid_sizes[-1], 2 * out_size))
        self._optimizer = optim.Adam(  # pylint: disable=attribute-defined-outside-init
            self.parameters(),
            lr=CFG.learning_rate)

    @staticmethod
    def _split_prediction(x: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.split(x, x.shape[-1] // 2, dim=-1)  # type: ignore

    def _predict_mean_var(self, inputs: Array) -> Tuple[Array, Array]:
        x = torch.from_numpy(np.array(inputs, dtype=np.float32))
        x = x.unsqueeze(dim=0)
        # Normalize input
        x = (x - self._input_shift) / self._input_scale
        mean, variance = self._split_prediction(self(x))
        # Normalize output
        mean = (mean * self._output_scale) + self._output_shift
        variance = variance * (torch.square(self._output_scale))
        mean = mean.squeeze(dim=0)
        mean = mean.detach()  # type: ignore
        np_mean = mean.numpy()
        variance = variance.squeeze(dim=0)
        variance = variance.detach()  # type: ignore
        np_variance = variance.numpy()
        return np_mean, np_variance

    @staticmethod
    def _normalize_data(data: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        shift = torch.min(data, dim=0, keepdim=True).values
        scale = torch.max(data - shift, dim=0, keepdim=True).values
        scale = torch.clip(scale, min=CFG.normalization_scale_clip)
        return (data - shift) / scale, shift, scale


class Classifier(abc.ABC):
    """ABC for classifier types."""

    @abc.abstractmethod
    def fit(self, X: Array, y: Array) -> None:
        """Train classifier on the given data.

        X is multi-dimensional, y is single-dimensional.
        """
        raise NotImplementedError("Override me")

    @abc.abstractmethod
    def classify(self, x: Array) -> bool:
        """Return a classification of the given datapoint.

        x is single-dimensional.
        """
        raise NotImplementedError("Override me")


class MLPClassifier(Classifier, nn.Module):
    """MLPClassifier definition."""

    def __init__(self,
                 in_size: int,
                 max_itr: int,
                 seed: Optional[int] = None) -> None:
        super().__init__()  # type: ignore
        if seed is None:
            self._rng = np.random.default_rng(CFG.seed)
            torch.manual_seed(CFG.seed)
        else:
            self._rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
        hid_sizes = CFG.mlp_classifier_hid_sizes
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(in_size, hid_sizes[0]))
        for i in range(len(hid_sizes) - 1):
            self._linears.append(nn.Linear(hid_sizes[i], hid_sizes[i + 1]))
        self._linears.append(nn.Linear(hid_sizes[-1], 1))
        self._input_shift = np.zeros(1, dtype=np.float32)
        self._input_scale = np.zeros(1, dtype=np.float32)
        self._max_itr = max_itr
        self._do_single_class_prediction = False
        self._predicted_single_class = False

    def fit(self, X: Array, y: Array) -> None:
        assert X.ndim == 2
        assert y.ndim == 1
        # If there is only one class in the data, then there's no point in
        # learning a NN, since any predictions other than that one class
        # could only be strange generalization issues.
        if np.all(y == 0):
            self._do_single_class_prediction = True
            self._predicted_single_class = False
            return
        if np.all(y == 1):
            self._do_single_class_prediction = True
            self._predicted_single_class = True
            return
        X, self._input_shift, self._input_scale = self._normalize_data(X)
        # Balance the classes
        if CFG.mlp_classifier_balance_data and len(y) // 2 > sum(y):
            old_len = len(y)
            pos_idxs_np = np.argwhere(np.array(y) == 1).squeeze()
            neg_idxs_np = np.argwhere(np.array(y) == 0).squeeze()
            pos_idxs = ([pos_idxs_np.item()]
                        if not pos_idxs_np.shape else list(pos_idxs_np))
            neg_idxs = ([neg_idxs_np.item()]
                        if not neg_idxs_np.shape else list(neg_idxs_np))
            assert len(pos_idxs) + len(neg_idxs) == len(y) == len(X)
            keep_neg_idxs = list(
                self._rng.choice(neg_idxs, replace=False, size=len(pos_idxs)))
            keep_idxs = pos_idxs + keep_neg_idxs
            X_lst = [X[i] for i in keep_idxs]
            y_lst = [y[i] for i in keep_idxs]
            X = np.array(X_lst)
            y = np.array(y_lst)
            print(f"Reduced dataset size from {old_len} to {len(y)}")
        self._fit(X, y)

    def forward(self, inputs: Array) -> Tensor:
        """Pytorch forward method."""
        assert not self._do_single_class_prediction
        x = torch.from_numpy(np.array(inputs, dtype=np.float32))
        for _, linear in enumerate(self._linears[:-1]):
            x = F.relu(linear(x))
        x = self._linears[-1](x)
        return torch.sigmoid(x.squeeze(dim=-1))

    def classify(self, x: Array) -> bool:
        assert x.ndim == 1
        if self._do_single_class_prediction:
            classification = self._predicted_single_class
        else:
            x = self.normalize(x)
            classification = self._classify(x)
        assert classification in [False, True]
        return classification

    @staticmethod
    def _normalize_data(data: Array) -> Tuple[Array, Array, Array]:
        shift = np.min(data, axis=0)  # type: ignore
        scale = np.max(data - shift, axis=0)  # type: ignore
        scale = np.clip(scale, CFG.normalization_scale_clip, None)
        return (data - shift) / scale, shift, scale

    def normalize(self, x: Array) -> Array:
        """Apply shift and scale to the input."""
        return (x - self._input_shift) / self._input_scale

    def _classify(self, x: Array) -> bool:
        return self(x).item() > 0.5

    def _fit(self, inputs: Array, outputs: Array) -> None:
        # Convert data to torch
        X = torch.from_numpy(np.array(inputs, dtype=np.float32))
        y = torch.from_numpy(np.array(outputs, dtype=np.float32))
        # Train
        print(f"Training {self.__class__.__name__} on {X.shape[0]} datapoints")
        self.train()  # switch to train mode
        itr = 0
        best_loss = float("inf")
        best_itr = 0
        n_iter_no_change = CFG.mlp_classifier_n_iter_no_change
        model_name = tempfile.NamedTemporaryFile(delete=False).name
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=CFG.learning_rate)
        while True:
            yhat = self(X)
            loss = loss_fn(yhat, y)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_itr = itr
                # Save this best model
                torch.save(self.state_dict(), model_name)
            if itr % 100 == 0:
                print(f"Loss: {loss:.5f}, iter: {itr}/{self._max_itr}",
                      end="\r",
                      flush=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if itr == self._max_itr:
                print()
                break
            if itr - best_itr > n_iter_no_change:
                print(f"\nLoss did not improve after {n_iter_no_change} "
                      f"itrs, terminating at itr {itr}.")
                break
            itr += 1
        # Load best model
        self.load_state_dict(torch.load(model_name))  # type: ignore
        os.remove(model_name)
        self.eval()  # switch to eval mode
        yhat = self(X)
        loss = loss_fn(yhat, y)
        print(f"Loaded best model with loss: {loss:.5f}")


class MLPClassifierEnsemble(Classifier):
    """MLPClassifierEnsemble definition."""

    def __init__(self, in_size: int, max_itr: int, n: int) -> None:
        self._members = [
            MLPClassifier(in_size, max_itr, CFG.seed + i) for i in range(n)
        ]

    def fit(self, X: Array, y: Array) -> None:
        for i, member in enumerate(self._members):
            print(f"Fitting member {i} of ensemble...")
            member.fit(X, y)

    def classify(self, x: Array) -> bool:
        avg = np.mean(self.predict_proba(x))
        classification = avg > 0.5
        assert classification in [True, False]
        return classification

    def predict_proba(self, x: Array) -> Array:
        """Return logits calculated by each member."""
        assert x.ndim == 1
        ps = []
        for member in self._members:
            x_normalized = member.normalize(x)
            ps.append(member(x_normalized).item())
        return np.array(ps)


@dataclass(frozen=True, eq=False, repr=False)
class LearnedPredicateClassifier:
    """A convenience class for holding the model underlying a learned
    predicate."""
    _model: Classifier

    def classifier(self, state: State, objects: Sequence[Object]) -> bool:
        """The classifier corresponding to the given model.

        May be used as the _classifier field in a Predicate.
        """
        v = state.vec(objects)
        return self._model.classify(v)
