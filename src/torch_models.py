"""Models useful for classification/regression."""

import abc
import functools
import logging
import os
import tempfile
import dill as pkl
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from predicators.src import utils
from predicators.src.gnn.gnn import setup_graph_net
from predicators.src.gnn.gnn_utils import GraphDictDataset, \
    compute_normalizers, get_single_model_prediction, graph_batch_collate, \
    normalize_graph, train_model
from predicators.src.settings import CFG
from predicators.src.structs import Array, Object, State, GroundAtom, \
    _Option, Dict, ParameterizedOption

torch.use_deterministic_algorithms(mode=True)  # type: ignore


class Regressor(abc.ABC):
    """ABC for regressor classes."""

    @abc.abstractmethod
    def fit(self, X: Array, Y: Array) -> None:
        """Train regressor on the given data.

        X and Y are both two-dimensional.
        """
        raise NotImplementedError("Override me")

    @abc.abstractmethod
    def predict(self, arr: Array) -> Array:
        """Return a prediction for the given datapoint.

        arr is single-dimensional.
        """
        raise NotImplementedError("Override me")


class MLPRegressor(Regressor, nn.Module):
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

        Both X and Y are two-dimensional.
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

    def predict(self, arr: Array) -> Array:
        """Normalize, predict, un-normalize, and convert to array."""
        x = torch.from_numpy(np.array(arr, dtype=np.float32))
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
        logging.info(f"Training {self.__class__.__name__} on {num_data} "
                     "datapoints")
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
            if itr % 1000 == 0:
                logging.info(f"Loss: {loss:.5f}, iter: {itr}/{max_itrs}")
            self._optimizer.zero_grad()
            loss.backward()
            if CFG.mlp_regressor_clip_gradients:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), CFG.mlp_regressor_gradient_clip_value)
            self._optimizer.step()
            if itr == max_itrs:
                break
            itr += 1
        # Load best model
        self.load_state_dict(torch.load(model_name))  # type: ignore
        os.remove(model_name)
        self.eval()  # switch to eval mode
        yhat = self(X)
        loss = self._loss_fn(yhat, Y)
        logging.info(f"Loaded best model with loss: {loss:.5f}")


class ImplicitMLPRegressor(Regressor):
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

    def __init__(self) -> None:
        self._rng = np.random.default_rng(CFG.seed)
        # Set in fit().
        self._x_dim = 0
        self._y_dim = 0
        self._input_shift = np.zeros(1, dtype=np.float32)
        self._input_scale = np.zeros(1, dtype=np.float32)
        self._output_shift = np.zeros(1, dtype=np.float32)
        self._output_scale = np.zeros(1, dtype=np.float32)
        self._classifier = MLPClassifier(1, 1)

    def fit(self, X: Array, Y: Array) -> None:
        # Normalize everything right off the bat for simplicity.
        X, self._input_shift, self._input_scale = self._normalize_data(X)
        Y, self._output_shift, self._output_scale = self._normalize_data(Y)
        # Initialize the classifier.
        num_data, self._x_dim = X.shape
        assert Y.shape[0] == num_data
        logging.info(f"Training {self.__class__.__name__} on {num_data} "
                     "datapoints")
        self._y_dim = Y.shape[1]
        max_itr = CFG.implicit_mlp_regressor_max_itr
        self._classifier = MLPClassifier(in_size=(self._x_dim + self._y_dim),
                                         max_itr=max_itr,
                                         balance_data=False)
        # Create the negative data.
        neg_X, neg_Y = self._create_negative_data(X, Y)
        # Set up the data for the classifier.
        pos_concat_inputs = np.hstack([X, Y])
        neg_concat_inputs = np.hstack([neg_X, neg_Y])
        concat_inputs = np.vstack([pos_concat_inputs, neg_concat_inputs])
        targets = np.array([1 for _ in pos_concat_inputs] +
                           [0 for _ in neg_concat_inputs],
                           dtype=np.float32)
        # Train the classifier.
        self._classifier.fit(concat_inputs, targets)

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
        num_samples = CFG.implicit_mlp_regressor_num_negative_data_per_input
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

    def predict(self, arr: Array) -> Array:
        # This sampling-based inference method is okay in 1 dimension, but
        # won't work well with higher dimensions.
        assert arr.shape == (self._x_dim, )
        # Normalize.
        x = (arr - self._input_shift) / self._input_scale
        num_samples = CFG.implicit_mlp_regressor_num_samples_per_inference
        sample_ys = self._rng.uniform(size=(num_samples, self._y_dim))
        # Concatenate the x and ys.
        concat_xy = np.array([np.hstack([x, y]) for y in sample_ys],
                             dtype=np.float32)
        assert concat_xy.shape == (num_samples, self._x_dim + self._y_dim)
        # Pass through network.
        scores = self._classifier.predict_proba(concat_xy)
        # Find the highest probability sample.
        sample_idx = np.argmax(scores)
        norm_y = sample_ys[sample_idx]
        # Denormalize.
        denorm_y = (norm_y * self._output_scale) + self._output_shift
        return denorm_y

    @staticmethod
    def _normalize_data(data: Array) -> Tuple[Array, Array, Array]:
        shift = np.min(data, axis=0)  # type: ignore
        scale = np.max(data - shift, axis=0)  # type: ignore
        scale = np.clip(scale, CFG.normalization_scale_clip, None)
        return (data - shift) / scale, shift, scale


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

        Both X and Y are two-dimensional.
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
            y_i = rng.normal(loc=mu, scale=np.sqrt(sigma_sq))
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
        logging.info(f"Training {self.__class__.__name__} on {num_data} "
                     "datapoints")
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
            if itr % 1000 == 0:
                logging.info(f"Loss: {loss:.5f}, iter: {itr}/{max_itrs}")
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            if itr == max_itrs:
                break
            itr += 1
        # Load best model
        self.load_state_dict(torch.load(model_name))  # type: ignore
        os.remove(model_name)
        self.eval()  # switch to eval mode
        pred_mean, pred_var = self._split_prediction(self(X))
        loss = self._loss_fn(pred_mean, Y, pred_var)
        logging.info(f"Loaded best model with loss: {loss:.5f}")

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

        X is two-dimensional, y is single-dimensional.
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
                 seed: Optional[int] = None,
                 balance_data: bool = CFG.mlp_classifier_balance_data) -> None:
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
        self._balance_data = balance_data

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
        if self._balance_data and len(y) // 2 > sum(y):
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
            logging.info(f"Reduced dataset size from {old_len} to {len(y)}")
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

    def predict_proba(self, X: Array) -> Array:
        """Get the predicted probability that the input classifies to 1."""
        return self(X).detach().numpy()

    def _classify(self, x: Array) -> bool:
        return self(x).item() > 0.5

    def _fit(self, inputs: Array, outputs: Array) -> None:
        # Convert data to torch
        X = torch.from_numpy(np.array(inputs, dtype=np.float32))
        y = torch.from_numpy(np.array(outputs, dtype=np.float32))
        # Train
        logging.info(f"Training {self.__class__.__name__} on {X.shape[0]} "
                     "datapoints")
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
            if itr % 1000 == 0:
                logging.info(f"Loss: {loss:.5f}, iter: {itr}/{self._max_itr}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if itr == self._max_itr:
                break
            if itr - best_itr > n_iter_no_change:
                logging.info(f"Loss did not improve after {n_iter_no_change} "
                             f"itrs, terminating at itr {itr}.")
                break
            itr += 1
        # Load best model
        self.load_state_dict(torch.load(model_name))  # type: ignore
        os.remove(model_name)
        self.eval()  # switch to eval mode
        yhat = self(X)
        loss = loss_fn(yhat, y)
        logging.info(f"Loaded best model with loss: {loss:.5f}")


class MLPClassifierEnsemble(Classifier):
    """MLPClassifierEnsemble definition."""

    def __init__(self, in_size: int, max_itr: int, n: int) -> None:
        self._members = [
            MLPClassifier(in_size, max_itr, CFG.seed + i) for i in range(n)
        ]

    def fit(self, X: Array, y: Array) -> None:
        for i, member in enumerate(self._members):
            logging.info(f"Fitting member {i} of ensemble...")
            member.fit(X, y)

    def classify(self, x: Array) -> bool:
        avg = np.mean(self.predict_member_probas(x))
        classification = avg > 0.5
        assert classification in [True, False]
        return classification

    def predict_member_probas(self, x: Array) -> Array:
        """Return class probabilities predicted by each member."""
        assert x.ndim == 1
        ps = []
        for member in self._members:
            x_normalized = member.normalize(x)
            ps.append(member(x_normalized).item())
        return np.array(ps)

class GNNRegressor(): 
    """GNNRegressor definition."""

    def __init__(self, initial_options: Set[ParameterizedOption]) -> None:
        self._gnn: Any = None
        self._sorted_options = sorted(initial_options,
                                      key=lambda o: o.name)
        self._nullary_predicates: List[Predicate] = []
        self._max_option_objects = 0
        self._max_option_params = 0
        self._node_feature_to_index: Dict[Any, int] = {}
        self._edge_feature_to_index: Dict[Any, int] = {}
        self._input_normalizers: Dict = {}
        self._target_normalizers: Dict = {}
        self._data_exemplar: Tuple[Dict, Dict] = ({}, {})
        self._loss_fn = nn.GaussianNLLLoss()
        self._rng = np.random.default_rng(CFG.seed)

    @staticmethod
    def _split_prediction(x: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.split(x, x.shape[-1] // 2, dim=-1)  # type: ignore

    def predict_sample(self, state_feature, goal, goal_objs_to_states):
        print("input state_feature", state_feature)
        in_graph, object_to_node = self.graphify_single_input(
            state_feature, goal, goal_objs_to_states)
        node_to_object = {v: k for k, v in object_to_node.items()}
        type_to_node = defaultdict(set)
        for obj, node in object_to_node.items():
            type_to_node[obj.type.name].add(node)
        if CFG.gnn_policy_do_normalization:
            in_graph = normalize_graph(in_graph, self._input_normalizers)
        out_graph = get_single_model_prediction(self._gnn, in_graph)
        if CFG.gnn_policy_do_normalization:
            out_graph = normalize_graph(out_graph,
                                        self._target_normalizers,
                                        invert=True)
        mean, variance = self._split_prediction(torch.tensor(out_graph['globals']))
        variance = F.elu(variance) + 1
        y = []
        for mu, sigma_sq in zip(mean, variance):
            y_i = self._rng.normal(loc=mu, scale=np.sqrt(sigma_sq))
            y.append(y_i)
        print("mu", mu)
        print("sigma_sq", sigma_sq)
        print("output y", y)
        return np.array(y)
        
    def learn_from_graph_data(self, graph_data) -> None:
        
        # do I need to do the examples (?) 
        inputs, targets = graph_data
        example_inputs = [inputs[0]]
        example_targets = [targets[0]]
        self._data_exemplar = (example_inputs[0], example_targets[0])
        example_dataset = GraphDictDataset(example_inputs, example_targets)
        self._gnn = setup_graph_net(
            example_dataset,
            num_steps=CFG.gnn_policy_num_message_passing,
            layer_size=CFG.gnn_policy_layer_size)

        # Run training.
        if CFG.gnn_policy_use_validation_set:
            ## Split data, using 10% for validation.
            num_validation = max(1, int(len(inputs) * 0.1))
        else:
            num_validation = 0
        train_inputs = inputs[num_validation:]
        train_targets = targets[num_validation:]
        val_inputs = inputs[:num_validation]
        val_targets = targets[:num_validation]
        train_dataset = GraphDictDataset(train_inputs, train_targets)
        val_dataset = GraphDictDataset(val_inputs, val_targets)
        ## Set up Adam optimizer and dataloaders.
        optimizer = torch.optim.Adam(self._gnn.parameters(),
                                     lr=CFG.gnn_policy_learning_rate)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CFG.gnn_policy_batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      collate_fn=graph_batch_collate)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=CFG.gnn_policy_batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=graph_batch_collate)
        dataloaders = {"train": train_dataloader, "val": val_dataloader}
        ## Set up the optimization criteria.
        bce_loss = torch.nn.BCEWithLogitsLoss()
        crossent_loss = torch.nn.CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss()

        def _global_criterion(output: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
            # Combine losses from the one-hot option selection and
            # the continuous parameters.
            if self._max_option_params > 0:
                # params_loss = mse_loss(output, target)
                pred_mean, pred_var = self._split_prediction(output)
                pred_var = F.elu(pred_var) + 1
                params_loss = self._loss_fn(pred_mean, target, pred_var)
            else:
                params_loss = torch.tensor(0.0)
            return params_loss

        ## Launch training code.
        best_model_dict = train_model(
            self._gnn,
            dataloaders,
            optimizer=optimizer,
            criterion=None,
            global_criterion=_global_criterion,
            num_epochs=CFG.gnn_policy_num_epochs,
            do_validation=CFG.gnn_policy_use_validation_set)
        self._gnn.load_state_dict(best_model_dict)
        info = {
            "exemplar": self._data_exemplar,
            "state_dict": self._gnn.state_dict(),
            "nullary_predicates": self._nullary_predicates,
            "max_option_objects": self._max_option_objects,
            "max_option_params": self._max_option_params,
            "node_feature_to_index": self._node_feature_to_index,
            "edge_feature_to_index": self._edge_feature_to_index,
            "input_normalizers": self._input_normalizers,
            "target_normalizers": self._target_normalizers,
        }
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_None.gnn", "wb") as f:
            pkl.dump(info, f)

    def setup_fields(
        self, data: List[Tuple[State, Dict, _Option,
                               Set[GroundAtom]]]
    ) -> None:
        obj_types_set = set()
        nullary_predicates_set = set()
        unary_predicates_set = set()
        binary_predicates_set = set()
        obj_attrs_set = set()
        max_option_objects = 0
        max_option_params = 0

        # Go through the data, identifying the maximum number of option
        # objects and parameters, and the types/predicates/attributes.
        # For this version of setup_fields we don't use the state;
        # only the objects in the goal. We don't use the objects in sub either
        # which are implicitly used by being part of the globals vector. (?)
        for state, sub, option, goal in data:
            assert len(option.params.shape) == 1
            max_option_objects = max(max_option_objects, len(option.objects))
            max_option_params = max(max_option_params, option.params.shape[0])
            for atom in goal:
                arity = atom.predicate.arity
                assert arity <= 2, "Predicates with arity > 2 are not supported"
                if arity == 0:
                    nullary_predicates_set.add(atom.predicate)
                elif arity == 1:
                    unary_predicates_set.add(atom.predicate)
                elif arity == 2:
                    binary_predicates_set.add(atom.predicate)
                for obj in atom.objects:
                    obj_types_set.add(f"type_{obj.type.name}")
                    for feat in obj.type.feature_names:
                        obj_attrs_set.add(f"feat_{feat}")
        self._nullary_predicates = sorted(nullary_predicates_set)
        self._max_option_objects = max_option_objects
        self._max_option_params = max_option_params

        obj_types = sorted(obj_types_set)
        unary_predicates = sorted(unary_predicates_set)
        binary_predicates = sorted(binary_predicates_set)
        obj_attrs = sorted(obj_attrs_set)

        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        # Initialize input node features.
        self._node_feature_to_index = {}
        index = 0
        for obj_type in obj_types:
            self._node_feature_to_index[obj_type] = index
            index += 1
        for unary_predicate in unary_predicates:
            self._node_feature_to_index[unary_predicate] = index
            index += 1
        for unary_predicate in unary_predicates:
            self._node_feature_to_index[G(unary_predicate)] = index
            index += 1
        for obj_attr in obj_attrs:
            self._node_feature_to_index[obj_attr] = index
            index += 1

        # Initialize input edge features.
        self._edge_feature_to_index = {}
        index = 0
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[binary_predicate] = index
            index += 1
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[R(binary_predicate)] = index
            index += 1
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[G(binary_predicate)] = index
            index += 1
        for binary_predicate in binary_predicates:
            self._edge_feature_to_index[G(R(binary_predicate))] = index
            index += 1

    def graphify_data(
            self, inputs, targets) -> Tuple[List[Dict], List[Dict]]:
        graph_inputs = []
        graph_targets = []
        assert len(inputs) == len(targets)

        for (state_feature, goal, goal_objs_to_states), option in zip(inputs, targets):
            # Create input graph.
            graph_input, object_to_node = self.graphify_single_input(
                state_feature, goal, goal_objs_to_states)
            graph_inputs.append(graph_input)
            # Create target graph.
            ## First, copy over all unchanged fields.
            graph_target = {
                "n_node": graph_input["n_node"],
                "n_edge": graph_input["n_edge"],
                "edges": graph_input["edges"],
                "senders": graph_input["senders"],
                "receivers": graph_input["receivers"],
            }
            ## Next, set up the target node features.
            object_mask = np.zeros(
                (len(object_to_node), self._max_option_objects),
                dtype=np.int64)
            # for i, obj in enumerate(option.objects):
            #     object_mask[object_to_node[obj], i] = 1
            graph_target["nodes"] = object_mask
            ## Finally, set up the target globals.
            # option_index = self._sorted_options.index(option.parent)
            # onehot_target = np.zeros(len(self._sorted_options))
            # onehot_target[option_index] = 1
            # assert len(option.params.shape) == 1
            # params_target = np.zeros(self._max_option_params)
            # params_target[:option.params.shape[0]] = option.params
            graph_target["globals"] = torch.hstack((torch.tensor(option.params), torch.zeros(len(option.params)))) #np.r_[onehot_target, params_target] # correct for targets (?)
            graph_targets.append(graph_target)

        if CFG.gnn_policy_do_normalization:
            # Update normalization constants. Note that we do this for both
            # the input graph and the target graph.
            self._input_normalizers = compute_normalizers(graph_inputs)
            self._target_normalizers = compute_normalizers(graph_targets)
            graph_inputs = [
                normalize_graph(g, self._input_normalizers)
                for g in graph_inputs
            ]
            graph_targets = [
                normalize_graph(g, self._target_normalizers)
                for g in graph_targets
            ]

        return graph_inputs, graph_targets

    def graphify_single_input(self, state_feature, goal, goal_objs_to_states) -> Tuple[Dict, Dict]:
        # state is the concatenation of the affected objects in the operator
        # only pass in goal groundatoms
        # then pass in set of object states (for goal objects) 
        ## state is currently only used for its objs, so we can probably just pass in goal objs and 
        ## their states

        # import pdb; pdb.set_trace() 

        all_objects = list(goal_objs_to_states.keys())
        node_to_object = dict(enumerate(all_objects))
        object_to_node = {v: k for k, v in node_to_object.items()}
        num_objects = len(all_objects)
        num_node_features = len(self._node_feature_to_index)
        num_edge_features = len(self._edge_feature_to_index)

        G = functools.partial(utils.wrap_predicate, prefix="GOAL-")
        R = functools.partial(utils.wrap_predicate, prefix="REV-")

        graph = {}

        graph["globals"] = state_feature

        # Add nodes (one per object) and node features.
        graph["n_node"] = np.array(num_objects)
        node_features = np.zeros((num_objects, num_node_features))

        # ## Add node features for obj types.
        # for obj in all_objects:
        #     obj_index = object_to_node[obj]
        #     type_index = self._node_feature_to_index[f"type_{obj.type.name}"]
        #     node_features[obj_index, type_index] = 1
        
        # ## Add node features for unary atoms in goal.
        # for atom in goal:
        #     if atom.predicate.arity != 1:
        #         continue
        #     obj_index = object_to_node[atom.objects[0]]
        #     atom_index = self._node_feature_to_index[G(atom.predicate)]
        #     node_features[obj_index, atom_index] = 1

        # ## Add node features for state.
        # for obj in all_objects:
        #     obj_index = object_to_node[obj]
        #     for feat, val in zip(obj.type.feature_names, goal_objs_to_states[obj]):
        #         feat_index = self._node_feature_to_index[f"feat_{feat}"]
        #         node_features[obj_index, feat_index] = val

        graph["nodes"] = node_features

        # Deal with edge case (pun).
        num_edge_features = max(num_edge_features, 1)

        # Add edges (one between each pair of objects) and edge features.
        all_edge_features = np.zeros(
            (num_objects, num_objects, num_edge_features))

        # ## Add edge features for binary atoms in goal.
        # for atom in goal:
        #     if atom.predicate.arity != 2:
        #         continue
        #     pred_index = self._edge_feature_to_index[G(atom.predicate)]
        #     obj0_index = object_to_node[atom.objects[0]]
        #     obj1_index = object_to_node[atom.objects[1]]
        #     all_edge_features[obj0_index, obj1_index, pred_index] = 1

        # ## Add edge features for reversed binary atoms in goal.
        # for atom in goal:
        #     if atom.predicate.arity != 2:
        #         continue
        #     pred_index = self._edge_feature_to_index[G(R(atom.predicate))]
        #     obj0_index = object_to_node[atom.objects[0]]
        #     obj1_index = object_to_node[atom.objects[1]]
        #     # Note: the next line is reversed on purpose!
        #     all_edge_features[obj1_index, obj0_index, pred_index] = 1

        # Organize into expected representation.
        adjacency_mat = np.any(all_edge_features, axis=2)
        receivers, senders, edges = [], [], []
        for sender, receiver in np.argwhere(adjacency_mat):
            edge = all_edge_features[sender, receiver]
            senders.append(sender)
            receivers.append(receiver)
            edges.append(edge)

        n_edge = len(edges)
        graph["edges"] = np.reshape(edges, [n_edge, num_edge_features])
        graph["receivers"] = np.reshape(receivers, [n_edge]).astype(np.int64)
        graph["senders"] = np.reshape(senders, [n_edge]).astype(np.int64)
        graph["n_edge"] = np.reshape(n_edge, [1]).astype(np.int64)

        # import pdb; pdb.set_trace()

        return graph, object_to_node

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
