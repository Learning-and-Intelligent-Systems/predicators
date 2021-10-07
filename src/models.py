"""Models useful for classification/regression.
"""

import numpy as np
import tempfile
import os
import itertools
from collections import OrderedDict
import numpy as np
from scipy.stats import truncnorm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from predicators.src.settings import CFG


class NeuralGaussianRegressor(nn.Module):
    """NeuralGaussianRegressor definition.
    """
    def fit(self, X, Y):
        """Train regressor on the given data.
        Both X and Y are multi-dimensional.
        """
        assert X.ndim == 2
        assert Y.ndim == 2
        return self._fit(X, Y)

    def predict_mean(self, x):
        """Return a mean prediction on the given datapoint.
        x is single-dimensional.
        """
        assert x.ndim == 1
        mean, _ = self._predict_mean_var(x)
        return mean

    def predict_sample(self, x, rng):
        """Return a sampled prediction on the given datapoint.
        x is single-dimensional.
        """
        assert x.ndim == 1
        mean, variance = self._predict_mean_var(x)
        y = []
        for mu, sigma_sq in zip(mean, variance):
            y_i = truncnorm.rvs(-1.0*CFG.regressor_sample_clip,
                                CFG.regressor_sample_clip,
                                loc=mu, scale=np.sqrt(sigma_sq),
                                random_state=rng)
            y.append(y_i)
        return np.array(y)

    def forward(self, x):
        """Pytorch forward method.
        """
        x = torch.from_numpy(np.array(x, dtype=np.float32))
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

    def _fit(self, X, Y):
        torch.manual_seed(CFG.seed)
        # Infer input and output sizes from data
        num_data, input_size = X.shape
        _, output_size = Y.shape
        # Initialize net
        hid_sizes = CFG.regressor_hid_sizes
        self._initialize_net(input_size, hid_sizes, output_size)
        # Convert data to torch
        X = torch.from_numpy(np.array(X, dtype=np.float32))
        Y = torch.from_numpy(np.array(Y, dtype=np.float32))
        # Normalize data
        X, self._input_shift, self._input_scale = self._normalize_data(X)
        Y, self._output_shift, self._output_scale = self._normalize_data(Y)
        # Train
        print(f"Training {self.__class__.__name__} on {num_data} datapoints")
        self.train()  # switch to train mode
        itr = 0
        best_loss = float("inf")
        best_itr = 0
        model_name = tempfile.NamedTemporaryFile(delete=False).name
        while True:
            pred_mean, pred_var = self._split_prediction(self(X))
            loss = self._loss_fn(pred_mean, Y, pred_var)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_itr = itr
                # Save this best model
                torch.save(self.state_dict(), model_name)
            if itr % 100 == 0:
                print(f"Loss: {loss:.5f}, iter: {itr}/{CFG.regressor_max_itr}",
                      end="\r", flush=True)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            if itr == CFG.regressor_max_itr:
                print()
                break
            if itr-best_itr > CFG.n_iter_no_change:
                print("\nLoss did not improve after {CFG.n_iter_no_change} "
                      f"itrs, terminating at itr {itr}.")
                break
            itr += 1
        # Load best model
        self.load_state_dict(torch.load(model_name))
        os.remove(model_name)
        self.eval()  # switch to eval mode
        pred_mean, pred_var = self._split_prediction(self(X))
        loss = self._loss_fn(pred_mean, Y, pred_var)
        print(f"Loaded best model with loss: {loss:.5f}")

    def _initialize_net(self, in_size, hid_sizes, out_size):
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(in_size, hid_sizes[0]))
        for i in range(len(hid_sizes)-1):
            self._linears.append(nn.Linear(hid_sizes[i], hid_sizes[i+1]))
        # The 2 here is for mean and variance
        self._linears.append(nn.Linear(hid_sizes[-1], 2*out_size))
        self._optimizer = optim.Adam(self.parameters(), lr=CFG.learning_rate)
        self._loss_fn = nn.GaussianNLLLoss()

    @staticmethod
    def _split_prediction(x):
        return torch.split(x, x.shape[-1]//2, dim=-1)

    def _predict_mean_var(self, x):
        x = torch.from_numpy(np.array(x, dtype=np.float32))
        x = x.unsqueeze(dim=0)
        # Normalize input
        x = (x - self._input_shift) / self._input_scale
        mean, variance = self._split_prediction(self(x))
        # Normalize output
        mean = (mean * self._output_scale) + self._output_shift
        variance = variance * (self._output_scale**2)
        mean = mean.squeeze(dim=0).detach().numpy()
        variance = variance.squeeze(dim=0).detach().numpy()
        return mean, variance

    def _normalize_data(self, data):
        shift = torch.min(data, dim=0, keepdim=True).values
        scale = torch.max(data - shift, dim=0, keepdim=True).values
        scale = torch.clip(scale, min=CFG.normalization_scale_clip)
        return (data - shift) / scale, shift, scale


class MLPClassifier(nn.Module):
    """MLPClassifier definition.
    """
    def __init__(self, in_size) -> None:
        super().__init__()
        self._rng = np.random.default_rng(CFG.seed)
        hid_sizes = CFG.classifier_hid_sizes
        self._linears = nn.ModuleList()
        self._linears.append(nn.Linear(in_size, hid_sizes[0]))
        for i in range(len(hid_sizes)-1):
            self._linears.append(nn.Linear(hid_sizes[i], hid_sizes[i+1]))
        self._linears.append(nn.Linear(hid_sizes[-1], 1))

    def fit(self, X, y):
        """Train classifier on the given data.
        X is multi-dimensional, y is single-dimensional.
        """
        assert X.ndim == 2
        assert y.ndim == 1
        X, self._input_shift, self._input_scale = self._normalize_data(X)
        # Balance the classes
        if CFG.classifier_balance_data and len(y)//2 > sum(y):
            old_len = len(y)
            pos_idxs = list(np.argwhere(np.array(y) == 1).squeeze())
            neg_idxs = list(np.argwhere(np.array(y) == 0).squeeze())
            assert len(pos_idxs) + len(neg_idxs) == len(y) == len(X)
            keep_neg_idxs = list(self._rng.choice(neg_idxs, replace=False,
                                 size=len(pos_idxs)))
            keep_idxs = pos_idxs + keep_neg_idxs
            X = [X[i] for i in keep_idxs]
            y = [y[i] for i in keep_idxs]
            print(f"Reduced dataset size from {old_len} to {len(y)}")
        X = np.array(X)
        y = np.array(y)
        self._fit(X, y)
        return X, y

    def forward(self, x):
        """Pytorch forward method.
        """
        x = torch.from_numpy(np.array(x, dtype=np.float32))
        for _, linear in enumerate(self._linears[:-1]):
            x = F.relu(linear(x))
        x = self._linears[-1](x)
        return torch.sigmoid(x.squeeze(dim=-1))

    def classify(self, x):
        """Return a classification of the given datapoint.
        x is single-dimensional.
        """
        assert x.ndim == 1
        x = (x - self._input_shift) / self._input_scale
        classification = self._classify(x)
        assert classification in [False, True]
        return classification

    def _normalize_data(self, data):
        shift = np.min(data, axis=0)
        scale = np.max(data - shift, axis=0)
        scale = np.clip(scale, CFG.normalization_scale_clip, None)
        return (data - shift) / scale, shift, scale

    def _classify(self, x):
        return self(x).item() > 0.5

    def _fit(self, X, y):
        torch.manual_seed(CFG.seed)
        # Convert data to torch
        X = torch.from_numpy(np.array(X, dtype=np.float32))
        y = torch.from_numpy(np.array(y, dtype=np.float32))
        # Train
        print(f"Training {self.__class__.__name__} on {X.shape[0]} datapoints")
        self.train()  # switch to train mode
        itr = 0
        best_loss = float("inf")
        best_itr = 0
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
                print(f"Loss: {loss:.5f}, iter: {itr}/{CFG.classifier_max_itr}",
                      end="\r", flush=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if itr == CFG.classifier_max_itr:
                print()
                break
            if itr-best_itr > CFG.n_iter_no_change:
                print("\nLoss did not improve after {CFG.n_iter_no_change} "
                      f"itrs, terminating at itr {itr}.")
                break
            itr += 1
        # Load best model
        self.load_state_dict(torch.load(model_name))
        os.remove(model_name)
        self.eval()  # switch to eval mode
        yhat = self(X)
        loss = loss_fn(yhat, y)
        print(f"Loaded best model with loss: {loss:.5f}")
