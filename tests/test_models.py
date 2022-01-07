"""Tests for models."""

import time
import numpy as np
from predicators.src.torch_models import NeuralGaussianRegressor, MLPClassifier
from predicators.src.settings import CFG
from predicators.src import utils


def test_neural_gaussian_regressor():
    """Tests for NeuralGaussianRegressor."""
    utils.update_config({"seed": 123, "regressor_max_itr": 100})
    input_size = 3
    output_size = 2
    num_samples = 5
    model = NeuralGaussianRegressor()
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    mean = model.predict_mean(x)
    expected_y = np.zeros(output_size)
    assert mean.shape == expected_y.shape
    assert np.allclose(mean, expected_y, atol=1e-2)
    rng = np.random.default_rng(123)
    sample = model.predict_sample(x, rng)
    assert sample.shape == expected_y.shape


def test_mlp_classifier():
    """Tests for MLPClassifier."""
    utils.update_config({"seed": 123, "classifier_max_itr_sampler": 100})
    input_size = 3
    num_class_samples = 5
    X = np.concatenate([
        np.zeros((num_class_samples, input_size)),
        np.ones((num_class_samples, input_size))
    ])
    y = np.concatenate(
        [np.zeros((num_class_samples)),
         np.ones((num_class_samples))])
    model = MLPClassifier(input_size, CFG.classifier_max_itr_sampler)
    model.fit(X, y)
    prediction = model.classify(np.zeros(input_size))
    assert prediction == 0
    prediction = model.classify(np.ones(input_size))
    assert prediction == 1
    # Test for early stopping
    start_time = time.time()
    utils.update_config({
        "n_iter_no_change": 1,
        "classifier_max_itr_sampler": 10000,
        "learning_rate": 1e-2
    })
    model = MLPClassifier(input_size, CFG.classifier_max_itr_sampler)
    model.fit(X, y)
    assert time.time() - start_time < 3, "Didn't early stop"
