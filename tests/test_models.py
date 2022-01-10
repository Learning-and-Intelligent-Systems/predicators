"""Tests for models."""

import time
import numpy as np
from predicators.src.torch_models import (NeuralGaussianRegressor,
                                          MLPClassifier, MLPRegressor)
from predicators.src import utils


def test_basic_mlp_regressor():
    """Tests for MLPRegressor."""
    utils.update_config({
        "seed": 123,
        "mlp_regressor_max_itr": 100,
        "mlp_regressor_clip_gradients": True
    })
    input_size = 3
    output_size = 2
    num_samples = 5
    model = MLPRegressor()
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = np.zeros(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-2)
    # Test with nonzero outputs.
    Y = 75 * np.ones((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = 75 * np.ones(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-2)


def test_neural_gaussian_regressor():
    """Tests for NeuralGaussianRegressor."""
    utils.update_config({"seed": 123, "neural_gaus_regressor_max_itr": 100})
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
    utils.update_config({"seed": 123})
    input_size = 3
    num_class_samples = 5
    X = np.concatenate([
        np.zeros((num_class_samples, input_size)),
        np.ones((num_class_samples, input_size))
    ])
    y = np.concatenate(
        [np.zeros((num_class_samples)),
         np.ones((num_class_samples))])
    model = MLPClassifier(input_size, 100)
    model.fit(X, y)
    prediction = model.classify(np.zeros(input_size))
    assert prediction == 0
    prediction = model.classify(np.ones(input_size))
    assert prediction == 1
    # Test for early stopping
    start_time = time.time()
    utils.update_config({
        "mlp_classifier_n_iter_no_change": 1,
        "learning_rate": 1e-2
    })
    model = MLPClassifier(input_size, 10000)
    model.fit(X, y)
    assert time.time() - start_time < 3, "Didn't early stop"
