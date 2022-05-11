"""A standalone script to experiment with learning the function to open doors
in the doors environment.

Notes:
    * Not using [1.0] in input. I don't think it matters.
    * Not using a binary classifier. Train the main method with
      --sampler_disable_classifier True for direct comparison.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from predicators.src import utils
from predicators.src.ml_models import DegenerateMLPDistributionRegressor, \
    MLPRegressor, NeuralGaussianRegressor
from predicators.src.settings import CFG
from predicators.src.structs import Array

# Hardcoding these values for convenience.
# State dims: [rx, ry, x, y, theta, mass, friction, rot, target_rot, open]
NUM_STATE_DIMS = 10
MASS, FRICTION, ROT, TARGET_ROT, OPEN = range(5, 10)
# Param dims: [delta_rot, delta_open]
NUM_PARAM_DIMS = 2

NUM_TRAIN_DATA = [50, 100, 250, 500, 1000]
START_SEED = 678
NUM_SEEDS = 5

OTHER_SETTINGS: Dict[str, Any] = {
    # Uncomment to debug the pipeline.
    # "mlp_regressor_max_itr": 10,
    # "neural_gaus_regressor_max_itr": 10,
    # "sesame_max_samples_per_step": 1,
}


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    all_results = []
    metrics: Optional[List[str]] = None
    for seed in range(START_SEED, START_SEED + NUM_SEEDS):
        for num_train_data in NUM_TRAIN_DATA:
            results = _run_experiment(seed, num_train_data)
            if metrics is None:
                metrics = sorted(results)
            assert metrics == sorted(results)
            metric_values = [results[m] for m in metrics]
            row = (
                seed,
                num_train_data,
            ) + tuple(metric_values)
            all_results.append(row)
    assert metrics is not None
    # Display and save all data.
    df = pd.DataFrame(all_results)
    df.columns = ["Seed", "Num Train Data"] + metrics
    logging.info(df)
    raw_outfile = os.path.join(outdir, "door_opening_raw.csv")
    df.to_csv(raw_outfile)
    logging.info(f"Saved dataframe to {raw_outfile}")
    # Make a plot.
    plt.figure()
    x_key = "Num Train Data"
    param_df = df[["Seed", x_key, "Param Error"]]
    nonparam_df = df[["Seed", x_key, "Nonparam Error"]]
    xs = sorted(np.unique(df[x_key]))
    for model_df, y_key in zip([param_df, nonparam_df],
                               ["Param Error", "Nonparam Error"]):
        model_y_means = []
        model_y_stds = []
        for x in xs:
            df_x = model_df[model_df[x_key] == x][y_key]
            mean = np.mean(df_x)
            std = np.std(df_x)
            model_y_means.append(mean)
            model_y_stds.append(std)
        # Add a line to the plot.
        plt.errorbar(xs, model_y_means, yerr=model_y_stds, label=y_key)
    plt.xlabel(x_key)
    plt.ylabel("Error")
    plt.title("Standalone Door Learning")
    plt.legend()
    plt.tight_layout()
    outfile = os.path.join(outdir, "door_opening.png")
    plt.savefig(outfile)
    logging.info(f"Saved plot to {outfile}")


def _run_experiment(seed: int,
                    num_train_data: int,
                    num_test_data: int = 1000,
                    sampler_type: str = "gaussian") -> Dict[str, float]:
    utils.reset_config({"seed": seed, **OTHER_SETTINGS})
    logging.info(f"\nRunning experiment for seed={seed}, "
                 f"num_train_data={num_train_data}")
    metrics: Dict[str, float] = {}

    # Generate data.
    X, Y = _generate_data(num_train_data + num_test_data)
    train_X, full_test_X = X[:num_train_data], X[num_train_data:]
    train_Y, test_Y = Y[:num_train_data], Y[num_train_data:]

    # At test time, we don't have the parameterized dims.
    test_X = full_test_X[:, :NUM_STATE_DIMS]

    # Get the data for the nonparameterized model. This effectively removes
    # part of the inputs corresponding to the expected next state.
    assert train_X.shape == (num_train_data, NUM_STATE_DIMS + NUM_PARAM_DIMS)
    nonparam_train_X = train_X[:, :NUM_STATE_DIMS]
    # Train a nonparameterized model.
    nonparam_model = MLPRegressor(
        seed=CFG.seed,
        hid_sizes=CFG.mlp_regressor_hid_sizes,
        max_train_iters=CFG.mlp_regressor_max_itr,
        clip_gradients=CFG.mlp_regressor_clip_gradients,
        clip_value=CFG.mlp_regressor_gradient_clip_value,
        learning_rate=CFG.learning_rate)
    nonparam_model.fit(nonparam_train_X, train_Y)
    # Evaluate the nonparameterized model.
    pred_Y = np.array([nonparam_model.predict(x) for x in test_X])

    def error_fn(yhat: Array, y: Array) -> float:
        return np.sum((yhat - y)**2)

    errors = [error_fn(yhat, y) for yhat, y in zip(pred_Y, test_Y)]
    error = np.mean(errors)
    logging.info(f"Nonparam Error: {error}")
    metrics["Nonparam Error"] = error

    # Get the data for the sampler for the parameterized model.
    sampler_train_X = train_X[:, :NUM_STATE_DIMS]  # same as nonparam_train_X
    sampler_train_Y = train_X[:, NUM_STATE_DIMS:]
    assert sampler_train_Y.shape == (num_train_data, NUM_PARAM_DIMS)

    # Train the sampler for the parameterized model.
    if sampler_type == "gaussian":
        sampler = NeuralGaussianRegressor(
            seed=CFG.seed,
            hid_sizes=CFG.neural_gaus_regressor_hid_sizes,
            max_train_iters=CFG.neural_gaus_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate)
    else:
        assert sampler_type == "degenerate"
        sampler = DegenerateMLPDistributionRegressor(
            seed=CFG.seed,
            hid_sizes=CFG.mlp_regressor_hid_sizes,
            max_train_iters=CFG.mlp_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate)

    sampler.fit(sampler_train_X, sampler_train_Y)

    # Train the parameterized model.
    param_model = MLPRegressor(
        seed=CFG.seed,
        hid_sizes=CFG.mlp_regressor_hid_sizes,
        max_train_iters=CFG.mlp_regressor_max_itr,
        clip_gradients=CFG.mlp_regressor_clip_gradients,
        clip_value=CFG.mlp_regressor_gradient_clip_value,
        learning_rate=CFG.learning_rate)
    param_model.fit(train_X, train_Y)

    # Evaluate the parameterized model by sampling a fixed number of times and
    # keeping the min error over the samples.
    num_samples = CFG.sesame_max_samples_per_step
    errors = []
    rng = np.random.default_rng(CFG.seed)
    for x, y in zip(test_X, test_Y):
        samples = [sampler.predict_sample(x, rng) for _ in range(num_samples)]
        preds_for_x = [param_model.predict(np.hstack([x, s])) for s in samples]
        errors_for_x = [error_fn(yhat, y) for yhat in preds_for_x]
        min_error = min(errors_for_x)
        errors.append(min_error)
    error = np.mean(errors)
    logging.info(f"Param Error: {error}")
    metrics["Param Error"] = error
    return metrics


def _generate_data(num_data: int) -> Tuple[Array, Array]:
    rng = np.random.default_rng()
    # Generate random state inputs.
    state_inputs = rng.uniform(size=(num_data, NUM_STATE_DIMS))
    state_inputs[:, OPEN] = 0.0
    # Generate the targets based on the state inputs.
    mass = state_inputs[:, MASS]
    friction = state_inputs[:, FRICTION]
    target_rot = state_inputs[:, TARGET_ROT]
    ground_truth_outs = _ground_truth_function(mass, friction, target_rot)
    # Generate the parameterized inputs based on the outputs. For this example,
    # the parameterized outputs are the difference between the current rot and
    # the output, and then just 1.0 for the door opening.
    rot = state_inputs[:, ROT]
    delta_rot = ground_truth_outs - rot
    delta_open = np.ones_like(delta_rot)
    param_inputs = np.vstack([delta_rot, delta_open]).T
    concat_inputs = np.concatenate([state_inputs, param_inputs], axis=1)
    inputs = concat_inputs.astype(np.float32)
    outputs = np.reshape(ground_truth_outs, (num_data, 1)).astype(np.float32)
    return (inputs, outputs)


def _ground_truth_function(mass: NDArray, friction: NDArray,
                           target_rot: NDArray) -> NDArray:
    return np.tanh(target_rot) * (np.sin(mass) +
                                  np.cos(friction) * np.sqrt(mass))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s",
                        handlers=[logging.StreamHandler()])
    _main()
