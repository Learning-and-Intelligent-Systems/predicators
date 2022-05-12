"""Script to test the hypothesis that NNs can go haywire in the following case:

- Train on data where there is some constraint on the input dimensions, e.g.,
  the second dimension is a function of the first dimension.
- Test on data where that constraint is very slightly violated.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from predicators.src import utils
from predicators.src.ml_models import MLPRegressor
from predicators.src.settings import CFG

NUM_TRAIN_DATA = 1000
NUM_TEST_DATA = 1000
PERTURB_AMOUNTS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

START_SEED = 456
NUM_SEEDS = 5

OTHER_SETTINGS: Dict[str, Any] = {
    # Uncomment to debug the pipeline.
    # "mlp_regressor_max_itr": 10,
}


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    all_results = []
    for seed in range(START_SEED, START_SEED + NUM_SEEDS):
        seed_results = _run_experiment(seed)
        all_results.extend(seed_results)
    # Display and save all data.
    df = pd.DataFrame(all_results)
    logging.info(df)
    raw_outfile = os.path.join(outdir, "manifold_test.csv")
    df.to_csv(raw_outfile)
    logging.info(f"Saved dataframe to {raw_outfile}")
    # Make a plot.
    plt.figure()
    seed_key = "seed"
    x_key = "perturb"
    y_key = "error"
    df = df[[seed_key, x_key, y_key]]
    xs = sorted(np.unique(df[x_key]))
    y_means, y_stds = [], []
    for x in xs:
        df_x = df[(df[x_key] == x)][y_key]
        mean = np.mean(df_x)
        std = np.std(df_x)
        y_means.append(mean)
        y_stds.append(std)
    plt.errorbar(xs, y_means, yerr=y_stds)
    plt.xscale("log")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title("Learning a Regressor with Input Constraints")
    plt.tight_layout()
    outfile = os.path.join(outdir, "manifold_test.png")
    plt.savefig(outfile)
    logging.info(f"Saved plot to {outfile}")


def _run_experiment(seed: int) -> List[Dict[str, Any]]:
    logging.info(f"\nRunning experiment for seed={seed}")
    utils.reset_config({"seed": seed, **OTHER_SETTINGS})
    rng = np.random.default_rng(seed)

    # Generate train data.
    train_X, train_Y = _generate_data(NUM_TRAIN_DATA, rng, constraint_perb=0.0)

    # Train model.
    model = MLPRegressor(seed=CFG.seed,
                         hid_sizes=CFG.mlp_regressor_hid_sizes,
                         max_train_iters=CFG.mlp_regressor_max_itr,
                         clip_gradients=CFG.mlp_regressor_clip_gradients,
                         clip_value=CFG.mlp_regressor_gradient_clip_value,
                         learning_rate=CFG.learning_rate)
    model.fit(train_X, train_Y)

    # Generate test data with different amounts of perturbation.
    results = []
    for perturb_amount in PERTURB_AMOUNTS:
        test_X, test_Y = _generate_data(NUM_TEST_DATA,
                                        rng,
                                        constraint_perb=perturb_amount)
        # Test model.
        pred_Y = [model.predict(x) for x in test_X]
        error = np.mean(np.sum(np.subtract(test_Y, pred_Y)**2, axis=1))
        logging.info(f"MSE for perturb={perturb_amount}: {error}")

        results.append({
            "perturb": perturb_amount,
            "seed": seed,
            "error": error,
        })

    return results


def _generate_data(num_data: int, rng: np.random.Generator,
                   constraint_perb: float) -> Tuple[NDArray, NDArray]:
    # Generate "main" inputs.
    X_main = rng.uniform(size=(num_data, 5))
    # Generate some "constrained" inputs.
    X_constrained = _constraint_fn(X_main)
    # Add noise to the constrained inputs.
    if constraint_perb > 0:
        X_constrained_noise = rng.uniform(-constraint_perb, constraint_perb,
                                          size=X_constrained.shape)
        X_constrained = X_constrained + X_constrained_noise
    # Complete the inputs by concatenation.
    X = np.concatenate((X_main, X_constrained), axis=1)
    # Get the outputs.
    Y = _output_fn(X)
    return (X, Y)


def _constraint_fn(X_main: NDArray) -> NDArray:
    # An arbitrary function.
    assert X_main.shape[1] >= 3
    x0 = X_main[:, 0]
    x1 = X_main[:, 1]
    x2 = X_main[:, 2]
    new_x = np.tanh(x0) * (np.sin(x1) + np.cos(x2) * np.sqrt(x1))
    return np.reshape(new_x, (X_main.shape[0], 1))


def _output_fn(X: NDArray) -> NDArray:
    # Make the function "easy" in terms of the constraint.
    y0 = 2.0 * _constraint_fn(X) - 1.0
    y1 = np.sin(_constraint_fn(X))
    return np.concatenate((y0, y1), axis=1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s",
                        handlers=[logging.StreamHandler()])
    _main()
