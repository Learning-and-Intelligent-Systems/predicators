"""Analysis for spot cube placing with active sampler learning."""

import glob
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import dill as pkl
import imageio
import matplotlib.pyplot as plt
import numpy as np
from bosdyn.client import math_helpers
from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators import utils
from predicators.ml_models import BinaryClassifier, MLPBinaryClassifier
from predicators.settings import CFG
from predicators.structs import Array, Image


def _main() -> None:
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    logging.basicConfig(level=CFG.loglevel,
                        format="%(message)s",
                        handlers=handlers)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    _analyze_saved_data()
    # _analyze_online_learning_cycles()


def _analyze_saved_data() -> None:
    """Use this to analyze the data saved in saved_datasets/."""
    nsrt_name = "PlaceToolNotHigh"
    objects_tuple_str = "spot:robot, cube:tool, extra_room_table:flat_surface"
    prefix = f"{CFG.data_dir}/{CFG.env}_{nsrt_name}({objects_tuple_str})_"
    filepath_template = f"{prefix}*.data"
    all_saved_files = glob.glob(filepath_template)
    X: List[Array] = []
    y: List[Array] = []
    times: List[int] = []
    for filepath in all_saved_files:
        with open(filepath, "rb") as f:
            datum = pkl.load(f)
        X_i, y_i = datum["datapoint"]
        time_i = datum["time"]
        X.append(X_i)
        y.append(y_i)
        times.append(time_i)
    idxs = [i for (i, _) in sorted(enumerate(times), key=lambda i: i[1])]
    X = [X[i] for i in idxs]
    y = [y[i] for i in idxs]
    img = _create_image(X, y)
    img_outfile = "videos/spot_cube_active_sampler_learning_saved_data.png"
    imageio.imsave(img_outfile, img)
    print(f"Wrote out to {img_outfile}")
    # Run sample efficiency analysis.
    _run_sample_efficiency_analysis(X, y)


def _analyze_online_learning_cycles() -> None:
    """Use this to analyze the datasets saved after each cycle."""
    # Set up videos.
    video_frames = []
    # Evaluate samplers for each learning cycle.
    online_learning_cycle = 0
    while True:
        try:
            img = _run_one_cycle_analysis(online_learning_cycle)
            video_frames.append(img)
        except FileNotFoundError:
            break
        online_learning_cycle += 1
    # Save the video.
    video_outfile = "spot_cube_active_sampler_learning.mp4"
    utils.save_video(video_outfile, video_frames)
    # Save the frames individually too.
    for t, img in enumerate(video_frames):
        img_outfile = f"videos/spot_cube_active_sampler_learning_{t}.png"
        imageio.imsave(img_outfile, img)


def _run_one_cycle_analysis(online_learning_cycle: Optional[int]) -> Image:
    option_name = "PlaceToolNotHigh"
    approach_save_path = utils.get_approach_save_path_str()
    save_path = f"{approach_save_path}_{option_name}_" + \
        f"{online_learning_cycle}.sampler_classifier"
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"File does not exist: {save_path}")
    with open(save_path, "rb") as f:
        classifier = pkl.load(f)
    print(f"Loaded sampler classifier from {save_path}.")
    save_path = f"{approach_save_path}_{option_name}_" + \
        f"{online_learning_cycle}.sampler_classifier_data"
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"File does not exist: {save_path}")
    with open(save_path, "rb") as f:
        data = pkl.load(f)
    print(f"Loaded sampler classifier training data from {save_path}.")
    X: List[Array] = data[0]
    y: List[Array] = data[1]
    return _create_image(X, y, classifier=classifier)


def _vec_to_xy(vec: Array) -> Tuple[float, float]:
    place_robot_xy = math_helpers.Vec2(*vec[-3:-1])

    world_fiducial = math_helpers.Vec2(
        vec[12],  # state.get(surface, "x"),
        vec[13],  # state.get(surface, "y"),
    )
    world_to_robot = math_helpers.SE2Pose(
        vec[3],  # state.get(robot, "x"),
        vec[4],  # state.get(robot, "y"),
        vec[6],  # state.get(robot, "yaw"))
    )
    fiducial_in_robot_frame = world_to_robot.inverse() * world_fiducial
    x, y = place_robot_xy - fiducial_in_robot_frame
    return (x, y)


def _create_image(X: List[Array],
                  y: List[Array],
                  classifier: Optional[BinaryClassifier] = None) -> Image:
    cmap = colormaps.get_cmap('RdYlGn')
    norm = Normalize(vmin=0.0, vmax=1.0)

    # x is [1.0, spot, tool, surface, params]
    # spot: gripper_open_percentage, curr_held_item_id, x, y, z, yaw
    # tool: x, y, z, lost, in_view
    # surface: x, y, z
    # params: dx, dy, dz
    assert np.array(X).shape[1] == 1 + 6 + 5 + 3 + 3

    fig, ax = plt.subplots(1, 1)

    x_min = -0.25
    x_max = 0.25
    y_min = -0.25
    y_max = 0.25
    density = 25
    radius = 0.025

    if classifier is not None:
        candidates = [(x, y) for x in np.linspace(x_min, x_max, density)
                      for y in np.linspace(y_min, y_max, density)]
        for candidate in candidates:
            # Average scores over other possible values...?
            scores = []
            for standard_x in X:
                cand_x = standard_x.copy()
                cand_x[-3:-1] = candidate
                score = classifier.predict_proba(cand_x)
                scores.append(score)
            mean_score = np.mean(scores)
            color = cmap(norm(mean_score))
            circle = plt.Circle(candidate, radius, color=color, alpha=0.1)
            ax.add_patch(circle)

    # plot real data
    for datum, label in zip(X, y):
        x_pt, y_pt = _vec_to_xy(datum)
        print("x_pt, y_pt:", x_pt, y_pt)
        print("label:", label)
        color = cmap(norm(label))
        circle = plt.Circle((x_pt, y_pt), radius, color=color, alpha=0.5)
        ax.add_patch(circle)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((x_min - 3 * radius, x_max + 3 * radius))
    plt.ylim((y_min - 3 * radius, y_max + 3 * radius))

    return utils.fig2data(fig, dpi=150)


class _OracleModel(BinaryClassifier):
    """Oracle hand-written model."""

    def fit(self, X: Array, y: Array) -> None:
        pass

    def classify(self, x: Array) -> bool:
        # Approximate.
        _, y_pt = _vec_to_xy(x)
        return y_pt > 0

    def predict_proba(self, x: Array) -> float:
        return 1.0 if self.classify(x) else 0.0


class _ConstantModel(BinaryClassifier):
    """Oracle hand-written model."""

    def __init__(self, seed: int, constant: bool) -> None:
        super().__init__(seed)
        self._constant = constant

    def fit(self, X: Array, y: Array) -> None:
        pass

    def classify(self, x: Array) -> bool:
        return self._constant

    def predict_proba(self, x: Array) -> float:
        return 1.0 if self.classify(x) else 0.0


def _run_sample_efficiency_analysis(X: List[Array], y: List[Array]) -> None:

    # Do k-fold cross validation.
    validation_frac = 0.1
    num_data = len(X)
    num_valid = int(num_data * validation_frac)
    num_trials = 10

    models: Dict[str, Callable[[], BinaryClassifier]] = {
        "oracle":
        lambda: _OracleModel(seed=CFG.seed),
        # "always-true": lambda: _ConstantModel(CFG.seed, True),
        "always-false":
        lambda: _ConstantModel(CFG.seed, False),
        "mlp":
        lambda: MLPBinaryClassifier(
            seed=CFG.seed,
            balance_data=CFG.mlp_classifier_balance_data,
            max_train_iters=CFG.sampler_mlp_classifier_max_itr,
            learning_rate=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
            use_torch_gpu=CFG.use_torch_gpu,
            train_print_every=CFG.pytorch_train_print_every,
            n_iter_no_change=CFG.mlp_classifier_n_iter_no_change,
            hid_sizes=CFG.mlp_classifier_hid_sizes,
            n_reinitialize_tries=CFG.
            sampler_mlp_classifier_n_reinitialize_tries,
            weight_init="default")
    }
    training_data_results = {}
    for training_frac in np.linspace(0.05, 1.0 - validation_frac, 5):
        num_training_data = int(len(X) * training_frac)
        model_accuracies = {}
        for model_name, create_model in models.items():
            print("Starting model:", model_name)
            rng = np.random.default_rng(CFG.seed)
            model_accuracy = []
            for i in range(num_trials):
                # Split the data randomly.
                idxs = list(range(num_data))
                rng.shuffle(idxs)
                train_idxs = idxs[num_valid:num_valid + num_training_data]
                valid_idxs = idxs[:num_valid]
                X_train = np.array([X[i] for i in train_idxs])
                y_train = np.array([y[i] for i in train_idxs])
                X_valid = [X[i] for i in valid_idxs]
                y_valid = [y[i] for i in valid_idxs]
                # Train.
                model = create_model()
                model.fit(X_train, y_train)
                # Predict.
                y_pred = [model.classify(x) for x in X_valid]
                acc = np.mean([(y == y_hat)
                               for y, y_hat in zip(y_valid, y_pred)])
                print(f"Trial {i} accuracy: {acc}")
                model_accuracy.append(acc)
            model_accuracies[model_name] = model_accuracy
        print(f"Overall accuracies for training_frac={training_frac}")
        print("------------------")
        for model_name, model_accuracy in model_accuracies.items():
            print(f"{model_name}: {np.mean(model_accuracy)}")
        training_data_results[num_training_data] = model_accuracies

    # Make num training data versus validation accuracy plot.
    plt.figure()
    plt.title("Spot Place Offline Sample Complexity Analysis")
    plt.xlabel("# Training Examples for Sampler")
    plt.ylabel("Validation Classification Accuracy")
    xs = sorted(training_data_results)
    for model_name in models:
        all_ys = np.array([training_data_results[x][model_name] for x in xs])
        ys = np.mean(all_ys, axis=1)
        ys_std = np.std(all_ys, axis=1)
        plt.plot(xs, ys, label=model_name)
        plt.fill_between(xs, ys - ys_std, ys + ys_std, alpha=0.2)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(-0.1, 1.1)
    outfile = "spot_place_sample_complexity.png"
    plt.savefig(outfile)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    _main()
