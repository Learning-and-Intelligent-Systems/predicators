"""Analysis for spot cube placing with active sampler learning."""

import glob
import os
from typing import List, Optional

from bosdyn.client import math_helpers

import dill as pkl
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators import utils
from predicators.ml_models import BinaryClassifier
from predicators.settings import CFG
from predicators.structs import Array, Image


def _main() -> None:
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
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
    X: List[Array] = data["datapoint"][0]
    y: List[Array] = data["datapoint"][1]
    return _create_image(X, y, classifier=classifier)


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
        place_robot_xy = math_helpers.Vec2(*datum[-3:-1])
        print("place_robot_xy:", place_robot_xy)
        world_to_robot = math_helpers.SE2Pose(datum[3], datum[4], datum[6])
        print("world_to_robot:", world_to_robot)
        world_surface_xy = math_helpers.Vec2(datum[12], datum[13])
        print("world_surface_xy:", world_surface_xy)
        place_world_xy = world_to_robot * place_robot_xy
        print("place_world_xy:", place_world_xy)
        place_surface_xy = place_world_xy - world_surface_xy
        print("place_surface_xy:", place_surface_xy)
        x_pt, y_pt = place_surface_xy
        print("label:", label)
        color = cmap(norm(label))
        circle = plt.Circle((x_pt, y_pt), radius, color=color, alpha=0.5)
        ax.add_patch(circle)

    plt.xlabel("x (surface frame)")
    plt.ylabel("y (surface frame)")
    plt.xlim((x_min - 3 * radius, x_max + 3 * radius))
    plt.ylim((y_min - 3 * radius, y_max + 3 * radius))

    return utils.fig2data(fig, dpi=150)


if __name__ == "__main__":
    _main()
