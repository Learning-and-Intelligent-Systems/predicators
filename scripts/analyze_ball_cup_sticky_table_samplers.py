"""Analyze learned place-cup samplers for ball and cup sticky table."""

import os
from typing import Optional

import dill as pkl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators import utils
from predicators.envs import BaseEnv, create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import Image, ParameterizedOption, State


def _main() -> None:
    """Loads the saved samplers."""
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    env = create_new_env(CFG.env, do_cache=True)
    # Create an example state that includes the objects of interest. The actual
    # state should not be used.
    state = _create_example_state(env)
    # Load the parameterized option of interest.
    skill_name = "PlaceCupWithoutBallOnTable"
    options = get_gt_options(env.get_name())
    option = next(o for o in options if o.name == skill_name)
    # Set up videos.
    video_frames = []
    # Evaluate samplers for each learning cycle.
    for online_learning_cycle in range(CFG.num_online_learning_cycles):
        img = _run_one_cycle_analysis(online_learning_cycle, option)
        video_frames.append(img)
    # Save the videos.
    video_outfile = "place_on_table_sampler_learning.mp4"
    utils.save_video(video_outfile, video_frames)


def _run_one_cycle_analysis(online_learning_cycle: Optional[int],
                            param_option: ParameterizedOption) -> Image:
    option_name = param_option.name
    approach_save_path = utils.get_approach_save_path_str()
    save_path = f"{approach_save_path}_{option_name}_{online_learning_cycle}"
    assert CFG.active_sampler_learning_object_specific_samplers
    suffix = "(robot:robot, ball:ball, cup:cup, sticky-table-0:table)"
    save_path = f"{save_path}_{suffix}"
    classifier_save_path = f"{save_path}.sampler_classifier"
    if os.path.exists(classifier_save_path):
        with open(classifier_save_path, "rb") as f:
            classifier = pkl.load(f)
        print(f"Loaded sampler classifier from {classifier_save_path}.")
    data_save_path = f"{save_path}.sampler_classifier_data"
    if os.path.exists(data_save_path):
        with open(data_save_path, "rb") as f:
            data = pkl.load(f)
            print(f"Loaded classifier training data from {data_save_path}.")
            candidates = list(data[0])
    else:
        candidates = []

    cmap = colormaps.get_cmap('RdYlGn')
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Classify the candidates.
    predictions = []
    for x in candidates:
        prediction = classifier.predict_proba(x)
        predictions.append(prediction)

    # Visualize the classifications.
    fig, axes = plt.subplots(1, 2)
    plt.suptitle(f"Cycle {online_learning_cycle}")

    radius = 5e-3
    for i, ax in enumerate(axes.flat):
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
            ax.set_title("Ground Truth")
        else:
            ax.set_title("Predictions")
        ax.set_xlim((-0.15, 0.15))
        ax.set_ylim((-0.15, 0.15))

        for candidate, prediction in zip(candidates, predictions):

            _, table_radius, sticky, sticky_region_x, sticky_region_y, \
                sticky_region_radius, table_x, table_y, param_x, param_y = candidate

            assert table_radius > 0
            assert sticky

            # Get (x, y) in table frame.
            act_x_table = param_x - table_x
            act_y_table = param_y - table_y

            # Get (x, y) in sticky region frame.
            act_x_sticky = act_x_table - sticky_region_x
            act_y_sticky = act_y_table - sticky_region_y

            # Check if less than radius away.
            in_sticky_region = (act_x_sticky**2 + act_y_sticky**2 <
                                sticky_region_radius**2)

            if i == 0:
                color = cmap(norm(in_sticky_region))
            else:
                color = cmap(norm(prediction))

            circle = plt.Circle((act_x_sticky, act_y_sticky),
                                radius,
                                color=color,
                                alpha=1.0)
            ax.add_patch(circle)

    return utils.fig2data(fig, dpi=150)


def _create_example_state(env: BaseEnv) -> State:
    init_obs = env.reset("train", 0)
    assert isinstance(init_obs, State)
    return init_obs.copy()


if __name__ == "__main__":
    _main()
