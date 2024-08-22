"""Analyze learned samplers for spot dropping."""

import os
from typing import Optional

import dill as pkl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.spot_env import SpotRearrangementEnv, \
    _movable_object_type
from predicators.ground_truth_models import get_gt_options
from predicators.perception.spot_perceiver import SpotPerceiver
from predicators.settings import CFG
from predicators.structs import Image, Object, ParameterizedOption, State


def _main() -> None:
    """Loads the saved samplers."""
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    env = create_new_env(CFG.env, do_cache=True)
    assert isinstance(env, SpotRearrangementEnv)
    # Create an example state that includes the objects of interest. The actual
    # state should not be used.
    state = _create_example_state(env)
    # Load the parameterized option of interest.
    skill_name = "DropObjectInside"
    options = get_gt_options(env.get_name())
    option = next(o for o in options if o.name == skill_name)
    # Create separate plots for each movable object in the environment.
    graspable_objs = {o for o in state if o.is_instance(_movable_object_type)}
    for obj in graspable_objs:
        print(f"Starting analysis for {obj.name}")
        obj_id = state.get(obj, "object_id")
        # Set up videos.
        video_frames = []
        # Evaluate samplers for each learning cycle.
        for online_learning_cycle in range(CFG.num_online_learning_cycles):
            img = _run_one_cycle_analysis(online_learning_cycle, obj, obj_id,
                                          option)
            video_frames.append(img)
        # Save the videos.
        video_outfile = f"spot_drop_sampler_learning_{obj.name}.mp4"
        utils.save_video(video_outfile, video_frames)


def _run_one_cycle_analysis(online_learning_cycle: Optional[int],
                            target_object: Object, object_id: int,
                            param_option: ParameterizedOption) -> Image:
    option_name = param_option.name
    save_path = (f"{CFG.approach_dir}/{CFG.experiment_id}_"
                 f"{option_name}_{online_learning_cycle}")
    assert not CFG.active_sampler_learning_object_specific_samplers, "TODO"
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
            # Extract the candidates for this object.
            if not CFG.active_sampler_learning_object_specific_samplers:
                candidates = [x for x in data[0] if int(x[1]) == object_id]
            else:
                candidates = list(data[0])
    else:
        candidates = []

    cmap = colormaps.get_cmap('RdYlGn')
    norm = Normalize(vmin=0.0, vmax=1.0)

    assert not CFG.active_sampler_learning_object_specific_samplers, "TODO"
    r_idx = 3
    c_idx = 4

    # Classify the candidates.
    predictions = []
    for x in candidates:
        prediction = classifier.predict_proba(x)
        predictions.append(prediction)

    # Visualize the classifications.
    fig, axes = plt.subplots(1, 2)
    plt.suptitle(f"{target_object.name} cycle {online_learning_cycle}")

    radius = 0.01
    min_x, max_x, pad_x = -0.4, 0.4, 0.01
    min_y, max_y, pad_y = -0.4, 0.4, 0.01
    for i, ax in enumerate(axes.flat):
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
        ax.set_xlim((min_x - pad_x, max_x + pad_x))
        ax.set_ylim((min_y - pad_y, max_y + pad_y))

    ax = axes.flat[0]
    ax.set_title("Data")
    for candidate, prediction in zip(candidates, predictions):
        r = candidate[r_idx]
        c = candidate[c_idx]
        color = cmap(norm(prediction))
        circle = plt.Circle((c, r), radius, color=color, alpha=1.0)
        ax.add_patch(circle)

    ax = axes.flat[1]
    if candidates:
        size = 25
        ax.set_title("Predictions")
        for r in np.linspace(min_y, max_y, num=size):
            for c in np.linspace(min_x, max_x, num=size):
                x = candidates[0].copy()
                x[r_idx] = r
                x[c_idx] = c
                y = classifier.predict_proba(x)
                color = cmap(norm(y))
                circle = plt.Circle((c, r), radius, color=color, alpha=1.0)
                ax.add_patch(circle)
    else:
        ax.set_title("Waiting for Data")

    return utils.fig2data(fig, dpi=150)


def _create_example_state(env: SpotRearrangementEnv) -> State:
    perceiver = SpotPerceiver()
    empty_task = env.get_task("train", 0)
    perceiver.reset(empty_task)
    init_obs = env.reset("train", 0)
    init_state = perceiver.step(init_obs)
    return init_state.copy()


if __name__ == "__main__":
    _main()
