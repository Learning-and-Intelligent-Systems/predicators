"""Functions to help visualize learned samplers."""

import os
from typing import List

import dill as pkl
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators import utils
from predicators.settings import CFG
from predicators.spot_utils.utils import load_spot_metadata
from predicators.structs import Video


def _main() -> None:
    """Loads the saved samplers."""
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    cycles_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    imgs = visualize_cup_table_place_samplers(cycles_to_plot, 25)
    for t, img in zip(cycles_to_plot, imgs):
        img_outfile = f"videos/cup_table_active_sampler_learning_cycle_{t}.png"
        imageio.imsave(img_outfile, img)
        print(f"Saved sampler analysis figure {img_outfile}.")


def visualize_cup_table_place_samplers(online_learning_cycles: List,
                                       num_samples_len_and_wid: int) -> Video:
    """Create a visualization of the learned place cup on table sampler at
    different online learning cycles."""
    assert CFG.env == "spot_ball_and_cup_sticky_table_env"
    assert CFG.active_sampler_learning_feature_selection == "oracle"
    cmap = colormaps.get_cmap('RdYlGn')
    norm = Normalize(vmin=0.0, vmax=1.0)
    drafting_table_feats = load_spot_metadata(
    )["static-object-features"]["drafting_table"]
    drafting_table_len = drafting_table_feats["length"]
    drafting_table_wid = drafting_table_feats["width"]
    sticky_region_x = drafting_table_feats["sticky-region-x"]
    sticky_region_y = drafting_table_feats["sticky-region-y"]
    len_vals = np.linspace(-drafting_table_len / 2, drafting_table_len / 2,
                           num_samples_len_and_wid)
    wid_vals = np.linspace(-drafting_table_wid / 2, drafting_table_wid / 2,
                           num_samples_len_and_wid)
    imgs = []
    # Produce a new image for every online learning cycle.
    for online_learning_cycle in online_learning_cycles:
        fig, axes = plt.subplots(1, 2)
        gt_ax, sampled_ax = axes
        gt_ax.set_title("Real-world Data")
        sampled_ax.set_title("Learned Sampler")
        gt_ax.set_ylim(-drafting_table_wid / 2 - 0.1,
                       drafting_table_wid / 2 + 0.1)
        gt_ax.set_xlim(-drafting_table_len / 2 - 0.1,
                       drafting_table_len / 2 + 0.1)
        sampled_ax.set_ylim(-drafting_table_wid / 2 - 0.1,
                            drafting_table_wid / 2 + 0.1)
        sampled_ax.set_xlim(-drafting_table_len / 2 - 0.1,
                            drafting_table_len / 2 + 0.1)
        gt_ax.set_aspect('equal', adjustable='box')
        sampled_ax.set_aspect('equal', adjustable='box')
        table_geom = utils.Rectangle.from_center(0.0, 0.0, drafting_table_len,
                                                 drafting_table_wid, 0.0)
        table_geom.plot(gt_ax, **{'fill': None, 'alpha': 1})
        table_geom.plot(sampled_ax, **{'fill': None, 'alpha': 1})
        option_name = "PlaceObjectOnTop"
        option_args = "(robot:robot, cup:container, " + \
            "drafting_table:drafting_table)"
        save_path = f"{CFG.approach_dir}/{CFG.experiment_id}_{option_name}"
        cls_save_path = save_path + \
            f"_{online_learning_cycle}_{option_args}.sampler_classifier"
        cls_data_save_path = save_path + \
            f"_{online_learning_cycle}_{option_args}.sampler_classifier_data"
        if not os.path.exists(cls_save_path) or not os.path.exists(
                cls_data_save_path):
            print(f"Didn't find data for cycle {online_learning_cycle}")
            continue
        with open(cls_save_path, "rb") as f:
            classifier = pkl.load(f)
        with open(cls_data_save_path, "rb") as f:
            classifier_data = pkl.load(f)
        print(f"Loaded sampler classifier and data from {cls_save_path}.")
        assert len(classifier_data[0][0]) == 5

        # Plot actual training data.
        for i in range(len(classifier_data[0])):
            datapoint = classifier_data[0][i]
            correctness = classifier_data[1][i]
            color = cmap(norm(correctness))
            circle = plt.Circle((datapoint[3], datapoint[4]),
                                0.009,
                                color=color,
                                alpha=0.9)
            gt_ax.add_patch(circle)

        # Plot sampled points with correctness.
        for x in len_vals:
            for y in wid_vals:
                sampler_input = [1.0, sticky_region_x, sticky_region_y, x, y]
                score = classifier.predict_proba(np.array(sampler_input))
                color = cmap(norm(score))
                circle = plt.Circle((x, y), 0.015, color=color, alpha=0.1)
                sampled_ax.add_patch(circle)
        img = utils.fig2data(fig, dpi=150)
        imgs.append(img)

    return imgs


if __name__ == "__main__":
    _main()
