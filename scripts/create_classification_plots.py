"""Create 2D plots for investigating interactively-learned predicate classifier
ensembles."""

import os
from typing import List, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from predicators.scripts.evaluate_interactive_approach_classifiers import \
    create_states_cover
from predicators.src import utils
from predicators.src.approaches import create_approach
from predicators.src.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.src.envs import create_new_env
from predicators.src.envs.cover import CoverEnv
from predicators.src.settings import CFG
from predicators.src.structs import Array, Predicate, Task


def _main() -> None:
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    # Create classes.
    env = create_new_env(CFG.env, do_cache=True)
    preds, excluded_preds = utils.parse_config_excluded_predicates(env)
    # Don't need actual train tasks.
    train_tasks: List[Task] = []
    # Create the agent (approach).
    approach = create_approach(CFG.approach, preds, env.options, env.types,
                               env.action_space, train_tasks)
    assert isinstance(approach, InteractiveLearningApproach)
    # Load approach
    approach.load(online_learning_cycle=None)
    if CFG.env == "cover":
        assert isinstance(env, CoverEnv)
        _plot_cover(env, approach, excluded_preds)
    else:
        raise NotImplementedError(
            f"Plotting not yet implemented for {CFG.env}")


DPI = 500
GRID_SIZE = 40
TICKS_PER = 8
COLOR = "Greys"


def _plot_cover(env: CoverEnv, approach: InteractiveLearningApproach,
                excluded_preds: Set[Predicate]) -> None:
    PRED_NAME = "Covers"
    Covers = [p for p in excluded_preds if p.name == PRED_NAME][0]
    # Create state and objects
    states, blocks, targets = create_states_cover(env)
    block = blocks[0]
    target = targets[0]
    # Get original labelled data points
    dataset = approach._dataset  # pylint: disable=protected-access
    # ([target_pose values], [block_pose values])
    neg_examples: Tuple[List[float], List[float]] = ([], [])
    pos_examples: Tuple[List[float], List[float]] = ([], [])
    for (traj, traj_annotations) in zip(dataset.trajectories,
                                        dataset.annotations):
        assert len(traj.states) == len(traj_annotations)
        for (state, state_annotation) in zip(traj.states, traj_annotations):
            assert len(state_annotation) == 2
            for examples, annotations in zip((neg_examples, pos_examples),
                                             state_annotation):
                for atom in annotations:
                    if atom.predicate.name != PRED_NAME:
                        continue
                    block_pose = state.get(atom.objects[0], "pose")
                    target_pose = state.get(atom.objects[1], "pose")
                    examples[0].append(target_pose * GRID_SIZE)
                    examples[1].append(block_pose * GRID_SIZE)
    # Aggregate data
    state = states[0]
    axis_vals = np.linspace(0.0, 1.0, num=GRID_SIZE)
    means = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    stds = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    true_means = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for c, target_pose in enumerate(axis_vals):
        for r, block_pose in enumerate(axis_vals):
            new_state = state.copy()
            new_state.set(block, "pose", block_pose)
            new_state.set(target, "pose", target_pose)
            x = new_state.vec((block, target))
            ps = approach._pred_to_ensemble[PRED_NAME].predict_member_probas(x)  # pylint: disable=protected-access
            means[r][c] = np.mean(ps)
            stds[r][c] = np.std(ps)
            true_means[r][c] = 1 if Covers.holds(new_state,
                                                 (block, target)) else 0
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    # Plot means, stds, and true means
    heatmap(true_means, axes[0], axis_vals, axis_vals, "True Means")
    heatmap(means, axes[1], axis_vals, axis_vals, "Means")
    heatmap(stds,
            axes[2],
            axis_vals,
            axis_vals,
            "Stds",
            normalize_color_map=False)
    # Plot originally annotated data points
    for ax in axes[:2]:
        ax.scatter(pos_examples[0], pos_examples[1], marker="o", c="green")
        ax.scatter(neg_examples[0], neg_examples[1], marker="o", c="red")
    fig.suptitle(CFG.experiment_id.replace("_", " "))
    plt.tight_layout()
    # Write image
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)
    filename = f"ensemble_predictions__{utils.get_config_path_str()}.png"
    outfile = os.path.join(outdir, filename)
    plt.savefig(outfile, dpi=DPI)
    print(f"Wrote out to {outfile}")


def heatmap(data: Array,
            ax: matplotlib.axis,
            x_axis_vals: Array,
            y_axis_vals: Array,
            cbarlabel: str,
            normalize_color_map: bool = True) -> None:
    """Create a heatmap from a numpy array and two lists of labels."""
    # Plot the heatmap
    if normalize_color_map:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    else:
        norm = None
    im = ax.imshow(data, cmap=COLOR, norm=norm)
    # Create colorbar
    # Reference for magic numbers: https://stackoverflow.com/questions/18195758
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Determine axis ticks
    assert data.shape[0] % TICKS_PER == 0
    assert data.shape[1] % TICKS_PER == 0
    xticks = np.arange(0, data.shape[1], TICKS_PER)
    yticks = np.arange(0, data.shape[0], TICKS_PER)
    xtick_labels = map(lambda x: f"{x:.1f}", x_axis_vals[::TICKS_PER])
    ytick_labels = map(lambda x: f"{x:.1f}", y_axis_vals[::TICKS_PER])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels)
    # Let the horizontal axes labeling appear on top
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(),
             rotation=-90,
             ha="right",
             rotation_mode="anchor")
    ax.tick_params(which="minor", bottom=False, left=False)
    # Label axes and plot
    ax.set_title(cbarlabel)
    ax.set_xlabel("Target Pose")
    ax.set_ylabel("Block Pose")


if __name__ == "__main__":
    _main()
