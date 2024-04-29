"""Create 2D plots for investigating interactively-learned predicate classifier
ensembles."""

from typing import List, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from predicators import utils
from predicators.approaches import create_approach
from predicators.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.envs import create_new_env
from predicators.envs.cover import CoverEnv
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import Array, Image, Predicate, Task
from scripts.evaluate_interactive_approach_classifiers import \
    create_states_cover


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
    options = get_gt_options(env.get_name())
    approach = create_approach(CFG.approach, preds, options, env.types,
                               env.action_space, train_tasks)
    assert isinstance(approach, InteractiveLearningApproach)
    # Get plotting function
    if CFG.env in ("cover", "cover_handempty"):
        assert isinstance(env, CoverEnv)
        plot_fnc = _plot_cover
    else:
        raise NotImplementedError(
            f"Plotting not yet implemented for {CFG.env}")
    # Load approaches and gather images
    video = []
    approach.load(online_learning_cycle=None)
    image = plot_fnc(env, approach, excluded_preds)
    video.append(image)
    for i in range(CFG.num_online_learning_cycles):
        print(f"\n\nONLINE LEARNING CYCLE {i}\n")
        try:
            approach.load(online_learning_cycle=i)
            image = plot_fnc(env, approach, excluded_preds)
            video.append(image)
        except FileNotFoundError:
            break
    # Save video
    outfile = f"ensemble_predictions__{utils.get_config_path_str()}.mp4"
    utils.save_video(outfile, video)


DPI = 500
GRID_SIZE = 40
TICKS_PER = 8
COLOR = "Greys"


def _plot_cover(env: CoverEnv, approach: InteractiveLearningApproach,
                excluded_preds: Set[Predicate]) -> Image:
    PRED_NAME = "Covers"
    Covers = [p for p in excluded_preds if p.name == PRED_NAME][0]
    # Create state and objects
    states, blocks, targets = create_states_cover(env)
    block = blocks[0]
    target = targets[0]
    # Get labelled data points
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
    entropies = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    true_means = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for c, target_pose in enumerate(axis_vals):
        for r, block_pose in enumerate(axis_vals):
            new_state = state.copy()
            new_state.set(block, "pose", block_pose)
            new_state.set(target, "pose", target_pose)
            x = new_state.vec((block, target))
            ps = approach._pred_to_ensemble[PRED_NAME].predict_member_probas(x)  # pylint: disable=protected-access
            means[r][c] = np.mean(ps)
            entropies[r][c] = utils.entropy(float(np.mean(ps)))
            true_means[r][c] = 1 if Covers.holds(new_state,
                                                 (block, target)) else 0
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    # Plot means, entropies, and true means
    heatmap(true_means, axes[0], axis_vals, axis_vals, "True Means")
    heatmap(means, axes[1], axis_vals, axis_vals, "Means")
    heatmap(entropies,
            axes[2],
            axis_vals,
            axis_vals,
            "Entropies",
            cmap_max=0.3)
    # Plot originally annotated data points
    for ax in axes[:2]:
        ax.scatter(pos_examples[0],
                   pos_examples[1],
                   s=2,
                   marker="o",
                   c="green")
        ax.scatter(neg_examples[0], neg_examples[1], s=2, marker="o", c="red")
    fig.suptitle(CFG.experiment_id.replace("_", " "))
    plt.tight_layout()
    image = utils.fig2data(fig, dpi=DPI)
    plt.close(fig)
    return image


def heatmap(data: Array,
            ax: matplotlib.axis,
            x_axis_vals: Array,
            y_axis_vals: Array,
            cbarlabel: str,
            cmap_min: float = 0.0,
            cmap_max: float = 1.0) -> None:
    """Create a heatmap from a numpy array and two lists of labels."""
    # Plot the heatmap
    norm = matplotlib.colors.Normalize(vmin=cmap_min, vmax=cmap_max)
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
    # Let the horizontal axes labelling appear on top
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
