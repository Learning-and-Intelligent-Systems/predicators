"""Create learning curves showing percentage solved over online learning
iterations.

Shows how different approaches improve over online learning cycles, with
each line representing a different approach and x-axis showing
iterations.
"""

import os
from typing import Any, Callable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from scripts.analyze_results_directory import create_raw_dataframe, \
    pd_create_equal_selector

plt.style.use('ggplot')
pd.set_option('chained_assignment', None)

############################ Change below here ################################

# Details about the plt figure.
DPI = 500
FONT_SIZE = 30
Y_LIM = (-5, 105)

# Toggle to generate a single plot with subplots for each environment
GENERATE_A_SINGLE_PLOT = True

# Color palette for different approaches
COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

# All column names and keys to load into the pandas tables.
COLUMN_NAMES_AND_KEYS = [
    ("ENV", "env"),
    ("APPROACH", "approach"),
    ("EXCLUDED_PREDICATES", "excluded_predicates"),
    ("EXPERIMENT_ID", "experiment_id"),
    ("SEED", "seed"),
    ("AVG_TEST_TIME", "avg_suc_time"),
    ("AVG_NODES_CREATED", "avg_num_nodes_created"),
    ("LEARNING_TIME", "learning_time"),
    ("PERC_SOLVED", "perc_solved"),
    ("ONLINE_LEARNING_CYCLE", "cycle"),
]

DERIVED_KEYS = [("perc_solved",
                 lambda r: 100 * r["num_solved"] / r["num_test_tasks"])]

# The keys of the dict are labels for the legend, and the dict values are
# selectors to filter the dataframe for each approach that do online learning.
APPROACH_GROUPS = [
    # ("Ours",
    #  lambda df: df["EXPERIMENT_ID"].apply(
    #      lambda v: "predicate_invention" in v)),
    # ("Online NSRT",
    #  lambda df: df["EXPERIMENT_ID"].apply(
    #      lambda v: "online_nsrt_learning" in v)),
    ("MAPLE", lambda df: df["EXPERIMENT_ID"].apply(lambda v: "maple_q" in v)),
    # ("Ours",
    #  lambda df: df["EXPERIMENT_ID"].apply(
    #      lambda v: "predicate_invention" in v)),
    ("Ours",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "ours_always_test" in v)),
    ("VisPred", lambda df: df["EXPERIMENT_ID"].apply(
        lambda v: "online_nsrt_learning" in v)),
]

# Approaches that don't do online learning - show as
# horizontal lines at final performance
HORIZONTAL_LINE_GROUPS = [
    ("Manual", lambda df: df["EXPERIMENT_ID"].apply(lambda v: "oracle" in v)),
    # ("Ours",
    #  lambda df: df["EXPERIMENT_ID"].apply(
    #      lambda v: "predicate_invention" in v)),
    # ("ViLa (zs)",
    #  lambda df: df["EXPERIMENT_ID"].apply(
    #      lambda v: "vlm_plan_zero_shot" in v)),
    # ("ViLa (fs)",
    #  lambda df: df["EXPERIMENT_ID"].apply(
    #      lambda v: "vlm_plan_few_shot" in v)),
]

# Which environments to create plots for
PLOT_ENVS = [
    ("Coffee", pd_create_equal_selector("ENV", "pybullet_coffee")),
    ("Grow", pd_create_equal_selector("ENV", "pybullet_grow")),
    ("Boil", pd_create_equal_selector("ENV", "pybullet_boil")),
    ("Domino", pd_create_equal_selector("ENV", "pybullet_domino_grid")),
    ("Fan", pd_create_equal_selector("ENV", "pybullet_fan")),
]

#################### Should not need to change below here #####################


def _convert_cycle_to_numeric(cycle_str: Any) -> int:
    """Convert cycle string to numeric value, with None -> -1 for sorting."""
    if cycle_str == "None" or cycle_str is None:
        return -1
    try:
        return int(cycle_str)
    except (ValueError, TypeError):
        return -1


def _get_learning_curves_for_approach(
    df: Any,
    approach_selector: Callable[..., Any],
    env_selector: Callable[..., Any],
    max_iteration: Optional[int] = None
) -> Tuple[List[int], List[float], List[float]]:
    """Get learning curves for a specific approach and environment.

    For each seed, forward-fills missing iterations with the
    previous best performance.

    Args:
        max_iteration: If provided, extend curves to this
            iteration using forward-filling

    Returns:
        x_values: List of cycle numbers (starting from 0 for None)
        y_means: List of mean percentages solved
        y_stds: List of standard deviations
    """
    # Filter data for this approach and environment
    filtered_df = df[approach_selector(df) & env_selector(df)].copy()

    if filtered_df.empty:
        return [], [], []

    # Convert cycles to numeric and add offset so None (-1) becomes 0
    filtered_df['CYCLE_NUMERIC'] = filtered_df['ONLINE_LEARNING_CYCLE'].apply(
        _convert_cycle_to_numeric)
    filtered_df['X_VALUE'] = filtered_df[
        'CYCLE_NUMERIC'] + 1  # None (-1) -> 0, 0 -> 1, 1 -> 2, etc.

    # Get all unique seeds and iterations
    all_seeds = filtered_df['SEED'].unique()
    actual_iterations = sorted(filtered_df['X_VALUE'].unique())

    # If max_iteration is provided, extend to that iteration
    if max_iteration is not None:
        all_iterations = list(range(min(actual_iterations), max_iteration + 1))
    else:
        all_iterations = actual_iterations

    # Forward-fill missing values for each seed
    forward_filled_data = []

    for seed in all_seeds:
        seed_data = filtered_df[filtered_df['SEED'] == seed].copy()
        seed_data = seed_data.sort_values('X_VALUE')

        # Track best performance seen so far for this seed
        best_performance = 0

        for iteration in all_iterations:
            iteration_data = seed_data[seed_data['X_VALUE'] == iteration]

            if not iteration_data.empty:
                # We have data for this iteration, update best performance
                current_performance = iteration_data['PERC_SOLVED'].iloc[0]
                best_performance = max(best_performance, current_performance)
                forward_filled_data.append({
                    'SEED': seed,
                    'X_VALUE': iteration,
                    'PERC_SOLVED': current_performance
                })
            else:
                # Missing data for this iteration, use previous best
                forward_filled_data.append({
                    'SEED': seed,
                    'X_VALUE': iteration,
                    'PERC_SOLVED': best_performance
                })

    # Convert to DataFrame and compute mean/std across seeds for each iteration
    forward_filled_df = pd.DataFrame(forward_filled_data)

    if forward_filled_df.empty:
        return [], [], []

    grouped = forward_filled_df.groupby('X_VALUE')['PERC_SOLVED'].agg(
        ['mean', 'std']).reset_index()

    x_values = grouped['X_VALUE'].tolist()
    y_means = grouped['mean'].tolist()
    y_stds = grouped['std'].fillna(
        0).tolist()  # Fill NaN std with 0 for single data points

    return x_values, y_means, y_stds


def _get_final_performance_for_approach(
    df: Any, approach_selector: Callable[..., Any],
    env_selector: Callable[...,
                           Any]) -> Tuple[Optional[float], Optional[float]]:
    """Get final performance (mean and std) for approaches that don't do online
    learning.

    Returns:
        mean: Mean percentage solved across seeds
        std: Standard deviation across seeds
    """
    # Filter data for this approach and environment
    filtered_df = df[approach_selector(df) & env_selector(df)].copy()

    if filtered_df.empty:
        return None, None

    # For non-online learning approaches, we just take the
    # mean/std across all seeds
    mean = filtered_df['PERC_SOLVED'].mean()
    std = filtered_df['PERC_SOLVED'].std()
    if pd.isna(std):  # Single data point case
        std = 0

    return mean, std


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results", "learning_curves")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': FONT_SIZE})

    # Load all raw data (don't group/aggregate yet)
    df = create_raw_dataframe(COLUMN_NAMES_AND_KEYS, DERIVED_KEYS)

    if GENERATE_A_SINGLE_PLOT:
        # Create a single plot with subplots for each environment
        num_envs = len(PLOT_ENVS)
        fig, axes = plt.subplots(1, num_envs, figsize=(6 * num_envs, 7))

        # Handle case where there's only one subplot
        if num_envs == 1:
            axes = [axes]

        # Determine global max x-axis range
        global_max_x = 0
        for _, env_selector in PLOT_ENVS:
            for approach_label, approach_selector in APPROACH_GROUPS:
                x_vals, _, _ = _get_learning_curves_for_approach(
                    df, approach_selector, env_selector)
                if x_vals:
                    global_max_x = max(global_max_x, max(x_vals))
        global_max_x = max(global_max_x, 5)

        for env_idx, (env_name, env_selector) in enumerate(PLOT_ENVS):
            ax = axes[env_idx]

            # Plot horizontal lines for non-online learning approaches
            color_idx = 0
            for approach_label, approach_selector in HORIZONTAL_LINE_GROUPS:
                mean, std = _get_final_performance_for_approach(
                    df, approach_selector, env_selector)

                if mean is None:  # Skip if no data for this approach
                    continue

                color = COLORS[color_idx % len(COLORS)]
                color_idx += 1

                # Plot horizontal line spanning the full x range
                ax.axhline(y=mean,
                           label=approach_label if env_idx == 0 else "",
                           linestyle='--',
                           alpha=0.8,
                           color=color)

                # Add error band around the horizontal line
                if std is not None and std > 0:
                    ax.fill_between([0, global_max_x],
                                    mean - std,
                                    mean + std,
                                    alpha=0.2,
                                    color=color)

            # Plot each online learning approach as a separate line
            for approach_label, approach_selector in APPROACH_GROUPS:
                x_vals, y_means, y_stds = _get_learning_curves_for_approach(
                    df,
                    approach_selector,
                    env_selector,
                    max_iteration=global_max_x)

                if not x_vals:  # Skip if no data for this approach
                    continue

                color = COLORS[color_idx % len(COLORS)]
                color_idx += 1

                # Plot the line with shaded error region
                ax.plot(x_vals,
                        y_means,
                        label=approach_label if env_idx == 0 else "",
                        marker='o',
                        color=color)

                # Add shaded error region
                if any(std > 0 for std in y_stds):
                    y_lower = [
                        mean - std for mean, std in zip(y_means, y_stds)
                    ]
                    y_upper = [
                        mean + std for mean, std in zip(y_means, y_stds)
                    ]
                    ax.fill_between(x_vals,
                                    y_lower,
                                    y_upper,
                                    alpha=0.2,
                                    color=color)

            # Customize each subplot
            ax.set_xlabel('Online Learning Iteration',
                          color='black',
                          fontsize=FONT_SIZE)
            ax.set_title(f'{env_name}', color='black', fontsize=FONT_SIZE)
            ax.set_xlim(-0.5, global_max_x + 0.5)
            ax.set_ylim(Y_LIM[0], Y_LIM[1])

            # Only show y-axis ticks and labels on the leftmost subplot
            if env_idx > 0:
                ax.set_yticklabels([])

        # Add shared y-axis label to the left of the first plot
        fig.text(0.0,
                 0.5,
                 'Percentage Solved (%)',
                 va='center',
                 rotation='vertical',
                 color='black',
                 fontsize=FONT_SIZE + 2)

        # Add legend flat at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,
                   labels,
                   loc='lower center',
                   bbox_to_anchor=(0.5, -0.1),
                   ncol=len(labels))

        # Save the single plot
        plt.tight_layout()
        filename = "all_environments_learning_curves.png"
        outfile = os.path.join(outdir, filename)
        plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
        print(f"Wrote out to {outfile}")
        plt.close()

    else:
        # Create individual plots for each environment (original behavior)
        for env_name, env_selector in PLOT_ENVS:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Determine x-axis range by finding max iteration
            # across all online learning approaches
            max_x = 0
            for approach_label, approach_selector in APPROACH_GROUPS:
                x_vals, _, _ = _get_learning_curves_for_approach(
                    df, approach_selector, env_selector)
                if x_vals:
                    max_x = max(max_x, max(x_vals))

            # Use at least 5 as max for reasonable plot range
            max_x = max(max_x, 5)

            # Plot horizontal lines for non-online learning approaches
            color_idx = 0
            for approach_label, approach_selector in HORIZONTAL_LINE_GROUPS:
                mean, std = _get_final_performance_for_approach(
                    df, approach_selector, env_selector)

                if mean is None:  # Skip if no data for this approach
                    continue

                color = COLORS[color_idx % len(COLORS)]
                color_idx += 1

                # Plot horizontal line spanning the full x range
                ax.axhline(y=mean,
                           label=approach_label,
                           linestyle='--',
                           alpha=0.8,
                           color=color)

                # Add error band around the horizontal line
                if std is not None and std > 0:
                    ax.fill_between([0, max_x],
                                    mean - std,
                                    mean + std,
                                    alpha=0.2,
                                    color=color)

            # Plot each online learning approach as a separate line
            for approach_label, approach_selector in APPROACH_GROUPS:
                x_vals, y_means, y_stds = _get_learning_curves_for_approach(
                    df, approach_selector, env_selector, max_iteration=max_x)

                if not x_vals:  # Skip if no data for this approach
                    continue

                color = COLORS[color_idx % len(COLORS)]
                color_idx += 1

                # Plot the line with shaded error region
                ax.plot(x_vals,
                        y_means,
                        label=approach_label,
                        marker='o',
                        color=color)

                # Add shaded error region
                if any(std > 0 for std in y_stds):
                    y_lower = [
                        mean - std for mean, std in zip(y_means, y_stds)
                    ]
                    y_upper = [
                        mean + std for mean, std in zip(y_means, y_stds)
                    ]
                    ax.fill_between(x_vals,
                                    y_lower,
                                    y_upper,
                                    alpha=0.2,
                                    color=color)

            # Customize the plot
            ax.set_xlabel('Online Learning Iteration',
                          color='black',
                          fontsize=FONT_SIZE)
            ax.set_ylabel('Percentage Solved (%)',
                          color='black',
                          fontsize=FONT_SIZE)
            ax.set_title(f'{env_name}', color='black', fontsize=FONT_SIZE)
            ax.set_xlim(-0.5, max_x + 0.5)
            ax.set_ylim(Y_LIM[0], Y_LIM[1])
            ax.legend(bbox_to_anchor=(1, 1.02), loc='upper left')

            # Save the plot
            plt.tight_layout()
            filename = f"{env_name.lower()}_learning_curves.png"
            outfile = os.path.join(outdir, filename)
            plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
            print(f"Wrote out to {outfile}")
            plt.close()


if __name__ == "__main__":
    _main()
