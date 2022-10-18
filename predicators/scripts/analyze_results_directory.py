"""Script to analyze the results of experiments in the results/ directory."""

import argparse
import glob
from typing import Callable, Dict, Sequence, Tuple

import dill as pkl
import numpy as np
import pandas as pd

from predicators.settings import CFG

parser = argparse.ArgumentParser()
parser.add_argument("--sidelining", action="store_true")
args = parser.parse_args()

GROUPS = [
    # "ENV",
    # "APPROACH",
    # "EXCLUDED_PREDICATES",
    "EXPERIMENT_ID",
    # "NUM_TRAIN_TASKS",
    # "CYCLE"
]

COLUMN_NAMES_AND_KEYS = [
    ("ENV", "env"),
    ("APPROACH", "approach"),
    ("EXCLUDED_PREDICATES", "excluded_predicates"),
    ("EXPERIMENT_ID", "experiment_id"),
    ("SEED", "seed"),
    ("NUM_TRAIN_TASKS", "num_train_tasks"),
    ("CYCLE", "cycle"),
    ("NUM_SOLVED", "num_solved"),
    ("AVG_NUM_PREDS", "avg_num_preds"),
    ("AVG_TEST_TIME", "avg_suc_time"),
    ("AVG_NODES_CREATED", "avg_num_nodes_created"),
    ("LEARNING_TIME", "learning_time"),
    # ("AVG_SKELETONS", "avg_num_skeletons_optimized"),
    # ("MIN_SKELETONS", "min_skeletons_optimized"),
    # ("MAX_SKELETONS", "max_skeletons_optimized"),
    # ("AVG_NODES_EXPANDED", "avg_num_nodes_expanded"),
    # ("AVG_NUM_NSRTS", "avg_num_nsrts"),
    # ("AVG_DISCOVERED_FAILURES", "avg_num_failures_discovered"),
    # ("AVG_PLAN_LEN", "avg_plan_length"),
    # ("NUM_EXECUTION_FAILURES", "num_execution_failures"),
    # ("NUM_OFFLINE_TRANSITIONS", "num_offline_transitions"),
    # ("NUM_ONLINE_TRANSITIONS", "num_online_transitions"),
    # ("QUERY_COST", "query_cost"),
]

if args.sidelining:
    COLUMN_NAMES_AND_KEYS.remove(("AVG_NUM_PREDS", "avg_num_preds"))
    COLUMN_NAMES_AND_KEYS.append(
        ("SO_NUM_PLANS_UP_TO_N",
         "offline_learning_sidelining_obj_num_plans_up_to_n"))
    COLUMN_NAMES_AND_KEYS.append(
        ("SO_COMPLEXITY", "offline_learning_sidelining_obj_complexity"))


def pd_create_equal_selector(
        key: str, value: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Create a mask for a dataframe by checking key == value."""
    return lambda df: df[key] == value


def combine_selectors(
    selectors: Sequence[Callable[[pd.DataFrame], pd.Series]]
) -> Callable[[pd.DataFrame], pd.Series]:
    """And together multiple selectors."""
    assert len(selectors) > 0

    def _selector(df: pd.DataFrame) -> pd.Series:
        mask = selectors[0](df)
        for i in range(1, len(selectors)):
            mask = mask & selectors[i](df)
        return mask

    return _selector


def get_df_for_entry(
        x_key: str, df: pd.DataFrame,
        selector: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    """Create a dataframe with a subset selected by selector and with rows
    sorted by x_key."""
    df = df[selector(df)]
    # Handle CYCLE as a special case, since the offline learning phase is
    # logged as None. Note that we shift everything by 1 so the first data
    # point is 0, meaning 0 online learning cycles have happened so far.
    if "CYCLE" in df:
        df["CYCLE"].replace("None", "-1", inplace=True)
        df["CYCLE"] = df["CYCLE"].map(pd.to_numeric) + 1
    df = df.sort_values(x_key)
    df[x_key] = df[x_key].map(pd.to_numeric)
    return df


def create_raw_dataframe(
    column_names_and_keys: Sequence[Tuple[str, str]],
    derived_keys: Sequence[Tuple[str, Callable[[Dict[str, float]], float]]],
) -> pd.DataFrame:
    """Returns one dataframe with all data, not grouped."""
    all_data = []
    git_commit_hashes = set()
    column_names = [c for (c, _) in column_names_and_keys]
    for filepath in sorted(glob.glob(f"{CFG.results_dir}/*")):
        with open(filepath, "rb") as f:
            outdata = pkl.load(f)
        if "git_commit_hash" in outdata:
            git_commit_hashes.add(outdata["git_commit_hash"])
        if "config" in outdata:
            config = outdata["config"].__dict__.copy()
            run_data_defaultdict = outdata["results"]
            assert not set(config.keys()) & set(run_data_defaultdict.keys())
            run_data_defaultdict.update(config)
        else:
            run_data_defaultdict = outdata
        if len(filepath[8:-4].split("__")) > 7:  # pragma: no cover
            # Environment is a BEHAVIOR environment with task and scene name.
            (env, approach, seed, excluded_predicates, included_options,
             experiment_id, online_learning_cycle, task_name,
             scene_name) = filepath[8:-4].split("__")
            del task_name, scene_name
        else:
            (env, approach, seed, excluded_predicates, included_options,
             experiment_id, online_learning_cycle) = filepath[8:-4].split("__")
        if not excluded_predicates:
            excluded_predicates = "none"
        run_data = dict(
            run_data_defaultdict)  # want to crash if key not found!
        run_data.update({
            "env": env,
            "approach": approach,
            "seed": seed,
            "excluded_predicates": excluded_predicates,
            "included_options": included_options,
            "experiment_id": experiment_id,
            "cycle": online_learning_cycle,
        })
        for key, fn in derived_keys:
            run_data[key] = fn(run_data)
        data = [run_data.get(k, np.nan) for (_, k) in column_names_and_keys]
        all_data.append(data)
    if not all_data:
        raise ValueError(f"No data found in {CFG.results_dir}/")
    # Group & aggregate data.
    pd.set_option("display.max_rows", 999999)
    df = pd.DataFrame(all_data)
    df.rename(columns=dict(zip(df.columns, column_names)), inplace=True)
    print(f"Git commit hashes seen in {CFG.results_dir}/:")
    for commit_hash in git_commit_hashes:
        print(commit_hash)
    # Uncomment the next line to print out ALL the raw data.
    # print(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def create_dataframes(
    column_names_and_keys: Sequence[Tuple[str, str]],
    groups: Sequence[str],
    derived_keys: Sequence[Tuple[str, Callable[[Dict[str, float]], float]]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns means, standard deviations, and sizes."""
    df = create_raw_dataframe(column_names_and_keys, derived_keys)
    grouped = df.groupby(list(groups))
    means = grouped.mean()
    stds = grouped.std(ddof=0)
    sizes = grouped.size().to_frame()
    return means, stds, sizes


def _main() -> None:
    means, stds, sizes = create_dataframes(COLUMN_NAMES_AND_KEYS, GROUPS, [])
    # Add standard deviations to the printout.
    for col in means:
        for row in means[col].keys():
            mean = means.loc[row][col]
            std = stds.loc[row][col]
            means.loc[row, col] = f"{mean:.2f} ({std:.2f})"
    means["NUM_SEEDS"] = sizes
    pd.set_option("expand_frame_repr", False)
    print("\n\nAGGREGATED DATA OVER SEEDS:")
    print(means)
    means.to_csv("results_summary.csv")
    print("\n\nWrote out table to results_summary.csv")


if __name__ == "__main__":
    _main()
