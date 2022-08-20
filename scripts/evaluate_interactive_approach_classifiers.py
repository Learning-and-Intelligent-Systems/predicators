"""Script to evaluate interactively learned predicate classifiers on held-out
test cases."""

import os
from typing import Callable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from predicators import utils
from predicators.approaches import create_approach
from predicators.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.envs import BaseEnv, create_new_env
from predicators.envs.cover import CoverEnv
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, Task
from scripts.analyze_results_directory import get_df_for_entry


def evaluate_approach(evaluate_fn: Callable[
    [BaseEnv, InteractiveLearningApproach, Optional[int], List], None],
                      data: List) -> None:
    """Loads an approach and evaluates it using the given function."""
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    # Create classes.
    env = create_new_env(CFG.env, do_cache=True)
    preds, _ = utils.parse_config_excluded_predicates(env)
    # Don't need actual train tasks.
    train_tasks: List[Task] = []
    # Create the agent (approach).
    approach = create_approach(CFG.approach, preds, env.options, env.types,
                               env.action_space, train_tasks)
    assert isinstance(approach, InteractiveLearningApproach)
    _run_pipeline(env, approach, evaluate_fn, data)
    if data:
        _plot(data)


def _run_pipeline(
        env: BaseEnv, approach: InteractiveLearningApproach,
        evaluate_fn: Callable[
            [BaseEnv, InteractiveLearningApproach, Optional[int], List],
            None], data: List) -> None:
    approach.load(online_learning_cycle=None)
    evaluate_fn(env, approach, None, data)
    for i in range(CFG.num_online_learning_cycles):
        print(f"\n\nONLINE LEARNING CYCLE {i}\n")
        try:
            approach.load(online_learning_cycle=i)
            evaluate_fn(env, approach, i, data)
        except FileNotFoundError:
            break


def _evaluate_preds(env: BaseEnv, approach: InteractiveLearningApproach,
                    cycle_num: Optional[int], data: List) -> None:
    del cycle_num  # unused
    if CFG.env == "cover":
        assert isinstance(env, CoverEnv)
        return _evaluate_preds_cover(
            approach._get_current_predicates(),  # pylint: disable=protected-access
            env,
            data)
    raise NotImplementedError(
        f"Held out predicate test set not yet implemented for {CFG.env}")


def _evaluate_preds_cover(preds: Set[Predicate], env: CoverEnv,
                          data: List) -> None:
    del data  # unused
    Holding = [p for p in preds if p.name == "Holding"][0]
    Covers = [p for p in preds if p.name == "Covers"][0]
    HoldingGT = [p for p in env.predicates if p.name == "Holding"][0]
    CoversGT = [p for p in env.predicates if p.name == "Covers"][0]
    states, _, _ = create_states_cover(env)
    # Test 1: no blocks overlap any targets, none are held
    state = states[0]
    atoms = utils.abstract(state, (Holding, Covers))
    atoms_gt = utils.abstract(state, (HoldingGT, CoversGT))
    print(f"False positives: {atoms - atoms_gt}\n"
          f"False negatives: {atoms_gt - atoms}")
    # Test 2: block0 does not completely cover target0
    state = states[2]
    atoms = utils.abstract(state, (Holding, Covers))
    atoms_gt = utils.abstract(state, (HoldingGT, CoversGT))
    print(f"False positives: {atoms - atoms_gt}\n"
          f"False negatives: {atoms_gt - atoms}")
    # Test 3: block0 covers target0
    state = states[4]
    atoms = utils.abstract(state, (Holding, Covers))
    atoms_gt = utils.abstract(state, (HoldingGT, CoversGT))
    print(f"False positives: {atoms - atoms_gt}\n"
          f"False negatives: {atoms_gt - atoms}")


def create_states_cover(
        env: CoverEnv) -> Tuple[List[State], List[Object], List[Object]]:
    """Create a sequence of CoverEnv states to be used during evaluation."""
    states = []
    block_poses = [0.15, 0.605]
    target_poses = [0.375, 0.815]
    # State 0: no blocks overlap any targets
    task = env.get_test_tasks()[0]
    state = task.init
    block0 = [b for b in state if b.name == "block0"][0]
    block1 = [b for b in state if b.name == "block1"][0]
    target0 = [b for b in state if b.name == "target0"][0]
    target1 = [b for b in state if b.name == "target1"][0]
    robot = [b for b in state if b.name == "robby"][0]
    blocks = [block0, block1]
    targets = [target0, target1]
    assert len(blocks) == len(block_poses)
    for block, pose in zip(blocks, block_poses):
        # [is_block, is_target, width, pose, grasp]
        state.set(block, "pose", pose)
        # Make sure blocks are not held
        state.set(block, "grasp", -1)
    assert len(targets) == len(target_poses)
    for target, pose in zip(targets, target_poses):
        # [is_block, is_target, width, pose]
        state.set(target, "pose", pose)
    state.set(robot, "hand", 0.0)
    states.append(state)
    # State 1: block0 and target0 overlap a bit
    next_state = state.copy()
    next_state.set(block0, "pose", 0.31)
    states.append(next_state)
    # State 2: block and target overlap more
    next_state = state.copy()
    next_state.set(block0, "pose", 0.33)
    states.append(next_state)
    # State 3: block covers target, right edges align
    next_state = state.copy()
    next_state.set(block0, "pose", 0.35)
    states.append(next_state)
    # State 4: block0 covers target0, centered
    next_state = state.copy()
    next_state.set(block0, "pose", target_poses[0])
    states.append(next_state)
    return states, blocks, targets


DPI = 500
Y_LIM = (-0.05, 1.05)
X_KEY, X_LABEL = "CYCLE", "Cycle"
Y_KEY, Y_LABEL = "SCORE", "Model Score"

COLUMN_NAMES_AND_KEYS = [("TEST_ID", "test_id"), ("CYCLE", "cycle"),
                         ("SCORE", "score")]

PLOT_GROUPS = {
    "Entropies": [
        ("Far", lambda df: df["TEST_ID"].apply(lambda v: "entropy_0" in v)),
        ("Closer", lambda df: df["TEST_ID"].apply(lambda v: "entropy_1" in v)),
        ("Overlap a little",
         lambda df: df["TEST_ID"].apply(lambda v: "entropy_2" in v)),
        ("Overlap more",
         lambda df: df["TEST_ID"].apply(lambda v: "entropy_3" in v)),
        ("Overlap edges align",
         lambda df: df["TEST_ID"].apply(lambda v: "entropy_4" in v)),
        ("Overlap centered",
         lambda df: df["TEST_ID"].apply(lambda v: "entropy_5" in v)),
    ],
    "BALD Scores": [
        ("Far", lambda df: df["TEST_ID"].apply(lambda v: "BALD_0" in v)),
        ("Closer", lambda df: df["TEST_ID"].apply(lambda v: "BALD_1" in v)),
        ("Overlap a little",
         lambda df: df["TEST_ID"].apply(lambda v: "BALD_2" in v)),
        ("Overlap more",
         lambda df: df["TEST_ID"].apply(lambda v: "BALD_3" in v)),
        ("Overlap edges align",
         lambda df: df["TEST_ID"].apply(lambda v: "BALD_4" in v)),
        ("Overlap centered",
         lambda df: df["TEST_ID"].apply(lambda v: "BALD_5" in v)),
    ],
}


def _plot(all_data: List) -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)
    column_names = [c for (c, _) in COLUMN_NAMES_AND_KEYS]
    df_all = pd.DataFrame(all_data)
    df_all.rename(columns=dict(zip(df_all.columns, column_names)),
                  inplace=True)
    print(df_all)
    for plot_title, d in PLOT_GROUPS.items():
        _, ax = plt.subplots()
        for label, selector in d:
            df = get_df_for_entry(X_KEY, df_all, selector)
            xs = df[X_KEY].tolist()
            ys = df[Y_KEY].tolist()
            ax.plot(xs, ys, label=label)
        ax.set_title(plot_title)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        ax.set_ylim(Y_LIM)
        plt.legend()
        plt.tight_layout()
        filename = f"{plot_title}_{utils.get_config_path_str()}.png"
        filename = filename.replace(" ", "_").lower()
        outfile = os.path.join(outdir, filename)
        plt.savefig(outfile, dpi=DPI)
        print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    evaluate_approach(_evaluate_preds, [])
