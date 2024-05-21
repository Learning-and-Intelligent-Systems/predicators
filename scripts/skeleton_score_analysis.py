"""Make pretty plots by hill-climbing with respect to different score functions
that operator over skeletons, and different aggregation functions for combining
the results over skeletons.

Mostly used for decomposing the expected_nodes score function in grammar
search into skeleton length error and number of nodes expanded.
"""

import functools
import os
from typing import Callable, FrozenSet, List, Tuple

import dill as pkl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from predicators import utils
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.planning import PlanningFailure, PlanningTimeout, task_plan, \
    task_plan_grounding
from predicators.settings import CFG
from predicators.structs import Dataset, Predicate, Task

FORCE_REMAKE_RESULTS = False

ENV_NAMES = {
    "cover": "Cover",
    "cover_regrasp": "Cover Regrasp",
    "blocks": "Blocks",
    "painting": "Painting",
    "tools": "Tools",
}

SEEDS = list(range(10))

SCORE_AND_AGGREGATION_NAMES = {
    ("skeleton_len", "min"): "Min Skeleton Length Error",
    ("num_nodes", "sum"): "Total Nodes Created",
    ("expected_nodes", "sum"): "Expected Planning Time",
}


@functools.lru_cache(maxsize=None)
def _setup_data_for_env(env_name: str,
                        seed: int) -> Tuple[List[Task], Dataset, List[int]]:
    # Create data for this environment and seed.
    utils.reset_config({
        "seed": seed,
        "env": env_name,
        "offline_data_planning_timeout": 10
    })
    env = create_new_env(env_name)
    options = get_gt_options(env.get_name())
    predicates, _ = utils.parse_config_excluded_predicates(env)
    env_train_tasks = env.get_train_tasks()
    train_tasks = [t.task for t in env_train_tasks]
    dataset = create_dataset(env, train_tasks, options, predicates)
    assert all(traj.is_demo for traj in dataset.trajectories)
    demo_skeleton_lengths = [
        utils.num_options_in_action_sequence(t.actions)
        for t in dataset.trajectories
    ]
    return (train_tasks, dataset, demo_skeleton_lengths)


def _create_score_function(
    score_name: str
) -> Callable[[str, int, FrozenSet[Predicate]], NDArray[np.float64]]:
    if score_name == "skeleton_len":
        return _compute_skeleton_length_errors
    if score_name == "num_nodes":
        return _compute_num_nodes
    if score_name == "expected_nodes":
        return _compute_expected_nodes
    raise NotImplementedError("Unrecognized score function:"
                              f" {score_name}.")


def _create_aggregation_function(
        aggregation_name: str) -> Callable[[NDArray[np.float64]], float]:
    if aggregation_name == "min":
        return lambda result: result.min(axis=1).mean()
    if aggregation_name == "sum":
        return lambda result: result.sum(axis=1).mean()
    raise NotImplementedError("Unrecognized aggregation function:"
                              f" {aggregation_name}.")


def _order_predicate_sets(
    env_name: str,
    seed: int,
    score_name: str,
    aggregation_name: str,
) -> Tuple[FrozenSet[Predicate], ...]:
    utils.reset_config({"seed": seed, "env": env_name})
    score_function = _create_score_function(score_name)
    aggregate = _create_aggregation_function(aggregation_name)
    env = create_new_env(env_name)
    oracle_predicates = set(env.predicates)
    # Hack: remove these unnecessary predicates for cover.
    if env_name.startswith("cover"):
        oracle_predicates -= {
            p
            for p in oracle_predicates if p.name in ("IsTarget", "IsBlock")
        }
    # Starting with an empty predicate set (plus goal predicates),
    # build up to the oracle predicate set, adding one predicate at a time.
    current_predicate_set = frozenset(env.goal_predicates)
    order = [current_predicate_set]
    num_candidates = len(oracle_predicates) - len(current_predicate_set)
    for _ in range(num_candidates):
        # Evaluate each possible next candidate predicate set and keep the
        # best one.
        best_next_predicate_set: FrozenSet[Predicate] = frozenset()
        best_score = np.inf
        for next_predicate in oracle_predicates - current_predicate_set:
            next_predicate_set = current_predicate_set | {next_predicate}
            next_predicate_set_result = score_function(env_name, seed,
                                                       next_predicate_set)
            next_predicate_set_score = aggregate(next_predicate_set_result)
            if next_predicate_set_score < best_score:
                best_next_predicate_set = next_predicate_set
                best_score = next_predicate_set_score
        assert not np.isinf(best_score)
        print("Found next best predicate set:")
        print(best_next_predicate_set)
        current_predicate_set = best_next_predicate_set
        order.append(current_predicate_set)
    assert current_predicate_set == oracle_predicates
    return tuple(order)


def _skeleton_based_score_function(
        env_name: str,
        seed: int,
        frozen_predicate_set: FrozenSet[Predicate],
        skeleton_score_fn: Callable,  # too complicated...
        get_default_result: Callable[[int], float],
        max_skeletons: int = 8,
        timeout: float = 10) -> NDArray[np.float64]:
    current_predicate_set = set(frozen_predicate_set)
    # Load cached data for this env and seed.
    train_tasks, dataset, demo_skeleton_lengths = _setup_data_for_env(
        env_name, seed)
    # Learn operators.
    segmented_trajs = [
        segment_trajectory(traj, current_predicate_set)
        for traj in dataset.trajectories
    ]
    pnads = learn_strips_operators(dataset.trajectories,
                                   train_tasks,
                                   current_predicate_set,
                                   segmented_trajs,
                                   verify_harmlessness=False,
                                   verbose=False,
                                   annotations=dataset.annotations)
    strips_ops = [pnad.op for pnad in pnads]
    option_specs = [pnad.option_spec for pnad in pnads]
    per_skeleton_results = []  # shape (num tasks, max skeletons)
    for traj, demo_len in zip(dataset.trajectories, demo_skeleton_lengths):
        # Run task planning.
        train_task = train_tasks[traj.train_task_idx]
        init_atoms = utils.abstract(traj.states[0], current_predicate_set)
        objects = set(traj.states[0])
        dummy_nsrts = utils.ops_and_specs_to_dummy_nsrts(
            strips_ops, option_specs)
        ground_nsrts, reachable_atoms = task_plan_grounding(
            init_atoms, objects, dummy_nsrts)
        heuristic = utils.create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic, init_atoms, train_task.goal,
            ground_nsrts, current_predicate_set, objects)
        generator = task_plan(init_atoms, train_task.goal, ground_nsrts,
                              reachable_atoms, heuristic, seed, timeout,
                              max_skeletons)
        task_results = []
        try:
            for idx, (plan_skeleton, plan_atoms_sequence, metrics) in \
                enumerate(generator):
                result = skeleton_score_fn(plan_skeleton, plan_atoms_sequence,
                                           metrics, idx, demo_len)
                task_results.append(result)
        except (PlanningTimeout, PlanningFailure):
            # Use an upper bound on the error.
            for idx in range(len(task_results), max_skeletons):
                task_results.append(get_default_result(idx))
        per_skeleton_results.append(task_results)
    results_arr = np.array(per_skeleton_results, dtype=np.float64)
    assert results_arr.shape == (len(train_tasks), max_skeletons)
    return results_arr


@functools.lru_cache(maxsize=None)
def _compute_skeleton_length_errors(
        env_name: str,
        seed: int,
        frozen_predicate_set: FrozenSet[Predicate],
        max_skeletons: int = 8,
        timeout: float = 10,
        error_upper_bound: int = 100) -> NDArray[np.float64]:
    skeleton_score_fn = lambda plan_skeleton, _1, _2, _3, demo_len: \
        abs(len(plan_skeleton) - demo_len)
    get_default_result = lambda _: error_upper_bound
    return _skeleton_based_score_function(env_name, seed, frozen_predicate_set,
                                          skeleton_score_fn,
                                          get_default_result, max_skeletons,
                                          timeout)


@functools.lru_cache(maxsize=None)
def _compute_num_nodes(env_name: str,
                       seed: int,
                       frozen_predicate_set: FrozenSet[Predicate],
                       max_skeletons: int = 8,
                       timeout: float = 10) -> NDArray[np.float64]:
    skeleton_score_fn = lambda _1, _2, metrics, _3, _4: \
        metrics["num_nodes_created"]
    default_result = CFG.grammar_search_expected_nodes_upper_bound
    get_default_result = lambda _: default_result
    return _skeleton_based_score_function(env_name, seed, frozen_predicate_set,
                                          skeleton_score_fn,
                                          get_default_result, max_skeletons,
                                          timeout)


@functools.lru_cache(maxsize=None)
def _compute_expected_nodes(env_name: str,
                            seed: int,
                            frozen_predicate_set: FrozenSet[Predicate],
                            max_skeletons: int = 8,
                            timeout: float = 10) -> NDArray[np.float64]:
    # Horribly horribly hacky, but oh well...
    p = CFG.grammar_search_expected_nodes_optimal_demo_prob
    w = CFG.grammar_search_expected_nodes_backtracking_cost
    ub = CFG.grammar_search_expected_nodes_upper_bound
    refinable_skeleton_not_found_prob = 1.0

    def helper_fn(skeleton_len_error: int, skeleton_idx: int,
                  num_nodes_created: int) -> float:
        nonlocal refinable_skeleton_not_found_prob
        if skeleton_idx == max_skeletons - 1:
            return refinable_skeleton_not_found_prob * ub
        refinement_prob = p * (1 - p)**skeleton_len_error
        skeleton_prob = refinable_skeleton_not_found_prob * refinement_prob
        refinable_skeleton_not_found_prob *= (1 - refinement_prob)
        expected_planning_time = skeleton_prob * num_nodes_created
        if skeleton_idx > 0:
            expected_planning_time += skeleton_prob * w
        return expected_planning_time

    def get_default_result(_skeleton_idx: int) -> float:
        return refinable_skeleton_not_found_prob * ub
    skeleton_score_fn = lambda plan_skeleton, _1, metrics, idx, demo_len: \
        helper_fn(abs(len(plan_skeleton) - demo_len), idx,
        metrics["num_nodes_created"])
    return _skeleton_based_score_function(env_name, seed, frozen_predicate_set,
                                          skeleton_score_fn,
                                          get_default_result, max_skeletons,
                                          timeout)


def _create_predicate_labels(
        predicate_set_order: Tuple[FrozenSet[Predicate], ...]) -> List[str]:
    if len(predicate_set_order[0]) > 5:
        labels = ["[Goal Predicates]"]
    else:
        labels = [", ".join(p.name for p in predicate_set_order[0])]
    for i in range(len(predicate_set_order) - 1):
        new_predicates = predicate_set_order[i + 1] - predicate_set_order[i]
        assert len(new_predicates) == 1
        new_predicate = next(iter(new_predicates))
        labels.append(f"+{new_predicate.name}")
    return labels


def _create_heatmap(env_results: NDArray[np.float64], env_name: str,
                    score_name: str, aggregation_name: str,
                    predicate_set_order: Tuple[FrozenSet[Predicate],
                                               ...], outfile: str) -> None:
    # Env results shape is (seed, predicate set, task, skeleton idx).
    # Reorganize into heatmap array of shape (predicate set, skeleton idx)
    # by averaging out seed and task.
    heatmap_arr = np.mean(env_results, axis=(0, 2))
    num_predicate_sets, num_skeletons = heatmap_arr.shape
    assert num_predicate_sets == len(predicate_set_order)
    labels = _create_predicate_labels(predicate_set_order)
    env_label = ENV_NAMES[env_name]
    score_label = SCORE_AND_AGGREGATION_NAMES[(score_name, aggregation_name)]

    fig, ax = plt.subplots()
    ax.imshow(heatmap_arr)
    ax.set_xticks(np.arange(num_skeletons))
    ax.set_yticks(np.arange(num_predicate_sets), labels=labels)
    for i in range(num_skeletons):
        for j in range(num_predicate_sets):
            if heatmap_arr[j, i] >= 100:
                text = ">100"
            else:
                text = f"{heatmap_arr[j, i]:.2f}"
            ax.text(i, j, text, ha="center", va="center", color="w")

    ax.set_title(f"{env_label}: {score_label}")
    ax.set_xlabel("Skeleton Index")
    fig.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    print(f"Wrote out to {outfile}.")


def _create_plot(env_results: NDArray[np.float64], env_name: str,
                 score_name: str, aggregation_name: str,
                 predicate_set_order: Tuple[FrozenSet[Predicate],
                                            ...], outfile: str) -> None:
    # Env results shape is (seed, predicate set, task, skeleton idx).
    # Reorganize into array of shape (predicate set,) by scoring each seed's
    # result and then averaging out seed.
    aggregate = _create_aggregation_function(aggregation_name)
    seed_results = [[aggregate(r) for r in seed_rs] for seed_rs in env_results]
    arr = np.mean(seed_results, axis=0)
    # std_arr = np.std(seed_results, axis=0)
    num_predicate_sets, = arr.shape
    assert num_predicate_sets == len(predicate_set_order)
    labels = _create_predicate_labels(predicate_set_order)
    env_label = ENV_NAMES[env_name]
    score_label = SCORE_AND_AGGREGATION_NAMES[(score_name, aggregation_name)]

    fig, ax = plt.subplots()
    xs = np.arange(num_predicate_sets)
    ax.plot(xs, arr)
    # Pretty ugly and distracting, so disabling for now.
    # ax.fill_between(xs, arr - std_arr, arr + std_arr, alpha=0.5)
    ax.set_xticks(np.arange(num_predicate_sets), labels=labels)
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")
    ax.set_title(f"{env_label}: {score_label}")
    ax.set_ylabel(score_label)
    fig.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    print(f"Wrote out to {outfile}.")


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)

    for env_name in sorted(ENV_NAMES):
        outfile = os.path.join(outdir, f"skeleton_results_{env_name}.p")
        if not FORCE_REMAKE_RESULTS and os.path.exists(outfile):
            with open(outfile, "rb") as f:
                predicate_set_order, results = pkl.load(f)
            print(f"Loaded results from {outfile}.")
        else:
            # First determine the predicate sets that we want for this env.
            # Do this by hill climbing over all seeds and taking the mode.
            # Cache everything to avoid redundant computation.
            predicate_set_orders = [
                _order_predicate_sets(env_name,
                                      seed,
                                      score_name="expected_nodes",
                                      aggregation_name="sum") for seed in SEEDS
            ]
            predicate_set_order = max(predicate_set_orders,
                                      key=predicate_set_orders.count)
            # Now create the per-score-function and per-seed results that we
            # will actually plot.
            results = {}
            for score_name, aggregation_name in SCORE_AND_AGGREGATION_NAMES:
                score_function = _create_score_function(score_name)
                env_results = []
                for seed in SEEDS:
                    seed_results = [
                        score_function(env_name, seed, p)
                        for p in predicate_set_order
                    ]
                    env_results.append(seed_results)
                results_arr = np.array(env_results, dtype=np.float64)
                results[(score_name, aggregation_name)] = results_arr
            # Save raw results.
            with open(outfile, "wb") as f:
                pkl.dump((predicate_set_order, results), f)
            print(f"Wrote out results to {outfile}.")
        for (score_name, aggregation_name), results_arr in results.items():
            # Create heat map.
            outfile = os.path.join(outdir, f"{score_name}_{env_name}_heat.png")
            _create_heatmap(results_arr, env_name, score_name,
                            aggregation_name, predicate_set_order, outfile)
            # Create plot.
            outfile = os.path.join(outdir, f"{score_name}_{env_name}_plot.png")
            _create_plot(results_arr, env_name, score_name, aggregation_name,
                         predicate_set_order, outfile)


if __name__ == "__main__":
    _main()
