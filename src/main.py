"""Main entry point for running approaches in environments.

Example usage with learning NSRTs:
    python src/main.py --env cover --approach nsrt_learning --seed 0

Example usage with oracle NSRTs:
    python src/main.py --env cover --approach oracle --seed 0

To make videos:
    python src/main.py --env cover --approach oracle --seed 0 \
        --make_videos --num_test_tasks 1

To run interactive learning approach:
    python src/main.py --env cover --approach interactive_learning \
         --seed 0

To exclude predicates:
    python src/main.py --env cover --approach oracle --seed 0 \
         --excluded_predicates Holding

To run grammar search predicate invention (example):
    python src/main.py --env cover --approach grammar_search_invention \
        --seed 0 --excluded_predicates all
"""

from collections import defaultdict
import os
import subprocess
import time
import dill as pkl
from predicators.src.settings import CFG
from predicators.src.envs import create_env, EnvironmentFailure, BaseEnv
from predicators.src.approaches import create_approach, ApproachTimeout, \
    ApproachFailure, BaseApproach
from predicators.src.datasets import create_dataset
from predicators.src.structs import Metrics
from predicators.src import utils


def main() -> None:
    """Main entry point for running approaches in environments."""
    script_start = time.time()
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    print("Full config:")
    print(CFG)
    print(
        "Git commit hash:",
        subprocess.check_output(["git", "rev-parse",
                                 "HEAD"]).decode("ascii").strip())
    if not os.path.exists(CFG.results_dir):
        os.mkdir(CFG.results_dir)
    # Create classes. Note that seeding happens inside the env and approach.
    env = create_env(CFG.env)
    # The action space and options need to be seeded externally, because
    # env.action_space and env.options are often created during env __init__().
    env.action_space.seed(CFG.seed)
    for option in env.options:
        option.params_space.seed(CFG.seed)
    assert env.goal_predicates.issubset(env.predicates)
    if CFG.excluded_predicates:
        if CFG.excluded_predicates == "all":
            excludeds = {
                pred.name
                for pred in env.predicates if pred not in env.goal_predicates
            }
            print(f"All non-goal predicates excluded: {excludeds}")
            preds = env.goal_predicates
        else:
            excludeds = set(CFG.excluded_predicates.split(","))
            assert excludeds.issubset({pred.name for pred in env.predicates}), \
                "Unrecognized excluded_predicates!"
            preds = {
                pred
                for pred in env.predicates if pred.name not in excludeds
            }
            assert env.goal_predicates.issubset(preds), \
                "Can't exclude a goal predicate!"
    else:
        preds = env.predicates
    approach = create_approach(CFG.approach, env.simulate, preds, env.options,
                               env.types, env.action_space)
    # If approach is learning-based, get training datasets and do learning,
    # testing after each learning call. Otherwise, just do testing.
    if approach.is_learning_based:
        if CFG.load:
            approach.load()
            results = _run_testing(env, approach)
            _save_test_results(results, learning_time=0.0)
        else:
            # Iterate over the train_tasks lists coming from the generator.
            dataset_idx = 0
            for train_tasks in env.train_tasks_generator():
                dataset = create_dataset(env, train_tasks)
                print(f"\n\nDATASET INDEX: {dataset_idx}")
                dataset_idx += 1
                learning_start = time.time()
                approach.learn_from_offline_dataset(dataset)
                learning_time = time.time() - learning_start
                results = _run_testing(env, approach)
                _save_test_results(results, learning_time=learning_time)
    else:
        results = _run_testing(env, approach)
        _save_test_results(results, learning_time=0.0)
    script_time = time.time() - script_start
    print(f"\n\nMain script terminated in {script_time:.5f} seconds")


def _run_testing(env: BaseEnv, approach: BaseApproach) -> Metrics:
    test_tasks = env.get_test_tasks()
    num_found_policy = 0
    num_solved = 0
    approach.reset_metrics()
    total_suc_time = 0.0
    for i, task in enumerate(test_tasks):
        start = time.time()
        print(end="", flush=True)
        try:
            policy = approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure) as e:
            print(f"Task {i+1} / {len(test_tasks)}: Approach failed to "
                  f"solve with error: {e}")
            continue
        num_found_policy += 1
        try:
            _, video, solved = utils.run_policy_on_task(
                policy, task, env.simulate, env.predicates,
                CFG.max_num_steps_check_policy, CFG.make_videos, env.render)
        except EnvironmentFailure as e:
            print(f"Task {i+1} / {len(test_tasks)}: Environment failed "
                  f"with error: {e}")
            continue
        except (ApproachTimeout, ApproachFailure) as e:
            print(f"Task {i+1} / {len(test_tasks)}: Approach failed at policy "
                  f"execution time with error: {e}")
            continue
        if solved:
            print(f"Task {i+1} / {len(test_tasks)}: SOLVED")
            num_solved += 1
            total_suc_time += (time.time() - start)
        else:
            print(f"Task {i+1} / {len(test_tasks)}: Policy failed")
        if CFG.make_videos:
            outfile = f"{utils.get_config_path_str()}__task{i}.mp4"
            utils.save_video(outfile, video)
    metrics: Metrics = defaultdict(float)
    metrics["num_solved"] = num_solved
    metrics["num_total"] = len(test_tasks)
    metrics["avg_suc_time"] = (total_suc_time /
                               num_solved if num_solved > 0 else float("inf"))
    total_num_nodes_expanded = approach.metrics["total_num_nodes_expanded"]
    metrics["avg_nodes_expanded"] = (total_num_nodes_expanded /
                                     num_found_policy
                                     if num_found_policy > 0 else float("inf"))
    total_plan_length = approach.metrics["total_plan_length"]
    metrics["avg_plan_length"] = (total_plan_length / num_found_policy
                                  if num_found_policy > 0 else float("inf"))
    return metrics


def _save_test_results(results: Metrics, learning_time: float) -> None:
    num_solved = results["num_solved"]
    num_total = results["num_total"]
    avg_suc_time = results["avg_suc_time"]
    print(f"Tasks solved: {num_solved} / {num_total}")
    print(f"Average time for successes: {avg_suc_time:.5f} seconds")
    outfile = f"{CFG.results_dir}/{utils.get_config_path_str()}.pkl"
    outdata = results.copy()
    outdata["learning_time"] = learning_time
    with open(outfile, "wb") as f:
        pkl.dump(outdata, f)
    print(f"Wrote out results to {outfile}")


if __name__ == "__main__":  # pragma: no cover
    main()
