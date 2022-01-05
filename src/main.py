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
    python src/main.py --env blocks --approach grammar_search_invention \
        --seed 0 --excluded_predicates Holding,Clear,GripperOpen

"""

from collections import defaultdict
import os
import subprocess
import time
import pickle as pkl
from typing import Dict
from predicators.src.settings import CFG
from predicators.src.envs import create_env, EnvironmentFailure, BaseEnv
from predicators.src.approaches import create_approach, ApproachTimeout, \
    ApproachFailure, BaseApproach
from predicators.src.datasets import create_dataset
from predicators.src.structs import Metrics
from predicators.src import utils


def main() -> None:
    """Main entry point for running approaches in environments.
    """
    if not os.path.exists("results/"):
        os.mkdir("results/") # pragma: no cover
    start = time.time()
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    print("Full config:")
    print(CFG)
    print("Git commit hash:", subprocess.check_output(
        ["git", "rev-parse", "HEAD"]).decode("ascii").strip())
    # Create & seed classes
    env = create_env(CFG.env)
    assert env.goal_predicates.issubset(env.predicates)
    if CFG.excluded_predicates:
        if CFG.excluded_predicates == "all":
            excludeds = {pred.name for pred in env.predicates
                         if pred not in env.goal_predicates}
            print(f"All non-goal predicates excluded: {excludeds}")
            preds = env.goal_predicates
        else:
            excludeds = set(CFG.excluded_predicates.split(","))
            assert excludeds.issubset({pred.name for pred in env.predicates}), \
                "Unrecognized excluded_predicates!"
            preds = {pred for pred in env.predicates
                     if pred.name not in excludeds}
            assert env.goal_predicates.issubset(preds), \
                "Can't exclude a goal predicate!"
    else:
        preds = env.predicates
    approach = create_approach(CFG.approach, env.simulate, preds,
                               env.options, env.types, env.action_space)
    env.seed(CFG.seed)
    approach.seed(CFG.seed)
    env.action_space.seed(CFG.seed)
    for option in env.options:
        option.params_space.seed(CFG.seed)
    # If approach is learning-based, get training datasets and do learning,
    # testing after each learning call. Otherwise, just do testing.
    if approach.is_learning_based:
        if CFG.load:
            approach.load()
            results = _run_testing(env, approach)
            _save_test_results(results, start)
        else:
            # Iterate over the train_tasks lists coming from the generator.
            dataset_idx = 0
            for train_tasks in env.train_tasks_generator():
                dataset = create_dataset(env, train_tasks)
                print(f"\n\nDATASET INDEX: {dataset_idx}")
                dataset_idx += 1
                approach.learn_from_offline_dataset(dataset, train_tasks)
                results = _run_testing(env, approach)
                _save_test_results(results, start)
    else:
        results = _run_testing(env, approach)
        _save_test_results(results, start)


def _run_testing(env: BaseEnv, approach: BaseApproach) -> Dict[str, Metrics]:
    test_tasks = env.get_test_tasks()
    num_solved = 0
    approach.reset_metrics()
    start = time.time()
    for i, task in enumerate(test_tasks):
        print(end="", flush=True)
        try:
            policy = approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure) as e:
            print(f"Task {i+1} / {len(test_tasks)}: Approach failed to "
                  f"solve with error: {e}")
            continue
        try:
            _, video, solved = utils.run_policy_on_task(
                policy, task, env.simulate, env.predicates,
                CFG.max_num_steps_check_policy, CFG.make_videos, env.render)
        except EnvironmentFailure as e:
            print(f"Task {i+1} / {len(test_tasks)}: Environment failed "
                  f"with error: {e}")
            continue
        if solved:
            print(f"Task {i+1} / {len(test_tasks)}: SOLVED")
            num_solved += 1
        else:
            print(f"Task {i+1} / {len(test_tasks)}: Policy failed")
        if CFG.make_videos:
            outfile = f"{utils.get_config_path_str()}__task{i}.mp4"
            utils.save_video(outfile, video)
    total_test_time = time.time()-start
    test_metrics: Metrics = defaultdict(float)
    test_metrics["test_tasks_solved"] = num_solved
    test_metrics["test_tasks_total"] = len(test_tasks)
    test_metrics["total_test_time"] = total_test_time
    return {"test": test_metrics, "approach": approach.metrics.copy()}


def _save_test_results(results: Dict[str, Metrics], start_time: float) -> None:
    test_tasks_solved = results["test"]["test_tasks_solved"]
    test_tasks_total = results["test"]["test_tasks_total"]
    total_test_time = results["test"]["total_test_time"]
    approach_metrics = results["approach"]
    print(f"Tasks solved: {test_tasks_solved} / {test_tasks_total}")
    print(f"Approach metrics: {approach_metrics}")
    print(f"Total test time: {total_test_time:.5f} seconds")
    total_time = time.time() - start_time
    outfile = f"results/{utils.get_config_path_str()}.pkl"
    outdata = results["test"].copy()
    outdata["total_time"] = total_time
    with open(outfile, "wb") as f:
        pkl.dump(outdata, f)
    print(f"\n\nMain script terminated in {total_time:.5f} seconds")
    print(f"Wrote out results to {outfile}")


if __name__ == "__main__":  # pragma: no cover
    main()
