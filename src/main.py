"""Main entry point for running approaches in environments.

Example usage with learning NSRTs:
    python src/main.py --env cover --approach nsrt_learning --seed 0

Example usage with oracle NSRTs:
    python src/main.py --env cover --approach oracle --seed 0

Example with verbose logging:
    python src/main.py --env cover --approach oracle --seed 0 --debug

To load a saved approach:
    python src/main.py --env cover --approach nsrt_learning --seed 0 \
        --load_approach

To load saved data:
    python src/main.py --env cover --approach nsrt_learning --seed 0 \
        --load_data

To make videos of test tasks:
    python src/main.py --env cover --approach oracle --seed 0 \
        --make_test_videos --num_test_tasks 1

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

import logging
import os
import sys
import time
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import dill as pkl

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach, create_approach
from predicators.src.datasets import create_dataset
from predicators.src.envs import BaseEnv, create_new_env
from predicators.src.settings import CFG
from predicators.src.structs import Dataset, InteractionRequest, \
    InteractionResult, Metrics, Task
from predicators.src.teacher import Teacher, TeacherInteractionMonitorWithVideo

assert os.environ.get("PYTHONHASHSEED") == "0", \
        "Please add `export PYTHONHASHSEED=0` to your bash profile!"


def main() -> None:
    """Main entry point for running approaches in environments."""
    script_start = time.time()
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    str_args = " ".join(sys.argv)
    # Log to stderr.
    logging.basicConfig(level=CFG.loglevel,
                        format="%(message)s",
                        handlers=[logging.StreamHandler()])
    logging.info(f"Running command: python {str_args}")
    logging.info("Full config:")
    logging.info(CFG)
    logging.info(f"Git commit hash: {utils.get_git_commit_hash()}")
    # Create results directory.
    os.makedirs(CFG.results_dir, exist_ok=True)
    # Create classes. Note that seeding happens inside the env and approach.
    env = create_new_env(CFG.env, do_cache=True)
    # The action space and options need to be seeded externally, because
    # env.action_space and env.options are often created during env __init__().
    env.action_space.seed(CFG.seed)
    for option in env.options:
        option.params_space.seed(CFG.seed)
    assert env.goal_predicates.issubset(env.predicates)
    preds, _ = utils.parse_config_excluded_predicates(env)
    # Create the train tasks.
    train_tasks = env.get_train_tasks()
    # If train tasks have goals that involve excluded predicates, strip those
    # predicate classifiers to prevent leaking information to the approaches.
    stripped_train_tasks = [
        utils.strip_task(task, preds) for task in train_tasks
    ]
    # Don't pass in options if we are learning them.
    if CFG.option_learner == "no_learning":
        options = env.options
    else:
        options = set()
    # Create the agent (approach).
    approach = create_approach(CFG.approach, preds, options, env.types,
                               env.action_space, stripped_train_tasks)
    if approach.is_learning_based:
        # Create the offline dataset. Note that this needs to be done using
        # the non-stripped train tasks because dataset generation may need
        # to use the oracle predicates (e.g. demo data generation).
        offline_dataset = _generate_or_load_offline_dataset(env, train_tasks)
    else:
        offline_dataset = None
    # Run the full pipeline.
    _run_pipeline(env, approach, stripped_train_tasks, offline_dataset)
    script_time = time.time() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")


def _run_pipeline(env: BaseEnv,
                  approach: BaseApproach,
                  train_tasks: List[Task],
                  offline_dataset: Optional[Dataset] = None) -> None:
    # If agent is learning-based, generate an offline dataset, allow the agent
    # to learn from it, and then proceed with the online learning loop. Test
    # after each learning call. If agent is not learning-based, just test once.
    if approach.is_learning_based:
        assert offline_dataset is not None, "Missing offline dataset"
        total_num_transitions = sum(
            len(traj.actions) for traj in offline_dataset.trajectories)
        total_query_cost = 0.0
        if CFG.load_approach:
            approach.load(online_learning_cycle=None)
            learning_time = 0.0  # ignore loading time
        else:
            learning_start = time.time()
            approach.learn_from_offline_dataset(offline_dataset)
            learning_time = time.time() - learning_start
        # Run evaluation once before online learning starts.
        if CFG.skip_until_cycle < 0:
            results = _run_testing(env, approach)
            results["num_transitions"] = total_num_transitions
            results["query_cost"] = total_query_cost
            results["learning_time"] = learning_time
            _save_test_results(results, online_learning_cycle=None)
        teacher = Teacher(train_tasks)
        # The online learning loop.
        for i in range(CFG.num_online_learning_cycles):
            if i < CFG.skip_until_cycle:
                continue
            logging.info(f"\n\nONLINE LEARNING CYCLE {i}\n")
            logging.info("Getting interaction requests...")
            if total_num_transitions > CFG.online_learning_max_transitions:
                logging.info("Reached online_learning_max_transitions, "
                             "terminating")
                break
            interaction_requests = approach.get_interaction_requests()
            if not interaction_requests:
                logging.info("Did not receive any interaction requests, "
                             "terminating")
                break  # agent doesn't want to learn anything more; terminate
            interaction_results, query_cost = _generate_interaction_results(
                env, teacher, interaction_requests, i)
            total_num_transitions += sum(
                len(result.actions) for result in interaction_results)
            total_query_cost += query_cost
            logging.info(f"Query cost incurred this cycle: {query_cost}")
            if CFG.load_approach:
                approach.load(online_learning_cycle=i)
                learning_time += 0.0  # ignore loading time
            else:
                learning_start = time.time()
                approach.learn_from_interaction_results(interaction_results)
                learning_time += time.time() - learning_start
            # Evaluate approach after every online learning cycle.
            results = _run_testing(env, approach)
            results["num_transitions"] = total_num_transitions
            results["query_cost"] = total_query_cost
            results["learning_time"] = learning_time
            _save_test_results(results, online_learning_cycle=i)
    else:
        results = _run_testing(env, approach)
        results["num_transitions"] = 0
        results["query_cost"] = 0.0
        results["learning_time"] = 0.0
        _save_test_results(results, online_learning_cycle=None)


def _generate_or_load_offline_dataset(env: BaseEnv,
                                      train_tasks: List[Task]) -> Dataset:
    """Create offline dataset from training tasks."""
    dataset_filename = (
        f"{CFG.env}__{CFG.offline_data_method}__{CFG.num_train_tasks}"
        f"__{CFG.seed}.data")
    dataset_filepath = os.path.join(CFG.data_dir, dataset_filename)
    if CFG.load_data:
        assert os.path.exists(dataset_filepath), f"Missing: {dataset_filepath}"
        with open(dataset_filepath, "rb") as f:
            dataset = pkl.load(f)
        logging.info("\n\nLOADED DATASET")
    else:
        dataset = create_dataset(env, train_tasks)
        logging.info("\n\nCREATED DATASET")
        os.makedirs(CFG.data_dir, exist_ok=True)
        with open(dataset_filepath, "wb") as f:
            pkl.dump(dataset, f)
    return dataset


def _generate_interaction_results(
        env: BaseEnv,
        teacher: Teacher,
        requests: Sequence[InteractionRequest],
        cycle_num: Optional[int] = None
) -> Tuple[List[InteractionResult], float]:
    """Given a sequence of InteractionRequest objects, handle the requests and
    return a list of InteractionResult objects."""
    logging.info("Generating interaction results...")
    results = []
    query_cost = 0.0
    if CFG.make_interaction_videos:
        video = []
    for request in requests:
        monitor = TeacherInteractionMonitorWithVideo(env.render, request,
                                                     teacher)
        traj, _ = utils.run_policy(
            request.act_policy,
            env,
            "train",
            request.train_task_idx,
            request.termination_function,
            max_num_steps=CFG.max_num_steps_interaction_request,
            exceptions_to_break_on={
                utils.EnvironmentFailure, utils.OptionExecutionFailure
            },
            monitor=monitor)
        request_responses = monitor.get_responses()
        query_cost += monitor.get_query_cost()
        result = InteractionResult(traj.states, traj.actions,
                                   request_responses)
        results.append(result)
        if CFG.make_interaction_videos:
            video.extend(monitor.get_video())
    if CFG.make_interaction_videos:
        video_prefix = utils.get_config_path_str()
        outfile = f"{video_prefix}__cycle{cycle_num}.mp4"
        utils.save_video(outfile, video)
    return results, query_cost


def _run_testing(env: BaseEnv, approach: BaseApproach) -> Metrics:
    test_tasks = env.get_test_tasks()
    num_found_policy = 0
    num_solved = 0
    approach.reset_metrics()
    total_suc_time = 0.0
    total_num_execution_failures = 0
    total_num_solve_failures = 0
    video_prefix = utils.get_config_path_str()
    for test_task_idx, task in enumerate(test_tasks):
        solve_start = time.time()
        try:
            policy = approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure) as e:
            logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: "
                         f"Approach failed to solve with error: {e}")
            total_num_solve_failures += 1
            if CFG.make_failure_videos and e.info.get("partial_refinements"):
                video = utils.create_video_from_partial_refinements(
                    e.info["partial_refinements"], env, "test", test_task_idx,
                    CFG.horizon)
                outfile = f"{video_prefix}__task{test_task_idx+1}_failure.mp4"
                utils.save_video(outfile, video)
            continue
        solve_time = time.time() - solve_start
        num_found_policy += 1
        make_video = False
        solved = False
        caught_exception = False
        if CFG.make_test_videos or CFG.make_failure_videos:
            monitor = utils.VideoMonitor(env.render)
        else:
            monitor = None
        try:
            traj, execution_metrics = utils.run_policy(
                policy,
                env,
                "test",
                test_task_idx,
                task.goal_holds,
                max_num_steps=CFG.horizon,
                monitor=monitor)
            solved = task.goal_holds(traj.states[-1])
            solve_time += execution_metrics["policy_call_time"]
        except utils.EnvironmentFailure as e:
            log_message = f"Environment failed with error: {e}"
            caught_exception = True
        except (ApproachTimeout, ApproachFailure) as e:
            log_message = ("Approach failed at policy execution time with "
                           f"error: {e}")
            total_num_execution_failures += 1
            caught_exception = True
        if solved:
            log_message = "SOLVED"
            num_solved += 1
            total_suc_time += solve_time
            make_video = CFG.make_test_videos
            video_file = f"{video_prefix}__task{test_task_idx+1}.mp4"
        else:
            if not caught_exception:
                log_message = "Policy failed to reach goal"
            make_video = CFG.make_failure_videos
            video_file = f"{video_prefix}__task{test_task_idx+1}_failure.mp4"
        logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: "
                     f"{log_message}")
        if make_video:
            assert monitor is not None
            video = monitor.get_video()
            utils.save_video(video_file, video)
    metrics: Metrics = defaultdict(float)
    metrics["num_solved"] = num_solved
    metrics["num_total"] = len(test_tasks)
    metrics["avg_suc_time"] = (total_suc_time /
                               num_solved if num_solved > 0 else float("inf"))
    metrics["min_skeletons_optimized"] = approach.metrics[
        "min_num_skeletons_optimized"] if approach.metrics[
            "min_num_skeletons_optimized"] < float("inf") else 0
    metrics["max_skeletons_optimized"] = approach.metrics[
        "max_num_skeletons_optimized"]
    metrics["num_execution_failures"] = total_num_execution_failures
    metrics["num_solve_failures"] = total_num_solve_failures
    # Handle computing averages of total approach metrics wrt the
    # number of found policies. Note: this is different from computing
    # an average wrt the number of solved tasks, which might be more
    # appropriate for some metrics, e.g. avg_suc_time above.
    for metric_name in [
            "num_skeletons_optimized", "num_nodes_expanded",
            "num_nodes_created", "num_nsrts", "num_preds", "plan_length",
            "num_failures_discovered"
    ]:
        total = approach.metrics[f"total_{metric_name}"]
        metrics[f"avg_{metric_name}"] = (
            total / num_found_policy if num_found_policy > 0 else float("inf"))
    return metrics


def _save_test_results(results: Metrics,
                       online_learning_cycle: Optional[int]) -> None:
    num_solved = results["num_solved"]
    num_total = results["num_total"]
    avg_suc_time = results["avg_suc_time"]
    logging.info(f"Tasks solved: {num_solved} / {num_total}")
    logging.info(f"Average time for successes: {avg_suc_time:.5f} seconds")
    outfile = (f"{CFG.results_dir}/{utils.get_config_path_str()}__"
               f"{online_learning_cycle}.pkl")
    # Save CFG alongside results.
    outdata = {
        "config": CFG,
        "results": results.copy(),
        "git_commit_hash": utils.get_git_commit_hash()
    }
    with open(outfile, "wb") as f:
        pkl.dump(outdata, f)
    logging.info(f"Test results: {outdata['results']}")
    logging.info(f"Wrote out test results to {outfile}")


if __name__ == "__main__":  # pragma: no cover
    main()
