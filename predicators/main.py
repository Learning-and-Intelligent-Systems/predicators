"""Main entry point for running approaches in environments.

Example usage with learning NSRTs:
    python predicators/main.py --env cover --approach nsrt_learning --seed 0

Example usage with oracle NSRTs:
    python predicators/main.py --env cover --approach oracle --seed 0

Example with verbose logging:
    python predicators/main.py --env cover --approach oracle --seed 0 --debug

To load a saved approach:
    python predicators/main.py --env cover --approach nsrt_learning --seed 0 \
        --load_approach

To load saved data:
    python predicators/main.py --env cover --approach nsrt_learning --seed 0 \
        --load_data

To make videos of test tasks:
    python predicators/main.py --env cover --approach oracle --seed 0 \
        --make_test_videos --num_test_tasks 1

To run interactive learning approach:
    python predicators/main.py --env cover --approach interactive_learning \
         --seed 0

To exclude predicates:
    python predicators/main.py --env cover --approach oracle --seed 0 \
         --excluded_predicates Holding

To run grammar search predicate invention (example):
    python predicators/main.py --env cover --approach grammar_search_invention \
        --seed 0 --excluded_predicates all
"""

import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import dill as pkl

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach, create_approach
from predicators.datasets import create_dataset
from predicators.envs import BaseEnv, create_new_env
from predicators.settings import CFG
from predicators.structs import Dataset, InteractionRequest, \
    InteractionResult, Metrics, Task
from predicators.teacher import Teacher, TeacherInteractionMonitorWithVideo

assert os.environ.get("PYTHONHASHSEED") == "0", \
        "Please add `export PYTHONHASHSEED=0` to your bash profile!"


def main() -> None:
    """Main entry point for running approaches in environments."""
    script_start = time.perf_counter()
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    str_args = " ".join(sys.argv)
    # Log to stderr.
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if CFG.log_file:
        handlers.append(logging.FileHandler(CFG.log_file, mode='w'))
    logging.basicConfig(level=CFG.loglevel,
                        format="%(message)s",
                        handlers=handlers)
    if CFG.log_file:
        logging.info(f"Logging to {CFG.log_file}")
    logging.info(f"Running command: python {str_args}")
    logging.info("Full config:")
    logging.info(CFG)
    logging.info(f"Git commit hash: {utils.get_git_commit_hash()}")
    # Create results directory.
    os.makedirs(CFG.results_dir, exist_ok=True)
    # Create the eval trajectories directory.
    os.makedirs(CFG.eval_trajectories_dir, exist_ok=True)
    # Create classes. Note that seeding happens inside the env and approach.
    env = create_new_env(CFG.env, do_cache=True, use_gui=CFG.use_gui)
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
    if CFG.option_learner == "no_learning":
        # If we are not doing option learning, pass in all the environment's
        # oracle options.
        options = env.options
    else:
        # Determine from the config which oracle options to include, if any.
        options = utils.parse_config_included_options(env)
    # Create the agent (approach).
    approach = create_approach(CFG.approach, preds, options, env.types,
                               env.action_space, stripped_train_tasks)
    if approach.is_learning_based:
        # Create the offline dataset. Note that this needs to be done using
        # the non-stripped train tasks because dataset generation may need
        # to use the oracle predicates (e.g. demo data generation).
        offline_dataset = create_dataset(env, train_tasks, options)
    else:
        offline_dataset = None
    # Run the full pipeline.
    _run_pipeline(env, approach, stripped_train_tasks, offline_dataset)
    script_time = time.perf_counter() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")


def _run_pipeline(env: BaseEnv,
                  approach: BaseApproach,
                  train_tasks: List[Task],
                  offline_dataset: Optional[Dataset] = None) -> None:
    # If agent is learning-based, allow the agent to learn from the generated
    # offline dataset, and then proceed with the online learning loop. Test
    # after each learning call. If agent is not learning-based, just test once.
    if approach.is_learning_based:
        assert offline_dataset is not None, "Missing offline dataset"
        num_offline_transitions = sum(
            len(traj.actions) for traj in offline_dataset.trajectories)
        num_online_transitions = 0
        total_query_cost = 0.0
        if CFG.load_approach:
            approach.load(online_learning_cycle=None)
            learning_time = 0.0  # ignore loading time
        else:
            learning_start = time.perf_counter()
            approach.learn_from_offline_dataset(offline_dataset)
            learning_time = time.perf_counter() - learning_start
        offline_learning_metrics = {
            f"offline_learning_{k}": v
            for k, v in approach.metrics.items()
        }
        # Run evaluation once before online learning starts.
        if CFG.skip_until_cycle < 0:
            results = _run_testing(env, approach)
            results["num_offline_transitions"] = num_offline_transitions
            results["num_online_transitions"] = num_online_transitions
            results["query_cost"] = total_query_cost
            results["learning_time"] = learning_time
            results.update(offline_learning_metrics)
            _save_test_results(results, online_learning_cycle=None)
        teacher = Teacher(train_tasks)
        # The online learning loop.
        for i in range(CFG.num_online_learning_cycles):
            if i < CFG.skip_until_cycle:
                continue
            logging.info(f"\n\nONLINE LEARNING CYCLE {i}\n")
            logging.info("Getting interaction requests...")
            if num_online_transitions >= CFG.online_learning_max_transitions:
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
            num_online_transitions += sum(
                len(result.actions) for result in interaction_results)
            total_query_cost += query_cost
            logging.info(f"Query cost incurred this cycle: {query_cost}")
            if CFG.load_approach:
                approach.load(online_learning_cycle=i)
                learning_time += 0.0  # ignore loading time
            else:
                learning_start = time.perf_counter()
                logging.info("Learning from interaction results...")
                approach.learn_from_interaction_results(interaction_results)
                learning_time += time.perf_counter() - learning_start
            # Evaluate approach after every online learning cycle.
            results = _run_testing(env, approach)
            results["num_offline_transitions"] = num_offline_transitions
            results["num_online_transitions"] = num_online_transitions
            results["query_cost"] = total_query_cost
            results["learning_time"] = learning_time
            results.update(offline_learning_metrics)
            _save_test_results(results, online_learning_cycle=i)
    else:
        results = _run_testing(env, approach)
        results["num_offline_transitions"] = 0
        results["num_online_transitions"] = 0
        results["query_cost"] = 0.0
        results["learning_time"] = 0.0
        _save_test_results(results, online_learning_cycle=None)


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
        if request.train_task_idx < CFG.max_initial_demos and \
            not CFG.allow_interaction_in_demo_tasks:
            raise RuntimeError("Interaction requests cannot be on demo tasks "
                               "if allow_interaction_in_demo_tasks is False.")
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
                utils.EnvironmentFailure, utils.OptionExecutionFailure,
                utils.RequestActPolicyFailure
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
        save_prefix = utils.get_config_path_str()
        outfile = f"{save_prefix}__cycle{cycle_num}.mp4"
        utils.save_video(outfile, video)
    return results, query_cost


def _run_testing(env: BaseEnv, approach: BaseApproach) -> Metrics:
    test_tasks = env.get_test_tasks()
    num_found_policy = 0
    num_solved = 0
    approach.reset_metrics()
    total_suc_time = 0.0
    total_num_solve_timeouts = 0
    total_num_solve_failures = 0
    total_num_execution_timeouts = 0
    total_num_execution_failures = 0

    save_prefix = utils.get_config_path_str()
    metrics: Metrics = defaultdict(float)
    for test_task_idx, task in enumerate(test_tasks):
        # Run the approach's solve() method to get a policy for this task.
        solve_start = time.perf_counter()
        try:
            policy = approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure) as e:
            logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: "
                         f"Approach failed to solve with error: {e}")
            if isinstance(e, ApproachTimeout):
                total_num_solve_timeouts += 1
            elif isinstance(e, ApproachFailure):
                total_num_solve_failures += 1
            if CFG.make_failure_videos and e.info.get("partial_refinements"):
                video = utils.create_video_from_partial_refinements(
                    e.info["partial_refinements"], env, "test", test_task_idx,
                    CFG.horizon)
                outfile = f"{save_prefix}__task{test_task_idx+1}_failure.mp4"
                utils.save_video(outfile, video)
            if CFG.crash_on_failure:
                raise e
            continue
        solve_time = time.perf_counter() - solve_start
        metrics[f"PER_TASK_task{test_task_idx}_solve_time"] = solve_time
        num_found_policy += 1
        make_video = False
        solved = False
        caught_exception = False
        if CFG.make_test_videos or CFG.make_failure_videos:
            monitor = utils.VideoMonitor(env.render)
        else:
            monitor = None
        try:
            # Now, measure success by running the policy in the environment.
            traj, execution_metrics = utils.run_policy(
                policy,
                env,
                "test",
                test_task_idx,
                task.goal_holds,
                max_num_steps=CFG.horizon,
                monitor=monitor)
            solved = task.goal_holds(traj.states[-1])
            exec_time = execution_metrics["policy_call_time"]
            metrics[f"PER_TASK_task{test_task_idx}_exec_time"] = exec_time
            # Save the successful trajectory, e.g., for playback on a robot.
            traj_file = f"{save_prefix}__task{test_task_idx+1}.traj"
            traj_file_path = Path(CFG.eval_trajectories_dir) / traj_file
            # Include the original task too so we know the goal.
            traj_data = {
                "task": task,
                "trajectory": traj,
                "pybullet_robot": CFG.pybullet_robot
            }
            with open(traj_file_path, "wb") as f:
                pkl.dump(traj_data, f)
        except utils.EnvironmentFailure as e:
            log_message = f"Environment failed with error: {e}"
            caught_exception = True
        except (ApproachTimeout, ApproachFailure) as e:
            log_message = ("Approach failed at policy execution time with "
                           f"error: {e}")
            if isinstance(e, ApproachTimeout):
                total_num_execution_timeouts += 1
            elif isinstance(e, ApproachFailure):
                total_num_execution_failures += 1
            caught_exception = True
        if solved:
            log_message = "SOLVED"
            num_solved += 1
            total_suc_time += (solve_time + exec_time)
            make_video = CFG.make_test_videos
            video_file = f"{save_prefix}__task{test_task_idx+1}.mp4"
        else:
            if not caught_exception:
                log_message = "Policy failed to reach goal"
            if CFG.crash_on_failure:
                raise RuntimeError(log_message)
            make_video = CFG.make_failure_videos
            video_file = f"{save_prefix}__task{test_task_idx+1}_failure.mp4"
        logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: "
                     f"{log_message}")
        if make_video:
            assert monitor is not None
            video = monitor.get_video()
            utils.save_video(video_file, video)
    metrics["num_solved"] = num_solved
    metrics["num_total"] = len(test_tasks)
    metrics["avg_suc_time"] = (total_suc_time /
                               num_solved if num_solved > 0 else float("inf"))
    metrics["min_num_samples"] = approach.metrics[
        "min_num_samples"] if approach.metrics["min_num_samples"] < float(
            "inf") else 0
    metrics["max_num_samples"] = approach.metrics["max_num_samples"]
    metrics["min_skeletons_optimized"] = approach.metrics[
        "min_num_skeletons_optimized"] if approach.metrics[
            "min_num_skeletons_optimized"] < float("inf") else 0
    metrics["max_skeletons_optimized"] = approach.metrics[
        "max_num_skeletons_optimized"]
    metrics["num_solve_timeouts"] = total_num_solve_timeouts
    metrics["num_solve_failures"] = total_num_solve_failures
    metrics["num_execution_timeouts"] = total_num_execution_timeouts
    metrics["num_execution_failures"] = total_num_execution_failures
    # Handle computing averages of total approach metrics wrt the
    # number of found policies. Note: this is different from computing
    # an average wrt the number of solved tasks, which might be more
    # appropriate for some metrics, e.g. avg_suc_time above.
    for metric_name in [
            "num_samples", "num_skeletons_optimized", "num_nodes_expanded",
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
    # Dump the CFG, results, and git commit hash to a pickle file.
    with open(outfile, "wb") as f:
        pkl.dump(outdata, f)
    # Before printing the results, filter out keys that start with the
    # special prefix "PER_TASK_", to prevent an annoyingly long printout.
    del_keys = [k for k in results if k.startswith("PER_TASK_")]
    for k in del_keys:
        del results[k]
    logging.info(f"Test results: {results}")
    logging.info(f"Wrote out test results to {outfile}")


if __name__ == "__main__":  # pragma: no cover
    # Write out the exception to the log file.
    try:
        main()
    except Exception as _err:  # pylint: disable=broad-except
        logging.exception("main.py crashed")
        raise _err
