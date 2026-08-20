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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dill as pkl

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    create_approach
from predicators.approaches.base_approach import BaseApproach
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.datasets import create_dataset
from predicators.envs import BaseEnv, create_new_env
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options, \
    parse_config_included_options
from predicators.perception import create_perceiver
from predicators.pybullet_helpers.real_robot_executor import attach_real_robot
from predicators.settings import CFG, get_allowed_query_type_names
from predicators.structs import Action, Dataset, EnvironmentTask, \
    InteractionRequest, InteractionResult, Metrics, Observation, Response, \
    Task
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

    # Setup logging and directories
    utils.configure_logging()
    os.makedirs(CFG.results_dir, exist_ok=True)
    os.makedirs(CFG.eval_trajectories_dir, exist_ok=True)

    # Log initial info
    utils.log_initial_info(str_args)

    # Setup environment and tasks
    env, approach_train_tasks, train_tasks = setup_environment()

    # Setup predicates
    included_preds, excluded_preds = utils.parse_config_excluded_predicates(
        env)
    preds = utils.replace_goals_with_agent_specific_goals(
        included_preds, excluded_preds,
        env) if CFG.approach != "oracle" else included_preds

    # Create approach
    approach = setup_approach(env, preds, approach_train_tasks)

    # Create dataset and cognitive manager
    offline_dataset = create_offline_dataset(env, train_tasks, preds, approach)
    execution_monitor = create_execution_monitor(CFG.execution_monitor)
    cogman = CogMan(approach, create_perceiver(CFG.perceiver),
                    execution_monitor)

    # Run pipeline
    _run_pipeline(env, cogman, approach_train_tasks, offline_dataset)

    # Log completion
    script_time = time.perf_counter() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")


# ── Setup helpers ────────────────────────────────────────────────


def setup_environment() -> Tuple[BaseEnv, List[Task], List[Task]]:
    """Create and setup the environment and tasks.

    Returns:
        Tuple containing:
        - The environment
        - The training tasks for the approach
        - The original training tasks
    """
    # Create environment. Under real_robot_execute an executor is attached so
    # its rollouts drive the arm; deliberately HERE and not inside
    # create_new_env, because the planner builds its own envs through that
    # factory (the option model's private simulator, the shared skill
    # simulator) and those must stay pure simulation.
    env = create_new_env(CFG.env, do_cache=True, use_gui=CFG.use_gui)
    attach_real_robot(env)
    env.action_space.seed(CFG.seed)
    assert env.goal_predicates.issubset(env.predicates)

    # Setup predicates
    included_preds, excluded_preds = utils.parse_config_excluded_predicates(
        env)
    preds = utils.replace_goals_with_agent_specific_goals(
        included_preds, excluded_preds,
        env) if CFG.approach != "oracle" else included_preds

    # Create train tasks
    env_train_tasks = env.get_train_tasks()
    perceiver = create_perceiver(CFG.perceiver)
    train_tasks = [perceiver.reset(t) for t in env_train_tasks]

    # Strip excluded predicates and prepare approach tasks
    stripped_train_tasks = [
        utils.strip_task(task, preds) for task in train_tasks
    ]
    approach_train_tasks = [
        task.replace_goal_with_alt_goal() for task in stripped_train_tasks
    ]

    return env, approach_train_tasks, train_tasks


def setup_approach(env: BaseEnv, preds: set,
                   approach_train_tasks: List[Task]) -> 'BaseApproach':
    """Create and setup the approach/agent.

    Returns:
        The configured approach
    """
    # Setup options
    if CFG.option_learner == "no_learning":
        options = get_gt_options(env.get_name())
    else:
        options = parse_config_included_options(env)

    # Create approach
    approach_name = CFG.approach
    if CFG.approach_wrapper:
        approach_name = f"{CFG.approach_wrapper}[{approach_name}]"

    return create_approach(approach_name, preds, options, env.types,
                           env.action_space, approach_train_tasks)


def create_offline_dataset(env: BaseEnv, train_tasks: List[Task], preds: set,
                           approach: BaseApproach) -> Optional[Dataset]:
    """Create offline dataset if needed.

    Returns:
        Dataset if required, None otherwise
    """
    if approach.is_learning_based or CFG.make_demo_videos or \
        CFG.make_demo_images:

        options = get_gt_options(env.get_name()) if \
                        CFG.option_learner == "no_learning" \
                    else parse_config_included_options(env)
        return create_dataset(env, train_tasks, options, preds)
    return None


# ── Pipeline ─────────────────────────────────────────────────────


def _run_pipeline(env: BaseEnv,
                  cogman: CogMan,
                  train_tasks: List[Task],
                  offline_dataset: Optional[Dataset] = None) -> None:
    """Main pipeline for running the learning and testing process."""
    if cogman.is_learning_based:
        assert offline_dataset is not None, "Missing offline dataset"

        # Handle offline learning phase
        num_offline_trans, num_online_trans, learning_time, offline_metrics = \
            _handle_offline_learning(cogman, offline_dataset)

        # Run initial evaluation if needed
        initial_test_summary: Optional[Tuple[str, Metrics]] = None
        if CFG.skip_until_cycle < 0 and \
           not CFG.skip_test_until_last_ite_or_early_stopping and \
           not CFG.skip_initial_test:
            results = _run_testing(env, cogman, online_learning_cycle=None)
            results.update({
                "num_offline_transitions": num_offline_trans,
                "num_online_transitions": num_online_trans,
                "query_cost": 0.0,
                "learning_time": learning_time,
                **offline_metrics
            })
            _save_test_results(results, online_learning_cycle=None)
            initial_test_summary = ("the pre-loop test", results)

        # Run online learning loop
        _run_online_learning_loop(env, cogman, train_tasks, num_offline_trans,
                                  learning_time, offline_metrics,
                                  initial_test_summary)
    else:
        # Handle non-learning case
        results = _run_testing(env, cogman, online_learning_cycle=None)
        results.update({
            "num_offline_transitions": 0,
            "num_online_transitions": 0,
            "query_cost": 0.0,
            "learning_time": 0.0
        })
        _save_test_results(results, online_learning_cycle=None)


def _handle_offline_learning(
        cogman: CogMan,
        offline_dataset: Dataset) -> Tuple[int, float, float, dict]:
    """Handle offline learning phase and initial evaluation."""
    num_offline_transitions = sum(
        len(traj.actions) for traj in offline_dataset.trajectories)
    if CFG.load_approach:
        cogman.load(online_learning_cycle=None)
        learning_time = 0.0  # ignore loading time
    else:
        learning_start = time.perf_counter()
        cogman.learn_from_offline_dataset(offline_dataset)
        learning_time = time.perf_counter() - learning_start

    offline_learning_metrics = {
        f"offline_learning_{k}": v
        for k, v in cogman.metrics.items()
    }

    return num_offline_transitions, 0.0, learning_time, offline_learning_metrics


def _run_online_learning_loop(
        env: BaseEnv,
        cogman: CogMan,
        train_tasks: List[Task],
        num_offline_transitions: int,
        learning_time: float,
        offline_learning_metrics: dict,
        initial_test_summary: Optional[Tuple[str, Metrics]] = None) -> None:
    """Run the online learning loop.

    ``initial_test_summary`` is ``(label, results)`` from the pre-loop
    test, if one ran; it seeds the last-test summary that gets logged
    when early stopping triggers.
    """
    num_online_transitions = 0
    total_query_cost = 0.0
    test_solve_rate = 0.0
    # Train-driven early stopping certifies the *learned* model, so it is
    # only eligible once the scored attempts were generated by a model that
    # has actually learned: from offline demos, a loaded approach, or a
    # prior online learning update. Otherwise (e.g. cycle 0 with no demos)
    # the explorer's successes reflect only the initial mental model, and
    # stopping would skip learning entirely.
    model_has_learned = CFG.load_approach or num_offline_transitions > 0
    # (label, results) of the most recent test evaluation, re-logged on
    # early stopping so the final solve rate and rewards are visible at
    # the end of the log instead of cycles back.
    last_test_summary = initial_test_summary

    # Create teacher if needed
    teacher = Teacher(train_tasks) if get_allowed_query_type_names() else None
    load_approach = CFG.load_approach

    for i in range(CFG.num_online_learning_cycles):
        if i < CFG.skip_until_cycle:
            continue

        # Handle loading approach
        if load_approach and i > 0:
            cogman.load(online_learning_cycle=i - 1)
            if CFG.restart_learning:
                load_approach = False

        # Run online interaction
        logging.info(f"\n\nONLINE LEARNING CYCLE {i}\n")
        if num_online_transitions >= CFG.online_learning_max_transitions:
            logging.info(
                "Reached online_learning_max_transitions, terminating")
            break

        interaction_requests = cogman.get_interaction_requests()
        if not interaction_requests:
            logging.info(
                "Did not receive any interaction requests, terminating")
            break

        (interaction_results, query_cost,
         task_solved_status) = \
            _generate_interaction_results(
                cogman, env, teacher,
                interaction_requests, i)

        # Track every solve attempt per task. The first attempt is used for
        # the legacy solve-rate metric; the full list is used when
        # online_learning_early_stopping_require_all_attempts is on.
        task_first_solve_attempts: Dict[int, bool] = {}
        task_all_solve_attempts: Dict[int, List[bool]] = {}
        for request, solved in zip(interaction_requests, task_solved_status):
            task_idx = request.train_task_idx
            task_all_solve_attempts.setdefault(task_idx, []).append(solved)
            if task_idx not in task_first_solve_attempts:
                task_first_solve_attempts[task_idx] = solved

        num_online_transitions += sum(
            len(result.actions) for result in interaction_results)
        total_query_cost += query_cost
        logging.info(f"Query cost incurred this cycle: {query_cost}")

        # Calculate train task solve rate. When require_all_attempts is on,
        # report over every attempt this cycle so the denominator matches the
        # early-stop criterion (which inspects task_all_solve_attempts).
        if CFG.online_learning_early_stopping_require_all_attempts:
            all_attempts = [
                solved for attempts in task_all_solve_attempts.values()
                for solved in attempts
            ]
            if all_attempts:
                train_task_solve_rate = sum(all_attempts) / len(all_attempts)
                logging.info(
                    f"Train task solve rate: {train_task_solve_rate:.3f} "
                    f"({sum(all_attempts)}/{len(all_attempts)})")
            else:
                train_task_solve_rate = 0.0
        elif task_first_solve_attempts:
            train_task_solve_rate = sum(task_first_solve_attempts.values()
                                        ) / len(task_first_solve_attempts)
            logging.info(f"Train task solve rate: {train_task_solve_rate:.3f} "
                         f"({sum(task_first_solve_attempts.values())}/"
                         f"{len(task_first_solve_attempts)})")

        else:
            train_task_solve_rate = 0.0

        # Determine if we should run testing
        is_last_iteration = (i == CFG.num_online_learning_cycles - 1)
        should_run_testing = (
            is_last_iteration
            or not CFG.skip_test_until_last_ite_or_early_stopping)
        # Early stopping has two mutually-exclusive modes, selected by
        # CFG.online_learning_early_stopping_by_test_solve_rate:
        #
        # (A) Train-driven (default; require online_learning_early_stopping
        #     to be True). Stop once this cycle's interaction requests cover
        #     every train task and all of those attempts succeeded, provided
        #     the model generating those attempts has learned at least once
        #     (model_has_learned above). Sub-mode
        #     controlled by online_learning_early_stopping_require_all_attempts:
        #       - False: only the first attempt per task must succeed
        #                (legacy behaviour).
        #       - True:  every attempt must succeed. Combined with multiple
        #                interaction requests per cycle and the explorer's
        #                advancing rng (so each request samples differently)
        #                this guards against a single lucky sample masking
        #                a buggy learned model.
        #
        # (B) Test-driven
        #     (CFG.online_learning_early_stopping_by_test_solve_rate).
        #     Stop once test_solve_rate hits 1.0. Note: testing for cycle i
        #     happens AFTER this check (see _run_testing below), so the
        #     test_solve_rate we read here is from cycle i-1 (or 0.0 before
        #     the first test run). This mode ignores
        #     online_learning_early_stopping itself.
        early_stopping = False
        if CFG.online_learning_early_stopping_require_all_attempts:
            train_tasks_all_attempts_solved = (
                len(task_all_solve_attempts) == len(train_tasks)
                and all(attempts and all(attempts)
                        for attempts in task_all_solve_attempts.values()))
            train_early_stop_msg = (
                "All training tasks solved on every attempt this cycle, "
                "triggering early stopping.\n")
        else:
            train_tasks_all_attempts_solved = (
                len(task_first_solve_attempts) == len(train_tasks)
                and all(task_first_solve_attempts.values()))
            train_early_stop_msg = (
                "All training tasks solved on first attempt, "
                "triggering early stopping.\n")
        train_driven_early_stop = (
            CFG.online_learning_early_stopping
            and not CFG.online_learning_early_stopping_by_test_solve_rate
            and model_has_learned and train_tasks_all_attempts_solved)
        if (CFG.online_learning_early_stopping
                and not CFG.online_learning_early_stopping_by_test_solve_rate
                and train_tasks_all_attempts_solved and not model_has_learned):
            logging.info(
                "All training tasks solved this cycle, but the model has "
                "not learned yet, so early stopping is not eligible; "
                "continuing to learning.\n")
        test_driven_early_stop = (
            CFG.online_learning_early_stopping_by_test_solve_rate
            and test_solve_rate == 1.0)
        # On the early-stopping cycle, force a test of the final model UNLESS we
        # have been testing every cycle AND the user opted into skipping the
        # redundant re-test. Because learning is skipped on the early-stopping
        # cycle (see below), the model is identical to the one the previous
        # cycle already tested, so re-testing only re-samples test-time
        # stochasticity at full test-set cost. When
        # skip_test_until_last_ite_or_early_stopping is True the early-stopping
        # cycle is the model's only test, so we must still run it. Likewise,
        # when no test has run yet at all (e.g. skip_initial_test and early
        # stopping on cycle 0), there is no prior result the re-test would
        # duplicate, so run it to get at least one evaluation.
        force_early_stop_test = not (
            CFG.online_learning_early_stopping_skip_redundant_test
            and not CFG.skip_test_until_last_ite_or_early_stopping
            and last_test_summary is not None)
        if train_driven_early_stop:
            logging.info(train_early_stop_msg)
            early_stopping = True
            should_run_testing = force_early_stop_test
        elif test_driven_early_stop:
            logging.info("Test solve rate from the previous cycle is 1.0, "
                         "triggering early stopping.\n")
            early_stopping = True
            should_run_testing = force_early_stop_test
        # Learn from results if appropriate
        if (not CFG.load_approach or CFG.restart_learning) and \
            not early_stopping:
            learning_start = time.perf_counter()
            logging.info("Learning from interaction results...")
            cogman.learn_from_interaction_results(interaction_results)
            learning_time += time.perf_counter() - learning_start
            model_has_learned = True

        # Evaluate if needed
        if should_run_testing:
            results = _run_testing(env, cogman, online_learning_cycle=i)
            results.update({
                "num_offline_transitions": num_offline_transitions,
                "num_online_transitions": num_online_transitions,
                "query_cost": total_query_cost,
                "learning_time": learning_time,
                **offline_learning_metrics
            })
            _save_test_results(results, online_learning_cycle=i)
            test_solve_rate = results["test_solve_rate"]
            last_test_summary = (f"cycle {i}", results)
        elif early_stopping:
            prev_test_label = f"cycle {i - 1}" if i > 0 else "the pre-loop test"
            logging.info(
                f"Skipping testing for early-stopping cycle {i}: model is "
                f"unchanged from {prev_test_label} (learning skipped this "
                "cycle), which was already tested. See "
                "online_learning_early_stopping_skip_redundant_test.")
        else:
            logging.info("Skipping testing for cycle "
                         f"{i} due to "
                         "skip_test_until_last_ite_or"
                         "_early_stopping flag")

        if early_stopping:
            if last_test_summary is not None:
                label, test_results = last_test_summary
                logging.info(
                    f"Early stopping: last test evaluation ({label}): "
                    f"{_format_test_results_line(test_results)}")
            break


def _early_stop_below_bar_msg(episode_reward: float,
                              env_task: EnvironmentTask) -> Optional[str]:
    """Check a solved episode's reward against the task's early-stopping bar.

    Returns a log-ready description when the episode reward falls short
    of ``env_task.early_stop_min_reward`` (minus the configured slack),
    meaning the solve must NOT count toward early stopping; returns None
    when the task sets no bar, the bar is ignored via
    ``CFG.online_learning_early_stopping_ignore_reward_bar``, or the
    reward clears it. The comparison carries a small tolerance so a
    reward computed exactly at the bar is never rejected on float
    rounding.
    """
    reward_bar = env_task.early_stop_min_reward
    if reward_bar is None:
        return None
    if CFG.online_learning_early_stopping_ignore_reward_bar:
        return None
    slack = CFG.online_learning_early_stopping_reward_slack
    if episode_reward >= reward_bar - slack - 1e-9:
        return None
    return (f"below the early-stop reward bar (reward={episode_reward:g} < "
            f"min_reward={reward_bar:g} - slack {slack:g})")


def _generate_interaction_results(
    cogman: CogMan,
    env: BaseEnv,
    teacher: Optional[Teacher],
    requests: Sequence[InteractionRequest],
    cycle_num: Optional[int] = None
) -> Tuple[List[InteractionResult], float, List[bool]]:
    """Given a sequence of InteractionRequest objects, handle the requests and
    return a list of InteractionResult objects."""
    logging.info("Generating interaction results...")
    results = []
    query_cost = 0.0
    task_solved_status = []
    for episode_idx, request in enumerate(requests):
        if request.train_task_idx < CFG.max_initial_demos and \
            not CFG.allow_interaction_in_demo_tasks:
            raise RuntimeError("Interaction requests cannot be on demo tasks "
                               "if allow_interaction_in_demo_tasks is False.")
        monitor: Optional[utils.VideoMonitor] = None
        if teacher is not None:
            monitor = TeacherInteractionMonitorWithVideo(
                env.render, request, teacher)
        elif CFG.make_interaction_videos:
            monitor = utils.VideoMonitor(env.render)

        # Used to check if our think the approach is unsolvable.
        if CFG.env_has_impossible_goals:
            planning_explorer_generated_a_plan = True
            if 'RandomNSRTsExplorer' in request.act_policy.__qualname__:
                planning_explorer_generated_a_plan = False
        cogman.set_override_policy(request.act_policy)
        cogman.set_termination_function(request.termination_function)
        env_task = env.get_train_tasks()[request.train_task_idx]
        cogman.reset(env_task)
        observed_traj, solved, _ = run_episode_and_get_observations(
            cogman,
            env,
            "train",
            request.train_task_idx,
            max_num_steps=(CFG.max_num_steps_interaction_request + 1),
            terminate_on_goal_reached=False,
            exceptions_to_break_on={
                utils.EnvironmentFailure,
                utils.OptionExecutionFailure,
                utils.RequestActPolicyFailure,
            },
            monitor=monitor)
        if CFG.env_has_impossible_goals:
            task_solvable = env.is_task_solvable(env_task)
            if not task_solvable:
                solved = not planning_explorer_generated_a_plan
        # A planning explorer may report that its mental model could NOT
        # reach the goal during refinement (it then ran the plan as an
        # experiment). Don't certify such a task as solved for early
        # stopping even if real-env execution happened to reach the goal —
        # the learned model still can't be planned with. None ⇒ no verdict.
        if request.mental_model_solved is False:
            solved = False
        task_solved_status.append(solved)

        # Debug final state (mirrors _run_testing). Lets us inspect the real
        # env state at the end of the rollout — e.g. whether SwitchBurnerOff
        # actually flipped the burner — separately from what the agent's
        # mental model believes happened.
        # pylint: disable=protected-access
        final_obs = env.get_observation()
        logging.debug(f"Interaction goal:\n{env_task.task.goal}")
        if hasattr(cogman._approach, "_get_current_predicates"):
            abstract_state = utils.abstract(
                final_obs, cogman._approach._get_current_predicates())
            logging.debug(f"Interaction final abstract state:\n"
                          f"{abstract_state}")
        # pylint: enable=protected-access
        logging.debug(f"Interaction final state (solved={solved}):\n"
                      f"{final_obs.pretty_str()}")
        cogman.unset_override_policy()
        cogman.unset_termination_function()
        traj = cogman.get_current_history()
        request_responses: List[Optional[Response]] = [
            None for _ in traj.states
        ]
        if isinstance(monitor, TeacherInteractionMonitorWithVideo):
            request_responses = monitor.get_responses()
            query_cost += monitor.get_query_cost()
        assert len(traj.states) == len(observed_traj[0])
        assert len(traj.actions) == len(observed_traj[1])
        # The env evaluator's verdict on this episode. Only the (reward,
        # terminated) pair travels to the agent side (rejection is
        # decodable from it) - the agent must infer the violated rule
        # from the task's NL description; the specific reason stays here
        # in the logs.
        episode_eval = env.evaluate_episode(observed_traj[0], observed_traj[1])
        accepted = episode_eval.terminated and not episode_eval.rejected
        logging.info(
            "Interaction episode on train task %d: reward=%.2f, "
            "terminated=%s, accepted=%s", request.train_task_idx,
            episode_eval.reward, episode_eval.terminated, accepted)
        if episode_eval.rejected:
            logging.info(
                "Interaction episode on train task %d REJECTED by the "
                "env: %s", request.train_task_idx, episode_eval.reason)
            # A rule-breaking episode must never count as solved for the
            # early-stopping criterion, regardless of how the cogman solve
            # gate scored it (today the gate runs the same certificate, so
            # this is belt-and-braces; it keeps the invariant local and
            # explicit).
            task_solved_status[-1] = False
        if task_solved_status[-1]:
            below_bar_msg = _early_stop_below_bar_msg(episode_eval.reward,
                                                      env_task)
            if below_bar_msg is not None:
                logging.info(
                    "Interaction episode on train task %d solved but %s: "
                    "does NOT count as solved for early stopping.",
                    request.train_task_idx, below_bar_msg)
                task_solved_status[-1] = False
        result = InteractionResult(traj.states,
                                   traj.actions,
                                   request_responses,
                                   episode_reward=episode_eval.reward,
                                   episode_terminated=episode_eval.terminated)
        results.append(result)
        if CFG.make_interaction_videos:
            assert monitor is not None
            # One video per interaction episode, saved as soon as the
            # episode ends so a mid-cycle crash keeps earlier footage.
            # scripts/log_viewer.py parses the __ep<i>__cycle<C>.mp4 tail
            # to pair each explore transcript with its episode's video.
            save_prefix = utils.get_config_path_str()
            outfile = f"{save_prefix}__ep{episode_idx}__cycle{cycle_num}.mp4"
            utils.save_video(outfile, monitor.get_video())
    return results, query_cost, task_solved_status


def _run_testing(env: BaseEnv,
                 cogman: CogMan,
                 online_learning_cycle: Optional[int] = None) -> Metrics:
    """Run testing on the environment's test tasks using the cogman approach,
    measuring both solve and execution metrics, and recording
    successes/failures.

    ``online_learning_cycle`` is the cycle this test round belongs to (``None``
    for the pre-learning baseline, ``i`` for the test after cycle ``i``). It is
    woven into the saved image/video filenames so successive test rounds do NOT
    overwrite each other -- matching how ``_save_test_results`` already suffixes
    the metrics pkl with the cycle.

    Returns a Metrics object populated with aggregated statistics.
    """
    test_tasks = env.get_test_tasks()
    if CFG.approach != "oracle":
        test_tasks = [task.replace_goal_with_alt_goal() for task in test_tasks]

    # Initialize counters and per-run metrics
    cogman.reset_metrics()
    save_prefix = utils.get_config_path_str()
    # Per-cycle tag so each test round's rendered artifacts are distinct.
    cycle_tag = f"__cycle{online_learning_cycle}"
    metrics: Metrics = defaultdict(float)

    num_found_policy = 0
    num_solved = 0
    # Sum of per-task episode rewards. Tasks that never execute (solve
    # failure/timeout) contribute 0.0, so the average below is over ALL
    # test tasks.
    total_test_reward = 0.0
    total_suc_time = 0.0
    total_low_level_action_cost = 0.0

    # Summaries for approach/execution failures
    total_num_solve_timeouts = 0
    total_num_solve_failures = 0
    total_num_execution_timeouts = 0
    total_num_execution_failures = 0

    # Track the running totals for nodes created/expanded
    curr_num_nodes_created = 0.0
    curr_num_nodes_expanded = 0.0

    # --------------------------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------------------------
    def _save_video(monitor: Optional[utils.LoggingMonitor], is_failure: bool,
                    task_idx: int) -> None:
        """Save a video from the monitor if the current config calls for it."""
        if monitor is None:
            return
        if CFG.use_counterfactual_dataset_path_name:
            suffix = ""
        else:
            suffix = "_failure" if is_failure else ""
        outfile = f"{save_prefix}__task{task_idx+1}{suffix}{cycle_tag}.mp4"
        if isinstance(monitor, utils.StreamingVideoMonitor):
            monitor.finalize(outfile)
        else:
            assert isinstance(monitor, utils.VideoMonitor)
            utils.save_video(outfile, monitor.get_video())

    def _save_images(monitor: Optional[utils.LoggingMonitor], is_failure: bool,
                     task_idx: int) -> None:
        """Save images from the monitor if the current config calls for it."""
        if monitor is None:
            return
        assert isinstance(monitor, utils.VideoMonitor)
        video = monitor.get_video()
        if CFG.use_counterfactual_dataset_path_name:
            experiment_id = CFG.experiment_id.split("-")[0]
            outfile = (f"{experiment_id}/seed{CFG.seed}/query/"
                       f"cycle{online_learning_cycle}/task{task_idx+1}/")
        else:
            suffix = "_failure" if is_failure else ""
            outfile = f"{save_prefix}__task{task_idx+1}{suffix}{cycle_tag}"
        utils.save_images(outfile, video)

    def _handle_solve_exception(
        e: Union[ApproachTimeout, ApproachFailure],
        task_idx: int,
        partial_refinements: Any,
    ) -> Tuple[int, int]:
        """Handle approach exceptions during the solve step, returning
        (updated_num_solve_timeouts, updated_num_solve_failures)."""
        nonlocal total_num_solve_timeouts, total_num_solve_failures
        if isinstance(e, ApproachTimeout):
            total_num_solve_timeouts += 1
        else:
            total_num_solve_failures += 1

        # Optionally save partial-refinement-based video
        if (CFG.make_failure_videos or CFG.make_failure_images) and\
              partial_refinements:
            logging.info("Creating video from partial "
                         "refinements...")
            video = utils.create_video_from_partial_refinements(
                partial_refinements, env, "test", task_idx, CFG.horizon)
            if CFG.make_failure_images:
                experiment_id = CFG.experiment_id.split("-")[0]
                outfile = f"{experiment_id}/seed{CFG.seed}/query/"+\
                            f"cycle{online_learning_cycle}/task{task_idx+1}/"
                utils.save_images(outfile, video)
            if CFG.make_failure_videos:
                outfile = (f"{save_prefix}__task{task_idx+1}_failure"
                           f"{cycle_tag}.mp4")
                utils.save_video(outfile, video)

        if CFG.crash_on_failure:
            raise e
        return total_num_solve_timeouts, total_num_solve_failures

    def _solve_task(_task_idx: int, env_task: EnvironmentTask) -> float:
        """Try to solve the given env_task using cogman, returning the solve
        time."""
        solve_start = time.perf_counter()
        logging.debug(f"[main.py] Solving task w. goal: {env_task.goal}")
        cogman.reset(env_task)  # May raise ApproachTimeout or ApproachFailure
        return time.perf_counter() - solve_start

    def _execute_policy(
        task_idx: int,
        env_task: EnvironmentTask,
        episode_env: BaseEnv,
        monitor: Optional[utils.LoggingMonitor] = None
    ) -> Tuple[bool, bool, float, int, Tuple[List[Observation], List[Action]]]:
        """Execute the cogman policy in ``episode_env`` to see if the goal is
        solved.

        Returns:
            (solved, caught_exception, exec_time,
             num_options_executed, low_level_action_cost)
        """
        solved = False
        caught_exception = False
        exec_time = 0.0
        num_options_executed = 0

        try:
            traj, solved, execution_metrics = run_episode_and_get_observations(
                cogman,
                episode_env,
                "test",
                task_idx,
                max_num_steps=CFG.horizon,
                monitor=monitor,
                terminate_on_goal_reached=CFG.terminate_on_goal_reached)
            exec_time = execution_metrics["policy_call_time"]
            num_options_executed = int(
                execution_metrics["num_options_executed"])

            # Optionally save a successful trajectory
            if CFG.save_eval_trajs:
                os.makedirs(CFG.eval_trajectories_dir, exist_ok=True)
                traj_file = f"{save_prefix}__task{task_idx+1}.traj"
                traj_file_path = Path(CFG.eval_trajectories_dir) / traj_file
                traj_data = {
                    "task": env_task,
                    "trajectory": traj,
                    "pybullet_robot": CFG.pybullet_robot
                }
                with open(traj_file_path, "wb") as f:
                    pkl.dump(traj_data, f)
        except utils.EnvironmentFailure as e:
            logging.info(f"Environment failed with error: {e}")
            caught_exception = True
        except (ApproachTimeout, ApproachFailure) as e:
            logging.info(f"Approach failed at execution time with error: {e}")
            if isinstance(e, ApproachTimeout):
                nonlocal total_num_execution_timeouts
                total_num_execution_timeouts += 1
            else:
                nonlocal total_num_execution_failures
                total_num_execution_failures += 1
            caught_exception = True

        # Debug final state
        # pylint: disable=protected-access
        if hasattr(cogman._approach, "_get_current_predicates"):
            abstract_state = utils.abstract(
                episode_env.get_observation(),
                cogman._approach._get_current_predicates())
            # pylint: enable=protected-access
            logging.debug(f"Final abstract state:\n{abstract_state}")
        logging.debug(
            f"Final state:\n{episode_env.get_observation().pretty_str()}")

        # if traj is defined
        if 'traj' not in locals():
            traj = ([], [])

        return solved, caught_exception, exec_time, num_options_executed, traj

    # --------------------------------------------------------------------------
    # Main testing loop
    # --------------------------------------------------------------------------
    cogman._approach.begin_test_phase()  # pylint: disable=protected-access
    for test_task_idx, env_task in enumerate(test_tasks):
        # ---------------------
        # 1) Solve phase
        # ---------------------
        try:
            logging.info(f"[main.py] Solving task {test_task_idx+1}/"
                         f"{len(test_tasks)}...")
            solve_time = _solve_task(test_task_idx, env_task)
        except (ApproachTimeout, ApproachFailure) as e:
            # Handle solve failure/timeouts
            partial_refinements = getattr(e, "info",
                                          {}).get("partial_refinements")
            logging.info(f"[main.py] Task {test_task_idx+1} / "
                         f"{len(test_tasks)}: approach failed with error: {e}")
            _handle_solve_exception(e, test_task_idx, partial_refinements)
            # Handle impossible goals here
            if CFG.env_has_impossible_goals:
                task_solvable = env.is_task_solvable(env_task)
                if not task_solvable:
                    if "not dr-reachable" in str(e):
                        logging.info("[main.py] Task is unsolvable and is "
                                     "recognized")
                        num_solved += 1
                        logging.info(f"Task {test_task_idx+1} / "
                                     f"{len(test_tasks)}: SOLVED")
            continue

        # Update solve-time metrics
        metrics[f"PER_TASK_task{test_task_idx}_solve_time"] = solve_time
        created = cogman.metrics["total_num_nodes_created"]
        expanded = cogman.metrics["total_num_nodes_expanded"]
        metrics[
            f"PER_TASK_task{test_task_idx}_nodes_created"] = created - \
                curr_num_nodes_created
        metrics[
            f"PER_TASK_task{test_task_idx}_nodes_expanded"] = expanded - \
                curr_num_nodes_expanded
        curr_num_nodes_created, curr_num_nodes_expanded = created, expanded

        num_found_policy += 1

        # ---------------------
        # 2) Execution phase
        # ---------------------
        # Run the episode in a fresh env instance when the env supports
        # it (see BaseEnv.make_fresh_test_instance): a long-lived
        # PyBullet world carries history that state-level resets do not
        # clear, so the episode's physics would depend on everything the
        # run executed before it.
        episode_env: BaseEnv = env
        fresh_env: Optional[BaseEnv] = None
        if CFG.test_fresh_env_per_episode:
            fresh_env = env.make_fresh_test_instance()
            if fresh_env is not None:
                episode_env = fresh_env
            else:
                logging.info(
                    "test_fresh_env_per_episode: env does not support a "
                    "fresh instance here (GUI/real-robot/base env); "
                    "executing in the shared long-lived env.")

        monitor: Optional[utils.LoggingMonitor] = None
        try:
            # Decide if we need to record video. Image saving needs the
            # raw frames after the episode, so it gets the buffering
            # monitor; video-only runs stream frames to disk as they are
            # rendered, keeping peak memory at one frame instead of a
            # whole episode.
            need_images = CFG.make_test_images or CFG.make_failure_images
            need_video = CFG.make_test_videos or CFG.make_failure_videos
            if need_images:
                monitor = utils.VideoMonitor(episode_env.render)
            elif need_video:
                monitor = utils.StreamingVideoMonitor(episode_env.render)

            logging.info("Executing policy...")
            solved, caught_exception, exec_time, num_opts, traj = \
                _execute_policy(test_task_idx, env_task, episode_env, monitor)

            # Record execution metrics
            metrics[f"PER_TASK_task{test_task_idx}_exec_time"] = exec_time
            metrics[
                f"PER_TASK_task{test_task_idx}_options_executed"] = num_opts

            # Task-evaluator verdict + offline metrics (e.g. domino
            # k_used), plus per-task oracle quantities (e.g. domino
            # k_star) stored on the EnvironmentTask. Offline-only:
            # reported in results, never agent-visible.
            if traj[0]:
                episode_eval = episode_env.evaluate_episode(traj[0], traj[1])
                metrics[f"PER_TASK_task{test_task_idx}_reward"] = \
                    episode_eval.reward
                total_test_reward += episode_eval.reward
                for metric_name, value in episode_eval.offline_metrics.items():
                    metrics[
                        f"PER_TASK_task{test_task_idx}_{metric_name}"] = value
                for metric_name, value in env_task.offline_task_metrics.items(
                ):
                    metrics[
                        f"PER_TASK_task{test_task_idx}_{metric_name}"] = value

            # Add cost for low-level actions if configured
            if CFG.refinement_data_include_execution_cost:
                total_low_level_action_cost += (
                    len(traj[1]) *
                    CFG.refinement_data_low_level_execution_cost)

            # ---------------------
            # 3) Post-execution handling
            # ---------------------
            if solved and not caught_exception:
                # The plan reached the goal
                log_msg = "SOLVED"
                num_solved += 1
                total_suc_time += (solve_time + exec_time)
                # If solved, we may want to save a video if
                # make_test_videos is True
                if CFG.make_test_videos:
                    _save_video(monitor,
                                is_failure=False,
                                task_idx=test_task_idx)
                if CFG.make_test_images:
                    _save_images(monitor,
                                 is_failure=False,
                                 task_idx=test_task_idx)
                # Count how many steps we took
                # (We rely on the last trajectory from
                # run_episode_and_get_observations)
                # If you need the real trajectory, you'd store
                # it as in `_execute_policy`.
                # Suppose we do that here (execution_metrics / logging):
                metrics[f"PER_TASK_task{test_task_idx}_num_steps"] = len(
                    traj[1])
            else:
                # The plan did not reach the goal, or an exception occurred
                if not caught_exception:
                    log_msg = "Policy failed to reach goal"
                else:
                    log_msg = "Policy/Env encountered an exception"
                if CFG.crash_on_failure:
                    raise RuntimeError(log_msg)
                if CFG.make_failure_videos:
                    _save_video(monitor,
                                is_failure=True,
                                task_idx=test_task_idx)
                if CFG.make_failure_images:
                    _save_images(monitor,
                                 is_failure=True,
                                 task_idx=test_task_idx)

        finally:
            # Drop the streamed clip when no branch above finalized it
            # (a solved episode with only make_failure_videos on, or an
            # exception past the save calls); no-op otherwise. In the
            # finally so a raise inside the try cannot leak the
            # monitor's temp file and open writer.
            if isinstance(monitor, utils.StreamingVideoMonitor):
                monitor.discard()
            if fresh_env is not None:
                fresh_env.dispose()

        logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: {log_msg}")

    cogman._approach.end_test_phase()  # pylint: disable=protected-access

    # --------------------------------------------------------------------------
    # Aggregate final metrics
    # --------------------------------------------------------------------------
    metrics["num_solved"] = num_solved
    metrics["num_total"] = len(test_tasks)
    metrics["avg_test_reward"] = (total_test_reward /
                                  len(test_tasks) if test_tasks else 0.0)
    metrics["avg_suc_time"] = (total_suc_time /
                               num_solved if num_solved > 0 else float("inf"))
    metrics["avg_ref_cost"] = ((total_low_level_action_cost +
                                cogman.metrics["total_refinement_time"]) /
                               num_solved if num_solved > 0 else float("inf"))

    # Skeleton / sample info
    metrics["min_num_samples"] = (
        cogman.metrics["min_num_samples"]
        if cogman.metrics["min_num_samples"] < float("inf") else 0)
    metrics["max_num_samples"] = cogman.metrics["max_num_samples"]
    metrics["min_skeletons_optimized"] = (
        cogman.metrics["min_num_skeletons_optimized"]
        if cogman.metrics["min_num_skeletons_optimized"] < float("inf") else 0)
    metrics["max_skeletons_optimized"] = cogman.metrics[
        "max_num_skeletons_optimized"]

    # Failure/timeouts
    metrics["num_solve_timeouts"] = total_num_solve_timeouts
    metrics["num_solve_failures"] = total_num_solve_failures
    metrics["num_execution_timeouts"] = total_num_execution_timeouts
    metrics["num_execution_failures"] = total_num_execution_failures

    # Compute averages of certain CogMan metrics wrt # of found policies
    for metric_name in [
            "num_samples", "num_skeletons_optimized", "num_nodes_expanded",
            "num_nodes_created", "num_nsrts", "num_preds", "plan_length",
            "num_failures_discovered"
    ]:
        total = cogman.metrics[f"total_{metric_name}"]
        metrics[f"avg_{metric_name}"] = (
            total / num_found_policy if num_found_policy > 0 else float("inf"))

    return metrics


def _format_per_task_rewards(results: Metrics) -> str:
    """Comma-joined per-task episode rewards of one test round.

    Tasks that never executed (solve failure/timeout) have no reward
    entry and show as ``n/a``.
    """
    parts = []
    for i in range(int(results["num_total"])):
        reward = results.get(f"PER_TASK_task{i}_reward")
        parts.append(
            f"task{i}={reward:.2f}" if reward is not None else f"task{i}=n/a")
    return ", ".join(parts)


def _format_test_results_line(results: Metrics) -> str:
    """Summarize a test round: solve rate, average reward, per-task rewards."""
    num_solved = int(results["num_solved"])
    num_total = int(results["num_total"])
    rate = num_solved / num_total if num_total else 0.0
    return (f"solve rate {rate:.3f} ({num_solved} / {num_total}), "
            f"avg reward {results['avg_test_reward']:.3f}, "
            f"per-task rewards: {_format_per_task_rewards(results)}")


def _save_test_results(results: Metrics,
                       online_learning_cycle: Optional[int]) -> None:
    num_solved = results["num_solved"]
    num_total = results["num_total"]
    avg_suc_time = results["avg_suc_time"]
    logging.info(f"Tasks solved: {num_solved} / {num_total}")
    logging.info(f"Average test reward: {results['avg_test_reward']:.3f}")
    logging.info(f"Per-task rewards: {_format_per_task_rewards(results)}")
    logging.info(f"Average time for successes: {avg_suc_time:.5f} seconds")
    os.makedirs(CFG.results_dir, exist_ok=True)
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
