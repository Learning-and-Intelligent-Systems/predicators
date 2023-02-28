"""Scripts for collecting refinement data and for training learned refinement
cost estimators.

Example usage to generate data and train:
    python predicators/train_refinement_estimator.py --env narrow_passage \
        --approach refinement_estimation --refinement_estimator tabular \
        --seed 0

To save a generated dataset under a name different from the default:
    python predicators/train_refinement_estimator.py --env narrow_passage \
        --approach refinement_estimation --refinement_estimator tabular \
        --seed 0 --refinement_data_file_name my_data_file.data

To skip data collection and load a dataset file instead:
    python predicators/train_refinement_estimator.py --env narrow_passage \
        --approach refinement_estimation --refinement_estimator tabular \
        --seed 0 --load_data

To specify a dataset file name different from the default:
    python predicators/train_refinement_estimator.py --env narrow_passage \
        --approach refinement_estimation --refinement_estimator tabular \
        --seed 0 --load_data --refinement_data_file_name my_data_file.data

To skip training the approach:
    python predicators/train_refinement_estimator.py --env narrow_passage \
        --approach refinement_estimation --refinement_estimator tabular \
        --seed 0 --skip_refinement_estimator_training
"""

import logging
import os
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List, Set

import dill as pkl

from predicators import utils
from predicators.approaches.refinement_estimation_approach import \
    RefinementEstimationApproach
from predicators.envs import BaseEnv, create_new_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.option_model import _OptionModelBase, create_option_model
from predicators.planning import PlanningTimeout, _MaxSkeletonsFailure, \
    _skeleton_generator, _SkeletonSearchTimeout, filter_nsrts, \
    run_low_level_search, sesame_ground_nsrts
from predicators.settings import CFG
from predicators.structs import NSRT, Metrics, ParameterizedOption, \
    Predicate, RefinementDatapoint, Task, Type


def _train_refinement_estimation_approach() -> None:
    """Main function, setting up env/approach and then performing data
    collection and/or training as required."""
    # Following is copied/adapted from predicators/main.py
    script_start = time.perf_counter()

    # Parse & validate args
    parser = _get_refinement_estimation_parser()
    args = utils.parse_args_with_parser(parser)
    utils.update_config_with_parser(parser, args)
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

    # Create environment
    env = create_new_env(CFG.env, do_cache=True, use_gui=CFG.use_gui)
    # The action space needs to be seeded externally, because env.action_space
    # is often created during env __init__().
    env.action_space.seed(CFG.seed)
    assert env.goal_predicates.issubset(env.predicates)
    preds, _ = utils.parse_config_excluded_predicates(env)

    # Create the train tasks.
    train_tasks = env.get_train_tasks()
    # Assume we're not doing option learning, pass in all the environment's
    # oracle options.
    assert CFG.option_learner == "no_learning"
    options = get_gt_options(env.get_name())

    # Get dataset to train on
    if CFG.load_data:
        data_file_path = _get_data_file_path()
        with open(data_file_path, "rb") as f:
            dataset = pkl.load(f)
        logging.info(f"Loaded dataset from {data_file_path}")
    else:
        logging.info("Generating refinement data using "
                     f"{len(train_tasks)} train tasks...")
        data_gen_start_time = time.perf_counter()
        dataset = _generate_refinement_data(env, preds, options, train_tasks)
        data_gen_time = time.perf_counter() - data_gen_start_time
        logging.info(f"Generated {len(dataset)} datapoints in "
                     f"{data_gen_time:.5f} seconds")

    # Terminate early if training should be skipped
    if CFG.skip_refinement_estimator_training:
        script_time = time.perf_counter() - script_start
        logging.info(f"\n\nScript terminated in {script_time:.5f} seconds")
        return

    # Create approach
    assert CFG.approach == "refinement_estimation", \
        "Approach (--approach) must be set to refinement_estimation"
    approach = RefinementEstimationApproach(preds, options, env.types,
                                            env.action_space, train_tasks)
    refinement_estimator = approach.refinement_estimator
    assert refinement_estimator.is_learning_based, \
        "Refinement estimator (--refinement_estimator) must be learning-based"

    # Train estimator
    train_start_time = time.perf_counter()
    refinement_estimator.train(dataset)
    train_time = time.perf_counter() - train_start_time
    logging.info(f"Finished training in {train_time:.5f} seconds")
    # Save the training model to a file
    # Create saved data directory.
    os.makedirs(CFG.approach_dir, exist_ok=True)
    config_path_str = utils.get_config_path_str()
    model_file = f"{CFG.refinement_estimator}_{config_path_str}.estimator"
    model_file_path = Path(CFG.approach_dir) / model_file
    refinement_estimator.save_model(model_file_path)
    logging.info(f"Saved trained estimator to {model_file_path}")

    script_time = time.perf_counter() - script_start
    logging.info(f"\n\nScript terminated in {script_time:.5f} seconds")


def _get_refinement_estimation_parser() -> ArgumentParser:
    """Get default parser and add script-specific flags for refinement
    estimation data collection and training."""
    parser = utils.create_arg_parser()
    # Add script-specific flags to the parser
    parser.add_argument("--refinement_data_file_name", default="", type=str)
    parser.add_argument("--skip_refinement_estimator_training",
                        action="store_true")
    return parser


def _generate_refinement_data(
        env: BaseEnv, preds: Set[Predicate], options: Set[ParameterizedOption],
        train_tasks: List[Task]) -> List[RefinementDatapoint]:
    """Collect refinement data and save the dataset to a file."""
    num_tasks = len(train_tasks)
    nsrts = get_gt_nsrts(CFG.env, preds, options)
    option_model = create_option_model(CFG.option_model_name)

    # Generate the dataset and save it to file.
    dataset: List[RefinementDatapoint] = []
    for test_task_idx, task in enumerate(train_tasks):
        try:
            _collect_refinement_data_for_task(task, option_model, nsrts, preds,
                                              env.types,
                                              CFG.seed + test_task_idx,
                                              dataset)
            logging.info(f"Task {test_task_idx+1} / {num_tasks}: Success")
        except (PlanningTimeout, _SkeletonSearchTimeout) as e:
            logging.info(f"Task {test_task_idx+1} / {num_tasks} failed by "
                         f"timing out: {e}")
    logging.info(f"Got {len(dataset)} data points.")
    # Create saved data directory.
    os.makedirs(CFG.data_dir, exist_ok=True)
    # Create file path.
    data_file_path = _get_data_file_path()
    # Store the train tasks just in case we need it in the future.
    # (Note: unpickling this doesn't work...)
    # data_content = {
    #     "tasks": train_tasks,
    #     "data": dataset,
    # }
    logging.info(f"Writing dataset to {data_file_path}")
    with open(data_file_path, "wb") as f:
        pkl.dump(dataset, f)
    return dataset


def _collect_refinement_data_for_task(task: Task,
                                      option_model: _OptionModelBase,
                                      nsrts: Set[NSRT],
                                      predicates: Set[Predicate],
                                      types: Set[Type], seed: int,
                                      data: List[RefinementDatapoint]) -> None:
    """Generate refinement data from given train task and append to data.

    Adapted from _sesame_plan_with_astar() from predicators/planning.py.
    """
    ground_nsrt_timeout = CFG.timeout
    init_atoms = utils.abstract(task.init, predicates)
    objects = list(task.init)
    ground_nsrt_start_time = time.perf_counter()
    ground_nsrts = sesame_ground_nsrts(task, init_atoms, nsrts, objects,
                                       predicates, types,
                                       ground_nsrt_start_time,
                                       ground_nsrt_timeout)
    metrics: Metrics = defaultdict(float)
    # Optionally exclude NSRTs with empty effects, because they can slow
    # the search significantly, so we may want to exclude them.
    reachable_nsrts = filter_nsrts(task, init_atoms, ground_nsrts)
    heuristic = utils.create_task_planning_heuristic(
        CFG.sesame_task_planning_heuristic, init_atoms, task.goal,
        reachable_nsrts, predicates, objects)
    try:
        gen = _skeleton_generator(
            task, reachable_nsrts, init_atoms, heuristic, seed,
            CFG.refinement_data_skeleton_generator_timeout, metrics,
            CFG.refinement_data_num_skeletons)
        for skeleton, atoms_sequence in gen:
            necessary_atoms_seq = utils.compute_necessary_atoms_seq(
                skeleton, atoms_sequence, task.goal)
            refinement_start_time = time.perf_counter()
            _, suc = run_low_level_search(
                task, option_model, skeleton, necessary_atoms_seq, seed,
                CFG.refinement_data_low_level_search_timeout, metrics,
                CFG.horizon)
            # Calculate time taken for refinement.
            refinement_time = time.perf_counter() - refinement_start_time
            # Add datapoint to dataset
            data.append((
                task,
                skeleton,
                atoms_sequence,
                suc,
                refinement_time,
            ))
    except _MaxSkeletonsFailure:
        # Done finding skeletons
        return


def _get_data_file_path() -> Path:
    if len(CFG.refinement_data_file_name):
        file_name = CFG.refinement_data_file_name
    else:
        config_path_str = utils.get_config_path_str()
        file_name = f"refinement_data_{config_path_str}.data"
    data_file_path = Path(CFG.data_dir) / file_name
    return data_file_path


if __name__ == "__main__":  # pragma: no cover
    # Write out the exception to the log file.
    try:
        _train_refinement_estimation_approach()
    except Exception as _err:  # pylint: disable=broad-except
        logging.exception("train_refinement_estimator.py crashed")
        raise _err
