"""Create offline datasets by collecting demonstrations."""

import functools
import logging
import os
import re
from typing import Callable, List, Set

import dill as pkl
import matplotlib
import matplotlib.pyplot as plt

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.base_approach import BaseApproach
from predicators.approaches.oracle_approach import OracleApproach
from predicators.approaches.pp_oracle_approach import \
    OracleBilevelProcessPlanningApproach
from predicators.cogman import CogMan, run_episode_and_get_states
from predicators.envs import BaseEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Action, Dataset, LowLevelTrajectory, \
    ParameterizedOption, State, Task


def create_demo_data(env: BaseEnv, train_tasks: List[Task],
                     known_options: Set[ParameterizedOption],
                     annotate_with_gt_ops: bool) -> Dataset:
    """Create offline datasets by collecting demos."""
    assert CFG.demonstrator in ("oracle", "human", "oracle_process_planning")
    dataset_fname, dataset_fname_template = utils.create_dataset_filename_str(
        saving_ground_atoms=False)
    os.makedirs(CFG.data_dir, exist_ok=True)
    if CFG.load_data:
        dataset = _create_demo_data_with_loading(env, train_tasks,
                                                 known_options,
                                                 dataset_fname_template,
                                                 dataset_fname)
    else:
        dataset = _generate_demonstrations(
            env,
            train_tasks,
            known_options,
            train_tasks_start_idx=0,
            annotate_with_gt_ops=annotate_with_gt_ops)
        logging.info(f"\n\nCREATED {len(dataset.trajectories)} DEMONSTRATIONS")

        with open(dataset_fname, "wb") as f:
            pkl.dump(dataset, f)
    return dataset


def _create_demo_data_with_loading(env: BaseEnv, train_tasks: List[Task],
                                   known_options: Set[ParameterizedOption],
                                   dataset_fname_template: str,
                                   dataset_fname: str) -> Dataset:
    """Create demonstration data while handling loading from disk.

    This method takes care of three cases: the demonstrations on disk
    are exactly the desired number, too many, or too few. Note that we
    can only load datasets with annotations of exactly the right size;
    attempting to load annotations for smaller or bigger datasets will
    fail.
    """
    if os.path.exists(dataset_fname):
        # Case 1: we already have a file with the exact name that we need
        # (i.e., the correct amount of data).
        with open(dataset_fname, "rb") as f:
            dataset = pkl.load(f)
        logging.info(f"\n\nLOADED DATASET OF {len(dataset.trajectories)} "
                     "DEMONSTRATIONS")
        return dataset
    fnames_with_less_data = {}  # used later, in Case 3
    for fname in os.listdir(CFG.data_dir):
        regex_match = re.match(dataset_fname_template, fname)
        if not regex_match:
            continue
        num_train_tasks = int(regex_match.groups()[0])
        assert num_train_tasks != CFG.num_train_tasks  # would be Case 1
        # Case 2: we already have a file with MORE data than we need. Load
        # and truncate this data.
        if num_train_tasks > CFG.num_train_tasks:
            with open(os.path.join(CFG.data_dir, fname), "rb") as f:
                dataset = pkl.load(f)
            logging.info("\n\nLOADED AND TRUNCATED DATASET OF "
                         f"{len(dataset.trajectories)} DEMONSTRATIONS")
            assert not dataset.has_annotations
            # To truncate, note that we can't simply take the first
            # `CFG.num_train_tasks` elements of `dataset.trajectories`,
            # because some of these might have a `train_task_idx` that is
            # out of range (if there were errors in the course of
            # collecting those demonstrations). The correct thing to do
            # here is to truncate based on the value of `train_task_idx`.
            return Dataset([
                traj for traj in dataset.trajectories
                if traj.train_task_idx < CFG.num_train_tasks
            ])
        # Save the names of all datasets that have less data than
        # we need, to be used in Case 3.
        fnames_with_less_data[num_train_tasks] = fname
    if not fnames_with_less_data:
        # Give up: we did not find any data file we can load from.
        raise ValueError(f"Cannot load data: {dataset_fname}")
    # Case 3: we already have a file with LESS data than we need. Load
    # this data and generate some more. Specifically, we load from the
    # file with the maximum data among all files that have less data
    # than we need, then we generate the remaining demonstrations.
    train_tasks_start_idx = max(fnames_with_less_data)
    fname = fnames_with_less_data[train_tasks_start_idx]
    with open(os.path.join(CFG.data_dir, fname), "rb") as f:
        dataset = pkl.load(f)
    loaded_trajectories = dataset.trajectories
    generated_dataset = _generate_demonstrations(
        env,
        train_tasks,
        known_options,
        train_tasks_start_idx=train_tasks_start_idx,
        annotate_with_gt_ops=False)
    generated_trajectories = generated_dataset.trajectories
    logging.info(f"\n\nLOADED DATASET OF {len(loaded_trajectories)} "
                 "DEMONSTRATIONS")
    logging.info(
        f"CREATED {len(generated_dataset.trajectories)} DEMONSTRATIONS")
    dataset = Dataset(loaded_trajectories + generated_trajectories)
    with open(dataset_fname, "wb") as f:
        pkl.dump(dataset, f)
    return dataset


def _generate_demonstrations(env: BaseEnv, train_tasks: List[Task],
                             known_options: Set[ParameterizedOption],
                             train_tasks_start_idx: int,
                             annotate_with_gt_ops: bool) -> Dataset:
    """Use the demonstrator to generate demonstrations, one per training task
    starting from train_tasks_start_idx."""
    # Create a non-GUI option model for the demonstrator to avoid conflicts
    # with any existing GUI connection (PyBullet only allows one GUI).
    # pylint: disable=import-outside-toplevel
    from predicators.option_model import create_option_model

    # pylint: enable=import-outside-toplevel
    demo_option_model = create_option_model(CFG.option_model_name,
                                            use_gui=False)

    if CFG.demonstrator == "oracle":
        # Instantiate CogMan with the oracle approach (to be used as the
        # demonstrator). This requires creating a perceiver and
        # execution monitor according to settings from CFG.
        options = get_gt_options(env.get_name())
        oracle_approach: BaseApproach = OracleApproach(
            env.predicates,
            options,
            env.types,
            env.action_space,
            train_tasks,
            task_planning_heuristic=CFG.offline_data_task_planning_heuristic,
            max_skeletons_optimized=CFG.offline_data_max_skeletons_optimized,
            bilevel_plan_without_sim=CFG.offline_data_bilevel_plan_without_sim,
            option_model=demo_option_model)
        perceiver = create_perceiver(CFG.perceiver)
        execution_monitor = create_execution_monitor(CFG.execution_monitor)
        cogman = CogMan(oracle_approach, perceiver, execution_monitor)
    elif CFG.demonstrator == "oracle_process_planning":
        options = get_gt_options(env.get_name())
        oracle_approach = OracleBilevelProcessPlanningApproach(
            env.predicates,
            options,
            env.types,
            env.action_space,
            train_tasks,
            task_planning_heuristic=CFG.offline_data_task_planning_heuristic,
            max_skeletons_optimized=CFG.offline_data_max_skeletons_optimized,
            bilevel_plan_without_sim=CFG.offline_data_bilevel_plan_without_sim,
            option_model=demo_option_model)
        perceiver = create_perceiver(CFG.perceiver)
        execution_monitor = create_execution_monitor(CFG.execution_monitor)
        cogman = CogMan(oracle_approach, perceiver, execution_monitor)
    else:  # pragma: no cover
        # Disable all built-in keyboard shortcuts.
        keymaps = {k for k in plt.rcParams if k.startswith("keymap.")}
        for k in keymaps:
            plt.rcParams[k].clear()
        # Create the environment-specific method for turning events into
        # actions. This should also log instructions.
        event_to_action = env.get_event_to_action_fn()
    trajectories = []
    if annotate_with_gt_ops:
        annotations = []
    num_tasks = min(len(train_tasks), CFG.max_initial_demos)
    for idx, task in enumerate(train_tasks):
        if idx < train_tasks_start_idx:  # ignore demos before this index
            continue
        if CFG.make_demo_videos or CFG.make_demo_images:
            video_monitor = utils.VideoMonitor(env.render)
        else:
            video_monitor = None

        # Note: we assume in main.py that demonstrations are only generated
        # for train tasks whose index is less than CFG.max_initial_demos. If
        # you modify code around here, make sure that this invariant holds.
        if idx >= CFG.max_initial_demos:
            break
        # There are two kinds of failures we need to deal
        # with when generating demos for the mara
        # counterfactual dataset:
        #   1. exception from solving;
        #   2. exception from run_episode_and_get_states.
        # Point 1:
        # Exception during solving from reset. In this case,
        # the video is empty. To solve this, need to run with
        # create_video_from_partial_refinments or something.
        # Point 2:
        # When generating mara counterfactual videos, we want
        # it to save the video even with ApproachFailure,
        # which would happen when terminate_on_goal_reached
        # is False.
        # --- Try to solve the task
        succeed_in_solving = True
        try:
            if CFG.demonstrator in ("oracle", "oracle_process_planning"):
                # In this case, we use the instantiated cogman to generate
                # demonstrations. Importantly, we want to access state-action
                # trajectories, not observation-action ones.
                env_task = env.get_train_tasks()[idx]
                logging.info(f"Solving task {idx}...")
                cogman.reset(env_task)
            else:  # pragma: no cover
                # Otherwise, we get human input demos.
                caption = (f"Task {idx+1} / {num_tasks}\nPlease demonstrate "
                           f"achieving the goal:\n{task.goal}")
                policy = functools.partial(human_demonstrator_policy, env,
                                           caption, event_to_action)
                termination_function = task.goal_holds
        except (ApproachTimeout, ApproachFailure,
                utils.EnvironmentFailure) as e:
            succeed_in_solving = False
            logging.warning("WARNING: Approach failed to solve with error: "
                            f"{e}")
            # Make a policy from partial refinments
            if CFG.keep_failed_demos:
                partial_refinements = getattr(e, "info",
                                              {}).get("partial_refinements")
                if partial_refinements is None:
                    plan = []
                else:
                    _, plan = max(partial_refinements, key=lambda x: len(x[1]))
                policy = utils.option_plan_to_policy(
                    plan)  # type: ignore[assignment]
                termination_function = (  # type: ignore[assignment]
                    lambda state, vlm=None: False)

        # If solving failed and we are not keeping failed demos there is
        # no policy to execute (the except branch above only assigns
        # ``policy`` when ``CFG.keep_failed_demos`` is True), so skip the
        # task entirely. Without this guard, the else branch below hits
        # an ``UnboundLocalError`` on ``policy``.
        if not succeed_in_solving and not CFG.keep_failed_demos:
            continue

        # --- Execute the policy to generate a demonstration.
        try:
            logging.info("Executing policy...")
            if CFG.demonstrator in ("oracle", "oracle_process_planning") and \
                    succeed_in_solving:
                traj, _, _ = run_episode_and_get_states(
                    cogman,
                    env,
                    "train",
                    idx,
                    max_num_steps=CFG.horizon,
                    exceptions_to_break_on={
                        utils.OptionExecutionFailure,
                        utils.HumanDemonstrationFailure,
                    },
                    monitor=video_monitor,
                    terminate_on_goal_reached=CFG.terminate_on_goal_reached)
            else:  # pragma: no cover
                traj, _ = utils.run_policy(
                    policy,
                    env,
                    "train",
                    idx,
                    termination_function=termination_function,
                    max_num_steps=CFG.horizon,
                    exceptions_to_break_on={
                        utils.OptionExecutionFailure,
                        utils.HumanDemonstrationFailure,
                    },
                    monitor=video_monitor)

                if CFG.keep_failed_demos:
                    logging.info(
                        "Keeping failed demonstration from run_policy.")
                    if CFG.make_demo_videos and video_monitor is not None:
                        make_demo_videos(video_monitor, idx)
                    if CFG.make_demo_images and video_monitor is not None:
                        make_demo_images(video_monitor, idx, num_tasks)
        except (ApproachTimeout, ApproachFailure,
                utils.EnvironmentFailure) as e:
            logging.warning("WARNING: Approach failed to solve with error: "
                            f"{e}")
            if CFG.keep_failed_demos:
                logging.info("Keeping failed demonstration.")
                if CFG.make_demo_videos and video_monitor is not None:
                    make_demo_videos(video_monitor, idx)
                if CFG.make_demo_images and video_monitor is not None:
                    make_demo_images(video_monitor, idx, num_tasks)
            else:
                continue

        # Check that the goal holds at the end. Print a warning if not.
        if not task.goal_holds(traj.states[-1]):  # pragma: no cover
            logging.warning(f"WARNING: Oracle failed on training task {idx}.")
            if not CFG.keep_failed_demos:
                continue
        if CFG.demonstrator == "human":  # pragma: no cover
            logging.info("Successfully collected human demonstration of "
                         f"length {len(traj.states)} for task {idx+1} / "
                         f"{num_tasks}.")
        # Add is_demo flag and task index information into the trajectory.
        traj = LowLevelTrajectory(traj.states,
                                  traj.actions,
                                  _is_demo=True,
                                  _train_task_idx=idx)
        # To prevent cheating by option learning approaches, remove all oracle
        # options from the trajectory actions, unless the options are known
        # (via CFG.included_options or CFG.option_learner = 'no_learning').
        if CFG.demonstrator in ("oracle", "oracle_process_planning"):
            for act in traj.actions:
                if act.get_option().parent not in known_options:
                    assert CFG.option_learner != "no_learning"
                    act.unset_option()
        trajectories.append(traj)
        # If we're also annotating with ground truth operators,
        # then get the last nsrt_plan and add the name of the
        # nsrt used to the list of annotations.
        if annotate_with_gt_ops:
            last_nsrt_plan = (
                oracle_approach  # type: ignore[attr-defined]
                .get_last_nsrt_plan())
            annotations.append(list(last_nsrt_plan))
        if CFG.make_demo_videos and video_monitor is not None:
            make_demo_videos(video_monitor, idx)
        if CFG.make_demo_images and video_monitor is not None:
            make_demo_images(video_monitor, idx, num_tasks)
    if annotate_with_gt_ops:
        dataset = Dataset(trajectories, annotations)
    else:
        dataset = Dataset(trajectories)
    return dataset


def make_demo_images(video_monitor: utils.VideoMonitor, idx: int,
                     num_train_tasks: int) -> None:
    """Save demo images from the video monitor."""
    assert video_monitor is not None
    video = video_monitor.get_video()
    width = len(str(num_train_tasks))
    task_number = str(idx).zfill(width)
    if CFG.use_counterfactual_dataset_path_name:
        experiment_id = CFG.experiment_id.split("-")[0]
        outfile_prefix = f"{experiment_id}/seed{CFG.seed}/support/task{idx+1}/"
    else:
        outfile_prefix = f"{CFG.env}__{CFG.seed}__demo__task{task_number}"
    utils.save_images(outfile_prefix, video)


def make_demo_videos(video_monitor: utils.VideoMonitor, idx: int) -> None:
    """Save demo videos from the video monitor."""
    assert video_monitor is not None
    video = video_monitor.get_video()
    if CFG.use_counterfactual_dataset_path_name:
        outfile = f"{CFG.env}__{CFG.seed}__{CFG.experiment_id}"+\
                        f"__support__task{idx+1}.mp4"
    else:
        outfile = f"{CFG.env}__{CFG.seed}__demo__task{idx}.mp4"
    utils.save_video(outfile, video)


def human_demonstrator_policy(env: BaseEnv, caption: str,
                              event_to_action: Callable[
                                  [State, matplotlib.backend_bases.Event],
                                  Action],
                              state: State) -> Action:  # pragma: no cover
    """Collect actions from a human interacting with a GUI."""
    # Temporarily change the backend to one that supports a GUI.
    # We do this here because we don't want the rest of the codebase
    # to use GUI-based Matplotlib.
    cur_backend = matplotlib.get_backend()
    matplotlib.use("Qt5Agg")
    # Render the state.
    fig = env.render_plt(caption=caption)
    container = {}

    def _handler(event: matplotlib.backend_bases.Event) -> None:
        container["action"] = event_to_action(state, event)

    keyboard_cid = fig.canvas.mpl_connect("key_press_event", _handler)
    mouse_cid = fig.canvas.mpl_connect("button_press_event", _handler)
    # Hang until either a mouse press or a keyboard press.
    plt.waitforbuttonpress()
    fig.canvas.mpl_disconnect(keyboard_cid)
    fig.canvas.mpl_disconnect(mouse_cid)
    plt.close()
    if "action" not in container:
        logging.warning("WARNING: Event handler failed. Its error message "
                        "should be printed above. Terminating task.")
        raise utils.HumanDemonstrationFailure("Event handler failed!")
    # Revert to the previous backend.
    matplotlib.use(cur_backend)
    return container["action"]
