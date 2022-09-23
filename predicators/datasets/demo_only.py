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
from predicators.approaches.oracle_approach import OracleApproach
from predicators.envs import BaseEnv
from predicators.envs.behavior import BehaviorEnv
from predicators.planning import _run_plan_with_option_model
from predicators.settings import CFG
from predicators.structs import Action, Dataset, LowLevelTrajectory, \
    ParameterizedOption, State, Task


def create_demo_data(env: BaseEnv, train_tasks: List[Task],
                     known_options: Set[ParameterizedOption]) -> Dataset:
    """Create offline datasets by collecting demos."""
    assert CFG.demonstrator in ("oracle", "human")
    dataset_fname, dataset_fname_template = utils.create_dataset_filename_str(
        saving_ground_atoms=False)
    os.makedirs(CFG.data_dir, exist_ok=True)
    if CFG.load_data:
        dataset = _create_demo_data_with_loading(env, train_tasks,
                                                 known_options,
                                                 dataset_fname_template,
                                                 dataset_fname)
    else:
        trajectories = _generate_demonstrations(env,
                                                train_tasks,
                                                known_options,
                                                train_tasks_start_idx=0)
        logging.info(f"\n\nCREATED {len(trajectories)} DEMONSTRATIONS")
        dataset = Dataset(trajectories)

        # NOTE: This is necessary because BEHAVIOR options save
        # the BEHAVIOR environment object in their memory, and this
        # can't be pickled.
        if CFG.env == "behavior":  # pragma: no cover
            for traj in dataset.trajectories:
                for act in traj.actions:
                    act.get_option().memory = {}

        with open(dataset_fname, "wb") as f:
            pkl.dump(dataset, f)
        # Pickle information about dataset created.
        if CFG.env == "behavior":  # pragma: no cover
            assert isinstance(env, BehaviorEnv)
            info = {}
            info["behavior_task_list"] = CFG.behavior_task_list
            info["behavior_scene_name"] = CFG.behavior_scene_name
            info["seed"] = CFG.seed
            if len(CFG.behavior_task_list) != 1:
                info["task_list_indices"] = env.task_list_indices
                info["scene_list"] = env.scene_list
            info[
                "task_num_task_instance_id_to_igibson_seed"] = \
                    env.task_num_task_instance_id_to_igibson_seed
            with open(dataset_fname.replace(".data", ".info"), "wb") as f:
                pkl.dump(info, f)

    # NOTE: This is necessary because we replace BEHAVIOR
    # options with dummy options in order to pickle them, so
    # when we load them, we need to make sure they have the
    # correct options from the environment.
    if CFG.env == "behavior":  # pragma: no cover
        assert isinstance(env, BehaviorEnv)
        option_name_to_option = env.option_name_to_option
        for traj in dataset.trajectories:
            for act in traj.actions:
                dummy_opt = act.get_option()
                gt_param_opt = option_name_to_option[dummy_opt.name]
                gt_opt = gt_param_opt.ground(dummy_opt.objects,
                                             dummy_opt.params)
                act.set_option(gt_opt)
    return dataset


def _create_demo_data_with_loading(env: BaseEnv, train_tasks: List[Task],
                                   known_options: Set[ParameterizedOption],
                                   dataset_fname_template: str,
                                   dataset_fname: str) -> Dataset:
    """Create demonstration data while handling loading from disk.

    This method takes care of three cases: the demonstrations on disk
    are exactly the desired number, too many, or too few.
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
    generated_trajectories = _generate_demonstrations(
        env,
        train_tasks,
        known_options,
        train_tasks_start_idx=train_tasks_start_idx)
    logging.info(f"\n\nLOADED DATASET OF {len(loaded_trajectories)} "
                 "DEMONSTRATIONS")
    logging.info(f"CREATED {len(generated_trajectories)} DEMONSTRATIONS")
    dataset = Dataset(loaded_trajectories + generated_trajectories)
    with open(dataset_fname, "wb") as f:
        pkl.dump(dataset, f)
    return dataset


def _generate_demonstrations(
        env: BaseEnv, train_tasks: List[Task],
        known_options: Set[ParameterizedOption],
        train_tasks_start_idx: int) -> List[LowLevelTrajectory]:
    """Use the demonstrator to generate demonstrations, one per training task
    starting from train_tasks_start_idx."""
    if CFG.demonstrator == "oracle":
        oracle_approach = OracleApproach(
            env.predicates,
            env.options,
            env.types,
            env.action_space,
            train_tasks,
            task_planning_heuristic=CFG.offline_data_task_planning_heuristic,
            max_skeletons_optimized=CFG.offline_data_max_skeletons_optimized)
    else:  # pragma: no cover
        # Disable all built-in keyboard shortcuts.
        keymaps = {k for k in plt.rcParams if k.startswith("keymap.")}
        for k in keymaps:
            plt.rcParams[k].clear()
        # Create the environment-specific method for turning events into
        # actions. This should also log instructions.
        event_to_action = env.get_event_to_action_fn()
    trajectories = []
    num_tasks = min(len(train_tasks), CFG.max_initial_demos)
    for idx, task in enumerate(train_tasks):
        if idx < train_tasks_start_idx:  # ignore demos before this index
            continue
        # Note: we assume in main.py that demonstrations are only generated
        # for train tasks whose index is less than CFG.max_initial_demos. If
        # you modify code around here, make sure that this invariant holds.
        if idx >= CFG.max_initial_demos:
            break
        try:
            if CFG.demonstrator == "oracle":
                timeout = CFG.offline_data_planning_timeout
                if timeout == -1:
                    timeout = CFG.timeout
                oracle_approach.recompute_nsrts(env)
                oracle_approach.solve(task, timeout=timeout)
                # Since we're running the oracle approach, we know that
                # the policy is actually a plan under the hood, and we
                # can retrieve it with get_last_plan(). We do this
                # because we want to run the full plan.
                last_plan = oracle_approach.get_last_plan()
                policy = utils.option_plan_to_policy(last_plan)
                # We will stop run_policy() when OptionExecutionFailure()
                # is hit, which should only happen when the goal has been
                # reached, as verified by the assertion later.
                termination_function = lambda s: False
            else:  # pragma: no cover
                policy = functools.partial(_human_demonstrator_policy, env,
                                           idx, num_tasks, task,
                                           event_to_action)
                termination_function = task.goal_holds
            if CFG.env == "behavior":  # pragma: no cover
                # For BEHAVIOR we are generating the trajectory by running
                # our plan on our option models. Since option models
                # return only states, we will add dummy actions to the
                # states to create our low-level trajectories.
                last_traj = oracle_approach.get_last_traj()
                traj, success = _run_plan_with_option_model(
                    task, idx, oracle_approach.get_option_model(), last_plan,
                    last_traj)
                # Is successful if we found a low-level plan that achieves
                # our goal using option models.
                if not success:
                    raise ApproachFailure(
                        "Falied execution of low-level plan on option model")
            else:
                if CFG.make_demo_videos:
                    monitor = utils.VideoMonitor(env.render)
                else:
                    monitor = None
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
                    monitor=monitor)
        except (ApproachTimeout, ApproachFailure,
                utils.EnvironmentFailure) as e:
            logging.warning("WARNING: Approach failed to solve with error: "
                            f"{e}")
            continue
        # Check that the goal holds at the end. Print a warning if not.
        if not task.goal_holds(traj.states[-1]):  # pragma: no cover
            logging.warning("WARNING: Oracle failed on training task.")
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
        if CFG.demonstrator == "oracle":
            for act in traj.actions:
                if act.get_option().parent not in known_options:
                    assert CFG.option_learner != "no_learning"
                    act.unset_option()
        trajectories.append(traj)
        if CFG.make_demo_videos:
            assert monitor is not None
            video = monitor.get_video()
            outfile = f"{CFG.env}__{CFG.seed}__demo__task{idx}.mp4"
            utils.save_video(outfile, video)
    return trajectories


def _human_demonstrator_policy(env: BaseEnv, idx: int, num_tasks: int,
                               task: Task, event_to_action: Callable[
                                   [State, matplotlib.backend_bases.Event],
                                   Action],
                               state: State) -> Action:  # pragma: no cover
    # Temporarily change the backend to one that supports a GUI.
    # We do this here because we don't want the rest of the codebase
    # to use GUI-based Matplotlib.
    cur_backend = matplotlib.get_backend()
    matplotlib.use("Qt5Agg")
    # Render the state.
    caption = (f"Task {idx+1} / {num_tasks}\nPlease demonstrate "
               f"achieving the goal:\n{task.goal}")
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
