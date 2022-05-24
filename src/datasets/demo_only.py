"""Create offline datasets by collecting demonstrations."""

from bdb import set_trace
import imp
import functools
import logging
from typing import Callable, List, Set

import matplotlib
import matplotlib.pyplot as plt

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.approaches.oracle_approach import OracleApproach
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Dataset, LowLevelTrajectory, \
    ParameterizedOption, Task, State, Action
from predicators.src.planning import _run_low_level_plan


def create_demo_data(env: BaseEnv, train_tasks: List[Task],
                     known_options: Set[ParameterizedOption]) -> Dataset:
    """Create offline datasets by collecting demos."""
    assert CFG.demonstrator in ("oracle", "human")
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
        # Note: we assume in main.py that demonstrations are only generated
        # for train tasks whose index is less than CFG.max_initial_demos. If
        # you modify code around here, make sure that this invariant holds.
        if idx >= CFG.max_initial_demos:
            break
        try:
            # ## TODO Uncomment after debugging simulator
            # if CFG.demonstrator == "oracle":
            #     oracle_approach.solve(
            #         task, timeout=CFG.offline_data_planning_timeout)
            #     # Since we're running the oracle approach, we know that the
            #     # policy is actually a plan under the hood, and we can
            #     # retrieve it with get_last_plan(). We do this because we want
            #     # to run the full plan.
            #     last_plan = oracle_approach.get_last_plan()
            #     policy = utils.option_plan_to_policy(last_plan)
            #     # We will stop run_policy() when OptionExecutionFailure() is
            #     # hit, which should only happen when the goal has been
            #     # reached, as verified by the assertion later.
            #     termination_function = lambda s: False
            # else:  # pragma: no cover
            #     policy = functools.partial(_human_demonstrator_policy, env,
            #                                idx, num_tasks, task,
            #                                event_to_action)
            #     termination_function = task.goal_holds

            # import ipdb; ipdb.set_trace()
            #  
            import dill as pickle
            file = open('plan.pkl', 'rb')
            pickled_plan = pickle.load(file)
            file.close()
            plan = []
            for i in range(len(pickled_plan)):
                curr_option = None
                for option in env.options:
                    if option.name == pickled_plan[i][0]:
                        curr_option = option
                plan.append(curr_option.ground(pickled_plan[i][1], pickled_plan[i][2]))
            #

            traj, suc = _run_low_level_plan(
                    task, oracle_approach._option_model, plan, oracle_approach._seed,
                    CFG.offline_data_planning_timeout, CFG.horizon)
            assert suc
            #

            if CFG.make_demo_videos:
                monitor = utils.VideoMonitor(env.render)
            else:
                monitor = None
            # traj, _ = utils.run_policy(
            #     utils.option_plan_to_policy(plan),
            #     env,
            #     "train",
            #     idx,
            #     termination_function=lambda s: False,
            #     max_num_steps=CFG.horizon,
            #     exceptions_to_break_on={utils.OptionExecutionFailure},
            #     monitor=monitor)
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
        
    return Dataset(trajectories)


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
    assert "action" in container, "Event handler failed. Its " \
        "error message should be printed above."
    # Revert to the previous backend.
    matplotlib.use(cur_backend)
    return container["action"]
