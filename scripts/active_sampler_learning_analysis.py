"""Script to analyze samplers learned through the active sampler learning
approach."""

import os
from typing import Any, List, Optional, Tuple

import dill as pkl
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators import utils
from predicators.envs import BaseEnv, create_new_env
from predicators.envs.cover import BumpyCoverEnv
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, Object, State, Video


def _main() -> None:
    """Loads the saved samplers."""
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    env = create_new_env(CFG.env, do_cache=True)
    # Set up the test cases.
    test_cases = _create_test_cases(env)
    # Set up videos.
    video_frames = []
    # Evaluate samplers before offline learning.
    imgs = _run_one_cycle_analysis(None, test_cases, env)
    video_frames.append(imgs)
    # Evaluate samplers for each learning cycle.
    online_learning_cycle = 1
    while True:
        try:
            imgs = _run_one_cycle_analysis(online_learning_cycle, test_cases,
                                           env)
            video_frames.append(imgs)
        except FileNotFoundError:
            break
        online_learning_cycle += 1
    # Save the videos.
    for i, video in enumerate(np.swapaxes(video_frames, 0, 1)):
        video_outfile = f"active_sampler_learning_case_{i}.mp4"
        utils.save_video(video_outfile, video)
        # Save the frames individually too.
        for t, img in enumerate(video):
            img_outfile = f"videos/active_sampler_learning_case_{i}_{t}.png"
            imageio.imsave(img_outfile, img)


def _create_test_cases(env: BaseEnv) -> List[Tuple[State, List[Object]]]:
    assert isinstance(env, BumpyCoverEnv)

    test_cases = []

    for block_poses, target_poses in [
        ([0.15, 0.605], [0.375, 0.815]),
        ([0.5, 0.2], [0.375, 0.815]),
        ([0.75, 0.1], [0.375, 0.815]),
        ([0.3, 0.8], [0.375, 0.815]),
    ]:
        env_task = env.get_test_tasks()[0]
        state = env_task.init.copy()
        block0 = [b for b in state if b.name == "block0"][0]
        block1 = [b for b in state if b.name == "block1"][0]

        # target0 = [b for b in state if b.name == "target0"][0]
        # target1 = [b for b in state if b.name == "target1"][0]
        # robot = [b for b in state if b.name == "robby"][0]
        # blocks = [block0, block1]
        # targets = [target0, target1]
        # assert len(blocks) == len(block_poses)
        # for block, pose in zip(blocks, block_poses):
        #     # [is_block, is_target, width, pose, grasp]
        #     state.set(block, "pose", pose)
        #     # Make sure blocks are not held
        #     state.set(block, "grasp", -1)
        # assert len(targets) == len(target_poses)
        # for target, pose in zip(targets, target_poses):
        #     # [is_block, is_target, width, pose]
        #     state.set(target, "pose", pose)
        # state.set(robot, "hand", 0.0)
        
        test_cases.append((state, [block0]))
        test_cases.append((state, [block1]))

    return test_cases


def _run_one_cycle_analysis(online_learning_cycle: Optional[int],
                            test_cases: List[Tuple[State, List[Object]]],
                            env: BaseEnv) -> Video:
    option_name = "Pick"
    approach_save_path = utils.get_approach_save_path_str()
    save_path = f"{approach_save_path}_{option_name}_" + \
        f"{online_learning_cycle}.sampler_regressor"
    if not os.path.exists(save_path):
        raise FileNotFoundError
    with open(save_path, "rb") as f:
        regressor = pkl.load(f)
    print(f"Loaded sampler regressor from {save_path}.")

    cmap = colormaps.get_cmap('RdYlGn')
    norm = Normalize(vmin=-1.0, vmax=0.0)

    imgs = []

    for state, objects in test_cases:
        assert len(objects) == 1
        obj = objects[0]
        dummy_task = EnvironmentTask(state, set())
        fig = env.render_state_plt(state, dummy_task)  # task ignored
        ax = fig.axes[0]

        # Construct flat state.
        x_lst: List[Any] = [1.0]  # start with bias term
        for obj in objects:
            x_lst.extend(state[obj])
        assert not CFG.sampler_learning_use_goals
        x = np.array(x_lst)

        # Evenly space along object.
        obj_pose = state.get(obj, "pose")
        obj_width = state.get(obj, "width")
        lo = obj_pose - obj_width / 2
        hi = obj_pose + obj_width / 2
        candidates = np.linspace(lo, hi, num=100)
        for candidate in candidates:
            score = regressor.predict(np.r_[x, [candidate]])[0]
            color = cmap(norm(score))
            circle = plt.Circle((candidate, -0.16),
                                0.005,
                                color=color,
                                alpha=0.1)
            ax.add_patch(circle)
        img = utils.fig2data(fig, dpi=150)
        imgs.append(img)

    return imgs


if __name__ == "__main__":
    _main()
