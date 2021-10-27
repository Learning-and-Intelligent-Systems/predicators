"""Contains global, immutable settings.
Anything that varies between runs should be a command-line arg (args.py).
"""

import os
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, Any
import numpy as np


class GlobalSettings:
    """Unchanging settings.
    """
    # cover env parameters
    cover_num_blocks = 2
    cover_num_targets = 2
    cover_block_widths = [0.1, 0.07]
    cover_target_widths = [0.05, 0.03]

    # cluttered table env parameters
    cluttered_table_num_cans_train = 5
    cluttered_table_num_cans_test = 10
    cluttered_table_can_radius = 0.01
    cluttered_table_collision_angle_thresh = np.pi / 4

    # blocks env parameters
    blocks_num_blocks_train = [3, 4]
    blocks_num_blocks_test = [5, 6]
    blocks_block_size = 0.1

    # parameters for approaches
    random_options_max_tries = 100

    # SeSamE parameters
    propagate_failures = True
    max_samples_per_step = 10
    max_num_steps_option_rollout = 100
    max_skeletons_optimized = 8  # if 1, can only solve downward refinable tasks

    # evaluation parameters
    save_dir = "saved_data"
    video_dir = "videos"
    video_fps = 2

    # teacher dataset parameters
    teacher_dataset_label_ratio = 0.3

    # operator learning parameters
    min_data_for_operator = 3
    max_rejection_sampling_tries = 100

    # sampler learning parameters
    do_sampler_learning = True
    normalization_scale_clip = 1
    classifier_hid_sizes = [32, 32]
    classifier_max_itr = 10000
    classifier_balance_data = True
    regressor_hid_sizes = [32, 32]
    regressor_max_itr = 10000
    regressor_sample_clip = 1
    n_iter_no_change = 5000
    learning_rate = 1e-3

    # interactive learning parameters
    interactive_known_predicates = {'HandEmpty'}

    @staticmethod
    def get_arg_specific_settings(args: Dict[str, Any]) -> Dict[str, Any]:
        """A workaround for global settings that are
        derived from the experiment-specific args
        """
        if "env" not in args:
            args["env"] = ""
        if "approach" not in args:
            args["approach"] = ""
        return dict(
            # Number of training tasks in each environment.
            num_train_tasks=defaultdict(int, {
                "cover": 10,
                "cover_typed": 10,
                "cover_augmented": 10,
                "cluttered_table": 50,
                "blocks": 50,
            })[args["env"]],

            # Number of test tasks in each environment.
            num_test_tasks=defaultdict(int, {
                "cover": 10,
                "cover_typed": 10,
                "cover_augmented": 10,
                "cluttered_table": 50,
                "blocks": 50,
            })[args["env"]],

            # Maximum number of steps to run a policy when checking whether
            # it solves a task.
            max_num_steps_check_policy=defaultdict(int, {
                "cover": 10,
                "cover_typed": 10,
                "cover_augmented": 10,
                "cluttered_table": 25,
                "blocks": 25,
            })[args["env"]],

            # For learning-based approaches, whether to include ground truth
            # options in the offline dataset.
            include_options_in_offline_data=defaultdict(bool, {
                "trivial_learning": True,
                "operator_learning": True,
                "interactive_learning": True,
            })[args["approach"]],

            # For learning-based approaches, the data collection strategy.
            offline_data_method=defaultdict(str, {
                "trivial_learning": "demo",
                "operator_learning": "demo+replay",
                "interactive_learning": "demo",
            })[args["approach"]],

            # For learning-based approaches, the data collection timeout
            # used for planning.
            offline_data_planning_timeout=defaultdict(int, {
                "trivial_learning": 500,
                "operator_learning": 500,
                "interactive_learning": 500,
            })[args["approach"]],

            # For learning-based approaches, the number of replays used
            # when the data generation method is data+replays.
            offline_data_num_replays=defaultdict(int, {
                "trivial_learning": 10,
                "operator_learning": 10,
                "interactive_learning": 10,
            })[args["approach"]],
        )


def get_save_path() -> str:
    """Create a path for this experiment that can be used to save
    and load results.
    """
    if not os.path.exists(CFG.save_dir):
        os.makedirs(CFG.save_dir)
    return f"{CFG.save_dir}/{CFG.env}___{CFG.approach}___{CFG.seed}.saved"


_attr_to_value = {}
for _attr, _value in GlobalSettings.__dict__.items():
    if _attr.startswith("_"):
        continue
    assert _attr not in _attr_to_value  # duplicate attributes
    _attr_to_value[_attr] = _value
CFG = SimpleNamespace(**_attr_to_value)
