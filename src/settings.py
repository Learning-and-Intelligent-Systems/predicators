"""Contains global, immutable settings.
Anything that varies between runs should be a command-line arg (args.py).
"""

from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, Any


class GlobalSettings:
    """Unchanging settings.
    """
    # cover env parameters
    cover_num_blocks = 2
    cover_num_targets = 2
    cover_block_widths = [0.05, 0.03]
    cover_target_widths = [0.1, 0.07]

    # SeSamE parameters
    max_samples_per_step = 10
    max_num_steps_option_rollout = 100
    max_skeletons = 1  # if 1, can only solve downward refinable tasks

    # evaluation parameters
    video_dir = "videos"
    video_fps = 2

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
            })[args["env"]],

            # Number of test tasks in each environment.
            num_test_tasks=defaultdict(int, {
                "cover": 10,
            })[args["env"]],

            # Maximum number of steps to run a policy when checking whether
            # it solves a task.
            max_num_steps_check_policy=defaultdict(int, {
                "cover": 10,
            })[args["env"]],

            # For learning-based approaches, whether to include ground truth
            # options in the offline dataset.
            include_options_in_offline_data=defaultdict(bool, {
                "trivial_learning": True,
                "operator_learning": True,
            })[args["approach"]],

            # For learning-based approaches, the data collection strategy.
            offline_data_method=defaultdict(str, {
                "trivial_learning": "demo",
                "operator_learning": "demo+replay",
            })[args["approach"]],

            # For learning-based approaches, the data collection timeout
            # used for planning.
            offline_data_planning_timeout=defaultdict(int, {
                "trivial_learning": 500,
                "operator_learning": 500,
            })[args["approach"]],

            # For learning-based approaches, the number of replays used
            # when the data generation method is data+replays.
            offline_data_num_replays=defaultdict(int, {
                "trivial_learning": 10,
                "operator_learning": 10,
            })[args["approach"]],
        )

_attr_to_value = {}
for _attr, _value in GlobalSettings.__dict__.items():
    if _attr.startswith("_"):
        continue
    assert _attr not in _attr_to_value  # duplicate attributes
    _attr_to_value[_attr] = _value
CFG = SimpleNamespace(**_attr_to_value)
