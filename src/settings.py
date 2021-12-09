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

    # cover_multistep_options parameters
    cover_multistep_action_limits = [-np.inf, np.inf]

    # cluttered table env parameters
    cluttered_table_num_cans_train = 5
    cluttered_table_num_cans_test = 10
    cluttered_table_can_radius = 0.01
    cluttered_table_collision_angle_thresh = np.pi / 4

    # repeated nextto env parameters
    repeated_nextto_num_dots = 25

    # painting env parameters
    painting_train_families = [
        "box_and_shelf",  # placing into both box and shelf
        # "box_only",  # just placing into the box
        # "shelf_only",  # just placing into the shelf
    ]

    # behavior env parameters
    behavior_config_file = os.path.join(  # relative to igibson.root_path
        "examples", "configs",
        "njk_re-shelving_library_books_full_obs.yaml",
        # "njk_sorting_books_full_obs.yaml"
    )
    behavior_mode = "headless"  # headless, pbgui, iggui
    behavior_action_timestep = 1.0 / 10.0
    behavior_physics_timestep = 1.0 / 120.0

    # parameters for approaches
    random_options_max_tries = 100

    # SeSamE parameters
    propagate_failures = True
    max_num_steps_option_rollout = 100
    max_skeletons_optimized = 8  # if 1, can only solve downward refinable tasks

    # evaluation parameters
    save_dir = "saved_data"
    video_dir = "videos"
    video_fps = 2

    # dataset parameters
    offline_data_planning_timeout = 500  # for learning-based approaches, the
                                         # data collection timeout for planning
    offline_data_num_replays = 500  # for learning-based approaches, the
                                    # number of replays used when the data
                                    # generation method is data+replays

    # teacher dataset parameters
    teacher_dataset_label_ratio = 1.0

    # NSRT learning parameters
    min_data_for_nsrt = 3

    # option learning parameters
    do_option_learning = False  # if False, uses ground truth options
    option_learner = "oracle"  # only used if do_option_learning is True

    # sampler learning parameters
    do_sampler_learning = True  # if False, uses random samplers
    max_rejection_sampling_tries = 100
    normalization_scale_clip = 1
    classifier_hid_sizes = [32, 32]
    classifier_max_itr_sampler = 10000
    classifier_max_itr_predicate = 1000
    classifier_balance_data = True
    regressor_hid_sizes = [32, 32]
    regressor_max_itr = 10000
    regressor_sample_clip = 1
    n_iter_no_change = 5000
    learning_rate = 1e-3

    # iterative invention parameters
    iterative_invention_accept_score = 1-1e-3

    # interactive learning parameters
    interactive_known_predicates = {'HandEmpty', 'Covers'}
    interactive_num_episodes = 3
    interactive_max_steps = 10
    interactive_relearn_every = 3
    interactive_num_babbles = 10
    interactive_max_num_atoms_babbled = 1
    interactive_num_tasks_babbled = 5
    interactive_atom_type_babbled = "ground"
    interactive_ask_strategy = "all_seen_states"
    interactive_ask_strategy_threshold = 0.0
    interactive_ask_strategy_pct = 20.0

    # grammar search invention parameters
    grammar_search_max_evals = 250
    grammar_search_direction = "smalltolarge"
    grammar_search_true_pos_weight = 10
    grammar_search_false_pos_weight = 1
    grammar_search_size_weight = 1e-2
    grammar_search_pred_complexity_weight = 1
    grammar_search_grammar_name = "forall_single_feat_ineqs"
    grammar_search_max_predicates = 50

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
                "cover_typed_options": 10,
                "cover_hierarchical_types": 10,
                "cover_multistep_options": 10,
                "cluttered_table": 50,
                "blocks": 50,
                "painting": 50,
                "repeated_nextto": 50,
                "playroom": 50,
                "behavior": 10,
            })[args["env"]],

            # Number of test tasks in each environment.
            num_test_tasks=defaultdict(int, {
                "cover": 10,
                "cover_typed_options": 10,
                "cover_hierarchical_types": 10,
                "cover_multistep_options": 10,
                "cluttered_table": 50,
                "blocks": 50,
                "painting": 50,
                "repeated_nextto": 50,
                "playroom": 50,
                "behavior": 10,
            })[args["env"]],

            # Maximum number of steps to run a policy when checking whether
            # it solves a task.
            max_num_steps_check_policy=defaultdict(int, {
                "cover": 10,
                "cover_typed_options": 10,
                "cover_hierarchical_types": 10,
                "cover_multistep_options": 100,
                "cluttered_table": 25,
                "blocks": 25,
                "painting": 100,
                "repeated_nextto": 10,
                "playroom": 25,
                "behavior": 100,
            })[args["env"]],

            # Name of the option model to use.
            option_model_name=defaultdict(str, {
                "cover": "default",
                "cover_typed_options": "default",
                "cover_hierarchical_types": "default",
                "cover_multistep_options": "default",
                "cluttered_table": "default",
                "blocks": "default",
                "painting": "default",
                "repeated_nextto": "default",
            })[args["env"]],

            max_samples_per_step=defaultdict(int, {
                "cover": 10,
                "cover_typed_options": 10,
                "cover_hierarchical_types": 10,
                "cover_multistep_options": 10,
                "cluttered_table": 10,
                "blocks": 10,
                "painting": 1,
                "repeated_nextto": 10,
                "playroom": 10,
                "behavior": 10,
            })[args["env"]],

            # For learning-based approaches, the data collection strategy.
            offline_data_method=defaultdict(str, {
                "nsrt_learning": "demo+replay",
                "interactive_learning": "demo",
                "iterative_invention": "demo+replay",
                "grammar_search_invention": "demo+replay",
            })[args["approach"]],
        )


def get_save_path() -> str:
    """Create a path for this experiment that can be used to save
    and load results.
    """
    if not os.path.exists(CFG.save_dir):
        os.makedirs(CFG.save_dir)
    return (f"{CFG.save_dir}/{CFG.env}___{CFG.approach}___{CFG.seed}___"
            f"{CFG.excluded_predicates}.saved")


_attr_to_value = {}
for _attr, _value in GlobalSettings.__dict__.items():
    if _attr.startswith("_"):
        continue
    assert _attr not in _attr_to_value  # duplicate attributes
    _attr_to_value[_attr] = _value
CFG = SimpleNamespace(**_attr_to_value)
