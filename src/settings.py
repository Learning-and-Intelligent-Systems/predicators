"""Contains global, immutable settings.

Anything that varies between runs should be a command-line arg
(args.py).
"""

import os
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, Any
import numpy as np


class GlobalSettings:
    """Unchanging settings."""
    # parameters for all envs
    num_train_tasks = 15
    num_test_tasks = 50
    max_num_steps_check_policy = 100  # maximum number of steps to run a policy
    # when checking whether it solves a task

    # cover env parameters
    cover_num_blocks = 2
    cover_num_targets = 2
    cover_block_widths = [0.1, 0.07]
    cover_target_widths = [0.05, 0.03]

    # cover_multistep_options parameters
    cover_multistep_action_limits = [-np.inf, np.inf]
    cover_multistep_use_learned_equivalents = True

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
        "examples",
        "configs",
        "njk_re-shelving_library_books_full_obs.yaml",
        # "njk_sorting_books_full_obs.yaml"
    )
    behavior_mode = "headless"  # headless, pbgui, iggui
    behavior_action_timestep = 1.0 / 10.0
    behavior_physics_timestep = 1.0 / 120.0

    # parameters for approaches
    random_options_max_tries = 100

    # SeSamE parameters
    option_model_name = "default"
    max_num_steps_option_rollout = 1000
    max_skeletons_optimized = 8  # if 1, can only solve downward refinable tasks
    max_samples_per_step = 10  # max effort on sampling a single skeleton

    # evaluation parameters
    results_dir = "results"
    save_dir = "saved_data"
    video_dir = "videos"
    video_fps = 2

    # dataset parameters
    offline_data_planning_timeout = 500  # for learning-based approaches, the
    # data collection timeout for planning

    # teacher dataset parameters
    teacher_dataset_label_ratio = 1.0

    # NSRT learning parameters
    min_data_for_nsrt = 3
    learn_side_predicates = False

    # torch model parameters
    normalization_scale_clip = 1
    learning_rate = 1e-3
    mlp_regressor_max_itr = 10000
    mlp_regressor_hid_sizes = [32, 32]
    mlp_regressor_clip_gradients = False
    mlp_regressor_gradient_clip_value = 5
    mlp_classifier_hid_sizes = [32, 32]
    mlp_classifier_balance_data = True
    neural_gaus_regressor_hid_sizes = [32, 32]
    neural_gaus_regressor_max_itr = 10000
    neural_gaus_regressor_sample_clip = 1
    mlp_classifier_n_iter_no_change = 5000

    # option learning parameters
    option_learner = "no_learning"  # "no_learning" or "oracle" or "simple"
    mlp_regressor_max_itr = 10000
    mlp_regressor_hid_sizes = [32, 32]
    mlp_regressor_clip_gradients = False
    mlp_regressor_gradient_clip_value = 5

    # sampler learning parameters
    sampler_learner = "neural"  # "neural" or "random" or "oracle"
    max_rejection_sampling_tries = 100
    sampler_mlp_classifier_max_itr = 10000

    # iterative invention parameters
    iterative_invention_accept_score = 1 - 1e-3
    predicate_mlp_classifier_max_itr = 1000

    # interactive learning parameters
    interactive_known_predicates = {"HandEmpty", "Covers"}
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
    grammar_search_grammar_includes_givens = True
    grammar_search_grammar_includes_foralls = True
    grammar_search_true_pos_weight = 10
    grammar_search_false_pos_weight = 1
    grammar_search_bf_weight = 1
    grammar_search_size_weight = 1e-2
    grammar_search_pred_complexity_weight = 1
    grammar_search_max_predicates = 50
    grammar_search_predicate_cost_upper_bound = 6
    grammar_search_score_function = "hff_lookahead_depth0"
    grammar_search_heuristic_based_weight = 10.
    grammar_search_heuristic_based_max_demos = 5
    grammar_search_lookahead_based_temperature = 10.
    grammar_search_task_planning_timeout = 1.0
    grammar_search_hill_climbing_depth = 0

    @staticmethod
    def get_arg_specific_settings(args: Dict[str, Any]) -> Dict[str, Any]:
        """A workaround for global settings that are derived from the
        experiment-specific args."""
        if "env" not in args:
            args["env"] = ""
        if "approach" not in args:
            args["approach"] = ""
        return dict(
            # Task planning heuristic to use in SeSamE.
            task_planning_heuristic=defaultdict(
                # Use HAdd by default.
                lambda: "hadd",
                {
                    # In the playroom domain, HFF works better.
                    "playroom": "hff",
                })[args["env"]],

            # In SeSamE, when to propagate failures back up to the high level
            # search. Choices are: {"after_exhaust", "immediately", "never"}.
            sesame_propagate_failures=defaultdict(
                # Use "immediately" by default.
                lambda: "immediately",
                {
                    # We use a different strategy for cluttered_table because
                    # of the high likelihood of getting cyclic failures if you
                    # immediately raise failures, leading to unsolvable tasks.
                    "cluttered_table": "after_exhaust",
                })[args["env"]],

            # For learning-based approaches, the data collection strategy.
            offline_data_method=defaultdict(
                # Use both demonstrations and random replays by default.
                lambda: "demo+replay",
                {
                    # No replays for active learning project.
                    "interactive_learning": "demo",
                })[args["approach"]],

            # Number of replays used when offline_data_method is demo+replay.
            offline_data_num_replays=defaultdict(
                # Default number of random replays.
                lambda: 500,
                {
                    # For the repeated_nextto environment, too many
                    # replays makes learning slow.
                    "repeated_nextto": 50,
                })[args["env"]],
        )


_attr_to_value = {}
for _attr, _value in GlobalSettings.__dict__.items():
    if _attr.startswith("_"):
        continue
    assert _attr not in _attr_to_value  # duplicate attributes
    _attr_to_value[_attr] = _value
CFG = SimpleNamespace(**_attr_to_value)
