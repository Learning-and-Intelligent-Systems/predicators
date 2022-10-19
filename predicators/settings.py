"""Contains global, immutable settings.

Anything that varies between runs should be a command-line arg
(args.py).
"""

import os
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, Set

import numpy as np


class GlobalSettings:
    """Unchanging settings."""
    # global parameters
    num_train_tasks = 50
    num_test_tasks = 50
    # Perform online learning for this many cycles or until this many
    # transitions have been collected, whichever happens first.
    num_online_learning_cycles = 10
    online_learning_max_transitions = float("inf")
    # Maximum number of training tasks to give a demonstration for, if the
    # offline_data_method is demo-based.
    max_initial_demos = float("inf")
    # If this is False, then environment interactions can only take place
    # on tasks that have no demonstrations.
    allow_interaction_in_demo_tasks = True
    # Maximum number of steps to run an InteractionRequest policy.
    max_num_steps_interaction_request = 100
    # Whether to pretty print predicates and NSRTs when NSRTs are loaded.
    pretty_print_when_loading = False
    # Used for random seeding in test environment.
    test_env_seed_offset = 10000
    # The method to use for segmentation. By default, segment using options.
    # If you are learning options, you should change this via the command line.
    segmenter = "option_changes"
    # The method to use for generating demonstrations: "oracle" or "human".
    demonstrator = "oracle"
    # DPI for rendering the state. Increase this if video quality is poor.
    # Note that for unit testing, we use a much smaller value by default,
    # which is set in utils.reset_config(). If you want higher-quality videos
    # in unit tests, make sure to pass in a value for `render_state_dpi` into
    # your call to utils.reset_config().
    render_state_dpi = 150
    # If this is True, then we will not execute our final plan in simulation
    # and we will say a task is solved successfully if we were able to find
    # a plan. Note, we might solve the task even if the plan we found would
    # not necessarily work in simulation (this is especially true of option
    # models that don't model the simulator well). This can only be True if
    # the approach is a subclass of BilevelPlanningApproach.
    plan_only_eval = False

    # cover env parameters
    cover_num_blocks = 2
    cover_num_targets = 2
    cover_block_widths = [0.1, 0.07]
    cover_target_widths = [0.05, 0.03]
    cover_initial_holding_prob = 0.75

    # cover_multistep_options env parameters
    cover_multistep_action_limits = [-np.inf, np.inf]
    cover_multistep_degenerate_oracle_samplers = False
    cover_multistep_max_tb_placements = 100  # max placements of targets/blocks
    cover_multistep_max_hr_placements = 100  # max placements of hand regions
    cover_multistep_thr_percent = 0.4  # target hand region percent of width
    cover_multistep_bhr_percent = 0.4  # block hand region percent of width
    cover_multistep_bimodal_goal = False
    cover_multistep_goal_conditioned_sampling = False  # assumes one goal

    # blocks env parameters
    blocks_num_blocks_train = [3, 4]
    blocks_num_blocks_test = [5, 6]
    blocks_test_task_json_dir = None

    # playroom env parameters
    playroom_num_blocks_train = [3]
    playroom_num_blocks_test = [3]

    # cluttered table env parameters
    cluttered_table_num_cans_train = 5
    cluttered_table_num_cans_test = 10
    cluttered_table_can_radius = 0.01
    cluttered_table_collision_angle_thresh = np.pi / 4
    cluttered_table_place_goal_conditioned_sampling = True

    # repeated nextto env parameters
    repeated_nextto_num_dots = 15

    # painting env parameters
    painting_initial_holding_prob = 0.5
    painting_lid_open_prob = 0.3
    painting_num_objs_train = [2, 3]
    painting_num_objs_test = [3, 4]
    painting_max_objs_in_goal = float("inf")
    painting_goal_receptacles = "box_and_shelf"  # box_and_shelf, box, shelf

    # repeated_nextto_painting (rnt_painting) env parameters
    rnt_painting_num_objs_train = [8, 9, 10]
    rnt_painting_num_objs_test = [11, 12, 13]
    rnt_painting_max_objs_in_goal = 2

    # tools env parameters
    tools_num_items_train = [2]
    tools_num_items_test = [2, 3]
    tools_num_contraptions_train = [2]
    tools_num_contraptions_test = [3]

    # BEHAVIOR env parameters
    behavior_config_file = os.path.join(  # relative to igibson.root_path
        "examples",
        "configs",
        "wbm3_modifiable_full_obs.yaml",
    )
    behavior_mode = "headless"  # headless, pbgui, iggui
    behavior_action_timestep = 1.0 / 10.0
    behavior_physics_timestep = 1.0 / 120.0
    behavior_task_list = ["re-shelving_library_books"]
    behavior_scene_name = "Pomaria_1_int"
    behavior_randomize_init_state = True
    behavior_option_model_eval = False
    behavior_option_model_rrt = False

    # general pybullet parameters
    pybullet_use_gui = False  # must be True to make videos
    pybullet_draw_debug = False  # useful for annotating in the GUI
    pybullet_camera_width = 335  # for high quality, use 1674
    pybullet_camera_height = 180  # for high quality, use 900
    pybullet_sim_steps_per_action = 20
    pybullet_max_ik_iters = 100
    pybullet_ik_tol = 1e-3
    pybullet_robot = "fetch"
    pybullet_birrt_num_attempts = 10
    pybullet_birrt_num_iters = 100
    pybullet_birrt_smooth_amt = 50
    pybullet_birrt_extend_num_interp = 10
    pybullet_control_mode = "position"

    # IKFast parameters
    ikfast_max_time = 0.05
    ikfast_max_candidates = 100
    ikfast_max_attempts = np.inf
    ikfast_max_distance = np.inf
    ikfast_norm = np.inf  # norm ord for np.linalg.norm

    # pddl blocks env parameters
    pddl_blocks_procedural_train_min_num_blocks = 3
    pddl_blocks_procedural_train_max_num_blocks = 4
    pddl_blocks_procedural_train_min_num_blocks_goal = 2
    pddl_blocks_procedural_train_max_num_blocks_goal = 3
    pddl_blocks_procedural_test_min_num_blocks = 5
    pddl_blocks_procedural_test_max_num_blocks = 6
    pddl_blocks_procedural_test_min_num_blocks_goal = 2
    pddl_blocks_procedural_test_max_num_blocks_goal = 5
    pddl_blocks_procedural_new_pile_prob = 0.5
    pddl_blocks_fixed_train_indices = list(range(1, 6))
    pddl_blocks_fixed_test_indices = list(range(6, 11))

    # pddl delivery env parameters
    pddl_delivery_procedural_train_min_num_locs = 5
    pddl_delivery_procedural_train_max_num_locs = 10
    pddl_delivery_procedural_train_min_want_locs = 2
    pddl_delivery_procedural_train_max_want_locs = 4
    pddl_delivery_procedural_train_min_extra_newspapers = 0
    pddl_delivery_procedural_train_max_extra_newspapers = 1
    pddl_delivery_procedural_test_min_num_locs = 31
    pddl_delivery_procedural_test_max_num_locs = 40
    pddl_delivery_procedural_test_min_want_locs = 20
    pddl_delivery_procedural_test_max_want_locs = 30
    pddl_delivery_procedural_test_min_extra_newspapers = 0
    pddl_delivery_procedural_test_max_extra_newspapers = 10
    pddl_easy_delivery_procedural_train_min_num_locs = 3
    pddl_easy_delivery_procedural_train_max_num_locs = 5
    pddl_easy_delivery_procedural_train_min_want_locs = 1
    pddl_easy_delivery_procedural_train_max_want_locs = 2
    pddl_easy_delivery_procedural_train_min_extra_newspapers = 0
    pddl_easy_delivery_procedural_train_max_extra_newspapers = 1
    pddl_easy_delivery_procedural_test_min_num_locs = 4
    pddl_easy_delivery_procedural_test_max_num_locs = 6
    pddl_easy_delivery_procedural_test_min_want_locs = 2
    pddl_easy_delivery_procedural_test_max_want_locs = 3
    pddl_easy_delivery_procedural_test_min_extra_newspapers = 0
    pddl_easy_delivery_procedural_test_max_extra_newspapers = 1

    # pddl spanner env parameters
    pddl_spanner_procedural_train_min_nuts = 1
    pddl_spanner_procedural_train_max_nuts = 3
    pddl_spanner_procedural_train_min_extra_spanners = 0
    pddl_spanner_procedural_train_max_extra_spanners = 2
    pddl_spanner_procedural_train_min_locs = 2
    pddl_spanner_procedural_train_max_locs = 4
    pddl_spanner_procedural_test_min_nuts = 10
    pddl_spanner_procedural_test_max_nuts = 20
    pddl_spanner_procedural_test_min_extra_spanners = 0
    pddl_spanner_procedural_test_max_extra_spanners = 10
    pddl_spanner_procedural_test_min_locs = 20
    pddl_spanner_procedural_test_max_locs = 30

    # pddl forest env parameters
    pddl_forest_procedural_train_min_size = 8
    pddl_forest_procedural_train_max_size = 10
    pddl_forest_procedural_test_min_size = 10
    pddl_forest_procedural_test_max_size = 12

    # stick button env parameters
    stick_button_num_buttons_train = [1, 2]
    stick_button_num_buttons_test = [3, 4]
    stick_button_disable_angles = True
    stick_button_holder_scale = 0.1

    # screws env parameters
    screws_num_screws_train = [15, 20]
    screws_num_screws_test = [25, 30]

    # doors env parameters
    doors_room_map_size = 5
    doors_min_obstacles_per_room = 0
    doors_max_obstacles_per_room = 3
    doors_min_room_exists_frac = 0.25
    doors_max_room_exists_frac = 0.75
    doors_birrt_num_attempts = 10
    doors_birrt_num_iters = 100
    doors_birrt_smooth_amt = 50
    doors_draw_debug = False

    # coffee env parameters
    coffee_num_cups_train = [1, 2]
    coffee_num_cups_test = [2, 3]
    coffee_jug_init_rot_amt = 2 * np.pi / 3

    # satellites env parameters
    satellites_num_sat_train = [2, 3]
    satellites_num_obj_train = [3, 4]
    satellites_num_sat_test = [3, 4]
    satellites_num_obj_test = [4, 5]

    # parameters for random options approach
    random_options_max_tries = 100

    # parameters for abstract GNN approach
    gnn_num_message_passing = 3
    gnn_layer_size = 16
    gnn_learning_rate = 1e-3
    gnn_num_epochs = 25000
    gnn_batch_size = 128
    gnn_do_normalization = False  # performs worse in Cover when True
    gnn_use_validation_set = True

    # parameters for GNN option policy approach
    gnn_option_policy_solve_with_shooting = True
    gnn_option_policy_shooting_variance = 0.1
    gnn_option_policy_shooting_max_samples = 100

    # parameters for metacontroller approaches
    metacontroller_max_samples = 100

    # parameters for PG3 approach
    pg3_heuristic = "policy_guided"
    pg3_search_method = "hill_climbing"
    pg3_task_planning_heuristic = "lmcut"
    pg3_gbfs_max_expansions = 100
    pg3_hc_enforced_depth = 0
    pg3_max_policy_guided_rollout = 50
    pg3_plan_compare_inapplicable_cost = 0.99

    # parameters for NSRT reinforcement learning approach
    nsrt_rl_reward_epsilon = 1e-2  # reward if in epsilon-ball from subgoal
    nsrt_rl_pos_reward = 0
    nsrt_rl_neg_reward = -1
    nsrt_rl_option_learner = "dummy_rl"
    nsrt_rl_valid_reward_steps_threshold = 10

    # parameters for large language models
    llm_prompt_cache_dir = "llm_cache"
    llm_openai_max_response_tokens = 700
    llm_use_cache_only = False
    llm_model_name = "text-curie-001"  # "text-davinci-002"
    llm_temperature = 0.5
    llm_num_completions = 1

    # SeSamE parameters
    sesame_task_planner = "astar"  # "astar" or "fdopt" or "fdsat"
    sesame_task_planning_heuristic = "lmcut"
    sesame_allow_noops = True  # recommended to keep this False if using replays
    sesame_check_expected_atoms = True
    sesame_use_visited_state_set = False
    # The algorithm used for grounding the planning problem. Choices are
    # "naive" or "fd_translator". The former does a type-aware cross product
    # of operators and objects to obtain ground operators, while the latter
    # calls Fast Downward's translator to produce an SAS task, then extracts
    # the ground operators from that. The latter is preferable when grounding
    # is a bottleneck in your environment, but will not work when operators
    # with no effects need to be part of the ground planning problem, like the
    # OpenLid() operator in painting. So, we'll keep the former as the
    # default.
    sesame_grounder = "naive"

    # evaluation parameters
    log_dir = "logs"
    results_dir = "results"
    eval_trajectories_dir = "eval_trajectories"
    approach_dir = "saved_approaches"
    data_dir = "saved_datasets"
    video_dir = "videos"
    video_fps = 2
    failure_video_mode = "longest_only"

    # dataset parameters
    # For learning-based approaches, the data collection timeout for planning.
    # If -1, defaults to CFG.timeout.
    offline_data_planning_timeout = -1
    # If "default", defaults to CFG.task_planning_heuristic.
    offline_data_task_planning_heuristic = "default"
    # If -1, defaults to CFG.sesame_max_skeletons_optimized.
    offline_data_max_skeletons_optimized = -1
    # Number of replays used when offline_data_method is replay-based.
    offline_data_num_replays = 500

    # teacher dataset parameters
    # Number of positive examples and negative examples per predicate.
    teacher_dataset_num_examples = 1

    # NSRT learning parameters
    min_data_for_nsrt = 0
    min_perc_data_for_nsrt = 0
    data_orderings_to_search = 1  # NSRT learning data ordering parameters
    # STRIPS learning algorithm. See get_name() functions in the directory
    # nsrt_learning/strips_learning/ for valid settings.
    strips_learner = "cluster_and_intersect"
    disable_harmlessness_check = True  # some methods may want this to be False
    enable_harmless_op_pruning = False  # some methods may want this to be True
    backchaining_check_intermediate_harmlessness = False
    compute_sidelining_objective_value = False
    clustering_learner_true_pos_weight = 10
    clustering_learner_false_pos_weight = 1
    cluster_and_intersect_prederror_max_groundings = 10
    cluster_and_search_inner_search_max_expansions = 2500
    cluster_and_search_inner_search_timeout = 30
    cluster_and_search_score_func_max_groundings = 10000
    cluster_and_search_var_count_weight = 0.1
    cluster_and_search_precon_size_weight = 0.01

    # torch model parameters
    learning_rate = 1e-3
    mlp_regressor_max_itr = 10000
    mlp_regressor_hid_sizes = [32, 32]
    mlp_regressor_clip_gradients = False
    mlp_regressor_gradient_clip_value = 5
    mlp_classifier_hid_sizes = [32, 32]
    mlp_classifier_balance_data = True
    neural_gaus_regressor_hid_sizes = [32, 32]
    neural_gaus_regressor_max_itr = 1000
    mlp_classifier_n_iter_no_change = 5000
    implicit_mlp_regressor_max_itr = 10000
    implicit_mlp_regressor_num_negative_data_per_input = 5
    implicit_mlp_regressor_num_samples_per_inference = 100
    implicit_mlp_regressor_temperature = 1.0
    implicit_mlp_regressor_inference_method = "derivative_free"
    implicit_mlp_regressor_derivative_free_num_iters = 3
    implicit_mlp_regressor_derivative_free_sigma_init = 0.33
    implicit_mlp_regressor_derivative_free_shrink_scale = 0.5
    implicit_mlp_regressor_grid_num_ticks_per_dim = 100

    # sampler learning parameters
    sampler_learner = "neural"  # "neural" or "random" or "oracle"
    max_rejection_sampling_tries = 100
    sampler_mlp_classifier_max_itr = 10000
    sampler_mlp_classifier_n_reinitialize_tries = 1
    sampler_learning_use_goals = False
    sampler_disable_classifier = False
    sampler_learning_regressor_model = "neural_gaussian"
    sampler_learning_max_negative_data = 100000

    # option learning parameters
    option_learning_action_converter = "identity"

    # interactive learning parameters
    interactive_num_ensemble_members = 10
    interactive_query_policy = "threshold"
    interactive_score_function = "entropy"
    interactive_score_threshold = 0.05
    interactive_random_query_prob = 0.5  # for query policy random
    interactive_num_requests_per_cycle = 10
    predicate_classifier_model = "mlp"  # "mlp" or "knn"
    predicate_mlp_classifier_max_itr = 100000
    predicate_mlp_classifier_n_reinitialize_tries = 1
    predicate_mlp_classifier_init = "default"  # or "normal"
    predicate_knn_classifier_n_neighbors = 1

    # online NSRT learning parameters
    online_nsrt_learning_requests_per_cycle = 10

    # glib explorer parameters
    glib_min_goal_size = 1
    glib_max_goal_size = 1
    glib_num_babbles = 10

    # greedy lookahead explorer parameters
    greedy_lookahead_max_num_trajectories = 100
    greedy_lookahead_max_traj_length = 2

    # grammar search invention parameters
    grammar_search_grammar_includes_givens = True
    grammar_search_grammar_includes_foralls = True
    grammar_search_use_handcoded_debug_grammar = False
    grammar_search_true_pos_weight = 10
    grammar_search_false_pos_weight = 1
    grammar_search_bf_weight = 1
    grammar_search_operator_complexity_weight = 0.0
    grammar_search_pred_complexity_weight = 1e-4
    grammar_search_max_predicates = 200
    grammar_search_predicate_cost_upper_bound = 6
    grammar_search_score_function = "expected_nodes_created"
    grammar_search_heuristic_based_weight = 10.
    grammar_search_max_demos = float("inf")
    grammar_search_max_nondemos = 50
    grammar_search_energy_based_temperature = 10.
    grammar_search_task_planning_timeout = 1.0
    grammar_search_search_algorithm = "hill_climbing"  # hill_climbing or gbfs
    grammar_search_hill_climbing_depth = 0
    grammar_search_parallelize_hill_climbing = False
    grammar_search_gbfs_num_evals = 1000
    grammar_search_off_demo_count_penalty = 1.0
    grammar_search_on_demo_count_penalty = 10.0
    grammar_search_suspicious_state_penalty = 10.0
    grammar_search_expected_nodes_upper_bound = 1e5
    grammar_search_expected_nodes_optimal_demo_prob = 1 - 1e-5
    grammar_search_expected_nodes_backtracking_cost = 1e3
    grammar_search_expected_nodes_allow_noops = True
    grammar_search_classifier_pretty_str_names = ["?x", "?y", "?z"]

    @staticmethod
    def get_arg_specific_settings(args: Dict[str, Any]) -> Dict[str, Any]:
        """A workaround for global settings that are derived from the
        experiment-specific args."""

        return dict(
            # Horizon for each environment. When checking if a policy solves a
            # task, we run the policy for at most this many steps.
            horizon=defaultdict(
                lambda: 100,
                {
                    # For certain environments, actions are lower level, so
                    # tasks take more actions to complete.
                    "behavior": 5000,
                    "pybullet_cover": 1000,
                    "pybullet_blocks": 1000,
                    "doors": 1000,
                    "coffee": 1000,
                    # For the very simple touch point environment, restrict
                    # the horizon to be shorter.
                    "touch_point": 15,
                })[args.get("env", "")],

            # Maximum number of steps to roll out an option policy.
            max_num_steps_option_rollout=defaultdict(
                lambda: 1000,
                {
                    # For the stick button environment, limit the per-option
                    # horizon.
                    "stick_button": 50,
                })[args.get("env", "")],

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
                    "cluttered_table_place": "after_exhaust",
                })[args.get("env", "")],

            # For learning-based approaches, the data collection strategy.
            offline_data_method=defaultdict(
                # Use only demonstrations by default.
                lambda: "demo",
                {
                    # Interactive learning project needs ground atom data.
                    "interactive_learning": "demo+ground_atoms",
                })[args.get("approach", "")],

            # The name of the option model used by the agent.
            option_model_name=defaultdict(
                lambda: "oracle",
                {
                    # For the BEHAVIOR environment, use a special option model.
                    "behavior": "oracle_behavior",
                    # For PyBullet environments, use non-PyBullet analogs.
                    "pybullet_cover": "oracle_cover",
                    "pybullet_blocks": "oracle_blocks",
                })[args.get("env", "")],

            # In SeSamE, the maximum number of skeletons optimized before
            # giving up. If 1, can only solve downward refinable tasks.
            sesame_max_skeletons_optimized=defaultdict(
                lambda: 8,
                {
                    # For these environments, allow more skeletons.
                    "behavior": 1000,
                    "coffee": 1000,
                    "tools": 1000,
                    "stick_button": 1000,
                })[args.get("env", "")],

            # In SeSamE, the maximum effort put into refining a single skeleton.
            # Concretely, this effort refers to the maximum number of calls to
            # the sampler on each step before backtracking.
            sesame_max_samples_per_step=defaultdict(
                lambda: 10,
                {
                    # For the tools environment, don't do any backtracking.
                    "tools": 1,
                })[args.get("env", "")],

            # Maximum number of skeletons used by ExpectedNodesScoreFunction.
            # If -1, defaults to CFG.sesame_max_skeletons_optimized.
            grammar_search_expected_nodes_max_skeletons=defaultdict(
                lambda: -1,
                {
                    # For the tools environment, keep it much lower.
                    "tools": 1,
                })[args.get("env", "")],

            # Used to save NSRTs as strings in pickle file.
            dump_nsrts_as_strings=defaultdict(
                lambda: False,
                {
                    # We cannot pickle BEHAVIOR NSRTs
                    "behavior": True,
                })[args.get("env", "")],
        )


def get_allowed_query_type_names() -> Set[str]:
    """Get the set of names of query types that the teacher is allowed to
    answer, computed based on the configuration CFG."""
    if CFG.option_learner == "direct_bc":
        return {"PathToStateQuery"}
    if CFG.approach == "interactive_learning":
        return {"GroundAtomsHoldQuery"}
    if CFG.approach == "unittest":
        return {
            "GroundAtomsHoldQuery",
            "DemonstrationQuery",
            "PathToStateQuery",
            "_MockQuery",
        }
    return set()


_attr_to_value = {}
for _attr, _value in GlobalSettings.__dict__.items():
    if _attr.startswith("_"):
        continue
    assert _attr not in _attr_to_value  # duplicate attributes
    _attr_to_value[_attr] = _value
CFG = SimpleNamespace(**_attr_to_value)
