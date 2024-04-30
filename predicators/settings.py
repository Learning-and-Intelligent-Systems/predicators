"""Contains global, immutable settings.

Anything that varies between runs should be a command-line arg
(args.py).
"""

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
    # Optionally define test tasks in JSON format
    test_task_json_dir = None
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
    approach_wrapper = None

    # cover_multistep_options env parameters
    cover_multistep_action_limits = [-np.inf, np.inf]
    cover_multistep_degenerate_oracle_samplers = False
    cover_multistep_max_tb_placements = 100  # max placements of targets/blocks
    cover_multistep_max_hr_placements = 100  # max placements of hand regions
    cover_multistep_thr_percent = 0.4  # target hand region percent of width
    cover_multistep_bhr_percent = 0.4  # block hand region percent of width
    cover_multistep_bimodal_goal = False
    cover_multistep_goal_conditioned_sampling = False  # assumes one goal

    # bumpy cover env parameters
    bumpy_cover_num_bumps = 2
    bumpy_cover_spaces_per_bump = 1
    bumpy_cover_right_targets = False
    bumpy_cover_bumpy_region_start = 0.8
    bumpy_cover_init_bumpy_prob = 0.25

    # regional bumpy cover env parameters
    regional_bumpy_cover_include_impossible_nsrt = False

    # blocks env parameters
    blocks_num_blocks_train = [3, 4]
    blocks_num_blocks_test = [5, 6]
    blocks_holding_goals = False
    blocks_block_size = 0.045  # use 0.0505 for real with panda

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
    repeated_nextto_nextto_thresh = 0.5

    # painting env parameters
    painting_initial_holding_prob = 0.5
    painting_lid_open_prob = 0.3
    painting_num_objs_train = [2, 3]
    painting_num_objs_test = [3, 4]
    painting_max_objs_in_goal = float("inf")
    painting_goal_receptacles = "box_and_shelf"  # box_and_shelf, box, shelf
    painting_raise_environment_failure = True

    # repeated_nextto_painting (rnt_painting) env parameters
    rnt_painting_num_objs_train = [8, 9, 10]
    rnt_painting_num_objs_test = [11, 12, 13]
    rnt_painting_max_objs_in_goal = 2

    # tools env parameters
    tools_num_items_train = [2]
    tools_num_items_test = [2, 3]
    tools_num_contraptions_train = [2]
    tools_num_contraptions_test = [3]

    # sandwich env parameters
    sandwich_ingredients_train = {
        "bread": [2],
        "patty": [1],
        "ham": [1],
        "egg": [1],
        "cheese": [1],
        "lettuce": [1],
        "tomato": [1],
        "green_pepper": [1],
    }
    sandwich_ingredients_test = {
        "bread": [2],
        "patty": [1],
        "ham": [1],
        "egg": [1],
        "cheese": [1],
        "lettuce": [1],
        "tomato": [1],
        "green_pepper": [1],
    }

    # general pybullet parameters
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
    pybullet_max_vel_norm = 0.05
    # env -> robot -> quaternion
    pybullet_robot_ee_orns = defaultdict(
        # Fetch and Panda gripper down and parallel to x-axis by default.
        lambda: {
            "fetch": (0.5, -0.5, -0.5, -0.5),
            "panda": (0.7071, 0.7071, 0.0, 0.0),
        },
        # In Blocks, Fetch gripper down since it's thin we don't need to
        # rotate 90 degrees.
        {
            "pybullet_blocks": {
                "fetch": (0.7071, 0.0, -0.7071, 0.0),
                "panda": (0.7071, 0.7071, 0.0, 0.0),
            }
        })

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

    # pddl gripper and prefixed gripper env parameters
    pddl_gripper_procedural_train_min_num_rooms = 3
    pddl_gripper_procedural_train_max_num_rooms = 5
    pddl_gripper_procedural_train_min_num_balls = 1
    pddl_gripper_procedural_train_max_num_balls = 2
    pddl_gripper_procedural_test_min_num_rooms = 3
    pddl_gripper_procedural_test_max_num_rooms = 5
    pddl_gripper_procedural_test_min_num_balls = 1
    pddl_gripper_procedural_test_max_num_balls = 2

    # pddl ferry env parameters
    pddl_ferry_procedural_train_min_num_locs = 3
    pddl_ferry_procedural_train_max_num_locs = 5
    pddl_ferry_procedural_train_min_num_cars = 1
    pddl_ferry_procedural_train_max_num_cars = 2
    pddl_ferry_procedural_test_min_num_locs = 3
    pddl_ferry_procedural_test_max_num_locs = 5
    pddl_ferry_procedural_test_min_num_cars = 1
    pddl_ferry_procedural_test_max_num_cars = 2

    # pddl miconic env parameters
    pddl_miconic_procedural_train_min_buildings = 1
    pddl_miconic_procedural_train_max_buildings = 2
    pddl_miconic_procedural_train_min_floors = 3
    pddl_miconic_procedural_train_max_floors = 5
    pddl_miconic_procedural_train_min_passengers = 1
    pddl_miconic_procedural_train_max_passengers = 2
    pddl_miconic_procedural_test_min_buildings = 1
    pddl_miconic_procedural_test_max_buildings = 2
    pddl_miconic_procedural_test_min_floors = 3
    pddl_miconic_procedural_test_max_floors = 5
    pddl_miconic_procedural_test_min_passengers = 1
    pddl_miconic_procedural_test_max_passengers = 2

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

    # narrow_passage env parameters
    narrow_passage_open_door_refine_penalty = 0
    narrow_passage_door_width_padding_lb = 1e-4
    narrow_passage_door_width_padding_ub = 0.015
    narrow_passage_passage_width_padding_lb = 5e-4
    narrow_passage_passage_width_padding_ub = 2e-2
    narrow_passage_birrt_num_attempts = 10
    narrow_passage_birrt_num_iters = 100
    narrow_passage_birrt_smooth_amt = 50

    # exit_garage env parameters
    exit_garage_clear_refine_penalty = 0
    exit_garage_min_num_obstacles = 2
    exit_garage_max_num_obstacles = 3  # inclusive
    exit_garage_rrt_extend_fn_threshold = 1e-3
    exit_garage_rrt_num_control_samples = 100
    exit_garage_rrt_num_attempts = 3
    exit_garage_rrt_num_iters = 100
    exit_garage_rrt_sample_goal_eps = 0.1
    exit_garage_motion_planning_ignore_obstacles = False
    exit_garage_raise_environment_failure = False

    # coffee env parameters
    coffee_num_cups_train = [1, 2]
    coffee_num_cups_test = [2, 3]
    coffee_jug_init_rot_amt = 2 * np.pi / 3

    # satellites env parameters
    satellites_num_sat_train = [2, 3]
    satellites_num_obj_train = [3, 4]
    satellites_num_sat_test = [3, 4]
    satellites_num_obj_test = [4, 5]

    # sokoban env parameters
    # use Sokoban-huge-v0 to show-off, the bottleneck is just the gym env
    # initialization and resetting. use Sokoban-small-v0 for tests
    sokoban_gym_name = "Sokoban-v0"

    # kitchen env parameters
    kitchen_use_perfect_samplers = False
    kitchen_goals = "all"

    # sticky table env parameters
    sticky_table_num_tables = 5
    sticky_table_place_smooth_fall_prob = 0.95
    sticky_table_place_sticky_fall_prob = 0.05
    sticky_table_pick_success_prob = 0.9
    sticky_table_tricky_floor_place_sticky_fall_prob = 0.5
    sticky_table_num_tables = 5  # cannot be less than 3
    sticky_table_place_smooth_fall_prob = 0.6
    sticky_table_place_sticky_fall_prob = 0.00
    sticky_table_place_ball_fall_prob = 1.00
    sticky_table_pick_success_prob = 1.00
    sticky_table_num_sticky_tables = 1  # must be less than the num_tables

    # grid row env parameters
    grid_row_num_cells = 100

    # parameters for random options approach
    random_options_max_tries = 100

    # option model parameters
    option_model_terminate_on_repeat = True
    option_model_use_gui = False

    # parameters for abstract GNN approach
    gnn_num_message_passing = 3
    gnn_layer_size = 16
    gnn_learning_rate = 1e-3
    gnn_weight_decay = 0
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
    pg3_add_condition_allow_new_vars = True
    pg3_max_analogies = 5

    # parameters for PG3 init approach
    # These need to be overridden via command line
    pg3_init_policy = None
    pg3_init_base_env = None

    # parameters for NSRT reinforcement learning approach
    nsrt_rl_reward_epsilon = 1e-2  # reward if in epsilon-ball from subgoal
    nsrt_rl_pos_reward = 0
    nsrt_rl_neg_reward = -1
    nsrt_rl_option_learner = "dummy_rl"
    nsrt_rl_valid_reward_steps_threshold = 10

    # parameters for large language models
    pretrained_model_prompt_cache_dir = "pretrained_model_cache"
    llm_openai_max_response_tokens = 700
    llm_use_cache_only = False
    llm_model_name = "text-curie-001"  # "text-davinci-002"
    llm_temperature = 0.5
    llm_num_completions = 1

    # parameters for vision language models
    vlm_model_name = "gemini-pro-vision"  # "gemini-1.5-pro-latest"

    # SeSamE parameters
    sesame_task_planner = "astar"  # "astar" or "fdopt" or "fdsat"
    sesame_task_planning_heuristic = "lmcut"
    sesame_allow_noops = True  # recommended to keep this False if using replays
    sesame_check_expected_atoms = True
    sesame_use_necessary_atoms = True
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
    sesame_check_static_object_changes = False
    # Warning: making this tolerance any lower breaks pybullet_blocks.
    sesame_static_object_change_tol = 1e-3
    # If True, then bilevel planning approaches will run task planning only,
    # and then greedily sample and execute in the environment. This avoids the
    # need for a simulator. In the future, we could check to see if the
    # observed states match (at the abstract level) the expected states, and
    # replan if not. But for now, we just execute each step without checking.
    bilevel_plan_without_sim = False

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
    # Default to bilevel_plan_without_sim.
    offline_data_bilevel_plan_without_sim = None

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
    disable_harmlessness_check = False  # some methods may want this to be True
    enable_harmless_op_pruning = False  # some methods may want this to be True
    backchaining_check_intermediate_harmlessness = False
    pnad_search_without_del = False
    pnad_search_timeout = 10.0
    compute_sidelining_objective_value = False
    clustering_learner_true_pos_weight = 10
    clustering_learner_false_pos_weight = 1
    cluster_and_intersect_prederror_max_groundings = 10
    # If a PNAD is learned by cluster and intersect such that
    # its datastore has less than the below fraction of data of the overall
    # dataset size for the PNADs option, then throw this PNAD out.
    cluster_and_intersect_min_datastore_fraction = 0.0
    cluster_and_search_inner_search_max_expansions = 2500
    cluster_and_search_inner_search_timeout = 30
    cluster_and_search_score_func_max_groundings = 10000
    cluster_and_search_var_count_weight = 0.1
    cluster_and_search_precon_size_weight = 0.01

    # torch GPU usage setting
    use_torch_gpu = False

    # torch model parameters
    learning_rate = 1e-3
    weight_decay = 0
    mlp_regressor_max_itr = 10000
    mlp_regressor_hid_sizes = [32, 32]
    mlp_regressor_clip_gradients = False
    mlp_regressor_gradient_clip_value = 5
    mlp_classifier_hid_sizes = [32, 32]
    mlp_classifier_balance_data = True
    cnn_regressor_max_itr = 500
    cnn_regressor_conv_channel_nums = [3, 3]
    cnn_regressor_conv_kernel_sizes = [5, 3]
    cnn_regressor_linear_hid_sizes = [32, 8]
    cnn_regressor_clip_gradients = True
    cnn_regressor_gradient_clip_value = 5
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

    # ml training parameters
    pytorch_train_print_every = 1000

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
    online_learning_max_novelty_count = 0

    # active sampler learning parameters
    active_sampler_learning_model = "myopic_classifier_mlp"
    active_sampler_learning_feature_selection = "all"
    active_sampler_learning_knn_neighbors = 3
    active_sampler_learning_use_teacher = True
    active_sampler_learning_num_samples = 100
    active_sampler_learning_score_gamma = 0.5
    active_sampler_learning_fitted_q_iters = 5
    active_sampler_learning_explore_pursue_goal_interval = 5
    active_sampler_learning_object_specific_samplers = False
    # shared with maple q function learning
    active_sampler_learning_n_iter_no_change = 5000
    active_sampler_learning_num_lookahead_samples = 5
    active_sampler_learning_explore_length_base = 2
    active_sampler_learning_num_ensemble_members = 10
    active_sampler_learning_exploration_sample_strategy = "epsilon_greedy"
    active_sampler_learning_exploration_epsilon = 0.5
    active_sampler_learning_replay_buffer_size = 1000000
    active_sampler_learning_batch_size = 64

    # skill competence model parameters
    skill_competence_model = "optimistic"
    skill_competence_model_num_em_iters = 3
    skill_competence_model_max_train_iters = 1000
    skill_competence_model_learning_rate = 1e-2
    skill_competence_model_lookahead = 1
    skill_competence_model_optimistic_window_size = 5
    skill_competence_model_optimistic_recency_size = 5
    skill_competence_default_alpha_beta = (10.0, 1.0)
    skill_competence_initial_prediction_bonus = 0.5

    # refinement cost estimation parameters
    refinement_estimator = "oracle"  # default refinement cost estimator
    refinement_estimation_num_skeletons_generated = 8

    # refinement data collection parameters
    refinement_data_num_skeletons = 8
    refinement_data_skeleton_generator_timeout = 20
    refinement_data_low_level_search_timeout = 5  # timeout for refinement try
    refinement_data_failed_refinement_penalty = 5  # added time on failure
    refinement_data_include_execution_cost = True
    refinement_data_low_level_execution_cost = 0.05  # per action cost to add

    # CNN refinement cost estimator image pre-processing parameters
    cnn_refinement_estimator_crop = False  # True
    cnn_refinement_estimator_crop_bounds = (320, 400, 100, 650)
    cnn_refinement_estimator_downsample = 2

    # bridge policy parameters
    bridge_policy = "learned_ldl"  # default bridge policy

    # glib explorer parameters
    glib_min_goal_size = 1
    glib_max_goal_size = 1
    glib_num_babbles = 10

    # greedy lookahead explorer parameters
    greedy_lookahead_max_num_trajectories = 100
    greedy_lookahead_max_traj_length = 2
    greedy_lookahead_max_num_resamples = 10

    # active sampler explorer parameters
    active_sampler_explore_use_ucb_bonus = True
    active_sampler_explore_bonus = 1e-1
    active_sampler_explore_task_strategy = "planning_progress"
    active_sampler_explorer_replan_frequency = 100
    active_sampler_explorer_planning_progress_max_tasks = 10
    active_sampler_explorer_planning_progress_max_replan_tasks = 5
    active_sampler_explorer_skip_perfect = True
    active_sampler_learning_init_cycles_to_pursue_goal = 1

    # grammar search invention parameters
    grammar_search_grammar_includes_givens = True
    grammar_search_grammar_includes_foralls = True
    grammar_search_grammar_use_diff_features = False
    grammar_search_grammar_use_euclidean_dist = False
    grammar_search_use_handcoded_debug_grammar = False
    grammar_search_pred_selection_approach = "score_optimization"
    grammar_search_pred_clusterer = "oracle"
    grammar_search_true_pos_weight = 10
    grammar_search_false_pos_weight = 1
    grammar_search_bf_weight = 1
    grammar_search_operator_complexity_weight = 0.0
    grammar_search_pred_complexity_weight = 1e-4
    grammar_search_max_predicates = 200
    grammar_search_predicate_cost_upper_bound = 6
    grammar_search_prune_redundant_preds = True
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
<<<<<<< HEAD
<<<<<<< HEAD
    grammar_search_vlm_atom_proposal_prompt_type = "options_labels_whole_traj"
    grammar_search_vlm_atom_label_prompt_type = "per_scene_naive"
    grammar_search_vlm_atom_proposal_use_debug = False
=======
    grammar_search_predicate_labelling_noise_prob = 0.0
>>>>>>> lots of changes with noise-based learning!
=======
>>>>>>> Just keep low-datastore operator pruning in cluster and intersect.

    # grammar search clustering algorithm parameters
    grammar_search_clustering_gmm_num_components = 10

    # filepath to be used if offline_data_method is set to
    # demo+labelled_atoms
    handmade_demo_filename = ""
    # filepath to be used if offline_data_method is set to
    # img_demos
    vlm_trajs_folder_name = ""

    @classmethod
    def get_arg_specific_settings(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """A workaround for global settings that are derived from the
        experiment-specific args."""

        return dict(
            # The method used for perception: now only "trivial" or "sokoban".
            perceiver=defaultdict(lambda: "trivial", {
                "sokoban": "sokoban",
                "kitchen": "kitchen",
            })[args.get("env", "")],
            # Horizon for each environment. When checking if a policy solves a
            # task, we run the policy for at most this many steps.
            horizon=defaultdict(
                lambda: 100,
                {
                    # For certain environments, actions are lower level, so
                    # tasks take more actions to complete.
                    "pybullet_cover": 1000,
                    "pybullet_blocks": 1000,
                    "doors": 1000,
                    "coffee": 1000,
                    "kitchen": 1000,
                    # For the very simple touch point environment, restrict
                    # the horizon to be shorter.
                    "touch_point": 15,
                    # Ditto for the simple grid row environment.
                    "grid_row": cls.grid_row_num_cells + 2,
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
                    "coffee": 1000,
                    "exit_garage": 1000,
                    "tools": 1000,
                    "stick_button": 1000,
                    "stick_button_move": 1000
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

            # Factor to divide feature range by when instantiating predicates
            # of the form |t1.f1 - t2.f2| < c to indicate that t1.f1 and
            # t2.f2 are "touching" or close. E.g. for the predicate
            # |robot.x - button.x| < c in the StickButtonMovement env,
            # we set this constant to 1/60.0 because that will yield
            # |robot.x - button.x| < ((ub - lb)/60.0) + ub, which corresponds
            # to a predicate that correctly classifies when the robot and
            # button are touching.
            grammar_search_diff_features_const_multiplier=defaultdict(
                lambda: 1e-6,
                {"stick_button_move": 1 / 30.0})[args.get("env", "")],

            # Feature names to use as part of the EuclideanPredicateGrammar.
            # Each entry is (type1_feature1name, type1_feature2name,
            # type2_feature1name, type2_feature2name)
            grammar_search_euclidean_feature_names=defaultdict(
                lambda: [("x", "y", "x", "y")], {
                    "stick_button_move": [("x", "y", "x", "y"),
                                          ("x", "y", "tip_x", "tip_y")]
                })[args.get("env", "")],

            # Factor to divide feature range by when instantiating euclidean
            # predicates of the form
            # (t1.f1 - t2.f1)^2 + (t1.f2 - t2.f2)^2 < c^2 to indicate that
            # the euclidean distance between f1 and f2 is close enough that.
            # the two objects are "touching".
            grammar_search_euclidean_const_multiplier=defaultdict(
                lambda: 1e-6,
                {"stick_button_move": 1 / 250.0})[args.get("env", "")],

            # Parameters specific to the cover environment.
            # cover env parameters
            cover_num_blocks=defaultdict(lambda: 2, {
                "cover_place_hard": 1,
            })[args.get("env", "")],
            cover_num_targets=defaultdict(lambda: 2, {
                "cover_place_hard": 1,
            })[args.get("env", "")],
            cover_block_widths=defaultdict(lambda: [0.1, 0.07], {
                "cover_place_hard": [0.1],
            })[args.get("env", "")],
            cover_target_widths=defaultdict(lambda: [0.05, 0.03], {
                "cover_place_hard": [0.05],
            })[args.get("env", "")],
            cover_initial_holding_prob=defaultdict(lambda: 0.75, {
                "cover_place_hard": 0.0,
            })[args.get("env", "")],
        )


def get_allowed_query_type_names() -> Set[str]:
    """Get the set of names of query types that the teacher is allowed to
    answer, computed based on the configuration CFG."""
    if CFG.option_learner == "direct_bc":
        return {"PathToStateQuery"}
    if CFG.approach == "interactive_learning":
        return {"GroundAtomsHoldQuery"}
    if CFG.approach == "bridge_policy":
        return {"DemonstrationQuery"}
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
