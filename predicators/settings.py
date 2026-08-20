"""Contains global, immutable settings.

Anything that varies between runs should be a command-line arg
(args.py).
"""

from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set

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
    online_learning_early_stopping = False
    skip_test_until_last_ite_or_early_stopping = False
    # When True, skip only the pre-loop (cycle-0) test that evaluates the
    # offline-learned model before any online learning. Per-cycle testing is
    # unaffected, so the learning-progression curve is still measured; only
    # the (usually predictable) evaluation of the uncalibrated initial model
    # is saved. Subsumed by skip_test_until_last_ite_or_early_stopping.
    skip_initial_test = False
    # just for plotting
    online_learning_early_stopping_by_test_solve_rate = False
    # When True, every interaction request in the cycle (not just the first
    # per task) must succeed before early stopping is triggered. Catches
    # "lucky single-sample" successes that mask a buggy learned model.
    online_learning_early_stopping_require_all_attempts = False
    # Slack (in reward units) below a task's ``early_stop_min_reward`` bar
    # that still counts as solved for early stopping. Only tasks that set
    # ``EnvironmentTask.early_stop_min_reward`` are affected (e.g. domino
    # min-block tasks set it to the optimal reward, 1 - block_cost * K*):
    # with the default 0.0, training continues until the agent solves at
    # the bar (or cycles run out) instead of stopping on an inefficient
    # solve. Tasks that leave the bar None keep the plain solved criterion.
    online_learning_early_stopping_reward_slack = 0.0
    # When True, ignore ``EnvironmentTask.early_stop_min_reward`` entirely:
    # any solved (env-accepted) training episode counts toward early
    # stopping, regardless of its reward. Episode legitimacy is still
    # enforced by the env's solved verdict itself; only the optimality
    # requirement is dropped. Subsumes any reward_slack setting.
    online_learning_early_stopping_ignore_reward_bar = False
    # When True, the early-stopping cycle does NOT re-run testing, provided
    # every cycle is already being tested (skip_test_until_last_ite_or_early
    # _stopping is False). On the early-stopping cycle learning is skipped, so
    # the model is identical to the one the previous cycle already tested;
    # re-testing it only re-measures test-time stochasticity at full test-set
    # cost. Has no effect when skip_test_until_last_ite_or_early_stopping is
    # True, since then the early-stopping cycle is the model's only test.
    online_learning_early_stopping_skip_redundant_test = False
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
    # Run each test episode in a freshly-constructed env instance (the
    # generated test tasks are shared, so the tasks are identical).
    # State-level resets on a long-lived PyBullet env leave
    # history-dependent residuals - velocities the reconstruction diff
    # skips, auxiliary joints no reset touches, contact-solver state
    # that survives ``restoreState`` - so by test time the world's
    # behavior depends on everything the run executed before it
    # (measured on run_20260721_205821 seed0: the captured plan's
    # cascade stalled mid-chain in the run's long-lived env but
    # completes deterministically, 10/10 across placement jitters, in a
    # fresh env). Same rationale as the fresh-env-per-rollout sysID fix
    # in code_sim_learning.rollout_env. Envs that cannot be duplicated
    # (GUI mode: one client only) fall back to the shared instance; see
    # ``BaseEnv.make_fresh_test_instance``.
    test_fresh_env_per_episode = True
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
    # Normally, excluding goal predicates does not make sense, because then
    # there is no goal for the agent to plan towards. This is intended to be
    # used by VLM predicate invention, where we want to invent goal predicates
    # and different task goals are provided to the agent and the demonstrator.
    allow_exclude_goal_predicates = False
    # Normally, State.allclose() raises an error if the simulator state of
    # either of its arguments is not None.
    allow_state_allclose_comparison_despite_simulator_state = False

    env_include_bbox_features = False

    # Cross-cutting partial-observability flag. When True, envs that
    # support it hide selected latent features in `get_observation()`
    # (e.g. pybullet_boil hides `heat_level` and exposes a derived
    # `bubbling_level` instead). Used by approaches such as
    # agent_po_sim_predicate_invention. Each env decides which
    # of its features count as latent.
    partially_observable = False
    # cover_multistep_options env parameters
    cover_multistep_action_limits = [-np.inf, np.inf]
    cover_multistep_degenerate_oracle_samplers = False
    cover_multistep_max_tb_placements = 100  # max placements of targets/blocks
    cover_multistep_max_hr_placements = 100  # max placements of hand regions
    cover_multistep_thr_percent = 0.4  # target hand region percent of width
    cover_multistep_bhr_percent = 0.4  # block hand region percent of width
    cover_multistep_bimodal_goal = False
    cover_multistep_goal_conditioned_sampling = False  # assumes one goal
    cover_blocks_change_color_when_cover = False

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
    blocks_high_towers_are_unstable = False

    # balance env parameters
    balance_num_blocks_train = [2, 4]
    balance_num_blocks_test = [4, 6]
    # balance_num_blocks_test = [2]
    balance_holding_goals = False
    balance_block_size = 0.045  # use 0.0505 for real with panda
    balance_wierd_balance = False

    # grow env parameters
    # Use skill-factory-based option implementations
    grow_use_skill_factories = True
    grow_plant_same_color_as_cup = False
    grow_weak_pour_terminate_condition = False
    grow_place_option_no_sampler = False
    grow_num_cups_train = [2]
    grow_num_cups_test = [2, 3]
    grow_num_jugs_train = [2]
    grow_num_jugs_test = [2]

    # laser env parameters
    laser_zero_reflection_angle = False
    laser_use_debug_line_for_beams = False

    # ants env params
    ants_ants_attracted_to_points = False

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
    # Override the sim gripper's closed-finger joint value (metres). None keeps
    # each robot's built-in default (Panda: 0.03). Lower it to clamp a thin
    # object the default gap is wider than (e.g. the real 0.029 m domino).
    pybullet_closed_fingers = None
    pybullet_birrt_num_attempts = 10
    pybullet_birrt_num_iters = 100
    pybullet_birrt_smooth_amt = 50
    pybullet_birrt_extend_num_interp = 10
    pybullet_birrt_path_subsample_ratio = 1
    pybullet_birrt_contact_margin = -0.001
    # During a lift after grasping, the held object can start in shallow
    # penetration from grasp settling. Allow escaping these initial contacts
    # only up to this depth; deeper penetration remains a collision.
    pybullet_birrt_shallow_held_contact_margin = -0.003
    # Required separation (metres) from "bystander" bodies during BiRRT -
    # bodies the plan neither starts nor deliberately ends in proximity of.
    # The hard contact margin above tolerates ~1mm of penetration (needed
    # for resting contacts), which lets a planned path physically graze a
    # bystander; against knife-edge objects (dominoes) that graze topples
    # them and voids the episode (run_20260712_122549 test task1). Bodies
    # already within this clearance of the robot/held object at the start
    # or goal configuration are treated as intended contact partners and
    # keep the hard margin. 0 disables the clearance entirely.
    pybullet_birrt_bystander_clearance = 0.003
    # Required separation (metres) between the HELD OBJECT and bystander
    # bodies during BiRRT. The held object hangs on a grasp constraint and
    # lags the end effector's mid-path orientation swings by ~0.05 rad
    # (~7 mm at the tip of a 15 cm domino), so a plan that clears a
    # bystander by only pybullet_birrt_bystander_clearance still
    # physically grazes it at execution (run_20260717_230436 test task1:
    # a transported domino toppled a standing one the plan cleared by
    # 3 mm). Bodies already within this clearance of the held object at
    # the start or goal configuration fall back to the plain bystander
    # clearance so deliberately tight placements stay plannable. Must stay
    # below Bullet's 0.02 contactBreakingThreshold. 0 disables (falls back
    # to the plain bystander clearance). Kept off globally because tight
    # workspaces (boil) cannot afford the margin; graze-sensitive envs
    # opt in per skill via SkillConfig.held_bystander_clearance (domino
    # uses 0.015).
    pybullet_birrt_held_bystander_clearance = 0.0
    # BiRRT replay tracking gate: a waypoint is re-commanded until every
    # arm joint is within this tolerance (radians) of it, so the executed
    # path stays on the collision-checked plan. Popping one waypoint per
    # control step regardless of tracking error lets the arm lag several
    # waypoints behind and cut corners - the EE tilted up to 0.28 rad off
    # the planned configs during a domino Place transport, swinging the
    # held domino centimetres past the planner's bystander clearance and
    # toppling a standing domino (run_20260717_230436 test task1). 0
    # disables the gate.
    pybullet_birrt_replay_track_tol = 0.03
    # Deadlock guard for the tracking gate: after this many consecutive
    # re-commands of the same waypoint, advance anyway (an unreachable
    # waypoint otherwise stalls the phase until the episode horizon).
    pybullet_birrt_replay_max_hold_steps = 10
    pybullet_control_mode = "position"
    pybullet_max_vel_norm = 0.05
    # env -> robot -> quaternion
    pybullet_robot_ee_orns = defaultdict(
        # Fetch and Panda gripper down and parallel to x-axis by default.
        lambda: {
            "fetch": (0.5, -0.5, -0.5, -0.5),
            "mobile_fetch": (0.5, -0.5, -0.5, -0.5),
            "panda": (0.7071, 0.7071, 0.0, 0.0),
        },
        # In Blocks, Fetch gripper down since it's thin we don't need to
        # rotate 90 degrees.
        {
            "pybullet_blocks": {
                "fetch": (0.7071, 0.0, -0.7071, 0.0),
                "mobile_fetch": (0.7071, 0.0, -0.7071, 0.0),
                "panda": (0.7071, 0.7071, 0.0, 0.0),
            },
            "pybullet_balance": {
                "fetch": (0.7071, 0.0, -0.7071, 0.0),
                "mobile_fetch": (0.7071, 0.0, -0.7071, 0.0),
                "panda": (0.7071, 0.7071, 0.0, 0.0),
            }
        })
    pybullet_ik_validate = True

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

    # doorknobs env parameters
    doorknobs_target_value = 0.75
    test_doors_room_map_size = 10

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

    # skill phase parameters
    skill_phase_use_motion_planning = False
    # EE yaw relative to the pushed object's yaw during Push. None (the
    # default) takes it from the robot.
    skill_push_ee_yaw_offset = None

    # coffee env parameters
    coffee_num_cups_train = [1, 2]
    coffee_num_cups_test = [2, 3]
    coffee_jug_init_rot_amt = 2 * np.pi / 3
    coffee_rotated_jug_ratio = 0.5
    coffee_twist_sampler = True
    coffee_combined_move_and_twist_policy = False
    coffee_move_back_after_place_and_push = False
    coffee_jug_pickable_pred = False
    coffee_render_grid_world = False
    coffee_simple_tasks = False
    coffee_machine_have_light_bar = True
    coffee_machine_has_plug = False
    coffee_use_pixelated_jug = False
    coffee_plug_break_after_plugged_in = False
    coffee_fill_jug_gradually = False
    # Use skill-factory-based option implementations
    coffee_use_skill_factories = True

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
    kitchen_render_set_of_marks = False
    kitchen_use_combo_move_nsrts = False
    kitchen_randomize_init_state = False

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

    # float
    float_water_level_doesnt_raise = False

    # domino
    domino_debug_layout = False
    domino_some_dominoes_are_connected = False
    domino_initialize_at_finished_state = True
    domino_use_domino_blocks_as_target = False
    domino_include_connected_predicate = False
    domino_has_glued_dominos = True
    domino_prune_actions = False  # Set to True to enable action pruning
    # Generate only straight sequences during training
    domino_only_straight_sequence_in_training = True
    domino_train_num_dominos = [2]
    domino_test_num_dominos = [3]
    domino_train_num_targets = [1]
    domino_test_num_targets = [1, 2]
    domino_train_num_pivots = [0]
    domino_test_num_pivots = [0]
    # Fraction of generated tasks that are L-shaped (contain one 90-degree
    # domino turn) rather than straight, shared by both task pipelines:
    # min-block generation fills its turn/straight quotas from it, and the
    # plain DominoTaskGenerator resamples each task's chain until it
    # contains (or avoids) a turn90 to meet the same quota, turn tasks
    # first. Turn tasks are the hard family (tighter topple reach ~0.11 vs
    # ~0.15 straight, corner-relay staging). 0.0 = all straight, 1.0 = all
    # turns. Split per task set: train tasks default to straight-only.
    domino_train_turn_ratio = 0.0
    domino_test_turn_ratio = 0.5
    domino_train_num_pos_x = 3
    domino_train_num_pos_y = 2
    domino_test_num_pos_x = 4  # 5 is too large for robot to reach sometimes
    domino_test_num_pos_y = 3
    domino_oracle_knows_glued_dominos = False
    # Use PlaceContinuous option instead of Place
    domino_use_continuous_place = False
    # When True, Push only targets the start block
    # (no domino arg)
    domino_restricted_push = False
    # Use skill_factories-based option implementations
    domino_use_skill_factories = True
    # --- real robot (any env driving a real arm; see
    # pybullet_helpers.real_robot_executor and .real_robot_bridge) -----------
    # When True, a RealRobotExecutor is attached to the executed env and its
    # rollouts drive the arm. Default False = safe dry-run (pure sim, no
    # motion).
    real_robot_execute = False
    # Construct the RealRobot without an arm: every method still runs (and the
    # gripper state is still tracked) but nothing moves. Only consulted when
    # real_robot_execute.
    real_robot_dry = False
    # Perception source handed to the RealRobot: "zed" (live cameras, held
    # open for the whole session), "scene_file" (replay domino_real_scene --
    # cameraless, but it always reports the captured layout), or "none" (no
    # cameras at all, blind open-loop run).
    real_robot_perception = "zed"
    # Look at the scene at each option boundary and correct the twin from what
    # was seen. This is the point of running on real hardware -- the learner
    # sees perceived transitions rather than the simulator's guesses.
    real_robot_observe_at_option_boundary = True
    # Ship the whole episode's motion in one batch once it has all been
    # simulated, instead of one option at a time as each is simulated. The arm
    # then runs the plan as one contiguous motion rather than idling through
    # the next option's motion planning. Mutually exclusive with
    # real_robot_observe_at_option_boundary: a boundary look has to happen
    # between the two options it separates, and here there is no such moment.
    # Off by default -- batching removes every natural stopping point, so a
    # bad plan runs to its end with the e-stop as the only intervention.
    real_robot_open_loop_episode = False
    # Record each episode's execution to an SVO take, for offline pose
    # estimation. Nothing is estimated during the run: the markerless pipeline
    # runs at roughly 3x real time, so a take is post-processed after the
    # episode that produced it. Mutually exclusive with a live "zed"
    # perception, which owns the same cameras.
    real_robot_record_episodes = False
    # Where takes are written; one directory per episode. Empty means
    # logs/zed_takes.
    real_robot_recording_dir = ""
    # HD720 is the resolution the markerless pipeline was measured on.
    real_robot_recording_resolution = "HD720"
    # 60, not 30: a real cascade's topple onsets came 6, 4 and 2 frames apart
    # at 30 fps, and those inter-domino intervals are what the friction fit is
    # scored on -- at 30 fps a one-frame detection error is half the shortest
    # interval. HD720 runs at 60, so the resolution is already paid for.
    real_robot_recording_fps = 60
    # Stop a take after this many frames per camera; 0 is unbounded. A guard
    # against a take nothing stops, not a disk-budget knob -- bundles are
    # ~48 MB now that depth is no longer stored.
    real_robot_recording_max_frames = 0
    # Rebuild each episode's task from a short markerless take instead of a
    # live look. The live "zed" perception is the MARKER pipeline, and the
    # markers are not resolvable at this camera distance, so this is how a
    # per-episode scene rebuild is served on this bench. Needs
    # real_robot_record_episodes: the snapshot is a second short take on the
    # recorder's already-open session, which is what keeps it from fighting the
    # episode recording for the cameras.
    real_robot_snapshot_rebuild = False
    # An earlier run's boxes.json, replayed as stage 2's prompt boxes. Empty
    # opens the drag window and waits for a human EVERY episode, which no
    # learning loop can sit through -- so this is what makes the rebuild
    # unattended.
    real_robot_snapshot_boxes_json = ""
    # Frames the snapshot exports. More than one is worth having: the fit seeds
    # each frame from the previous one.
    real_robot_snapshot_frames = 5
    # "contact" (z from the table) is right for a scene rebuild, where a human
    # has just arranged every domino upright on the table. "free" is for a
    # cascade, where dominoes come to rest on each other.
    real_robot_snapshot_z_mode = "contact"
    # Write the stage-2 box and stage-3 mask overlays. They are what show
    # whether a particular capture is trustworthy.
    real_robot_snapshot_viz = True
    # ZED serial the scene is fitted from. Empty uses the recorder's first
    # serial. Ignored when real_robot_snapshot_fuse_cameras is on, which fits
    # every camera the recorder holds.
    real_robot_snapshot_camera = ""
    # Fit the snapshot from BOTH of the recorder's cameras and fuse the results
    # into one scene, instead of fitting from one. The point is occlusion: a
    # domino hidden behind another (or behind the arm) in one view is usually
    # visible in the other, and single-camera stage 4 gates a mostly-occluded
    # object out and emits nothing at all for it. The take already contains
    # both recordings -- the recorder opens every camera it was given -- so the
    # second view costs no extra hardware time, only the pipeline run.
    #
    # Each camera is fitted by the ordinary single-camera chain into its own
    # sub-bundle, then the records are matched by base-frame position and
    # merged; ids come from the first camera. Every fused record carries
    # `disagreement_m`, how far the two cameras placed the same domino apart,
    # which is a per-observation confidence no single camera can produce. On
    # this bench that runs ~12 mm, and a scene where it jumps well above that
    # is one where a camera has been moved.
    #
    # Costs a second pipeline run per snapshot, so it roughly doubles the
    # rebuild's wall time.
    real_robot_snapshot_fuse_cameras = False
    # Per-camera stage-2 prompt boxes, as a JSON object mapping ZED serial to an
    # earlier run's boxes.json:
    #   {"30264679": "/path/cam1/boxes.json", "32294776": "/path/cam2/boxes.json"}
    # Boxes CANNOT be shared between cameras -- a box is pixel coordinates on
    # one camera's frame 0, and the other camera views the scene from somewhere
    # else, so the same numbers would prompt SAM-2 at whatever lies at those
    # there. Only read when real_robot_snapshot_fuse_cameras is on; the
    # single-camera path keeps using real_robot_snapshot_boxes_json. A camera
    # missing from the mapping opens the drag window every episode.
    real_robot_snapshot_boxes_json_by_camera = ""
    # Run the markerless pipeline over each episode's take automatically, in
    # the background, and write a manifest saying which track belongs to which
    # episode. Off leaves the takes on disk for a human to process. The job is
    # launched when a take closes and joined at the end of the run, so it
    # overlaps the next episode's scene reset rather than the episode loop.
    real_robot_process_takes = False
    # Draw the pipeline's prompt boxes once at the start of the run, in a
    # drag window, instead of requiring real_robot_snapshot_boxes_json to have
    # been produced beforehand. One human interaction per RUN rather than per
    # take, which is what makes an otherwise-interactive pipeline usable in a
    # learning loop -- and valid because a fixed-plan replay trains and tests
    # on one arrangement. Needs a display; over SSH that means X forwarding.
    # Ignored when boxes are supplied by file.
    real_robot_pick_boxes_at_start = False
    # Open the take in front of the option with this name, instead of in
    # front of the whole batch. Only the cascade is scored, so the
    # pick-and-place that arranges the row is recorded for nothing -- and it
    # is most of the take: on run_20260818_092302 the push landed 107 s into
    # a 131 s track, and post-processing scales with frames, not seconds.
    #
    # It also puts the track's first frame at the arrangement the push acts
    # on. Track ids are matched to objects by position against the scored
    # segment's first state, and that segment begins AFTER the dominoes have
    # been placed -- so a take starting at the reset has its frame 0 showing
    # them ~130 mm from where the twin says they are, and the match fails.
    #
    # Empty records the whole batch, as every episode did before this
    # existed, and so does a name the episode never runs: too much video is
    # slow, but too little is an episode whose first topple happened off
    # camera, and the first onset is what every interval is measured from.
    real_robot_record_from_option = ""
    # Where the per-episode bundles and the run manifest are written. Empty
    # means logs/zed_tracks.
    real_robot_track_dir = ""
    # z from the table, as for a scene capture. The scored quantity is when
    # each domino STARTS to fall, and at that moment it is still standing on
    # the table, so the mode that constrains z there is the right one. "free"
    # fits z as a 5th parameter, which matters for the poses a cascade ends in
    # -- dominoes at rest on each other -- and those are not scored.
    real_robot_track_z_mode = "contact"
    # ZED serial the poses are fitted from. Markerless is single-camera -- the
    # second's cloud is not fused -- and the two are NOT interchangeable: on
    # hand-measured ground truth 30264679 is 6x better on orientation (1.03 deg
    # median against 6.29) while 32294776 tracks 99.9% of frames against 82%.
    # Empty takes the session's first serial, which is arbitrary rather than
    # considered.
    real_robot_track_camera = "30264679"
    # Drop the still lead-in and tail at stage 1, so SAM-2 never sees frames
    # where nothing happens. An episode take makes its own dead air: recording
    # starts at the reset and the twin then simulates every option with the arm
    # parked. Measured on run_20260817_162250 -- 420 s recorded, 268 s of arm
    # motion, so 152 s (36%, ~6.5k frames) was a static scene. Both of the
    # scan's failure modes keep frames rather than lose them.
    #
    # Needs BabyRobotPredicator's --trim-motion; a driver that predates it
    # ignores the request rather than failing, so this is safe to leave on
    # before the submodule has it.
    real_robot_trim_still_frames = True
    # Extra --trim-* flags for tuning the scan. Empty uses its own defaults;
    # check a window with `svo_to_bundle.py --trim-dry-run` before pinning one.
    real_robot_trim_args = ""
    # Stage-4 worker processes. 0 leaves the driver's own default, which is 16
    # -- sized for the machine it was tuned on, not for this one. Stage 4 is
    # the pipeline's largest step and it is cleanly core-limited: on a
    # 7773-frame take it ran 486 frames per worker in 364 s, which is 16.0
    # cores busy for the whole stage, on a 32-core box. Raising it is the one
    # speedup available without changing any code.
    #
    # Set it to the cores you are willing to give the pipeline, remembering it
    # runs in the BACKGROUND while the next episode executes -- taking every
    # core would starve the run that is driving the robot.
    real_robot_track_jobs = 0
    # Render stage 3's masks_overlay.mp4. It is a debugging aid nothing
    # downstream reads, and it is written BEFORE stage 4, so its cost lands
    # directly on the wait for the track rather than after it: 163 s of a
    # 1008 s pipeline, 16% of time-to-track, for a 120 MB file no fit opens.
    #
    # Worth turning back on when the tracks themselves look wrong -- the
    # overlay is how id swaps are spotted. Needs BabyRobotPredicator's
    # TRACK_VIZ; a driver that predates it renders the overlay anyway rather
    # than failing, so this is safe to set before the submodule has it.
    real_robot_track_viz = False
    # Dwell before a capture, so the dominoes come to rest after the motion.
    real_robot_settle_s = 0.5
    # How far (metres) the scene may be from where the twin predicted before
    # the disagreement is worth logging.
    real_robot_divergence_atol = 0.02
    # Write every option-boundary look to this directory as JSON: what was
    # perceived, what the twin predicted, and the per-object disagreement.
    # Empty disables it. The divergence WARNING only fires above tolerance, so
    # without this a run leaves no record of the looks that behaved -- and no
    # way to tell a systematic offset from one bad capture after the fact.
    real_robot_observation_dump_dir = ""
    # Plan file the "fixed_plan" explorer replays every episode, in
    # replay_plan.py's format. Lets the online loop be exercised without
    # paying for a planning explorer.
    fixed_plan_explorer_path = ""
    # Between episodes, home the arm and wait for a human to rearrange the
    # scene, then rebuild that episode's task from what the cameras then see.
    # False keeps the captured scene, which is what a fixed-plan
    # replay wants (rebuilding would change the objects the plan named).
    real_robot_human_reset = True
    # --- real-world domino scene (pybullet_domino_real env) ------------------
    # The reconstructed-scene JSON (robot_base frame) the pybullet_domino_real
    # env builds its single train/test task from; the env sizes its domino
    # component from this scene's role counts.
    domino_real_scene = ("/home/amberli/babyrobot/BabyRobotPredicator/"
                         "scenes/domino_straight.json")
    # Raw capture JSONs have no per-domino role, so roles are keyed by domino
    # id: the id of the start (green) domino and of the target (purple) domino;
    # every other domino is movable (blue). Ignored if the scene already carries
    # an explicit 'role' field per domino.
    domino_real_start_id = 6
    domino_real_target_id = 5
    # Real-scene geometry.
    domino_real_table_z = -0.041  # real table height in the robot base frame
    domino_real_robot_init_tilt = np.pi
    domino_real_robot_init_wrist = np.pi / 2
    domino_real_domino_dims = [0.15, 0.07,
                               0.029]  # (L, W, H) -> height,width,depth
    domino_real_decorate = True  # spawn extended-table tile + robot pedestal
    # Allow a task built from the scene JSON to be used while the cameras are
    # live. Off by default, and deliberately: the JSON's poses are a snapshot,
    # so planning against them while the arm works a scene nobody looked at
    # plans for a world that is not there -- silently, because the twin only
    # jumps to the truth at the first option boundary. Replaying a recorded
    # plan is the one case that wants it (the plan was written against those
    # poses), which is why replay_plan turns it on.
    real_robot_allow_captured_scene_task = False
    # Reach-limited "minimum-blocks" task mode: generate start/target pairs
    # spaced so that toppling requires bridging near the reach limit, and
    # attach each task a ``DominoEvaluator``. Success = toppling the target
    # via a legitimate cascade; each toppled blue costs domino_block_cost
    # reward, so a solver with a miscalibrated (too-high) reach model
    # over-reaches and fails, while an over-builder succeeds at lower
    # reward. K* (the minimum blues needed at the true friction) is stored
    # env-side on each EnvironmentTask (offline_task_metrics) for offline
    # metrics only - it never enters the criterion the solver is scored
    # against, and never reaches the agent-facing Task. Off by default
    # (existing behavior).
    domino_min_block_tasks = False
    # The "real" domino lateral friction the env runs at. Applied to the live
    # PyBullet bodies at env init via set_domino_physical_params and used when
    # computing K* for min-block tasks. Default matches the ClassVar (0.5), so
    # leaving it unset preserves existing behavior; lower it to make a
    # high-friction (no-learning) planner over-reach.
    domino_true_friction = 0.5
    # Friction for the *planning* base sim only — envs created with
    # skip_residual_dynamics=True (the approaches' base envs / option models),
    # the same flag that already denies planners the ground-truth delayed
    # dynamics. The eval env (main.py) is created without that flag and keeps
    # domino_true_friction. Either mismatch direction defeats an uncalibrated
    # planner on min-block tasks (the task filters are direction-aware):
    #   * ABOVE true friction: planner over-estimates reach -> UNDER-builds
    #     -> chain dies short of the target;
    #   * BELOW true friction: planner under-estimates reach -> OVER-builds
    #     -> target topples but the per-block reward cost (or, when the
    #     staged blue budget binds, an under-reaching build) penalizes it.
    # A sysID learner must recover the true value either way. None (default)
    # = planning sim uses domino_true_friction (no mismatch).
    domino_planning_friction: Optional[float] = None
    # Start->target distance (metres) sampled per min-block task, uniformly
    # from [lo, hi]. Choose a range that lands K* in {1, 2} at the true
    # friction (calibrate with the reach probe). Two scalars (not a tuple) so
    # the value passes cleanly through the shell-based launch flag plumbing.
    domino_min_block_span_lo = 0.13
    domino_min_block_span_hi = 0.30
    # How many blue (movable) blocks to stage per min-block task. Must exceed
    # the largest expected K* so over-building is possible (and thus penalized
    # by the per-block reward cost) rather than budget-limited.
    domino_min_block_num_blues = 4
    # Per-blue-block cost in the DominoEvaluator's reward:
    # reward = 1.0 * (terminated AND certified) - cost * blues_toppled.
    # Must satisfy domino_block_cost * num_staged_blues < 1 so a
    # legitimate success always outscores any failure (the reward's sign
    # alone then separates them); the evaluator asserts this against the
    # min-block budget flag or, for plain chain tasks, the scene's actual
    # movable count. Public by design - the agent is told the
    # reward form; only the dynamics stay hidden. NOTE: as a domino_* flag
    # this enters the min-block task cache key, so tuning it regenerates
    # the cached tasks.
    domino_block_cost = 0.05
    # Directory for caching generated min-block tasks (with their simulated
    # K*). The cache key hashes the task-relevant CFG flags, the seed, AND
    # the domino env/skill source code, so tasks regenerate automatically
    # whenever the config or code changes. Lives with the other cached
    # datasets; note cluster prep (get_cmds_to_prep_repo) wipes SAVE_DIRS
    # including saved_datasets, so cluster runs regenerate once. Empty
    # string disables caching.
    domino_min_block_task_cache_dir = "saved_datasets/domino_min_block_tasks"
    # Turn-task leg bands: entry/exit leg lengths (metres) sampled uniformly
    # from [lo, hi] per attempt. None (default) = the legacy over_reach band
    # hardcoded in _make_turn_task (the low-friction arm); under_reach arms
    # MUST set these explicitly - the legacy under_reach band shipped
    # agent-intractable pair-corner tasks. Probe candidate bands with
    # scripts/domino_debug/probe_min_block_bands.py whenever the friction
    # pair changes (the differentiating cells move with the frictions).
    domino_min_block_turn_entry_lo: Optional[float] = None
    domino_min_block_turn_entry_hi: Optional[float] = None
    domino_min_block_turn_exit_lo: Optional[float] = None
    domino_min_block_turn_exit_hi: Optional[float] = None
    # Heavy-block (immovable obstacle) task type — the single switch for the
    # variant. A HEAVY gray domino-shaped block sits with natural alignment
    # in one of two shapes (mixed per domino_{train,test}_turn_ratio):
    #   * straight: start -> gray -> target on ONE line, all co-facing; the
    #     true solution is a half-circle swerve around the gray;
    #   * turn: an L whose believed-cheapest corner layout is found first,
    #     and the gray stands exactly where that natural corner blue would
    #     go; the true solution skips around it with an own corner.
    # The gray's true mass (DominoComponent.heavy_block_true_mass) makes it
    # untopple-able and unmovable, but planning sims believe it has normal
    # domino mass (env init sets the ``block_mass`` override), so the
    # believed-cheapest plan runs THROUGH the gray (a free link/corner) and
    # dies against it at execution. Run WITHOUT domino_planning_friction -
    # this task type isolates the MASS dimension. (2026-07-25: generation
    # verified at the default true friction 0.5 - turn lures and skip-around
    # detours both certify; an older note claimed corners cannot propagate
    # at 0.5, no longer true after the short-leg corner retune.) Reuses the
    # min-block machinery (DominoEvaluator; the offline k_star = the
    # STAGED blues: heavy tasks differentiate on topple-vs-not, and the
    # searched K* certifies solvability only — corner minima are
    # solver-history sensitive at the margin; quota loop, disk cache,
    # domino_min_block_num_blues); domino_min_block_tasks does not also
    # need to be set.
    domino_heavy_block_tasks = False

    # burger env parameters
    burger_render_set_of_marks = True
    # Which type of train/test tasks to generate. Options are "more_stacks",
    # "fatter_burger", "combo_burger".
    burger_no_move_task_type = "more_stacks"
    # Replace actual rendering with dummy rendering (black 16x16 image) to speed
    # up rendering -- used in testing or when debugging.
    burger_dummy_render = False
    # Number of test tasks where you start out holding a patty.
    burger_num_test_start_holding = 5

    # circuit
    circuit_light_doesnt_need_battery = False
    circuit_battery_in_box = False

    # fan env
    # Use skill-factory-based option implementations
    fan_use_skill_factories = True
    fan_fans_blow_opposite_direction = False
    fan_known_controls_relation = True
    fan_combine_switch_on_off = False
    fan_use_kinematic = False
    fan_train_num_pos_x = 3
    fan_train_num_pos_y = 3
    fan_test_num_pos_x = 6  # can do 9
    fan_test_num_pos_y = 6
    fan_train_num_walls_per_task = [1]
    fan_test_num_walls_per_task = [2, 3]  # can do 4
    # When True, 3x3 grids use curated task generation: ball on an edge
    # cell, target axis-aligned two cells away, and a single wall placed
    # to block the direct path. When False, all grid sizes use uniform
    # random placement of ball, target, and walls.
    fan_3x3_strategic_task_gen = False

    # domino_fan env (combined domino + fan environment)
    domino_domino_on_stairs = False
    domino_fan_use_grid = True
    domino_fan_train_num_dominos = [3, 4]
    domino_fan_test_num_dominos = [5, 6]
    domino_fan_train_num_targets = [1]
    domino_fan_test_num_targets = [1, 2]
    domino_fan_train_num_walls = [2, 3]
    domino_fan_test_num_walls = [3, 4]
    domino_fan_train_grid_size = (5, 5)
    domino_fan_test_grid_size = (6, 6)
    # Fraction of tasks with ball goals vs domino goals
    domino_fan_ball_task_ratio = 0.5
    # Include ball in domino tasks (as obstacle)
    domino_fan_include_ball_in_domino_tasks = True
    # Include dominoes in ball tasks
    domino_fan_include_dominoes_in_ball_tasks = False
    # Tolerance for ball reaching target
    domino_fan_ball_position_tolerance = 0.04
    # Use kinematic ball movement (vs dynamic forces)
    domino_fan_use_kinematic = True
    # Include immovable glued dominoes
    domino_fan_has_glued_dominoes = False

    # boil env
    # Use skill-factory-based option implementations
    boil_use_skill_factories = True
    boil_use_constant_delay = False
    boil_use_normal_delay = True
    boil_use_cmp_delay = False
    boil_goal = "simple"  # Can also be "task_completed", "human_happy"
    # Require a simpler condition for human happy
    boil_goal_simple_human_happy = False
    boil_use_derived_predicates = True
    boil_require_jug_full_to_heatup = False
    boil_goal_require_burner_off = True
    boil_add_jug_reached_capacity_predicate = False
    boil_num_jugs_train = [1]
    boil_num_jugs_test = [1, 2]
    boil_num_burner_train = [1]
    boil_num_burner_test = [1]
    boil_water_fill_speed = 0.002
    # For the mobile_fetch robot: park the base (x-aligned to each reach
    # target, a stand-off in front in y) before reaching, so the arm reaches
    # straight forward at a comfortable distance instead of sideways over the
    # burner or fully extended. No-op for fixed bases. Set False to disable
    # (e.g. to isolate base-positioning effects).
    boil_mobile_base_park = True
    # Forward (y) stand-off distance for the parked base. Smaller = closer to
    # the target = more reach margin (incl. sideways switch push-through),
    # bounded by the table-clear y cap.
    boil_mobile_base_standoff = 0.45
    # Align the parked base x with the reach target x. True is best for picks
    # (straight approach avoids sweeping over the burner); False keeps the base
    # at the home x (diagonal approach) which leaves room for sideways switch
    # push-throughs.
    boil_mobile_base_align_x = True

    # bridge env
    # Task sizes: "simple" = 4 blocks (1-block legs, 2-block span, 3 glue
    # joints); "full" = 7 blocks (2-block leg stacks, 3-block span, 6
    # glue joints). One spec is drawn per task from these lists.
    bridge_task_spec_train = ["simple"]
    bridge_task_spec_test = ["simple"]

    # parameters for random options approach
    random_options_max_tries = 100

    # Max steps an any-atom-change Wait may run without seeing a change
    # before it bails out (see option_policy_to_policy). Infinite by
    # default (legacy behavior); envs whose plans interleave work with
    # exogenous delays (e.g. bridge) should set a finite cap, because
    # the awaited change can complete during the PREVIOUS option and
    # strand the Wait until the horizon.
    wait_option_max_steps = float("inf")

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
    # supported provider: "google", "openai", or "openrouter"
    pretrained_model_service_provider = "openai"

    # parameters for vision language models
    # gemini-1.5-pro-latest, gpt-4-turbo, gpt-4o
    vlm_model_name = "gemini-pro-vision"
    vlm_temperature = 0.0
    vlm_num_completions = 1
    vlm_include_cropped_images = False
    use_hardcoded_vlm_atom_proposals = False
    vlm_double_check_output = False

    # parameters for the vlm_open_loop planning approach
    vlm_open_loop_use_training_demos = False
    vlm_open_loop_no_image = False  # Use object-centric state

    # parameters for the human_option_control_approach
    human_option_control_approach_use_scripted_option = False
    human_option_control_approach_use_all_options = False
    scripted_option_dir = "scripted_options"
    script_option_file_name = "scripted_plan.txt"

    # parameters for the human_low_level_control_approach
    # Note: actual movement is limited by pybullet_max_vel_norm (default 0.05)
    # For faster response, also increase pybullet_max_vel_norm
    human_control_move_speed = 0.15  # meters per step (target delta)
    human_control_rot_speed = 0.2  # radians per step

    # SeSamE parameters
    sesame_task_planner = "astar"  # "astar" or "fdopt" or "fdsat"
    sesame_task_planning_heuristic = "lmcut"
    sesame_allow_waits = True  # recommended to keep this False if using replays
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
    planning_filter_unreachable_nsrt = True
    planning_check_dr_reachable = True
    no_repeated_arguments_in_grounding = False
    # If True, replace per-attempt backtracking and option-execution log
    # output with a tqdm progress bar during run_backtracking_refinement.
    # Suppresses DEBUG/INFO/WARNING/ERROR on all handlers (terminal + log
    # files) for the duration of the search; only CRITICAL passes through.
    refinement_progress_bar = True

    # evaluation parameters
    log_dir = "logs"
    results_dir = "results"
    eval_trajectories_dir = "eval_trajectories"
    approach_dir = "saved_approaches"
    data_dir = "saved_datasets"
    video_dir = "videos"
    image_dir = "images"
    # Run-scoped subdir (approach/experiment_id/seed<N>/run_<timestamp>/) that
    # video_dir mirrors from the log dir, so two runs sharing a config write to
    # separate dirs instead of overwriting each other's videos. Set by
    # utils.configure_logging; stays empty when there is no log dir (ad-hoc
    # scripts, unit tests), which keeps writing flat as before.
    run_subdir = ""
    # How many runs of one approach/experiment_id/seed keep their videos. Those
    # run dirs no longer overwrite each other, so save_video prunes the oldest
    # instead, reclaiming space when an arm is re-run much as overwriting used
    # to. 0 disables pruning and lets videos accumulate forever.
    video_max_runs_kept = 3
    video_fps = 2
    failure_video_mode = "longest_only"
    terminate_on_goal_reached = True
    keep_failed_demos = False  # For saving videos
    terminate_on_goal_reached_and_option_terminated = False
    env_has_impossible_goals = False

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
    clustering_learner_check_effect_equality = True
    disable_harmlessness_check = False  # some methods may want this to be True
    enable_harmless_op_pruning = False  # some methods may want this to be True
    precondition_soft_intersection_threshold_percent = 0.8  # between 0 and 1
    backchaining_check_intermediate_harmlessness = False
    pnad_search_without_del = False
    pnad_search_timeout = 10.0
    compute_sidelining_objective_value = False
    clustering_learner_true_pos_weight = 10
    clustering_learner_false_pos_weight = 1
    cluster_and_intersect_prederror_max_groundings = 10
    cluster_and_search_inner_search_max_expansions = 2500
    cluster_and_search_inner_search_timeout = 30
    cluster_and_search_score_func_max_groundings = 10000
    cluster_and_search_var_count_weight = 0.1
    cluster_and_search_precon_size_weight = 0.01
    cluster_and_search_llm_propose_batch_size = 4
    cluster_and_intersect_prune_low_data_pnads = False
    # If cluster_and_intersect_prune_low_data_pnads is set to True, PNADs must
    # have at least this fraction of the segments produced by the option that is
    # associated with their PNAD in order to not be pruned during operator
    # learning.
    cluster_and_intersect_min_datastore_fraction = 0.0
    cluster_and_intersect_soft_intersection_for_preconditions = False
    find_best_matching_pnad_skip_if_effect_not_subset = True
    exogenous_process_learner = "cluster_and_intersect"
    exogenous_process_learner_do_intersect = False
    only_learn_exogenous_processes = False
    learn_process_parameters = False
    use_empirical_init_for_vi_params = False
    pause_after_process_learning_for_inspection = False
    learnable_delay_distribution = "cmp"  # "constant", "cmp", "normal"
    process_learner_check_false_positives = False
    cluster_and_search_process_learner_parallel_condition = True
    cluster_and_search_process_learner_parallel_pnad = False
    process_learner_ablate_bayes = False
    cluster_and_search_process_learner_llm_select_condition = False
    cluster_and_search_process_learner_llm_rank_atoms = False
    cluster_and_search_process_learner_llm_propose_top_conditions = False
    process_learner_llm_atom_ranking_max_atoms = 10
    process_learner_llm_propose_conditions_k = 5
    cluster_and_search_vi_steps = 200
    cluster_search_max_workers = -1
    # "all", "top_consistent"
    cluster_and_inverse_planning_candidates = \
        "top_consistent"
    # "number", "percentage", "cost",
    # "percentage_cost"
    cluster_and_inverse_planning_top_consistent_method \
        = "percentage"
    cluster_and_inverse_planning_top_consistent_num = \
        -1
    # percentage of top consistent candidates to use
    cluster_and_inverse_planning_top_p_percent = 3
    cluster_and_inverse_planning_top_consistent_max_cost = 3
    cluster_process_learner_top_n_conditions = -1
    process_scoring_method = "data_likelihood"  # "count_fp", "data_likelihood"
    process_condition_search_complexity_weight = 1e-4
    process_param_learning_num_steps = 200
    process_param_learning_use_empirical = False
    process_param_learning_patience = None
    process_param_learning_batch_size = 16
    process_learning_use_empirical = False
    process_condition_search_prune_with_fp_count = False
    process_learning_learn_strength = True
    # Physical core vs logical core
    process_learning_process_per_physical_core = True
    # Loading hasn't been very helpful
    process_learning_init_at_previous_results = False
    predicate_invent_neural_symbolic_predicates = False
    predicate_invent_invent_derived_predicates = False
    cluster_learning_one_effect_per_process = False
    use_derived_predicate_in_heuristic = True
    process_planning_heuristic_weight = 1.0
    build_exogenous_process_index_for_planning = True
    process_planning_use_abstract_policy = False
    process_planning_max_policy_guided_rollout = 10
    process_planning_set_parameters_one = False
    # On an execution-time option failure (e.g. a fresh BiRRT collision caused
    # by drift between the refinement simulator and the real environment),
    # re-refine from the current state and retry, up to this many times. 0
    # disables replanning (the option failure is terminal, as before).
    process_planning_max_execution_replans = 0
    # Whether non-oracle planning approaches augment with the ground-truth
    # helper types, predicates, and objects (e.g. the domino/fan grid). Shared
    # by both the process-planning family (process/param learning, predicate
    # invention, ExoPredicator, ...) and the agent-planning family; the oracle
    # always does regardless, the others opt in via this flag.
    use_gt_helpers = False
    process_task_planning_heuristic = 'h_ff'
    wait_option_terminate_on_atom_change = True
    running_no_invent_baseline = False

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
    online_nsrt_learning_number_of_tasks_to_try = 1
    online_nsrt_learning_requests_per_task = 3
    online_learning_assert_no_exclude_pred = True

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

    # maple q function parameters
    use_epsilon_annealing = True
    min_epsilon = 0.05
    maple_q_same_hla_option_param_space = True

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

    bilevel_planning_explorer_enumerate_plans = False

    exploit_bilevel_planning_explorer_fallback_explorer = "RandomOptions"

    # grammar search invention parameters
    grammar_search_grammar_use_single_feature = True
    grammar_search_grammar_includes_givens = True
    grammar_search_grammar_includes_negation = True
    grammar_search_grammar_includes_foralls = True
    grammar_search_grammar_use_diff_features = False
    grammar_search_grammar_use_euclidean_dist = False
    grammar_search_grammar_use_skip_grammar = True
    grammar_search_use_handcoded_debug_grammar = False
    grammar_search_forall_penalty = 1
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
    grammar_search_additional_bonus_for_matching_plan = 0
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
    grammar_search_expected_nodes_allow_waits = True
    grammar_search_classifier_pretty_str_names = ["?x", "?y", "?z", "?w"]
    grammar_search_vlm_atom_proposal_prompt_type = \
        "options_labels_whole_traj_diverse"
    grammar_search_vlm_atom_label_prompt_type = "per_scene_naive"
    grammar_search_vlm_atom_proposal_use_debug = False
    grammar_search_parallelize_vlm_labeling = True
    grammar_search_select_all_debug = False
    grammar_search_invent_geo_predicates_only = False
    grammar_search_early_termination_heuristic_thresh = 0.0
    grammar_search_recognizing_unsolvable_goals_bonus = 1000

    # grammar search clustering algorithm parameters
    grammar_search_clustering_gmm_num_components = 10

    # filepath to be used if offline_data_method is set to
    # demo+labelled_atoms
    handmade_demo_filename = ""
    # filepath to be used if offline_data_method is set to
    # saved_vlm_img_demos_folder
    vlm_trajs_folder_name = ""
    vlm_predicate_vision_api_generate_ground_atoms = False
    # At test-time, we will use the below number of states
    # as part of labelling the current state's VLM atoms.
    vlm_test_time_atom_label_prompt_type = "per_scene_naive"
    # Whether or not to save eval trajectories
    save_eval_trajs = True
    rgb_observation = False
    render_init_state = False
    use_counterfactual_dataset_path_name = False
    use_classification_problem_setting = False
    classification_has_counterfactual_support = True

    # dino similarity approach
    dino_model_name = "dinov2_vits14"
    distance_function = "dtw"

    # vlm predicate invention parameters
    vlm_predicator_oracle_base_predicates = False
    vlm_predicator_oracle_learned_predicates = False
    vlm_predicator_use_grammar = True
    vlm_predicator_num_proposal_batches = 1

    # agent SDK online abstraction learning parameters
    agent_sdk_model_name = "claude-sonnet-5"
    agent_sdk_max_agent_turns_per_iteration = 50
    # Consecutive agent queries that die without the agent doing ANY work
    # (an auth/billing banner as the only assistant text, an error result,
    # or a stream error before the first tool call) before the run
    # terminates with AgentSessionFatalError. Such failures make every
    # future query hopeless, but each one returns in ~1 s at $0.00 and is
    # otherwise indistinguishable from a no-capture attempt, so without
    # this check the solve restart / replan / online-cycle budgets grind
    # through hundreds of instant failures (run_20260721_161159: 300
    # "organization has disabled Claude subscription access" queries
    # across 10 cycles, agent never ran). 0 disables the check.
    agent_sdk_max_consecutive_fatal_queries = 3
    # Reasoning effort for the agent SDK's Claude agent. One of "low",
    # "medium", "high", "max" to set it, or "" / "default" to leave it unset
    # (the model's own default). With adaptive thinking this is the control
    # for how much the agent deliberates per response.
    agent_sdk_reasoning_effort = ""
    agent_sdk_agent_timeout = 300  # seconds per iteration
    # Longest side (px) of scene images rendered for the agent (saved to the
    # sandbox and, rarely, returned inline). Every image the agent Reads
    # stays in its conversation for the rest of the session, and vision
    # tokens scale with pixel count (~(w*h)/750), so 900px renders cost
    # ~1100 tokens per view vs ~350 at 512px. Agent-facing renders scope
    # pybullet_camera_width/height down to this cap at generation time
    # (see agent_render_resolution), which also makes the render itself
    # cheaper; 0 disables the cap. Videos are unaffected.
    agent_sdk_image_max_px = 512
    # Max size (bytes) of a single newline-delimited JSON message the agent SDK
    # subprocess transport will buffer. The SDK default is 1 MB, which a tool
    # result embedding a base64 scene image (e.g. inspect_train_tasks with
    # include_image=True at 900x900) can exceed -> "JSON message exceeded
    # maximum buffer size". 20 MB comfortably fits full-res scene images.
    agent_sdk_max_buffer_size = 20 * 1024 * 1024
    agent_sdk_resume_session = True  # resume previous session if available
    agent_sdk_propose_types = True
    agent_sdk_propose_predicates = True
    agent_sdk_propose_objects = True
    agent_sdk_propose_processes = True
    agent_sdk_propose_options = True
    agent_sdk_auto_select_predicates = True  # run hill-climbing after proposals
    agent_sdk_max_trajectories_in_context = 3
    agent_sdk_log_agent_responses = True

    # Sandbox settings for agent SDK
    agent_sdk_use_docker_sandbox = False  # run agent inside Docker container
    agent_sdk_docker_image = "predicators-sandbox"  # Docker image name
    # sandbox dir with built-in tools, no Docker
    agent_sdk_use_local_sandbox = False

    # Agent explorer settings
    agent_explorer_max_turns = 5  # max agent turns per exploration query
    agent_explorer_fallback_to_random = True  # fall back to random on failure

    # Agent planner approach settings
    agent_planner_use_scratchpad = False  # include notes.md scratchpad
    # Include the solve-phase explore_python tool: a persistent Python
    # namespace over the BeliefProbe exploration facade (set the sim to any task
    # state or a modified copy, run option plans from it, read full-precision
    # features, render, snapshot/restore) so the agent writes sweep loops in
    # one call instead of one evaluate_option_plan round-trip per probe.
    # Exploratory only: nothing run through it can be captured as the answer.
    agent_planner_use_explore_python = False
    # When explore_python is on, whether the tools it subsumes
    # (refine_plan_sketch -> sim.refine; inspect_trajectories /
    # inspect_train_tasks -> trajectories / describe_trajectory /
    # sim.task() in the probe namespace) are STILL offered alongside it.
    # Default False: one surface per capability, so the agent's habits
    # don't split across redundant tools. No effect when
    # agent_planner_use_explore_python is False.
    agent_planner_explore_python_keep_replaced_tools = False
    # Whether the planner is given a simulator to test candidate plans with
    # (the evaluate_option_plan tool / option-model rollouts). When False, the
    # agent must plan open-loop from trajectory data and LLM reasoning alone
    # -- the genuinely model-free baseline.
    agent_planner_use_simulator = True
    # When a simulator IS given, whether to wrap the *base* env
    # (skip_residual_dynamics=True -- delayed _domain_specific_step effects
    # such as boiling/heating are disabled) instead of the real env. Lets the
    # model-free planner be denied the ground-truth delayed dynamics that a
    # world-model learner has to reconstruct. No effect when
    # agent_planner_use_simulator is False.
    agent_planner_use_base_simulator = False

    # Agent bilevel approach settings
    agent_bilevel_max_samples_per_step = 50  # param samples per step
    # Total refine_plan_sketch attempts (fresh rng each) when a refined
    # plan reaches the goal atoms but the task evaluator scores it as a
    # non-solve; all attempts share the one tool-call timeout budget.
    agent_bilevel_refine_evaluator_attempts = 3
    agent_bilevel_check_subgoals = True  # check subgoal atoms after each step
    # When True, the agent proposes per-step continuous parameters inside the
    # plan sketch (`Option(obj:type)[p1, p2] -> {subgoals}`). Refinement tries
    # the proposed params first, then falls back to the registered sampler /
    # uniform backtracking on failure. Default False keeps the param-free
    # sketch (search finds all continuous params).
    agent_bilevel_use_llm_initial_params = False
    # When True, sketch steps may carry GROUND samplers - per-step, per-call
    # sampling priors that override any learned parameterized sampler for
    # that step (precedence: ground > parameterized > uniform). Two forms
    # after a step's `[params]`: a uniform window `~ [w1, w2]`
    # (per-dimension half-widths around the proposed params) or a named
    # code sampler `~ my_sampler` referencing GROUND_SAMPLERS in the
    # sandbox's ground_samplers.py, loaded fresh on each refine call
    # (signature (state, subgoal_atoms, rng, objects) -> params, same as a
    # parameterized sampler, so any state-conditioned region is
    # expressible). Default False hides the grammar from the agent and
    # rejects the annotations, keeping baseline arms free of the channel.
    agent_bilevel_ground_samplers = False
    # When True, close the agent SDK session at the start of each test task
    # so every test solve begins with a FRESH conversation (no context from
    # earlier test tasks). The sandbox filesystem and learned artifacts are
    # untouched. Default False keeps the current behavior: all test tasks
    # share one continuous agent conversation.
    agent_fresh_session_per_test_task = False
    # Restart loop for test-task solving. Solve-time outcomes are close to
    # heavy-tailed in agent-search quality (run_20260717 family split: the
    # same tasks solved in 9-32 min in one launch and burned 2-11 h without
    # solving in its identical sibling, anchored on wrong conclusions), so
    # several short, independent attempts beat one long one. Each attempt
    # above the first starts from a fresh conversation; the solve journal
    # (below) carries curated knowledge across attempts. An attempt ends
    # early with a validated (evaluator-solved) capture; otherwise its
    # best-effort capture is banked and the best across attempts executes.
    # Each attempt is exactly ONE agent query: however that query ends -
    # a spent budget, an unparseable sketch, or a session that simply
    # never submitted - the fresh-context restart is the only retry, so
    # this is the sole knob controlling how many shots a task gets. Only
    # the final attempt (no restart left) pays for the best-effort
    # submission nudge.
    agent_solve_max_attempts = 1
    # Wall-clock budget per solve attempt, in seconds (0 disables). The
    # turn cap bounds turns, not compute - one explore_python sweep hid
    # 47k rollouts (~7 h) inside a single turn. On expiry, exploration
    # tools refuse with a submit-now message and the approach runs the
    # same best-effort submission flow as turn-cap exhaustion.
    agent_solve_attempt_wall_clock = 0.0
    # When True, every solve attempt (including the first, i.e. every test
    # task) begins with a fresh agent conversation; cross-attempt and
    # cross-task knowledge travels through the solve journal instead of
    # raw transcript history, which also carries the *wrong* conclusions
    # of failed attempts.
    agent_solve_fresh_context = False
    # Persistent per-run solve journal (<sandbox>/journal.md): the harness
    # auto-records each attempt's outcome + captured plan, the agent adds
    # lessons via the record_journal tool, and the journal is injected
    # into every solve prompt. Entries are capped and guided to record
    # facts/measurements rather than verdicts, so failed attempts steer
    # later ones away from repeated sweeps without re-importing their
    # anchoring mistakes.
    agent_solve_use_journal = False
    # Per-call wall-clock limit for explore_python code execution, in
    # seconds (0 disables). Enforced cooperatively at every probe sim
    # call, plus a hard async-exception watchdog for sim-free code (a
    # pure-Python loop blocks the event loop, so nothing else can stop
    # it), so a combinatorial sweep stops with its printed output
    # returned (partial results + a cost lesson) instead of blocking the
    # session for hours. Synthesis sessions (candidate-simulator probes,
    # whose rollouts are far slower and whose reset can trigger a
    # refit) are exempt.
    agent_sdk_explore_python_call_timeout = 600.0
    # Test-time closed-loop recovery. After each option in the refined plan
    # finishes, the subgoal_annotations execution monitor checks the
    # sketch's subgoal annotation for that step against the REAL state; on
    # divergence (execution left the option-model rollout — e.g. a place
    # that settled off-target), CogMan re-invokes solve(), which resumes a
    # re-refined suffix of the executed sketch from the current state,
    # instead of running the rest of the stale plan open-loop. Value =
    # recoveries per test episode, shared across chained replans; 0
    # disables (legacy open-loop execution). Requires --execution_monitor
    # subgoal_annotations (enforced at approach construction).
    agent_bilevel_max_execution_replans = 0
    # When an execution replan's suffix refinement fails, whether to fall
    # back to querying the agent for a fresh sketch - a brand-new
    # full-turn-budget session. Default False: the cheap suffix replan is
    # the only recovery, and the episode fails when no suffix of the
    # executed sketch refines from the diverged state. Re-opening the
    # agent budget is especially wasteful after a best-effort (non-solve)
    # capture, whose execution diverges by construction.
    agent_bilevel_replan_agent_fallback = False
    # log state pretty_str before/after each step
    agent_bilevel_log_state = False
    # Load a plan sketch from a file instead of querying the LLM. The dir is
    # under scripts/; the file may be a bare name or an absolute path.
    agent_bilevel_plan_sketch_dir = "plan_sketches"
    agent_bilevel_plan_sketch_file = ""
    # When refine_plan_sketch is called without an explicit timeout,
    # the tool computes
    #   max(_min, _per_step * len(sketch))
    # so plans with more steps automatically get more wall-clock budget.
    agent_bilevel_refinement_timeout_per_step = 30.0  # seconds per step
    agent_bilevel_refinement_timeout_min = 30.0  # floor on auto-scaled timeout
    # Total number of belief-sim rollouts a goal-reaching plan must pass in
    # evaluate_option_plan before it is captured as the agent's answer. The
    # shared sim env is nondeterministic across repeats (motion-planner
    # sampling, physics-solver state), so repeats sample the same execution
    # variability the real rollout will - a flaky plan is reported to the
    # agent in-session (where it can add margin and resubmit) instead of
    # captured and discovered as a failed real episode. 1 disables repeats.
    agent_plan_validation_rollouts = 3
    # Escalated rollout count once a task has produced a FLAKY rejection.
    # A 3-rollout gate passes a plan with per-rollout success rate p with
    # probability p^3 (p=0.85 -> 61%), so marginal plans slip through and
    # die on the single real episode (run_20260717_182321: a 20/20-swept
    # relay placement validated 3/3, then missed the target for real). A
    # FLAKY rejection is direct evidence the agent is tuning in a marginal
    # region, so subsequent captures on that task must clear this stricter
    # gate instead. Never lowers the base count.
    agent_plan_validation_rollouts_after_flaky = 6
    # Run each validation rollout inside ``ctx.validation_env_scope`` (a
    # freshly constructed sim env) when the approach installs one. A shared
    # env's reset provably cannot reconstruct state exactly (solver
    # warm-start state, velocity residuals, near-matching bodies skipped by
    # the reconstruction diff - see rollout_states in physical_sysid.py), so
    # repeated rollouts on it are correlated with each other and offset from
    # the fresh real env; fresh envs make them honest i.i.d. samples of what
    # the real episode will draw. Costs one env construction per rollout.
    agent_plan_validation_fresh_env = True
    # Physics-margin gate on captures: after a goal-reaching plan passes
    # the execution-validation rollouts, re-run it at a grid of
    # perturbations spanning +-1 sigma of the identified physical
    # parameters (sigma = the posterior width the sysID fit reported,
    # floored by code_sim_learning_rollout_min_posterior_width). The
    # execution repeats above only sample motion-planner/physics-stepping
    # variability AT the fitted values; a plan can pass them all and
    # still have zero margin to the fit's parameter error
    # (run_20260723_091108: a capture validated 8/8 at fitted
    # lateral_friction 0.5319 failed deterministically at true 0.5 -
    # the design's success band started at the fitted value). A failing
    # perturbed rollout refuses the capture as PARAM-SENSITIVE so the
    # agent adds design margin in-session. Runs only when the approach
    # installs a fresh-env scope (perturbing the shared env would leak)
    # and a fit with nonzero posterior width has been applied. Default
    # False so existing arms keep their behavior; the main arm
    # (approaches/all.yaml agent_po_predicate_invention_al)
    # turns it on.
    agent_plan_validation_physics_margin = False
    # Number of grid points the margin gate (and the sim.run physics
    # sweep) spreads evenly across the +-1-sigma range, endpoints
    # included. Endpoints alone (2) are provably insufficient: near a
    # feasibility boundary success is a SPECKLED function of the
    # params, and run_20260724_140531's capture passed both +-1-sigma
    # endpoints (lateral_friction 0.4295/0.5246) while failing
    # deterministically at the true 0.5 between them. Replaying that
    # capture mapped the speckle: a hazard band [~0.494, 0.511] holding
    # ~30% failures at ~0.001 grain, so ANY even grid is a
    # probabilistic detector - a 16-point grid's two in-band points
    # both passed (would still have captured), while the 32-point
    # grid's 0.5046 fails (rejects it). Per-point rollouts are
    # deterministic measurements costing one rollout (~seconds), and
    # captures are infrequent, so density is cheap sensitivity; designs
    # with real margin pass every density identically.
    agent_plan_validation_physics_margin_points = 32
    # Agent bilevel explorer settings. Separate from the solve-path budget
    # above because the explorer runs full backtracking while looking for
    # the deepest subgoal-failure to truncate at. Denominated in
    # option-model rollouts per search node: plain steps spend one per
    # backtracking attempt (classic semantics); info-seeking steps spend
    # the same budget pooling candidates (see refine_sketch).
    agent_bilevel_explorer_max_samples_per_step = 50

    # Active-experiment-design exploration: refinement picks the feasible
    # continuous parameters the learned model is most *uncertain* about
    # (ensemble disagreement on the step's subgoal atoms) instead of the
    # first feasible sample, pushing probes toward learned decision
    # boundaries. Off ⇒ identical to plain feasibility search.
    agent_explorer_info_seeking = False
    # Feasible candidates pooled per step before proposing the most
    # informative; the pool doubles as the node's ranked retry stock and
    # attempt cap (see bilevel_sketch.refine_sketch). 1 disables.
    agent_explorer_info_n_feasible_target = 8
    # Ensemble size used to estimate disagreement. 1 disables scoring
    # (every candidate scores 0) and reduces to first-feasible.
    agent_explorer_info_ensemble_size = 6
    # Per-parameter jitter as a fraction of the ParamSpec box width, for
    # the uniform-fallback ensemble only (see calibrated flag below).
    agent_explorer_info_perturb_frac = 0.15
    # Prefer a *calibrated* ensemble when the fit provides one: posterior
    # subsample when MCMC ran, else a Laplace draw from the LM Jacobian
    # (per-transition or recurrent); uniform jitter only when neither is
    # available (e.g. oracle params, where no fit runs).
    agent_explorer_info_calibrated_ensemble = True
    # Extra MCMC budget for the once-per-cycle active-experiment posterior
    # fit. The solver/test-time fit still follows
    # code_sim_learning_num_mcmc_steps; this budget is used only when it
    # exceeds the global solver budget, and only to calibrate the
    # info-seeking ensemble. Keep >= ~250: emcee burn-in (200) eats the
    # budget first. See _exploration_fit_num_steps for the rationale
    # (posterior subsampling covers gate/threshold params that a Laplace
    # approximation cannot).
    agent_explorer_info_mcmc_steps = 300

    # Code sim-learning parameter fitting settings.
    # Set to 0 to skip MCMC and use initial parameter values directly.
    code_sim_learning_num_mcmc_steps = 0
    # Persist the raw rollout-fit trajectories (states + actions per
    # recorded episode) to <log_dir>/fit_data/ at every cycle-level
    # fit. The fit data otherwise lives only in memory, which made the
    # wrong fits of run_20260724_232411 (lateral_friction 1.0358 /
    # 0.3236 vs true 0.5) impossible to replay offline: approximate
    # re-execution from logged plans cannot reproduce mid-episode
    # replans or the warm-env recording context, the very channel
    # suspected of corrupting the fits. Cost: one small pickle per fit.
    code_sim_learning_persist_fit_data = True
    # Truncate each rollout-fit trajectory once the scored features have
    # settled (physical_sysid.truncate_settled_tail): keep everything up
    # to the last observed motion plus a margin, drop the static tail.
    # Rationale: a free-running rollout diverges chaotically from the
    # recording over hundreds of contact steps, and a long settled tail
    # only re-scores that accumulated divergence every step, drowning
    # the physical-parameter signal (run_20260705_203314: SSE at the
    # TRUE friction exceeded SSE at the wrong one on full 500-step
    # trajectories). Domino-style trajectories (push -> cascade ->
    # long static settle) lose no information to the cut.
    code_sim_learning_rollout_truncate_settled = True
    # Per-step observed feature delta that counts as "still moving"
    # (meters / radians; settled dominoes jitter ~1e-5, a toppling one
    # moves ~1e-2/step).
    code_sim_learning_rollout_settle_tol = 1e-3
    # Steps kept after the last observed motion, so the rollout is still
    # scored on coming to rest at the right pose.
    code_sim_learning_rollout_settle_margin = 20
    # Candidates per physical parameter for the coarse grid sweep that
    # seeds the rollout LM start (0 disables). Needed because the
    # rollout SSE can be locally flat at the declared init (domino
    # topple reach saturates above friction ~0.5), stalling LM's finite
    # differences even on informative data; the sweep costs
    # num_points+1 rollout evals per physical param.
    code_sim_learning_rollout_grid_seed_points = 7
    # Coordinate-sweep passes for the grid seeding. One pass sweeps each
    # physical param in declaration order with the others held fixed, so
    # a param swept early is stuck with its neighbors' pre-sweep values;
    # later passes re-sweep in the updated context, and the loop stops
    # early once a full pass moves nothing. Deterministic rollouts are
    # memoized within one seeding call, so a converged extra pass costs
    # no new rollouts; 3 passes (vs 2) buys a final sweep whose pools
    # were all evaluated in the settled context whenever pass 2 still
    # moved something.
    code_sim_learning_rollout_grid_sweep_passes = 3
    # Relative SSE tolerance defining the sweep's "data-equivalent" flat
    # set: candidates whose SSE is within max(same-theta noise floor,
    # frac * best SSE) of the best candidate are indistinguishable on
    # this data, and the value chosen among them is the one nearest the
    # anchor (the prior belief) in fit space. This keeps a compensating
    # param at its anchor instead of chasing insignificant SSE gains
    # (run_20260711_224624: spinning_friction was dragged 0.5 -> 0.024,
    # true 0.5, for a 1.6% SSE gain after lateral_friction over-moved),
    # and resolves a saturated landscape to its anchor-side edge instead
    # of an arbitrary interior grid point. 0 disables the flat set (the
    # raw per-candidate argmin wins, the legacy behavior).
    code_sim_learning_rollout_grid_flat_frac = 0.05
    # Bisection evaluations per moved param that refine the anchor-side
    # edge of its flat set to sub-grid resolution (0 disables). The
    # 7-point log grid has ~2.4x spacing, so a true value mid-gap is
    # unrepresentable: on run_20260711_224624 true lateral friction 0.5
    # sat between candidates 0.342 and 0.827 and the fit reported the
    # 0.827 grid point (+65%) every cycle, LM being unable to descend
    # a chaotic replay landscape from a coarse seed.
    code_sim_learning_rollout_grid_refine_evals = 4
    # Floor (in FIT space, so ~fractional for log-scale params) on the
    # posterior width the identifiability report assigns to a
    # rollout-fit parameter. Two systematic errors make a raw landscape
    # width dishonest at the low end: (1) the flat-edge bisection can
    # collapse the flat set to a single point (each midpoint lowers the
    # best SSE, shrinking the relative flat tolerance until only the
    # newest point survives), which reads as posterior_std = 0 -
    # certainty the sweep's finite evaluations cannot support; (2) the
    # free-running replay objective carries model bias that no local
    # landscape statistic can see (measured fits land 1-40% off truth
    # on clean data: lateral_friction 0.5319 vs 0.5 on
    # run_20260723_091108, 0.1414 vs 0.1 on run_20260708_213258). The
    # default 0.1 (~+-10% for log params) brackets the typical bias;
    # the consumers of the reported width (verdict contraction, the
    # capture gate's physics-margin sigma points) inherit the floor.
    # 0 disables.
    code_sim_learning_rollout_min_posterior_width = 0.1
    # Post-fit anchor-ablation backward elimination (False disables):
    # for each physical param the LM MAP moved off its env-registry
    # anchor, refit the REMAINING params with that param pinned at the
    # anchor; if the refit is data-equivalent (SSE within
    # max(noise floor, grid_flat_frac * SSE)), the move was compensatory
    # and the param is reverted to its anchor. This is a global
    # alternative-hypothesis test the local curvature probe cannot make:
    # a co-adapted MAP has real curvature in every direction, so the
    # probe stamps a compensating param "identified" even when an
    # anchor-consistent basin explains the data equally well (measured on
    # run_20260721_205821 seed1: the coordinate sweep overshot
    # lateral_friction to 0.827 (true 0.5) and restitution 0.02 -> 0.75
    # (true 0.02) / spinning_friction 0.5 -> 0.01 (true 0.5) then
    # compensated for it, all three declared "identified"; the resulting
    # belief sim invalidated every test plan). Uses the same
    # data-equivalence tolerance as the grid flat set, so a genuinely
    # identified param (whose revert destroys the SSE) is never touched.
    code_sim_learning_rollout_anchor_ablation = True
    # Goodness-of-fit trimming threshold for rollout sysID, as a
    # multiple of the fit's noise_sigma (0 disables): a segment whose
    # best-achievable RMS over the candidate param grid exceeds
    # factor*noise_sigma is unexplainable at ANY params (chaotic
    # recording / model misfit), so it is dropped before fitting.
    # With normalized residuals the clean vs chaotic clusters sit at
    # <=0.012 vs >=0.074 dimensionless RMS on domino replay data
    # (run_20260711_141026): factor 2.0 (threshold 0.10) separates
    # them, while 3.0 (0.15) admits robot-scraping segments that drag
    # the friction fit 40% low on sparse early-cycle data (measured:
    # cycle-1 replay fits 0.0564 at 3.0 vs 0.0988 at 2.0, true 0.1).
    code_sim_learning_rollout_trim_rms_factor = 2.0
    # Consistency requirement among surviving trajectories (0 disables):
    # at the joint fit, each survivor's RMS must be within this factor
    # of its own best-achievable RMS. A survivor fitting much worse than
    # it could means the set disagrees on the params (a recording can be
    # accidentally explainable at WRONG params and outvote clean data);
    # the survivor with the largest best-RMS is dropped and the fit
    # reruns, anchoring on the cleanest data.
    code_sim_learning_rollout_consistency_factor = 3.0
    # Normalize rollout residuals per (type, feature): each residual is
    # divided by the feature's observed motion span over the fit data
    # (floored below), and residuals of features marked angular on their
    # Type are wrapped to [-pi, pi] first and scaled by pi. Without
    # this, raw angle errors dominate the SSE (a settled domino at roll
    # -pi vs +pi reads as a (2*pi)^2 error per step; measured on
    # run_20260711_141026: roll+yaw were 83% of the post-fit SSE with
    # max errors of exactly 2*pi and pi) and position information is
    # drowned. With normalization the SSE and every RMS threshold below
    # are dimensionless fractions of typical motion.
    code_sim_learning_rollout_scale_residuals = True
    # Huber cap on each (scaled) rollout residual: residuals beyond
    # this many scale units contribute linearly instead of
    # quadratically to the SSE/LM objective (0 disables). Rationale:
    # per-step residuals through contact are chaotic in theta - a
    # replay can diverge QUALITATIVELY at one grid candidate and not
    # its neighbors (measured 2026-07-25 on a recorded single-domino
    # topple at true friction 0.5: SSE 248.7 at theta 0.4746 vs 0.0025
    # at 0.4743, the spike being a diverged replay that slid 0.22 m vs
    # the recorded 0.09 m) - and one such spike can steer the grid fit
    # to a wrong basin. Capping bounds a diverged replay's vote while
    # still penalizing it. Value in dimensionless scale units (typical
    # motion ~1 under scale_residuals).
    code_sim_learning_rollout_huber_delta = 1.0
    # Extra weight on the per-trajectory SUMMARY residuals appended to
    # the per-step objective (0 disables): the settled endpoint
    # features and the per-object motion-onset step. These are the
    # smooth-in-theta observables (an ABC-style summary-statistic
    # objective): slide distances and rest poses respond monotonically
    # to friction while mid-flight paths are chaos. Offline A/B on
    # run_20260724_232411's re-executed explore episodes: the summary
    # objective separates the true friction from the neighboring grid
    # point by ~21x where per-step SSE manages ~1.25x. Each summary
    # residual enters the SSE with this factor squared.
    code_sim_learning_rollout_summary_weight = 5.0
    # --- scoring against an external observation track (Step 3) --------------
    # Score the free-running rollout against a markerless per-frame pose track
    # instead of against every recorded state. Under open-loop execution the
    # recorded states are the TWIN's own simulation -- correcting nothing --
    # so a per-step SSE over them recovers the twin's friction by
    # construction. The track is the only real evidence in the episode.
    code_sim_learning_rollout_score_observed_only = False
    # The track to score against: the trajectory JSON the markerless pipeline
    # emits. Empty means none available, which is a fallback-and-warn rather
    # than a silent zero (see the flag above).
    code_sim_learning_rollout_track_path = ""
    # Restrict the scored feature scope to these object types; empty keeps
    # every type, which is what the global-fidelity report wants. ["domino"]
    # for the friction experiment: the arm is commanded, so it reproduces at
    # every friction and can only dilute -- and with it in scope the episode
    # never rests, so rest-point segmentation can never cut.
    code_sim_learning_rollout_scope_types: List[str] = []
    # Features that cannot carry physics signal, dropped from the scored scope
    # when scope_types is set. A colour channel does not move; a settle
    # tolerance is not meaningful applied to a boolean.
    code_sim_learning_rollout_nonkinematic_features: List[str] = [
        "r", "g", "b", "is_held"
    ]
    # A fall is believed only once it rises this far above a domino's own
    # first reading, and holds for onset_min_persist samples. Far above both
    # measured spurious effects: gripper occlusion produced 29 deg and
    # orientation drift 13 deg, so neither can manufacture a topple.
    code_sim_learning_onset_confirm_deg = 45.0
    # Having confirmed, the onset is backdated to where the angle last sat
    # within this much of the domino's baseline.
    code_sim_learning_onset_deg = 5.0
    # Consecutive samples required at or past the confirmation angle. Mirrors
    # cascade_certificate._TOPPLE_MIN_STEPS: one sample carries no
    # information.
    code_sim_learning_onset_min_persist = 3
    # Object-name prefix mapped onto the track's integer domino ids.
    code_sim_learning_track_object_prefix = "domino_"
    # Rigid transform taking the track's frame into the env's world frame,
    # applied to the track's positions before they are matched to objects.
    # The markerless pipeline emits poses in the ROBOT BASE frame while a
    # twin state is in the env's own world frame, and for the domino env
    # those differ by a quarter turn (see
    # pybullet_domino.real_geometry.base_to_world_transform). Matching is
    # invariant to a translation -- it votes over candidate offsets, which is
    # what absorbs the camera calibration error -- but NOT to a rotation, so
    # without this every pair sits hundreds of mm apart and nothing matches.
    # Identity by default: an env whose track is already in world frame, and
    # every test that builds both sides in one frame, must be untouched.
    code_sim_learning_track_frame_yaw = 0.0
    code_sim_learning_track_frame_xy = (0.0, 0.0)
    # Frames per second assumed when a track carries no per-frame timestamps.
    code_sim_learning_track_fallback_fps = 60.0
    # How long a fit waits for the tracks its manifest promised. The online
    # loop fits as soon as an episode ends, while post-processing is still
    # running -- about 3x the length of the take. Not waiting means falling
    # back to per-step scoring, which under open-loop scores the twin against
    # itself, so the wait is what keeps the flag meaning what it says. This is
    # the LEARNER waiting for its data, which is fine; open-loop exists to
    # stop the ROBOT waiting. 0 disables.
    code_sim_learning_track_wait_s = 900.0
    # Floor for the per-feature motion span used in residual
    # normalization, so a feature that never moves in the fit data does
    # not blow up its (noise) residuals. Units: the feature's own
    # (meters / radians / ...).
    code_sim_learning_rollout_feature_scale_floor = 0.05
    # Split each rollout-fit trajectory at rest points (all scored
    # features quiescent for at least segment_min_rest_steps) into
    # independently-scored segments, each re-anchored at an observed
    # at-rest state (multiple shooting). Free-running an entire
    # manipulation trajectory lets chaotic contact divergence compound
    # across phases and shifts the SSE minimum away from the true
    # parameters (replay-divergence bias, both directions observed);
    # segments bound the compounding horizon and let trimming drop only
    # the chaotic phase of a trajectory instead of the whole recording.
    # Rest anchoring keeps the zero-velocity reset exact.
    code_sim_learning_rollout_segment_on_rest = True
    # Consecutive settled steps (per settle_tol) required for a rest
    # point to become a segment boundary.
    code_sim_learning_rollout_segment_min_rest_steps = 10
    # Pre-fit sensitivity screen: a physical param whose SSE span over
    # its own grid sweep does not exceed factor * the same-theta SSE
    # noise floor is "insensitive" on this data - the rollouts do not
    # respond to it, so its fitted value is noise. It is reported as
    # such and its anchor (env-registry baseline) is kept instead of
    # the fitted value. 0 disables the screen.
    code_sim_learning_rollout_sensitivity_factor = 2.0
    # Cross-cycle consistency check on the final per-cycle fit: a param
    # whose MAP moved more than this many combined posterior sigmas
    # since the previous cycle's fit is flagged (and its "identified"
    # verdict downgraded) - mutually-incompatible confident fits are
    # the signature of an overconfident probe. 0 disables.
    code_sim_learning_rollout_cross_cycle_sigma = 3.0
    # Pooled-evidence arbitration of a cross-cycle conflict: when the
    # new fit is flagged (see above) but explains the fit's surviving
    # segments with an SSE at least this factor smaller than the
    # held value does, the jump is accepted instead of held - the two
    # fits are nested (the new one saw a superset of the data), so a
    # decisive pooled-objective gap is real evidence, not probe
    # overconfidence (run_20260727_210827 seed1: pooled SSE 0.14 at the
    # new 0.4748 vs ~4.4 at the held 0.9313, yet the hold kept 0.9313
    # for the rest of the run). 0 disables arbitration.
    code_sim_learning_rollout_consistency_sse_ratio = 3.0
    # Diagnostic: log the Hessian eigendecomposition at the MAP to
    # spot unidentifiable parameter combinations. Adds ~5-15s per fit.
    code_sim_learning_log_hessian_identifiability = False
    # If True, run an LM fit and center MCMC walkers on its MAP estimate
    # instead of init_values. Adds ~5-15s per fit.
    code_sim_learning_warm_start_with_lm = True

    # Sim-learning oracle flags (for ablation / debugging).
    # When True, load GT residual rules instead of running agent synthesis.
    # Parameters init_values are perturbed so MCMC still has work to do.
    agent_sim_learn_oracle_sim_program = False
    # Relative scale for perturbing oracle parameter init_values before MCMC.
    agent_sim_learn_oracle_sim_param_noise_scale = 0.2
    # When True, use GT parameter values directly, skipping MCMC fitting.
    # Also grants planning base sims the TRUE physical params (e.g. the true
    # domino friction even when domino_planning_friction is set) — as if all
    # param learning, rule-level and physical, had already succeeded. Task
    # generation still reads domino_planning_friction for the
    # differentiation filter, so the oracle, the no-learning baseline, and
    # the sysID learner all see IDENTICAL tasks (and share the task cache:
    # this agent_ flag is outside the cache key's
    # domino_/pybullet_/skill_phase_ prefixes on purpose).
    agent_sim_learn_oracle_sim_params = False
    # When True, the agent learns PARAMETERIZED samplers - per-option
    # (lifted-skill) functions that aim continuous option parameters at each
    # sketch step's subgoal, instead of bilevel refinement drawing them
    # uniformly from the option's box. The agent authors a versioned
    # ``samplers.py`` (LEARNED_SAMPLERS keyed by option name) and tunes it
    # with the ``evaluate_sampler`` tool. Sampler learning rides along in
    # the sim/predicate synthesis session when one runs
    # (oracle_sim_program=False); when no synthesis session runs
    # (oracle_sim_program=True) it gets a dedicated session of its own.
    # The GROUND level of the sampler hierarchy needs no flag: a sketch
    # step's ``~ [widths]`` region annotation compiles to a per-step
    # GroundSampler that overrides the parameterized sampler for that step
    # (ground > parameterized > uniform).
    agent_sim_learn_parameterized_samplers = False
    # When True (and parameterized_samplers is on), use ground-truth
    # per-skill samplers from the env's GroundTruthSamplerFactory instead of
    # having the agent learn them — if such samplers exist for the env;
    # otherwise warn and fall back to synthesis. Mirrors
    # agent_sim_learn_oracle_sim_program.
    agent_sim_learn_oracle_samplers = False

    # Allowlist of env predicate names surfaced to the agent for
    # agent_sim_learning and its subclasses (e.g.
    # agent_sim_predicate_invention). Empty list defers to the class's
    # KEPT_INITIAL_PREDICATE_NAMES attribute: None for agent_sim_learning
    # (keep every env predicate), {"Holding"} for the invention approach.
    # Setting it on agent_sim_learning strips the named-out predicates -
    # even goal predicates - from the agent's prompts/tools; tasks whose
    # goal atoms are stripped must then carry goal_nl.
    agent_sim_learn_kept_predicates_names: List[str] = []
    # Ablation axis ("the robot knows its own simulator"): when True,
    # copy the env's declared base-sim source modules
    # (``get_base_sim_source_files()``, e.g. pybullet_fan_base.py +
    # pybullet_env.py) into the sandbox's ./reference/base_sim/ for
    # every agent session (solve, explore, and synthesis). The
    # visibility split is structural: residual dynamics, task
    # generation, and goal semantics live in modules that are never
    # declared, so the provided files are byte-identical to the code
    # the base-sim rollouts execute. Envs that declare no source files
    # are unaffected.
    agent_sim_provide_base_sim_source = False

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
                    "pybullet_coffee": 2000,
                    "pybullet_balance": 2000,
                    "pybullet_grow": 2000,
                    "pybullet_circuit": 2000,
                    "pybullet_float": 2000,
                    "pybullet_domino_grid": 2000,
                    "pybullet_laser": 2000,
                    "pybullet_ants": 2000,
                    "pybullet_fan": 2000,
                    # Bridge plans are long (up to ~27 options in the
                    # full variant), each option ~60-100 low-level steps.
                    "pybullet_bridge": 3000,
                    "pybullet_switch": 2000,
                    "pybullet_barrier": 2000,
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
                    "pybullet_switch": 2000,
                    "pybullet_barrier": 2000,
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
                    "pybullet_coffee": 100,
                    "pybullet_coffee_pixel": 100,
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
