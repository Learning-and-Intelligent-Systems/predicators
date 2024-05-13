import os
import sys
import itertools
import subprocess
import inspect
from typing import List
from glob import glob

import resource
_, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
resource.setrlimit(resource.RLIMIT_NPROC, (20000, hard_limit))

sub_script = os.path.join(os.path.dirname(
    inspect.getabsfile(inspect.currentframe())), 'sub.sh')
max_num_objects_per_env = {
    'shelves2d': 24,
    'statue': 60,
    'donuts': 15,
    'wbox': 25,
    'pybullet_packing': 12,
}
env_size_var = {
    'shelves2d': '--shelves2d_test_num_boxes',
    'donuts': '--donuts_test_num_toppings',
    'statue': '--statue_test_world_size',
    'wbox': '--wbox_test_num_containers',
    'pybullet_packing': '--pybullet_packing_test_num_box_cols',
}
base_env_size = {
    'shelves2d': 5,
    'donuts': 3,
    'statue': 4,
    'wbox': 2,
    'pybullet_packing': 4,
}

def run_experiment(
    env: str,
    approach: str,
    sampler_regressor_model: str,
    seed: int,
    num_train_tasks: int,
    env_size: int,
    load_data: bool,
    load_approach: bool,
    results_dir: str = 'experiment-results',
) -> None:
    assert env in {'shelves2d', 'statue', 'donuts', 'wbox', 'pybullet_packing'}
    assert approach in {'nsrt_learning', 'search_pruning', 'gnn_action_policy', 'oracle'}
    assert sampler_regressor_model in {'neural_gaussian', 'diffusion'}
    assert seed >= 0
    assert num_train_tasks > 0

    approach_dir = os.path.join(results_dir,
                                  f'{env}-{approach}-{sampler_regressor_model}-{seed}-{num_train_tasks}-{base_env_size[env]}'
                                  )

    results_prefix = os.path.join(results_dir,
                                  f'{env}-{approach}-{sampler_regressor_model}-{seed}-{num_train_tasks}-{env_size}'
                                  )
    os.makedirs(results_prefix, exist_ok=True)
    if glob(os.path.join(results_prefix, "*.pkl")):
        print('='*20)
        print(f"------ experiment {results_prefix} already ran ------")
        return

    supercloud_args = [
        'LLsub',
        sub_script,
        '-J', results_prefix,
        '-s', '20',
        # '-T', '120',
        '-g', 'volta:1',
        # '-p', 'xeon-p8',
        # '-p', 'debug-gpu',
        # '-p', 'debug-cpu',
        # '-o', os.path.join(results_prefix, "log.out"),
        '--options', f'"--output={os.path.join(results_prefix, "log.out")}"',
        '--',
    ]
    variable_experiment_args = [
        # General parameters
        '--env', env,
        '--approach', approach,
        '--seed', str(seed),
        '--num_train_tasks', str(num_train_tasks),
        '--results_dir', results_prefix,
        '--approach_dir', approach_dir,
        env_size_var[env], str(env_size),

        # Gaussian + Diffusion Parameters
        '--learning_rate', {'diffusion': '0.0001', 'neural_gaussian': '0.001'}[sampler_regressor_model],
        '--sampler_learning_regressor_model', sampler_regressor_model,
        '--sampler_disable_classifier', str(sampler_regressor_model == 'diffusion'),

        # Feasibility Parameters
        '--feasibility_debug_directory', results_prefix,
        '--feasibility_max_object_count', str(max_num_objects_per_env[env]),

        # Rerunning Parameters
        # '--load_approach',
        # '--feasibility_load_path', '/dev/null',
        '--feasibility_load_path', os.path.join(approach_dir, "prefix-1", 'feasibility-classifier-model.pt'),
    ] + (['--load_data'] if load_data else []) + (['--load_approach'] if load_approach and approach != 'search_pruning' else [])
    static_experiment_args = [
        # General parameters
        '--num_test_tasks', '50',
        '--option_model_terminate_on_repeat', 'true',
        '--sesame_task_planner', 'fdsat',
        '--pybullet_max_vel_norm', '100000',
        '--pybullet_control_mode', 'reset',
        # '--make_failure_videos',
        # '--make_test_videos',
        # '--video_fps', '4',
        '--use_torch_gpu', 'true',

        # GNN Parameters
        '--horizon', '100',
        '--gnn_use_timeout', 'false',
        '--gnn_training_timeout', '7200',
        '--gnn_num_epochs', '1600',
        '--gnn_num_message_passing', '3',
        '--gnn_layer_size', '512',
        '--gnn_learning_rate', '1e-4',
        '--gnn_do_normalization', 'true',

        # NSRT Parameters
        '--strips_learner', 'oracle',
        '--option_learner', 'no_learning',
        '--sesame_max_samples_per_step', '20',
        '--timeout', '120',
        '--sesame_max_skeletons_optimized', '1',
        '--disable_harmlessness_check', 'true',

        # Gaussian Parameters
        '--mlp_classifier_hid_sizes', '[128,128]',
        '--neural_gaus_regressor_hid_sizes', '[1024,1024]',
        '--sampler_mlp_classifier_max_itr', '20000',
        '--neural_gaus_regressor_max_itr', '20000',

        # Diffusion Parameters
        '--diffusion_regressor_timesteps', '100',
        '--diffusion_regressor_hid_sizes', '[512,512]',
        '--diffusion_regressor_max_itr', '10000',

        # Feasibility Parameters
        '--feasibility_learning_strategy', 'load_model' if load_approach else 'backtracking',
        '--feasibility_num_datapoints_per_iter', '4000',
        '--feasibility_featurizer_sizes', '[256,256,256]',
        '--feasibility_embedding_max_idx', '130',
        '--feasibility_embedding_size', '128',
        '--feasibility_num_layers', '4',
        '--feasibility_num_heads', '8',
        '--feasibility_ffn_hid_size', '512',
        '--feasibility_token_size', '128',
        '--feasibility_max_itr', '5000',
        '--feasibility_batch_size', '4000',
        '--feasibility_general_lr', '1e-4',
        '--feasibility_transformer_lr', '1e-5',
        '--feasibility_l1_penalty', '0',
        '--feasibility_l2_penalty', '0',
        '--feasibility_threshold_recalibration_percentile', '0.0',
        '--feasibility_num_data_collection_threads', '30',
        '--feasibility_keep_model_params', 'true',
    ]
    result = subprocess.run(supercloud_args + variable_experiment_args +
                            static_experiment_args, capture_output=True)
    b = result.stdout.startswith(b'Submitted batch job') and not result.stderr
    print('='*20)
    print(result.stdout.decode('utf-8'))
    print(result.stderr.decode('utf-8'))
    assert b
    print(f"++++++ experiment {results_prefix} launched ++++++")

assert all(arg in {'--load_data'} for arg in sys.argv[1:])
load_data = '--load_data' in sys.argv[1:]

def iterate_with_env_size(env: str, size_min: int, size_max: int):
    return zip(itertools.repeat(env), range(size_min, size_max + 1))

for (env, env_size), (approach, sampler_regressor_model), seed, num_train_tasks in itertools.product(
    itertools.chain(
        iterate_with_env_size('shelves2d', 5, 5),
        iterate_with_env_size('donuts', 3, 3),
        iterate_with_env_size('statue', 4, 4),
        # iterate_with_env_size('shelves2d', 5, 10),
        # iterate_with_env_size('statue', 4, 8),
        # iterate_with_env_size('donuts', 3, 6),
        # iterate_with_env_size('pybullet_packing', 4, 4),
        # iterate_with_env_size('wbox', 2, 2),
        # iterate_with_env_size('wbox', 2, 4),
    ),  # ['shelves2d', 'statue', 'donuts', 'wbox'],
    [('gnn_action_policy', 'diffusion'), ('nsrt_learning', 'neural_gaussian'), ('nsrt_learning', 'diffusion'), ('search_pruning', 'diffusion')],
    # [('gnn_action_policy', 'diffusion'), ('nsrt_learning', 'neural_gaussian'), ('nsrt_learning', 'diffusion')],
    # [('search_pruning', 'diffusion')],
    # [('nsrt_learning', 'neural_gaussian')],
    # [('nsrt_learning', 'diffusion'), ('gnn_action_policy', 'diffusion')],
    range(8),
    [500, 1000, 1500],  # [400, 800, 1200, 1600, 2000],
):
    assert env_size >= base_env_size[env]
    run_experiment(env, approach, sampler_regressor_model,
                   seed, num_train_tasks, env_size, load_data, env_size > base_env_size[env])
