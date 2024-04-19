import os
import sys
import itertools
import subprocess
import inspect
from typing import List

sub_script = os.path.join(os.path.dirname(inspect.getabsfile(inspect.currentframe())), 'sub.sh')
max_num_objects_per_env = {
    'shelves2d': 24,
    'statue': 80,
    'donuts': 15,
    'wbox': 10,
}

def run_experiment(
    env: str,
    approach: str,
    sampler_regressor_model: str,
    seed: int,
    num_train_tasks: int,
    load_data: bool,
    results_dir: str = 'experiment-results',
) -> None:
    assert env in {'shelves2d', 'statue', 'donuts', 'wbox'}
    assert approach in {'nsrt_learning', 'search_pruning', 'gnn_action_policy'}
    assert sampler_regressor_model in {'neural_gaussian', 'diffusion'}
    assert seed >= 0
    assert num_train_tasks > 0

    results_prefix = os.path.join(results_dir,
        f'{env}-{approach}-{sampler_regressor_model}-{seed}-{num_train_tasks}'
    )
    os.makedirs(results_prefix, exist_ok=True)

    supercloud_args = [
        'LLsub',
        sub_script,
        '-s', '20',
        '-g', 'volta:1',
        # '-p', 'xeon-p8',
        # '-p', 'debug-gpu',
        '--options', f'"--output={os.path.join(results_prefix, "log.out")}"',
        '--',
    ]
    variable_experiment_args = [
        '--env', env,
        '--approach', approach,
        '--sampler_learning_regressor_model', sampler_regressor_model,
        '--sampler_disable_classifier', str(sampler_regressor_model == 'diffusion'),
        '--seed', str(seed),
        '--num_train_tasks', str(num_train_tasks),
        '--feasibility_debug_directory', results_prefix,
        '--results_dir', results_prefix,
        '--shelves2d_test_num_boxes', '1',
        '--donuts_test_num_toppings', '3',
        '--statue_test_world_size', '4',
        '--wbox_test_num_containers', '2',
    ] + (['--load_data'] if load_data else [])
    static_experiment_args = [
        # General parameters
        '--num_test_tasks', '50',
        '--option_model_terminate_on_repeat', 'false',
        '--sesame_task_planner', 'fdopt',
        '--make_failure_videos',
        '--make_test_videos',
        '--video_fps', '4',

        # GNN Parameters
        '--horizon', '1000',
        '--gnn_use_timeout', 'true',
        '--gnn_training_timeout', '2400',
        '--gnn_num_epochs', '2400',
        '--gnn_num_message_passing', '4',
        '--gnn_layer_size', '256',
        '--gnn_learning_rate', '1e-4',
        '--gnn_do_normalization', 'true',

        # NSRT Parameters
        '--strips_learner', 'oracle',
        '--option_learner', 'no_learning',
        '--sesame_max_samples_per_step', '30',
        '--use_torch_gpu', 'true',
        '--timeout', '90',
        '--sesame_max_skeletons_optimized', '1',
        '--disable_harmlessness_check', 'true',

        # Diffusion Parameters
        '--learning_rate', '0.0001',
        '--diffusion_regressor_timesteps', '100',
        '--diffusion_regressor_hid_sizes', '[128,128]',
        '--diffusion_regressor_max_itr', '10000',

        # Feasibility Parameters
        '--feasibility_learning_strategy', 'backtracking',
        '--feasibility_load_path', '/dev/null',
        '--feasibility_num_datapoints_per_iter', '2000',
        '--feasibility_max_object_count', str(max_num_objects_per_env[env]),
        '--feasibility_featurizer_sizes', '[128,128,128]',
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
        '--feasibility_keep_model_params', 'true',
    ]
    print(" ".join(variable_experiment_args + static_experiment_args))
    # result = subprocess.run(supercloud_args + variable_experiment_args + static_experiment_args, capture_output=True)
    # b = result.stdout.startswith(b'Submitted batch job') and not result.stderr
    # if not b:
    #     print(result.stdout.decode('utf-8'))
    #     print(result.stderr.decode('utf-8'))
    # assert b

if len(sys.argv) == 1:
    load_data = False
else:
    _, flag = sys.argv
    assert flag == '--load_data'
    load_data = True

for env, (approach, sampler_regressor_model), seed, num_train_tasks in itertools.product(
    ['wbox'],#['shelves2d', 'statue', 'donuts', 'wbox'],
    [('search_pruning', 'diffusion')],#[('nsrt_learning', 'diffusion'), ('search_pruning', 'diffusion')],
    range(1),
    [2000],
):
    run_experiment(env, approach, sampler_regressor_model, seed, num_train_tasks, load_data)