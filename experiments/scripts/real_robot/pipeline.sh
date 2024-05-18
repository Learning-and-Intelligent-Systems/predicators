#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
RESULTS_DIR="results"

mkdir $RESULTS_DIR

# rm -rf $RESULTS_DIR
# mkdir $RESULTS_DIR

if ! test -f $RESULTS_DIR/rough_capture_poses.pkl; then
	echo "GENERATING ROUGH CAPTURE POSES"
	python3.9 experiments/scripts/real_robot/generate_capture_poses.py $RESULTS_DIR/rough_capture_poses.pkl
fi

if ! test -f $RESULTS_DIR/rough_capture_image_data.pkl; then
	echo "CAPTURING THE ENVIRONMENT FOR ROUGH POSES"
	python3.9 experiments/scripts/real_robot/run_capture.py $RESULTS_DIR/rough_capture_poses.pkl $RESULTS_DIR/rough_capture_image_data.pkl
fi

if ! test -f $RESULTS_DIR/rough_block_info.pkl; then
	echo "GENERATING ROUGH BLOCK POSES"
	python3.9 experiments/envs/pybullet_packing/capture.py $RESULTS_DIR/rough_capture_image_data.pkl $RESULTS_DIR/rough_block_info.pkl
fi

if ! test -f $RESULTS_DIR/localized_capture_poses.pkl; then
	echo "GENERATING LOCALIZED RECAPTURE POSES"
	python3.9 experiments/scripts/real_robot/generate_recapture_poses.py $RESULTS_DIR/rough_block_info.pkl $RESULTS_DIR/localized_capture_poses.pkl
fi

if ! test -f $RESULTS_DIR/localized_capture_image_data.pkl; then
	echo "CAPTURING THE ENVIRONMENT FOR LOCALIZED POSES"
	python3.9 experiments/scripts/real_robot/run_capture.py $RESULTS_DIR/localized_capture_poses.pkl $RESULTS_DIR/localized_capture_image_data.pkl
fi

if ! test -f $RESULTS_DIR/localized_block_info.pkl; then
	echo "GENERATING LOCALIZED BLOCK POSES"
	python3.9 experiments/envs/pybullet_packing/capture.py $RESULTS_DIR/localized_capture_image_data.pkl $RESULTS_DIR/localized_block_info.pkl
fi

if ! test -f $RESULTS_DIR/*.traj; then
	echo "RUNNING THE SEARCH"
	FD_EXEC_PATH="downward" PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_LAUNCH_BLOCKING=1 python3.9 predicators/main.py --env pybullet_packing --approach $1 --seed 0 --num_train_tasks 0 --num_test_tasks 1 --results_dir "$RESULTS_DIR" --eval_trajectories_dir "$RESULTS_DIR" --pybullet_packing_test_num_box_cols 4 --learning_rate 0.0001 --sampler_learning_regressor_model diffusion --sampler_disable_classifier True --feasibility_debug_directory results --feasibility_max_object_count 12 --feasibility_load_path $2 --option_model_terminate_on_repeat true --sesame_task_planner "astar" --use_torch_gpu false --horizon 10000 --strips_learner oracle --option_learner no_learning --sesame_max_samples_per_step 20 --timeout 3600 --sesame_max_skeletons_optimized 1 --disable_harmlessness_check true  --diffusion_regressor_timesteps 200 --diffusion_regressor_hid_sizes [512,512] --diffusion_regressor_max_itr 10000 --feasibility_learning_strategy "load_model" --feasibility_num_datapoints_per_iter 4000 --feasibility_featurizer_sizes [256,256,256] --feasibility_embedding_max_idx 130 --feasibility_embedding_size 256 --feasibility_num_layers 4 --feasibility_num_heads 16 --feasibility_ffn_hid_size 1024 --feasibility_token_size 256 --feasibility_max_itr 4000 --feasibility_batch_size 4000 --feasibility_general_lr 1e-4 --feasibility_transformer_lr 1e-5 --feasibility_l1_penalty 0 --feasibility_l2_penalty 0 --feasibility_threshold_recalibration_percentile 0.0 --feasibility_num_data_collection_threads 20 --feasibility_keep_model_params true --pybullet_max_vel_norm 100000 --pybullet_control_mode "reset" --option_model_use_gui true --max_num_steps_option_rollout 10000 --pybullet_packing_task_info "$RESULTS_DIR/localized_block_info.pkl"
fi

if ! test -f $RESULTS_DIR/joint_angles.pkl; then
	echo "CONVERTING THE STATE TRAJECTORY TO JOINT ANGLES"
	python3.9 experiments/scripts/real_robot/convert_trajectory.py $RESULTS_DIR/*.traj $RESULTS_DIR/joint_angles.pkl
fi

echo "EXECUTING THE TRAJECTORY"
python3.9 experiments/scripts/real_robot/run_trajectory.py $RESULTS_DIR/joint_angles.pkl
