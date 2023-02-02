#!/bin/bash

set -e

# Constants.
DATA_DIR="blocks_vision_data"
TASK_NUM="0"
SEED="0"
IMG_SUFFIX="6blocks.png"
VIZ_PLANNING="True"
TASK_DIR="${DATA_DIR}/tasks"

# Set up file paths.
TASK_FILE="${TASK_DIR}/blocks-vision-task${TASK_NUM}.json"
EVAL_TRAJ_FILE="eval_trajectories/pybullet_blocks__oracle__${SEED}________task1.traj"
LISDF_PLAN_FILE="/tmp/pybullet_blocks__oracle__${SEED}________task1.json"
FINAL_PLAN_FILE="/tmp/file_plan.json"

# Start the pipeline.
mkdir -p $TASK_DIR

echo "Capturing images."
python scripts/realsense_helpers.py \
        --rgb $DATA_DIR/color-$TASK_NUM-$IMG_SUFFIX \
        --depth $DATA_DIR/depth-$TASK_NUM-$IMG_SUFFIX

echo "Running perception."
python scripts/run_blocks_perception.py \
        --rgb $DATA_DIR/color-$TASK_NUM-$IMG_SUFFIX \
        --depth $DATA_DIR/depth-$TASK_NUM-$IMG_SUFFIX \
        --goal $DATA_DIR/goal-$TASK_NUM.json \
        --extrinsics $DATA_DIR/extrinsics.json \
        --intrinsics $DATA_DIR/intrinsics.json \
        --output $TASK_FILE  # --debug_viz

echo "Running planning with oracle models."
python predicators/main.py --env pybullet_blocks --approach oracle \
    --seed $SEED --num_test_tasks 1 \
    --test_task_json_dir $TASK_DIR \
    --pybullet_robot panda \
    --option_model_use_gui $VIZ_PLANNING \
    --option_model_name oracle --option_model_terminate_on_repeat False \
    --blocks_block_size 0.0505 \
    --sesame_check_static_object_changes True \
    --crash_on_failure \
    --timeout 100 \
    --blocks_num_blocks_test [15] # just needs to be an upper bound

echo "Converting plan to LISDF."
python scripts/eval_trajectory_to_lisdf.py \
        --input $EVAL_TRAJ_FILE \
        --output $LISDF_PLAN_FILE

echo "Planning to reset the robot."
python scripts/lisdf_plan_to_reset.py \
        --lisdf $LISDF_PLAN_FILE \
        --output $FINAL_PLAN_FILE

echo "Visualizing LISDF plan."
python scripts/lisdf_pybullet_visualizer.py --lisdf $FINAL_PLAN_FILE

echo "To execute the LISDF plan on the real robot, run this command:"
echo "panda-client execute_lisdf_plan ${FINAL_PLAN_FILE}"
