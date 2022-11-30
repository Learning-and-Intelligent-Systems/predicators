#!/bin/bash

set -e

DATA_DIR="blocks_vision_data"
TASK_NUM="0"
SEED="0"
IMG_SUFFIX="6blocks.png"
TASK_DIR=$DATA_DIR/tasks
OUTPUT=$TASK_DIR/blocks-vision-task$TASK_NUM.json
VIZ_PLANNING="True"

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
        --output $OUTPUT

echo "Running planning with oracle models."
python predicators/main.py --env pybullet_blocks --approach oracle \
    --seed $SEED --num_test_tasks 1 \
    --blocks_test_task_json_dir $TASK_DIR \
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
        --input eval_trajectories/pybullet_blocks__oracle__${SEED}________task1.traj \
        --output /tmp/pybullet_blocks__oracle__${SEED}________task1.json

echo "Planning to reset the robot."
python scripts/lisdf_plan_to_reset.py \
        --lisdf /tmp/pybullet_blocks__oracle__${SEED}________task1.json \
        --output /tmp/final_plan.json

echo "Visualizing LISDF plan."
python scripts/lisdf_pybullet_visualizer.py --lisdf /tmp/final_plan.json

echo "To execute the LISDF plan on the real robot, run this command:"
echo "panda-client execute_lisdf_plan /tmp/final_plan.json"
