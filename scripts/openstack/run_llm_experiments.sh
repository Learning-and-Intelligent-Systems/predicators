#!/bin/bash

# General settings.
FLAGS="--num_train_tasks 5 --num_test_tasks 10 --strips_learner oracle --timeout 100"

# Approach settings.

# Oracle planning approach.
APPROACH_EXPERIMENT_ID="oracle"
APPROACH="oracle"
APPROACH_FLAGS=""

# # GNN option policy approach.
# APPROACH_EXPERIMENT_ID="gnn_mf"
# APPROACH="gnn_option_policy"
# APPROACH_FLAGS="--gnn_option_policy_solve_with_shooting False"

# # LLM approach (coming soon).


# Environment settings.

# Easy blocks environment.
ENV_EXPERIMENT_ID="easy-blocks"
ENV="pddl_blocks_procedural_tasks"
ENV_FLAGS="--pddl_blocks_procedural_train_min_num_blocks 3 \
    --pddl_blocks_procedural_train_max_num_blocks 4 \
    --pddl_blocks_procedural_train_min_num_blocks_goal 2 \
    --pddl_blocks_procedural_train_max_num_blocks_goal 3 \
    --pddl_blocks_procedural_test_min_num_blocks 5 \
    --pddl_blocks_procedural_test_max_num_blocks 6 \
    --pddl_blocks_procedural_test_min_num_blocks_goal 2 \
    --pddl_blocks_procedural_test_max_num_blocks_goal 5"

# # Medium blocks environment.
# ENV_EXPERIMENT_ID="medium-blocks"
# ENV="pddl_blocks_procedural_tasks"
# ENV_FLAGS="--pddl_blocks_procedural_train_min_num_blocks 7 \
#     --pddl_blocks_procedural_train_max_num_blocks 8 \
#     --pddl_blocks_procedural_train_min_num_blocks_goal 6 \
#     --pddl_blocks_procedural_train_max_num_blocks_goal 7 \
#     --pddl_blocks_procedural_test_min_num_blocks 9 \
#     --pddl_blocks_procedural_test_max_num_blocks 10 \
#     --pddl_blocks_procedural_test_min_num_blocks_goal 6 \
#     --pddl_blocks_procedural_test_max_num_blocks_goal 9"

# # Hard blocks environment.
# ENV_EXPERIMENT_ID="hard-blocks"
# ENV="pddl_blocks_procedural_tasks"
# ENV_FLAGS="--pddl_blocks_procedural_train_min_num_blocks 11 \
#     --pddl_blocks_procedural_train_max_num_blocks 12 \
#     --pddl_blocks_procedural_train_min_num_blocks_goal 10 \
#     --pddl_blocks_procedural_train_max_num_blocks_goal 11 \
#     --pddl_blocks_procedural_test_min_num_blocks 13 \
#     --pddl_blocks_procedural_test_max_num_blocks 14 \
#     --pddl_blocks_procedural_test_min_num_blocks_goal 10 \
#     --pddl_blocks_procedural_test_max_num_blocks_goal 13"

# # Easy delivery environment.
# ENV_EXPERIMENT_ID="easy-delivery"
# ENV="pddl_delivery_procedural_tasks"
# ENV_FLAGS="--pddl_delivery_procedural_train_min_num_locs 3 \
#     --pddl_delivery_procedural_train_max_num_locs 5 \
#     --pddl_delivery_procedural_train_min_want_locs 1 \
#     --pddl_delivery_procedural_train_max_want_locs 2 \
#     --pddl_delivery_procedural_test_min_num_locs 4 \
#     --pddl_delivery_procedural_test_max_num_locs 6 \
#     --pddl_delivery_procedural_test_min_want_locs 2 \
#     --pddl_delivery_procedural_test_max_want_locs 3 \
#     --pddl_delivery_procedural_test_max_extra_newspapers 1"

# # Medium delivery environment.
# ENV_EXPERIMENT_ID="medium-delivery"
# ENV="pddl_delivery_procedural_tasks"
# ENV_FLAGS="--pddl_delivery_procedural_train_min_num_locs 4 \
#     --pddl_delivery_procedural_train_max_num_locs 7 \
#     --pddl_delivery_procedural_train_min_want_locs 1 \
#     --pddl_delivery_procedural_train_max_want_locs 3 \
#     --pddl_delivery_procedural_test_min_num_locs 17 \
#     --pddl_delivery_procedural_test_max_num_locs 23 \
#     --pddl_delivery_procedural_test_min_want_locs 11 \
#     --pddl_delivery_procedural_test_max_want_locs 16 \
#     --pddl_delivery_procedural_test_max_extra_newspapers 5"

# # Hard delivery environment.
# ENV_EXPERIMENT_ID="hard-delivery"
# ENV="pddl_delivery_procedural_tasks"
# ENV_FLAGS="--pddl_delivery_procedural_train_min_num_locs 5 \
#     --pddl_delivery_procedural_train_max_num_locs 10 \
#     --pddl_delivery_procedural_train_min_want_locs 2 \
#     --pddl_delivery_procedural_train_max_want_locs 4 \
#     --pddl_delivery_procedural_test_min_num_locs 31 \
#     --pddl_delivery_procedural_test_max_num_locs 40 \
#     --pddl_delivery_procedural_test_min_want_locs 20 \
#     --pddl_delivery_procedural_test_max_want_locs 30 \
#     --pddl_delivery_procedural_test_max_extra_newspapers 10"

# Set this to the private key that has access to the machines.
SSH_PRIVATE_KEY_PATH="~/.ssh/cloud.key"

# Set this to the directory where downloaded results should get saved.
DOWNLOAD_DIR="/Users/tom/Dropbox/varun_llm/openstack"

# This script parallelizes over seeds, and assumes that the number of machines
# is equal to the desired number of seeds.
START_SEED=456
MACHINES=(
    "128.52.139.188"
    "128.52.139.191"
    "128.52.139.192"
    "128.52.139.193"
    "128.52.139.194"
)

# The main command (without the seed specified).
EXPERIMENT_ID="${ENV_EXPERIMENT_ID}-${APPROACH_EXPERIMENT_ID}"
CMD="python3.8 src/main.py --env ${ENV} --approach ${APPROACH} \
    --experiment_id $EXPERIMENT_ID \
    ${FLAGS} ${ENV_FLAGS} ${APPROACH_FLAGS} --debug"

# Run this script with either "launch" or "download" as the first argument.
if [[ $1 != "launch" && $1 != "download" ]]; then
echo "ERROR: Run this script with either 'launch' or 'download'."
exit
fi

# The main loop.
SEED=$START_SEED
for MACHINE in ${MACHINES[@]}; do

HOST=ubuntu@${MACHINE}

# Launch command.
if [ $1 == "launch" ]; then

# SSH into the machine.
SSH_CMD="ssh -tt -i $SSH_PRIVATE_KEY_PATH -o StrictHostKeyChecking=no $HOST"
echo "SSHing into ${MACHINE}"
$SSH_CMD << EOF

# Prepare the machine.
cd ~/predicators
mkdir -p logs
git checkout master
git pull

# Remove old results.
rm -f results/* logs/* saved_approaches/* saved_datasets/*

# Run the main command in the background and write out to a log.
$CMD --seed $SEED &> logs/${ENV}__${APPROACH}__${EXPERIMENT_ID}__${SEED}.log &

# Exit.
exit

EOF

# Download command.
else

# SCP the results, logs, saved_approaches, and saved_datasets.
SAVE_DIRS=(
    "results"
    "logs"
    "saved_datasets"
    "saved_approaches"
)
for SAVE_DIR in ${SAVE_DIRS[@]}; do

SCP_CMD="scp -r -i $SSH_PRIVATE_KEY_PATH -o StrictHostKeyChecking=no"
echo "SCPing from ${MACHINE} to ${DOWNLOAD_DIR}"
mkdir -p "${DOWNLOAD_DIR}/${SAVE_DIR}"
$SCP_CMD "${HOST}:~/predicators/${SAVE_DIR}/*" "${DOWNLOAD_DIR}/${SAVE_DIR}"

done

fi

# Increment the seed.
SEED=$(($SEED+1))

done

echo "Finished $1 for experiment: ${EXPERIMENT_ID}."
