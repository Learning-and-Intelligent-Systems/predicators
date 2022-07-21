#!/bin/bash

# To run this file, leave one approach section, one experiment section, and one
# machine group uncommented, with the rest commented out.

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

# # LLM standard approach.
# APPROACH_EXPERIMENT_ID="llm_standard"
# APPROACH="open_loop_llm"
# APPROACH_FLAGS="--open_loop_llm_model_name text-davinci-002"

# # LLM multi-response approach.
# APPROACH_EXPERIMENT_ID="llm_multi"
# APPROACH="open_loop_llm"
# APPROACH_FLAGS="--open_loop_llm_model_name text-davinci-002 \
#     --open_loop_llm_num_completions 5"


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

# # Easy spanner environment.
# ENV_EXPERIMENT_ID="easy-spanner"
# ENV="pddl_spanner_procedural_tasks"
# ENV_FLAGS="--pddl_spanner_procedural_train_min_nuts 1 \
#     --pddl_spanner_procedural_train_max_nuts 2 \
#     --pddl_spanner_procedural_train_min_extra_spanners 0 \
#     --pddl_spanner_procedural_train_max_extra_spanners 2 \
#     --pddl_spanner_procedural_train_min_locs 1 \
#     --pddl_spanner_procedural_train_max_locs 3 \
#     --pddl_spanner_procedural_test_min_nuts 2 \
#     --pddl_spanner_procedural_test_max_nuts 4 \
#     --pddl_spanner_procedural_test_min_extra_spanners 0 \
#     --pddl_spanner_procedural_test_max_extra_spanners 3 \
#     --pddl_spanner_procedural_test_min_locs 2 \
#     --pddl_spanner_procedural_test_max_locs 4"

# # Medium spanner environment.
# ENV_EXPERIMENT_ID="medium-spanner"
# ENV="pddl_spanner_procedural_tasks"
# ENV_FLAGS="--pddl_spanner_procedural_train_min_nuts 1 \
#     --pddl_spanner_procedural_train_max_nuts 3 \
#     --pddl_spanner_procedural_train_min_extra_spanners 0 \
#     --pddl_spanner_procedural_train_max_extra_spanners 2 \
#     --pddl_spanner_procedural_train_min_locs 2 \
#     --pddl_spanner_procedural_train_max_locs 4 \
#     --pddl_spanner_procedural_test_min_nuts 6 \
#     --pddl_spanner_procedural_test_max_nuts 7 \
#     --pddl_spanner_procedural_test_min_extra_spanners 0 \
#     --pddl_spanner_procedural_test_max_extra_spanners 5 \
#     --pddl_spanner_procedural_test_min_locs 5 \
#     --pddl_spanner_procedural_test_max_locs 6"


# Set this to the private key that has access to the machines.
SSH_PRIVATE_KEY_PATH="~/.ssh/cloud.key"

# Set this to the directory where downloaded results should get saved.
DOWNLOAD_DIR="/Users/tom/Dropbox/varun_llm/openstack"

# This script parallelizes over seeds, and assumes that the number of machines
# is equal to the desired number of seeds.
START_SEED=456
MACHINES=(
    # Group 1
    "128.52.139.188"
    "128.52.139.191"
    "128.52.139.192"
    "128.52.139.193"
    "128.52.139.194"

    # # Group 2
    # "128.52.139.195"
    # "128.52.139.196"
    # "128.52.139.197"
    # "128.52.139.198"
    # "128.52.139.199"

    # # Group 3
    # "128.52.139.2"
    # "128.52.139.201"
    # "128.52.139.202"
    # "128.52.139.203"
    # "128.52.139.205"

    # # Group 4
    # "128.52.139.206"
    # "128.52.139.207"
    # "128.52.139.208"
    # "128.52.139.209"
    # "128.52.139.210"

    # # Group 5
    # "128.52.139.215"
    # "128.52.139.216"
    # "128.52.139.217"
    # "128.52.139.218"
    # "128.52.139.222"

    # # Group 6
    # "128.52.139.223"
    # "128.52.139.224"
    # "128.52.139.226"
    # "128.52.139.227"
    # "128.52.139.228"

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

# SCP the results, logs, saved_approaches, saved_datasets, llm_cache.
SAVE_DIRS=(
    "results"
    "logs"
    "saved_datasets"
    "saved_approaches"
    # "llm_cache"
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

# Print final message.
if [ $1 == "launch" ]; then
echo "Finished launching experiment: ${EXPERIMENT_ID}."
else
echo "Finished downloading."
fi
