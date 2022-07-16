#!/bin/bash

# Experimental settings.
EXPERIMENT_ID="test"
ENV="blocks"
APPROACH="nsrt_learning"
OTHER_FLAGS=""

# Run this script with either "launch" or "download" as the first argument.
if [[ $1 != "launch" && $1 != "download" ]]; then
echo "ERROR: Run this script with either 'launch' or 'download'."
exit
fi

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
CMD="python3.8 src/main.py --env ${ENV} --approach ${APPROACH} --experiment_id ${EXPERIMENT_ID} ${OTHER_FLAGS}"

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
