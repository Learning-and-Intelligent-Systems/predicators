#!/bin/bash

# Set this to the private key that has access to the machines.
SSH_PRIVATE_KEY_PATH="~/.ssh/cloud.key"

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
CMD="python3.8 src/main.py --env cover --approach oracle"

# The main loop.
SEED=$START_SEED
for MACHINE in ${MACHINES[@]}; do

# SSH into the machine.
echo "SSHing into ${MACHINE}"
ssh -tt -i $SSH_PRIVATE_KEY_PATH -o StrictHostKeyChecking=no ubuntu@${MACHINE} << EOF

# Prepare the machine.
cd ~/predicators
git checkout master
git pull

# Remove old results.
rm -f results/* logs/* saved_approaches/* saved_datasets/*

# Run the main command in the background.
$CMD --seed $SEED &

# Exit.
exit

EOF

# Increment the seed.
SEED=$(($SEED+1))

done
