#!/bin/bash

# Replace this with the flags you want to test.
FLAGS="--approach oracle --env blocks --timeout 0.1"
START_SEED=0
NUM_SEEDS=100

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    if python src/main.py $FLAGS --num_test_tasks 1 --seed $SEED | grep -q 'Tasks solved: 0'
    then
        echo "Found failing seed: ${SEED}."
        break
    else
        echo "Seed ${SEED} did not fail."
    fi
done
