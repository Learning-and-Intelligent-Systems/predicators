#!/bin/bash

EXEC=python # python or profile or profile2

START_SEED=123
NUM_SEEDS=3
ENVS=(
    "cover"
    "blocks"
    "painting"
    "repeated_nextto"
    "cluttered_table"
)
APPROACHES=(
    "oracle"
    "nsrt_learning"
)

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    for ENV in ${ENVS[@]}; do
        for APPROACH in ${APPROACHES[@]}; do
            python analysis/submit.py --env $ENV --approach $APPROACH --seed $SEED
        done
    done
done
