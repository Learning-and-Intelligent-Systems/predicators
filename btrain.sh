#!/bin/bash

# Define the environments
# environments=("cover_multistep_options")
environments=("cover_multistep_options" "doors" "stick_button" "coffee")

# Define the number of training tasks
num_train_tasks=(50 200 500 750 1000)

# Loop over the environments
for env in "${environments[@]}"; do
    # Loop over the number of training tasks
    for num in "${num_train_tasks[@]}"; do
        sbatch -J "${env}_${num}" execute.sh python predicators/main.py \
            --env $env --num_train_tasks $num --seed 0 --experiment_id $num\
            --approach grammar_search_invention --option_learner direct_bc \
            --min_perc_data_for_nsrt 1 --segmenter contacts --timeout 300 \
            --excluded_predicates all
    done
done