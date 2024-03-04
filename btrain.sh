#!/bin/bash

# Define the environments
# environments=("doors")
environments=("doors" "coffee")
# environments=("cover_multistep_options" "doors" "stick_button" "coffee")

# Define the number of training tasks
# num_train_tasks=(50)
num_train_tasks=(50 200 500 750 1000)

# segmenter
# segmenter = "option_changes"
segmenter = "contact"

# Loop over the environments
for env in "${environments[@]}"; do
    # Loop over the number of training tasks
    for num in "${num_train_tasks[@]}"; do
        sbatch -J "oc_${env}_${num}_+" execute.sh python predicators/main.py \
            --env $env --num_train_tasks $num --seed 0 \
            --experiment_id "${num}_${segmenter}" \
            --approach grammar_search_invention --option_learner no_learning \
            --min_perc_data_for_nsrt 1 --segmenter $segmenter \
            --timeout 300 --excluded_predicates all 
    done
done