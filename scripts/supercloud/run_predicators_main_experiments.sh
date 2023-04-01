#!/bin/bash

FILE="scripts/supercloud/submit_supercloud_job.py"
NUM_TRAIN_TASKS="200"
ALL_ENVS=(
    "cover"
    "pybullet_blocks"
    "painting"
    "tools"
)

for ENV in ${ALL_ENVS[@]}; do
    # Main approach.
    python $FILE --experiment_id ${ENV}_main_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS

    # # Ablations.
    # # Note: downrefeval is main but with --sesame_max_skeletons_optimized 1 during evaluation only. We can only run this using `--load_approach` since we don't allow grammar_search_expected_nodes_max_skeletons to be greater than sesame_max_skeletons_optimized during invention.
    # python $FILE --experiment_id ${ENV}_downrefscore_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --grammar_search_expected_nodes_max_skeletons 1 --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS
    # python $FILE --experiment_id ${ENV}_noinventallexclude_${NUM_TRAIN_TASKS}demo --env $ENV --approach nsrt_learning --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS
    # python $FILE --experiment_id ${ENV}_noinventnoexclude_${NUM_TRAIN_TASKS}demo --env $ENV --approach nsrt_learning --num_train_tasks $NUM_TRAIN_TASKS

    # # Score function baselines.
    # python $FILE --experiment_id ${ENV}_prederror_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --grammar_search_score_function prediction_error --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS
    # python $FILE --experiment_id ${ENV}_branchfac_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --grammar_search_score_function branching_factor --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS
    # python $FILE --experiment_id ${ENV}_energy_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --grammar_search_score_function lmcut_energy_lookaheaddepth0 --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS

    # # Other baselines.
    # python $FILE --experiment_id ${ENV}_random --env $ENV --approach random_options --num_train_tasks 0
    # python $FILE --experiment_id ${ENV}_gnn_shooting_${NUM_TRAIN_TASKS}demo --env $ENV --approach gnn_option_policy --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS

    # # Blocks-specific experiments.
    # if [ $ENV = "blocks" ] || [ $ENV = "pybullet_blocks" ]; then
    #     python $FILE --experiment_id ${ENV}_mainhadd_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --sesame_task_planning_heuristic hadd --offline_data_task_planning_heuristic lmcut --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS
    #     python $FILE --experiment_id ${ENV}_noinventnoexcludehadd_${NUM_TRAIN_TASKS}demo --env $ENV --approach nsrt_learning --sesame_task_planning_heuristic hadd --offline_data_task_planning_heuristic lmcut --num_train_tasks $NUM_TRAIN_TASKS
    # fi
done
