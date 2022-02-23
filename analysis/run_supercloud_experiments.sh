#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # cover
    python $FILE --experiment_id cover_oracle --env cover --approach oracle --seed $SEED
    python $FILE --experiment_id cover_noinvent_noexclude --env cover --approach nsrt_learning --seed $SEED --num_train_tasks 50
    # python $FILE --experiment_id cover_noinvent_allexclude --env cover --approach nsrt_learning --excluded_predicates all --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id cover_invent_noexclude --env cover --approach grammar_search_invention --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id cover_invent_allexclude --env cover --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks 50

    # blocks
    python $FILE --experiment_id blocks_oracle --env blocks --approach oracle --seed $SEED
    python $FILE --experiment_id blocks_noinvent_noexclude --env blocks --approach nsrt_learning --seed $SEED --num_train_tasks 50
    # python $FILE --experiment_id blocks_noinvent_allexclude --env blocks --approach nsrt_learning --excluded_predicates all --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id blocks_invent_noexclude --env blocks --approach grammar_search_invention --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id blocks_invent_allexclude --env blocks --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks 50

    # painting
    python $FILE --experiment_id painting_oracle --env painting --approach oracle --seed $SEED
    python $FILE --experiment_id painting_noinvent_noexclude --env painting --approach nsrt_learning --seed $SEED --num_train_tasks 50
    # python $FILE --experiment_id painting_noinvent_allexclude --env painting --approach nsrt_learning --excluded_predicates all --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id painting_invent_noexclude --env painting --approach grammar_search_invention --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id painting_invent_allexclude --env painting --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks 50

    # tools
    python $FILE --experiment_id tools_oracle --env tools --approach oracle --seed $SEED
    python $FILE --experiment_id tools_noinvent_noexclude --env tools --approach nsrt_learning --seed $SEED --num_train_tasks 200
    # python $FILE --experiment_id tools_noinvent_allexclude --env tools --approach nsrt_learning --excluded_predicates all --seed $SEED --num_train_tasks 200
    python $FILE --experiment_id tools_invent_noexclude --env tools --approach grammar_search_invention --seed $SEED --num_train_tasks 200
    python $FILE --experiment_id tools_invent_allexclude --env tools --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks 200

    # repeated_nextto
    python $FILE --experiment_id repeated_nextto_oracle --env repeated_nextto --approach oracle --seed $SEED
    python $FILE --experiment_id repeated_nextto_noinvent_noexclude --env repeated_nextto --approach nsrt_learning --learn_side_predicates True --seed $SEED --num_train_tasks 50
    # python $FILE --experiment_id repeated_nextto_noinvent_allexclude --env repeated_nextto --approach nsrt_learning --learn_side_predicates True --excluded_predicates all --seed $SEED --num_train_tasks 50
done
