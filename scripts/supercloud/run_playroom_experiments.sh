#!/bin/bash

FILE="scripts/supercloud/submit_supercloud_job.py"
MAX_RATIO=1.0
MIN_RATIO=0.8
INCREMENT=-0.1

for RATIO in $(seq $MAX_RATIO $INCREMENT $MIN_RATIO); do
    # exclude On
    python $FILE --env playroom --approach interactive_learning --experiment_id exclude_On_$RATIO --teacher_dataset_label_ratio $RATIO --excluded_predicates On
    # exclude OnTable
    python $FILE --env playroom --approach interactive_learning --experiment_id exclude_OnTable_$RATIO --teacher_dataset_label_ratio $RATIO --excluded_predicates OnTable
    # exclude LightOn
    python $FILE --env playroom --approach interactive_learning --experiment_id exclude_LightOn_$RATIO --teacher_dataset_label_ratio $RATIO --excluded_predicates LightOn
    # exclude LightOff
    python $FILE --env playroom --approach interactive_learning --experiment_id exclude_LightOff_$RATIO --teacher_dataset_label_ratio $RATIO --excluded_predicates LightOff
    # exclude all four goal predicates
    python $FILE --env playroom --approach interactive_learning --experiment_id exclude_all_goals_$RATIO --teacher_dataset_label_ratio $RATIO --excluded_predicates On,OnTable,LightOn,LightOff
done
