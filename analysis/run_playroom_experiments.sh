#!/bin/bash

START_SEED=456
NUM_SEEDS=5
FILE="analysis/submit.py"
MAX_RATIO=1.0
MIN_RATIO=0.8
INCREMENT=-0.1

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    for RATIO in $(seq $MAX_RATIO $INCREMENT $MIN_RATIO); do
        # exclude On
        python $FILE --env playroom --approach interactive_learning --seed $SEED --experiment_id exclude_On_$RATIO --teacher_dataset_label_ratio $RATIO --interactive_known_predicates GripperOpen,Holding,Clear,NextToTable,NextToDoor,NextToDial,InRegion,Borders,Connects,IsBoringRoom,IsPlayroom,IsBoringRoomDoor,IsPlayroomDoor,DoorOpen,DoorClosed,LightOn,LightOff,OnTable
        # exclude OnTable
        python $FILE --env playroom --approach interactive_learning --seed $SEED --experiment_id exclude_OnTable_$RATIO --teacher_dataset_label_ratio $RATIO --interactive_known_predicates GripperOpen,Holding,Clear,NextToTable,NextToDoor,NextToDial,InRegion,Borders,Connects,IsBoringRoom,IsPlayroom,IsBoringRoomDoor,IsPlayroomDoor,DoorOpen,DoorClosed,LightOn,LightOff,On
        # exclude LightOn
        python $FILE --env playroom --approach interactive_learning --seed $SEED --experiment_id exclude_LightOn_$RATIO --teacher_dataset_label_ratio $RATIO --interactive_known_predicates GripperOpen,Holding,Clear,NextToTable,NextToDoor,NextToDial,InRegion,Borders,Connects,IsBoringRoom,IsPlayroom,IsBoringRoomDoor,IsPlayroomDoor,DoorOpen,DoorClosed,LightOff,On,OnTable
        # exclude LightOff
        python $FILE --env playroom --approach interactive_learning --seed $SEED --experiment_id exclude_LightOff_$RATIO --teacher_dataset_label_ratio $RATIO --interactive_known_predicates GripperOpen,Holding,Clear,NextToTable,NextToDoor,NextToDial,InRegion,Borders,Connects,IsBoringRoom,IsPlayroom,IsBoringRoomDoor,IsPlayroomDoor,DoorOpen,DoorClosed,LightOn,On,OnTable
        # exclude all four goal predicates
        python $FILE --env playroom --approach interactive_learning --seed $SEED --experiment_id exclude_all_goals_$RATIO --teacher_dataset_label_ratio $RATIO --interactive_known_predicates GripperOpen,Holding,Clear,NextToTable,NextToDoor,NextToDial,InRegion,Borders,Connects,IsBoringRoom,IsPlayroom,IsBoringRoomDoor,IsPlayroomDoor,DoorOpen,DoorClosed
    done
done
