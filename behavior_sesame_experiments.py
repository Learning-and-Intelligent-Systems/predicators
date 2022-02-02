import json
import os

test_tasks_wb = [
    "sorting_books",
    "re-shelving_library_books",
    "putting_leftovers_away",
    "putting_dishes_away_after_cleaning",
    "putting_away_toys",
    "packing_picnics",
    "packing_lunches",
    "packing_food_for_work",
    "packing_car_for_trip",
    "packing_boxes_for_household_move_or_trip",
    "packing_bags_or_suitcase",
    "packing_adult_s_bags",
    "organizing_file_cabinet",
    "organizing_boxes_in_garage",
    "opening_presents",
]

test_tasks_nk = [
    "opening_packages",
    "throwing_away_leftovers",
    "setting_up_candles",
    "picking_up_trash",
    "picking_up_take-out_food",
    "loading_the_dishwasher",
    "laying_tile_floors",
    "filling_an_Easter_basket",
    "filling_a_Christmas_stocking",
    "collecting_aluminum_cans",
    "collect_misplaced_items",
    "bringing_in_wood",
    "boxing_books_up_for_storage",
    "assembling_gift_baskets",
]

num_test_tasks = 5
timeout = 2000
experiment_id = None
test_tasks = test_tasks_nk + test_tasks_wb

json_file = open('scene_tasks.json', 'r')
scene_tasks = json.load(json_file)

for scene, tasks in scene_tasks.items():
    for task in tasks:
        if task in test_tasks:
            for online_sampling in ["True", "False"]:
                experiment_id = scene + "_" + task
                print(scene, task, "running...")
                cmd = f"python src/main.py --experiment_id {experiment_id} --env behavior --approach oracle --seed 0 --timeout {timeout} --max_samples_per_step 100 --behavior_mode simple --max_num_steps_check_policy 1000 --option_model_name behavior_oracle --num_train_tasks 2 --num_test_tasks {num_test_tasks} --behavior_randomize_init_state {online_sampling} --behavior_scene_name {scene} --behavior_task_name {task}"
                filename = "outputs/output_" + scene + "_" + task + "_" + online_sampling + ".txt"
                os.system(f'{cmd} 2>&1 | tee {filename}')

# grep -l "ValueError: No predicate found for name" * | xargs rm
