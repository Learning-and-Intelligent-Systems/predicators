import glob
import json
import os
import random
from collections import Counter
random.seed(0)

learnable_operators = ['Open', 'Close', 'ToggleOn', 'Cook-toggleable', 'Cook-notToggleable', 'Freeze-toggleable', 'Freeze-notToggleable', 'Soak-toggleable', 'Soak-notToggleable', 'CleanDusty', 'CleanStained', 'Slice']
jsonable_dict = {}
for operator in learnable_operators:
    tasks_for_operator = Counter()
    for fname in glob.glob('behavior2d_skeletons/skeletons/*/skeleton.txt'):
        print(fname)
        with open(fname, 'r') as f:
            for line in f:
                if line.startswith(operator):
                    task_name = fname.split('/')[-2]
                    tasks_for_operator[task_name] += 1
    jsonable_dict[operator] = tasks_for_operator

print(json.dumps(jsonable_dict, indent=4))
for operator, task_list in jsonable_dict.items():
    print('-',operator, ':', len(task_list))
    if len(task_list) <= 1:
        continue
    save_dirname = os.path.join('init_states_operator_tasks', operator)
    os.makedirs(save_dirname, exist_ok=True)
    for task, operator_count in task_list.items():
        load_state_fname_pattern = os.path.join('init_states', task, '*_state.json')
        load_state_path = random.choice(glob.glob(load_state_fname_pattern))
        load_goal_path = load_state_path.rstrip('_state.json') + '_goal.json'
        load_objects_path = os.path.join('init_states', task, 'objects.json')

        load_state_basename = os.path.basename(load_state_path)
        load_goal_basename = os.path.basename(load_goal_path)
        load_objects_basename = os.path.basename(load_state_path.rstrip('_state.json') + '_objects.json')
        
        save_state_fname = os.path.join(save_dirname, f'operator_count_{operator_count}_{task}_{load_state_basename}')
        save_goal_fname = os.path.join(save_dirname, f'operator_count_{operator_count}_{task}_{load_goal_basename}')
        save_objects_fname = os.path.join(save_dirname, f'operator_count_{operator_count}_{task}_{load_objects_basename}')

        os.system(f'cp {load_state_path} {save_state_fname}')
        os.system(f'cp {load_goal_path} {save_goal_fname}')
        os.system(f'cp {load_objects_path} {save_objects_fname}')

with open('tasks_for_operator.json', 'w') as f:
    json.dump(jsonable_dict, f, indent=4)

