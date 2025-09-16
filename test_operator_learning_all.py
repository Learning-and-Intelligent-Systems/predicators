import numpy as np
from gym.spaces import Box
import re
import pickle as pkl

from predicators import utils
from predicators.nsrt_learning.nsrt_learning_main import learn_nsrts_from_data
from predicators.structs import Action, LowLevelTrajectory, Predicate, State, \
    Type, GroundAtom, Task
import glob

demo_files = sorted([filename for filename in glob.glob("/Users/shashlik/Documents/GitHub/predicators/demos/*/*")])
demo_tasks = set([demo_file.split("/")[-1].split("_")[0] for demo_file in demo_files])

utils.reset_config({
        "strips_learner": "pnad_search",
        "segmenter": "every_step",
        "disable_harmlessness_check": True,
        "pnad_search_load_initial": False,
        "min_data_for_nsrt": 0,
        "min_perc_data_for_nsrt": 0,
        "pnad_search_timeout":1000.0
    })

# Load and do this from MiniBeahvior Demo

def get_demo_traj(demo_file, verbose=True):
    with open(demo_file, 'rb') as f:
        data = pkl.load(f)

    last_skill = "Move"
    state = [a for a in data[1][1] if "infovofrobot" not in a]
    states = [state]
    actions = []
    for step in data.keys():
        obs = data[step][0]['image']
        direction = data[step][0]['direction']
        action = data[step][2]
        skill = None

        if "forward" in str(action) or \
            "left" in str(action) or \
            "right" in str(action):

            skill = "Move"
        else:
            skill = str(action)
        
        has_effect = True
        try:
            next_obs = data[step][3]['image']
            next_direction = data[step][3]['direction']
            if np.allclose(obs, next_obs) and (direction == next_direction):
                has_effect = False  
        except:
            pass

        if has_effect:
            if last_skill != skill:
                if verbose:
                    print("#")
                    print(last_skill)
                try:
                    next_state = [a for a in data[step][1] if "infovofrobot" not in a]
                    if verbose:
                        print("PREV:", set(state))
                        print("ADD:", set(next_state) - set(state))
                        print("DEL:", set(state) - set(next_state))
                    state = next_state
                    actions.append(last_skill)
                    states.append(state)
                except:
                    pass
                last_skill = skill
    else:
        if verbose:
            print("#")
            print(last_skill)
        next_state = [a for a in data[step][4] if "infovofrobot" not in a]
        if verbose:
            print("PREV:", set(state))
            print("ADD:", set(next_state) - set(state))
            print("DEL:", set(state) - set(next_state))
        state = next_state
        if verbose:
            print("#")
        actions.append(last_skill)
        states.append(state)
    
    return LowLevelTrajectory(states, actions, _is_demo=True, _train_task_idx=0)

def parse_objs_preds_and_options(trajectory, train_task_idx=0):
    objs = set()
    preds = set()
    options = set()
    state = None
    states = []
    actions = []
    ground_atoms_traj = []
    obj_type = Type("obj_type", ["is_obj"])
    
    for i, s in enumerate(trajectory.states):
        ground_atoms = set()
        for pred_str in s:
            pred = None
            choice = []
            pattern = re.compile(r"(\w+)\((.*?)\)")
            match = pattern.match(pred_str)
            if match:
                func_name = match.group(1)
                args = match.group(2).split(',') if match.group(2) else []
                for arg in args:
                    obj = obj_type(arg.strip())
                    choice.append(obj)
                    objs.add(obj)
                if len(args) == 1:
                    pred = Predicate(func_name, [obj_type], lambda s, o: True)
                    preds.add(pred)
                elif len(args) == 2:
                    pred = Predicate(func_name, [obj_type, obj_type], lambda s, o: True)
                    preds.add(pred)
                else:
                    NotImplementedError("")
            ground_atoms.add(GroundAtom(pred, choice))
        states.append(state)
        ground_atoms_traj.append(ground_atoms)

        if i < len(trajectory.actions):
            a_name = trajectory.actions[i]
            name_to_actions = actions_dict = {
                "Move": 0,
                "Actions.pickup_0": 3,
                "Actions.pickup_1": 4,
                "Actions.pickup_2": 5,
                "Actions.drop_0": 6,
                "Actions.drop_1": 7,
                "Actions.drop_2": 8,
                "Actions.drop_in": 9,
                "Actions.toggle": 10,
                "Actions.close": 11,
                "Actions.open": 12,
                "Actions.cook": 13,
                "Actions.slice": 14
            }

            param_option = utils.SingletonParameterizedOption(
                a_name, lambda s, m, o, p: Action(name_to_actions[a_name]))
            options.add(param_option)
            option = param_option.ground([], [])
            action = option.policy(state)
            action.set_option(option)
            actions.append(action)

    return objs, preds, options, (LowLevelTrajectory([{obj:[0.0] for obj in objs} for _ in states], actions, _is_demo=True, _train_task_idx=train_task_idx), ground_atoms_traj)


# dataset = []
# ground_atom_dataset = []
# tasks = []
# action_space = Box(0, 7, (1, ))

# task_name = "MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0"
# for demo_file in demo_files:
#     if task_name in demo_file:
#         demo_traj = get_demo_traj(demo_file=demo_file, verbose=False)

#         idx = len(dataset)
#         demo_traj = LowLevelTrajectory(demo_traj.states[0:5], demo_traj.actions[0:4], _is_demo=True, _train_task_idx=idx)

#         dataset += [demo_traj]
#         objs, preds, options, ground_atoms_traj = parse_objs_preds_and_options(demo_traj, train_task_idx=idx)
#         ground_atom_dataset += [ground_atoms_traj]
#         goal = set([atom for atom in ground_atoms_traj[1][-1] if "inside(" in str(atom)])
#         tasks += [Task(State({}, None), goal)]

# print("#"*30)
# print(task_name)
# print("#"*30)

# nsrts, _, _ = learn_nsrts_from_data(dataset, tasks,
#                                     preds,
#                                     options,
#                                     action_space,
#                                     ground_atom_dataset,
#                                     sampler_learner="neural",
#                                     annotations=None)

# assert len(nsrts) == 3
# import ipdb; ipdb.set_trace()
# quit()


# dataset = []
# ground_atom_dataset = []
# tasks = []
# action_space = Box(0, 7, (1, ))

# task_name = "MiniGrid-CollectMisplacedItems-16x16-N2-v0"
# for demo_file in demo_files:
#     if task_name in demo_file:
#         demo_traj = get_demo_traj(demo_file=demo_file, verbose=False)

#         idx = len(dataset)
#         demo_traj = LowLevelTrajectory(demo_traj.states[0:5], demo_traj.actions[0:4], _is_demo=True, _train_task_idx=idx)

#         dataset += [demo_traj]
#         objs, preds, options, ground_atoms_traj = parse_objs_preds_and_options(demo_traj, train_task_idx=idx)
#         ground_atom_dataset += [ground_atoms_traj]
#         goal = set([atom for atom in ground_atoms_traj[1][-1] if "onTop(" in str(atom) and "table_1" in str(atom)])
#         tasks += [Task(State({}, None), goal)]

# print("#"*30)
# print(task_name)
# print("#"*30)

# nsrts, _, _ = learn_nsrts_from_data(dataset, tasks,
#                                     preds,
#                                     options,
#                                     action_space,
#                                     ground_atom_dataset,
#                                     sampler_learner="neural",
#                                     annotations=None)

# assert len(nsrts) == 3

# import ipdb; ipdb.set_trace()

# dataset = []
# ground_atom_dataset = []
# tasks = []
# action_space = Box(0, 7, (1, ))
# all_options = set()

# task_name = "MiniGrid-SortingBooks-16x16-N2-v0"
# for demo_file in demo_files:
#     if task_name in demo_file:
#         demo_traj = get_demo_traj(demo_file=demo_file, verbose=False)

#         idx = len(dataset)
#         demo_traj = LowLevelTrajectory(demo_traj.states[0:5], demo_traj.actions[0:4], _is_demo=True, _train_task_idx=idx)

#         dataset += [demo_traj]
#         objs, preds, options, ground_atoms_traj = parse_objs_preds_and_options(demo_traj, train_task_idx=idx)
#         all_options = all_options | options
#         ground_atom_dataset += [ground_atoms_traj]
#         goal = set([atom for atom in ground_atoms_traj[1][-1] if "onTop(" in str(atom) and "shelf" in str(atom)])
#         tasks += [Task(State({}, None), goal)]

# print("#"*30)
# print(task_name)
# print("#"*30)

# nsrts, _, _ = learn_nsrts_from_data(dataset, tasks,
#                                     preds,
#                                     all_options,
#                                     action_space,
#                                     ground_atom_dataset,
#                                     sampler_learner="neural",
#                                     annotations=None)

# import ipdb; ipdb.set_trace()
# # assert len(nsrts) == 3

# dataset = []
# ground_atom_dataset = []
# tasks = []
# action_space = Box(0, 7, (1, ))

# task_name = "MiniGrid-ThrowingAwayLeftovers-16x16-N2-v0"
# for demo_file in demo_files:
#     if task_name in demo_file:
#         demo_traj = get_demo_traj(demo_file=demo_file, verbose=False)

#         idx = len(dataset)
#         demo_traj = LowLevelTrajectory(demo_traj.states[0:5], demo_traj.actions[0:4], _is_demo=True, _train_task_idx=idx)

#         dataset += [demo_traj]
#         objs, preds, options, ground_atoms_traj = parse_objs_preds_and_options(demo_traj, train_task_idx=idx)
#         ground_atom_dataset += [ground_atoms_traj]
#         goal = set([atom for atom in ground_atoms_traj[1][-1] if "inside(" in str(atom)])
#         tasks += [Task(State({}, None), goal)]

# print("#"*30)
# print(task_name)
# print("#"*30)

# nsrts, _, _ = learn_nsrts_from_data(dataset, tasks,
#                                     preds,
#                                     options,
#                                     action_space,
#                                     ground_atom_dataset,
#                                     sampler_learner="neural",
#                                     annotations=None)

# # assert len(nsrts) == 3


# dataset = []
# ground_atom_dataset = []
# tasks = []
# action_space = Box(0, 7, (1, ))

# task_name = "MiniGrid-PuttingAwayDishesAfterCleaning-16x16-N2-v0"
# for demo_file in demo_files:
#     if task_name in demo_file:
#         demo_traj = get_demo_traj(demo_file=demo_file, verbose=False)

#         idx = len(dataset)
#         demo_traj = LowLevelTrajectory(demo_traj.states, demo_traj.actions, _is_demo=True, _train_task_idx=idx)

#         dataset += [demo_traj]
#         objs, preds, options, ground_atoms_traj = parse_objs_preds_and_options(demo_traj, train_task_idx=idx)
#         ground_atom_dataset += [ground_atoms_traj]
#         goal = set([atom for atom in ground_atoms_traj[1][-1] if "inside(" in str(atom)])
#         tasks += [Task(State({}, None), goal)]

# print("#"*30)
# print(task_name)
# print("#"*30)

# print("Skipped")

# # nsrts, _, _ = learn_nsrts_from_data(dataset, tasks,
# #                                     preds,
# #                                     options,
# #                                     action_space,
# #                                     ground_atom_dataset,
# #                                     sampler_learner="neural",
# #                                     annotations=None)

# # assert len(nsrts) == 2


# #### BROKEN #####
# dataset = []
# ground_atom_dataset = []
# tasks = []
# action_space = Box(0, 7, (1, ))

# task_name = "MiniGrid-CleaningACar-16x16-N2-v0"
# for demo_file in demo_files:
#     if task_name in demo_file:
#         demo_traj = get_demo_traj(demo_file=demo_file, verbose=False)

#         idx = len(dataset)
#         demo_traj = LowLevelTrajectory(demo_traj.states, demo_traj.actions, _is_demo=True, _train_task_idx=idx)

#         dataset += [demo_traj]
#         objs, preds, options, ground_atoms_traj = parse_objs_preds_and_options(demo_traj, train_task_idx=idx)
#         ground_atom_dataset += [ground_atoms_traj]
#         goal = set([atom for atom in ground_atoms_traj[1][-1] if "inside(" in str(atom) or "dustyable(" in str(atom)])
#         tasks += [Task(State({}, None), goal)]

# print("#"*30)
# print(task_name)
# print("#"*30)

# print("Broken - No dustyable")

# # import ipdb; ipdb.set_trace()

# # nsrts, _, _ = learn_nsrts_from_data(dataset, tasks,
# #                                     preds,
# #                                     options,
# #                                     action_space,
# #                                     ground_atom_dataset,
# #                                     sampler_learner="neural",
# #                                     annotations=None)

# # assert len(nsrts) == 2

# dataset = []
# ground_atom_dataset = []
# tasks = []
# action_space = Box(0, 7, (1, ))

# task_name = "MiniGrid-WateringHouseplants-16x16-N2-v0"
# for demo_file in demo_files:
#     if task_name in demo_file:
#         demo_traj = get_demo_traj(demo_file=demo_file, verbose=False)

#         idx = len(dataset)
#         demo_traj = LowLevelTrajectory(demo_traj.states, demo_traj.actions, _is_demo=True, _train_task_idx=idx)

#         dataset += [demo_traj]
#         objs, preds, options, ground_atoms_traj = parse_objs_preds_and_options(demo_traj, train_task_idx=idx)
#         ground_atom_dataset += [ground_atoms_traj]
#         goal = set([atom for atom in ground_atoms_traj[1][-1] if "soakable(" in str(atom)])
#         tasks += [Task(State({}, None), goal)]

# print("#"*30)
# print(task_name)
# print("#"*30)

# print("Skipped")

# # nsrts, _, _ = learn_nsrts_from_data(dataset, tasks,
# #                                     preds,
# #                                     options,
# #                                     action_space,
# #                                     ground_atom_dataset,
# #                                     sampler_learner="neural",
# #                                     annotations=None)

# # assert len(nsrts) == 2

# dataset = []
# ground_atom_dataset = []
# tasks = []
# action_space = Box(0, 7, (1, ))

# task_name = "MiniGrid-OpeningPackages-16x16-N2-v0"
# for demo_file in demo_files:
#     if task_name in demo_file:
#         demo_traj = get_demo_traj(demo_file=demo_file, verbose=False)

#         idx = len(dataset)
#         demo_traj = LowLevelTrajectory(demo_traj.states, demo_traj.actions, _is_demo=True, _train_task_idx=idx)

#         dataset += [demo_traj]
#         objs, preds, options, ground_atoms_traj = parse_objs_preds_and_options(demo_traj, train_task_idx=idx)
#         ground_atom_dataset += [ground_atoms_traj]
#         goal = set([atom for atom in ground_atoms_traj[1][-1] if "openable(" in str(atom)])
#         tasks += [Task(State({}, None), goal)]

# print("#"*30)
# print(task_name)
# print("#"*30)

# nsrts, _, _ = learn_nsrts_from_data(dataset, tasks,
#                                     preds,
#                                     options,
#                                     action_space,
#                                     ground_atom_dataset,
#                                     sampler_learner="neural",
#                                     annotations=None)

# assert len(nsrts) == 2


# ##########################################
# # Generate Random Operator Demos
# ##########################################


