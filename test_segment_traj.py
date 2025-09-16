import pickle as pkl
import numpy as np
from predicators.structs import Action, LowLevelTrajectory, Predicate, State, \
    Type
from test_operator_learning_all import get_demo_traj, demo_files
from test_colla_results import OperatorLearningAgent

completed = [
    'MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0',
    'MiniGrid-CleaningACar-16x16-N2-v0',
    'MiniGrid-CleaningShoes-16x16-N2-v0', #1
    'MiniGrid-CleaningUpTheKitchenOnly-16x16-N2-v0',
    'MiniGrid-CollectMisplacedItems-16x16-N2-v0',
    'MiniGrid-InstallingAPrinter-16x16-N2-v0',
    'MiniGrid-LayingWoodFloors-16x16-N2-v0',
    'MiniGrid-MakingTea-16x16-N2-v0',
    'MiniGrid-MovingBoxesToStorage-16x16-N2-v0',
    'MiniGrid-OpeningPackages-16x16-N2-v0',
    'MiniGrid-OrganizingFileCabinet-16x16-N2-v0',
    #[DEBUG]'MiniGrid-PreparingSalad-16x16-N2-v0',
    'MiniGrid-PuttingAwayDishesAfterCleaning-16x16-N2-v0',
    'MiniGrid-SettingUpCandles-16x16-N2-v0', #1
    'MiniGrid-SortingBooks-16x16-N2-v0',
    'MiniGrid-StoringFood-16x16-N2-v0',
    #[DEBUG]'MiniGrid-ThawingFrozenFood-16x16-N2-v0',
    'MiniGrid-ThrowingAwayLeftovers-16x16-N2-v0',
    'MiniGrid-WashingPotsAndPans-16x16-N2-v0',
    'MiniGrid-WateringHouseplants-16x16-N2-v0'
]

task_info = {}

for demo_file in demo_files:
    # print("#"*60)
    # print(demo_file.split("/")[-1])
    # print("#"*60)
    # print("# PLAN #")
    traj = get_demo_traj(demo_file, verbose=False)
    add_count = 0
    for i, action in enumerate(traj.actions):
        curr_state = set(traj.states[i])
        next_state = set(traj.states[i+1])
        del_effs = curr_state - next_state
        add_effs = next_state - curr_state
        # print(action)
        # print("DEL:", del_effs)
        # print("ADD:", add_effs)
        # print()
        add_count += len(add_effs)
        assert len(add_effs) != 0 or str(action) == "Move"
    task_name = demo_file.split("/")[-1].split("_")[0]
    agent = OperatorLearningAgent("cluster-intersect", strips_learner="cluster_and_intersect")
    agent.get_data(task_name=task_name)
    goal = agent.parse_goal(task_name=task_name, ground_atoms_state=agent.ground_atoms_traj[1][-1])
    task_info[demo_file.split("/")[-1]] = (len(traj.actions), len(goal), add_count)

i = 0
for k,v in sorted([(k,v) for k,v in task_info.items()], key=lambda x: x[1][2]): # by add effects
    i+=1
    print("|", v[0], "| goal length:",  v[1], "| add count:", v[2], "|", k.split("_")[0], i)
    # for atom in agent.parse_goal(task_name=task_name, ground_atoms_state=agent.ground_atoms_traj[1][-1]):
    #     print(atom)

#########################################

# ### NEED To Turn Images into Objects or Save Object Centric State
# from minigrid.wrappers import *
# from mini_behavior.states import *

# env = gym.make('MiniGrid-SortingBooks-16x16-N2-v0')
# env.reset()

# # AbilityState
# # AbsoluteObjectState
# # RelativeObjectState
# # ObjectProperty

# def get_lifted_state(env):
#     mb_state = env.get_state()
#     grid = mb_state['grid']
#     agent_pos = mb_state['agent_pos']
#     agent_dir = mb_state['agent_dir']
#     objs = mb_state['objs']
#     obj_instances = mb_state['obj_instances']
#     ground_atoms = []
#     for k, o in obj_instances.items():
#         for pred_name, pred in o.states.items():
#             if isinstance(o.states[pred_name], AbsoluteObjectState):
#                 if o.states[pred_name].get_value(env):
#                     ground_atoms.append(pred_name+'('+k+')')
#             elif isinstance(o.states[pred_name], AbilityState):
#                 if o.states[pred_name].get_value(env):
#                     ground_atoms.append(pred_name+'('+k+')')
#             elif isinstance(o.states[pred_name], ObjectProperty):
#                 if o.states[pred_name].get_value(env):
#                     ground_atoms.append(pred_name+'('+k+')')
#             elif isinstance(o.states[pred_name], RelativeObjectState):
#                 for k2, o2 in obj_instances.items():
#                     if o.states[pred_name].get_value(o2, env=env):
#                         ground_atoms.append(pred_name+'('+k+','+k2+')')
#     return ground_atoms