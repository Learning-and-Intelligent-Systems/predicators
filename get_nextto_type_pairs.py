import glob
import json
from predicators.envs.behavior2d import Behavior2DEnv

type_pairs = set()
nextto_types = set()
for file in glob.glob("/home/jmendez/Research/Behavior/predicators_behavior_v2/init_states/*/*goal.json"):
    with open(file, "r") as f:
        try:
            goal = json.load(f)
        except:
            print(file)
            raise
    for head_expr in goal:
        head_expr = head_expr[0]
        if head_expr[0] == "nextto":
            obj_0 = head_expr[1]
            obj_1 = head_expr[2]
            type_0 = Behavior2DEnv._get_object_typename(obj_0)
            type_1 = Behavior2DEnv._get_object_typename(obj_1)
            type_pairs.add((type_0, type_1))
            nextto_types.add(type_0)
            nextto_types.add(type_1)

print('NextTo')
print(nextto_types)
print(len(nextto_types), '(', len(nextto_types) ** 2,  ')')
print()
print(len(type_pairs))
print(type_pairs)
print()
# exit()

inside_types = set()
place_types = set()
for file in glob.glob("/home/jmendez/Research/Behavior/predicators_behavior_v2/init_states/*/*goal.json"):
    with open(file, "r") as f:
        goal = json.load(f)
    for head_expr in goal:
        head_expr = head_expr[0]
        if head_expr[0] == "inside" or head_expr[0] == "ontop" or head_expr[0] == "under":
            obj_0 = head_expr[1]
            obj_1 = head_expr[2]
            type_0 = Behavior2DEnv._get_object_typename(obj_0)
            type_1 = Behavior2DEnv._get_object_typename(obj_1)
            place_types.add(type_0)
            inside_types.add(type_1)

print('Inside')
print(place_types)
print(inside_types)
print(place_types & inside_types)
print()

notinside_type_pairs = set()
for file in glob.glob("/home/jmendez/Research/Behavior/predicators_behavior_v2/init_states/*/*goal.json"):
    with open(file, "r") as f:
        goal = json.load(f)
    for head_expr in goal:
        head_expr = head_expr[0]
        if head_expr[0] == "not" and head_expr[1] == "inside":
            obj_0 = head_expr[2]
            obj_1 = head_expr[3]
            type_0 = Behavior2DEnv._get_object_typename(obj_0)
            type_1 = Behavior2DEnv._get_object_typename(obj_1)
            notinside_type_pairs.add((type_0, type_1))
print('Not inside')
print(notinside_type_pairs)

type_pairs = set()
nextto_types = set()
for file in glob.glob("/home/jmendez/Research/Behavior/predicators_behavior_v2/init_states/*/*goal.json"):
    with open(file, "r") as f:
        goal = json.load(f)
    for head_expr in goal:
        head_expr = head_expr[0]
        if head_expr[0] == "touching":
            obj_0 = head_expr[1]
            obj_1 = head_expr[2]
            type_0 = Behavior2DEnv._get_object_typename(obj_0)
            type_1 = Behavior2DEnv._get_object_typename(obj_1)
            type_pairs.add((type_0, type_1))
            nextto_types.add(type_0)
            nextto_types.add(type_1)

print('Touching')
print(nextto_types)
print(len(nextto_types), '(', len(nextto_types) ** 2,  ')')
print()
print(len(type_pairs))
print(type_pairs)
print()