import numpy as np
from gym.spaces import Box
import re
import pickle as pkl

from predicators import utils
from predicators.nsrt_learning.nsrt_learning_main import learn_nsrts_from_data
from predicators.structs import Action, LowLevelTrajectory, Predicate, State, \
    Type, GroundAtom, Task, Variable, LiftedAtom, NSRT, Set
import glob

name_to_actions = {
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

demo_files = sorted([filename for filename in glob.glob("/Users/shashlik/Documents/GitHub/predicators/demos/*/*")])
demo_tasks = set([demo_file.split("/")[-1].split("_")[0] for demo_file in demo_files])

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

def parse_nsrt_block(block, segmented_trajs) -> NSRT:
        """Parses a single NSRT block into an PNAD object."""
        lines = block.strip().split("\n")
        
        name_match = re.match(r"(\S+):", lines[0])
        name = name_match.group(1) if name_match else ""

        parameters = re.findall(r"\?x\d+:\w+", lines[1])
        
        def extract_effects(label: str) -> Set[str]:
            """Extracts a list of predicates from labeled sections."""
            for line in lines:
                if line.strip().startswith(label):
                    return set(re.findall(r"\w+\(.*?\)", line))
            return set()
        
        preconditions = extract_effects("Preconditions")
        add_effects = extract_effects("Add Effects")
        delete_effects = extract_effects("Delete Effects")
        ignore_effects = extract_effects("Ignore Effects")

        option_spec_match = re.search(r"Option Spec:\s*(.*)", block)
        option_spec = option_spec_match.group(1) if option_spec_match else ""

        objects = set()
        atoms = set()
        option_specs = {}
        for traj in segmented_trajs:
            for segment in traj:
                for state in segment.states:
                    for k, v in state.items():
                        objects.add(k)
                atoms |= segment.init_atoms | segment.final_atoms
                option_specs[segment.get_option().parent.name] = segment.get_option().parent
        all_predicates_list = [(atom.predicate.name,atom.predicate) for atom in atoms]
        def get_predicate(name, entities):
            for pred_name, pred in all_predicates_list:
                if pred_name == pred_name and pred.arity == len(entities):
                    valid_types = True
                    for i, ent in enumerate(entities):
                        if ent.type != pred.types[i]:
                            valid_types = False
                    if valid_types:
                        return pred
            raise NotImplementedError
            
        types = {obj.type.name:obj.type for obj in objects}

        def extract_parameters(predicate: str) -> Set[str]:
            parameter_pattern = re.compile(r"\?x\d+:\w+")  # Matches variables like ?x0:obj_type
            matches = parameter_pattern.findall(predicate)
            return matches
        
        parameters = [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in parameters]
        preconditions = set([LiftedAtom(get_predicate(pre.split("(")[0], [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(pre)]), [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(pre)]) for pre in preconditions])
        add_effects = set([LiftedAtom(get_predicate(add.split("(")[0], [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(add)]), [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(add)]) for add in add_effects])
        delete_effects = set([LiftedAtom(get_predicate(dle.split("(")[0], [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(dle)]), [Variable(param.split(":")[0], types[param.split(":")[1]]) for param in extract_parameters(dle)]) for dle in delete_effects])
        ignore_effects = set([get_predicate(ige, None) for ige in ignore_effects])
        a_name = option_spec.split("(")[0]
        option_spec = utils.SingletonParameterizedOption(
                a_name, lambda s, m, o, p: Action(name_to_actions[a_name]))

        return NSRT(name, parameters, preconditions, add_effects, delete_effects, ignore_effects, option_spec, [], None)