# (1) implement the evaluation function evaluate(agent) returns dictionary of results
# (2) implment evaluation visualization visualize(results)
# (3) do whatever it takes to make results better (CI, BC, FF+BC, FF+BC+LLMs)

from test_colla_env import MiniBehaviorEnv
from test_colla_helpers import Box, LowLevelTrajectory, State, Task, \
    demo_files, get_demo_traj, learn_nsrts_from_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from predicators.planning import task_plan, task_plan_grounding, _SkeletonSearchTimeout, PlanningFailure
from predicators import utils
from predicators.structs import Action, LowLevelTrajectory, Predicate, State, \
    Type, GroundAtom, Task, STRIPSOperator
import numpy as np
from collections import Counter

from predicators.nsrt_learning.strips_learning.gen_to_spec_learner import parse_objs_preds_and_options

import pickle as pkl
import numpy as np
from predicators.structs import Action, LowLevelTrajectory, Predicate, State, \
    Type
from test_operator_learning_all import get_demo_traj, demo_files

opname_to_key = {
    'Actions.pickup_0': '0',
    'Actions.pickup_1': '1',
    'Actions.pickup_2': '2',
    'Actions.drop_0': '3',
    'Actions.drop_1': '4',
    'Actions.drop_2': '5',
    'Actions.drop_in': 'i',
    'Actions.toggle': 't',
    'Actions.close': 'c',
    'Actions.open': 'o',
    'Actions.cook': 'k',
    'Actions.slice': '6'
}

class RandomAgent():
    def __init__(self, name):
        self.name = name
        self.actions = None

    def reset(self, task_name):
        pass

    def policy(self, obs, env):
        #print(env.get_lifted_state())
        return env.key_to_action[random.choice(list(env.key_to_action.keys()))]

def evaluation(agents, tasks, num_iterations=10, start_seed=100, short_task=True):
    results = {}
    task_i = 0
    for i in range(num_iterations):
        for task in tasks:
            for agent in agents:
                env = MiniBehaviorEnv(env_id=task, seed=i+start_seed)
                env.short_task = short_task
                observation, _ = env.reset()
                agent.short_task = short_task
                agent.reset(env.env_id)
                steps = 0
                for _ in range(50):
                    action = agent.policy(observation, env)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    steps += 1
                    env.show()

                    if reward != 0:
                        break

                    if terminated or truncated:
                        break
                found_plan = 0
                plan_diff = -1
                if agent.actions is not None and agent.actions != []:
                    found_plan = 1
        
                    key_to_opname = {v:k  for k,v in opname_to_key.items()}
                    plan = [key_to_opname[action] if not action.startswith("moveto") else "Move" for action in agent.actions]
                    dataset_plan = agent.dataset[0].actions

                    def differing_reoccurring_counts(list1, list2):
                        count1 = Counter(list1)
                        count2 = Counter(list2)
                        all_keys = set(count1.keys()) | set(count2.keys())
                        result = {}
                        total_diff = 0
                        for key in all_keys:
                            c1 = count1.get(key, 0)
                            c2 = count2.get(key, 0)
                            if (c1 > 1 or c2 > 1) and c1 != c2:
                                diff = abs(c1 - c2)
                                result[key] = diff
                                total_diff += diff
                        result['total'] = total_diff
                        return result
                    plan_diff = differing_reoccurring_counts(plan, dataset_plan)['total']
                results[str(task_i) + "_" + task + "_" + agent.name] = (steps, reward, i, found_plan, plan_diff)
            task_i += 1
    return results

def structure_results(results_dict):
    data = []
    for key, (steps, reward, iteration, found_plan, plan_diff) in results_dict.items():
        task_idx, task_name, agent_name = key.split("_", 2)
        data.append({
            "task_name": task_name,
            "task_idx": int(task_idx),
            "iteration": int(iteration),
            "found_plan": int(found_plan),
            "plan_diff": int(plan_diff),
            "steps": steps,
            "reward": reward,
            "success": 1 if reward > 0 else 0,
            "agent": agent_name
        })
    df = pd.DataFrame(data)
    df["task_order"] = df["task_idx"]
    return df.sort_values(["agent", "iteration", "task_order"])

def plot_lifelong_success(df):
    plt.figure(figsize=(14, 5))
    
    sns.lineplot(
        data=df,
        x="task_order",
        y="success",
        hue="agent",
        marker="o"
    )

    # Set up x-ticks with task names, spaced across iterations
    xticks = df["task_order"]
    xticklabels = df["task_name"]
    plt.xticks(ticks=xticks, labels=xticklabels, rotation=45, ha='right')

    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ["Fail", "Success"])
    plt.ylabel("Success")
    plt.xlabel("Tasks over Lifelong Iterations")
    plt.title("Lifelong Learning Success per Task")
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("lifelong_learning_success.png", dpi=200)

class OperatorLearningAgent():
    def __init__(self, name, strips_learner, single_grounding=False):
        self.name = name
        self.num_demos = 1

        # Initialized once; populated in get_data()
        self.dataset = []
        self.ground_atom_dataset = []
        self.tasks = []
        self.action_space = Box(0, 7, (1,))
        self.objs = set()
        self.preds = set()
        self.options = set()
        self.ground_atoms_traj = []
        self.goal = None

        # Runtime variables
        self.nsrts = None
        self.actions = None
        self.i = 0
        self.seed_i = 0
        self.short_task = True

        # Learning Params
        self.strips_learner = strips_learner
        self.single_grounding = single_grounding
        utils.reset_config({
            "strips_learner": self.strips_learner,
            "segmenter": "every_step",
            "disable_harmlessness_check": True,
            "pnad_search_load_initial": True,
            "backward_forward_load_initial": True,
            "min_data_for_nsrt": 0,
            "min_perc_data_for_nsrt": 0,
            "pnad_search_timeout":1000.0,
            "single_grounding": self.single_grounding,
            "option_learner": "no_learning"
        })

    def reset(self, task_name):
        if False:
            self.dataset = []
            self.ground_atom_dataset = []
            self.tasks = []
            self.objs = set()
            self.preds = set()
            self.options = set()
            self.ground_atoms_traj = []
        self.action_space = Box(0, 7, (1,))
        self.seed_i = 0

        # Learning Params
        utils.reset_config({
            "strips_learner": self.strips_learner,
            "segmenter": "every_step",
            "disable_harmlessness_check": True,
            "pnad_search_load_initial": True,
            "backward_forward_load_initial": True,
            "min_data_for_nsrt": 0,
            "min_perc_data_for_nsrt": 0,
            "pnad_search_timeout":1000.0,
            "single_grounding": self.single_grounding,
            "option_learner": "no_learning"
        })

        self.nsrts = self.learn_nsrts(task_name)
        self.goal = self.parse_goal(task_name, self.ground_atoms_traj[1][-1])
        self.actions = None
        self.i = 0

    def parse_goal(self, task_name, ground_atoms_state):
        if task_name == "MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if  str(atom).startswith("inside(")])
        
        elif task_name == "MiniGrid-OpeningPackages-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if str(atom).startswith("openable(")])
        
        elif task_name == "MiniGrid-CleaningACar-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if str(atom).startswith("inside(")]) | set([atom for atom in ground_atoms_state if str(atom).startswith("~dustyable(")])
        
        elif task_name == "MiniGrid-CleaningShoes-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if str(atom).startswith("~stainable(") and "shoe" in str(atom)]) | \
                set([atom for atom in ground_atoms_state if str(atom).startswith("~dustyable(") and "shoe" in str(atom)]) | \
                set([atom for atom in ground_atoms_state if str(atom).startswith("onfloor(") and "towel" in str(atom)])

        elif task_name == "MiniGrid-CleaningUpTheKitchenOnly-16x16-N2-v0":
            return set([
                atom for atom in ground_atoms_state
                if (
                    str(atom).startswith("onTop(") and "blender" in str(atom) and "countertop" in str(atom)
                ) or (
                    str(atom).startswith("nextto(") and "soap" in str(atom) and "sink" in str(atom)
                ) or (
                    str(atom).startswith("inside(") and "vegetable_oil" in str(atom) and "cabinet" in str(atom)
                ) or (
                    str(atom).startswith("inside(") and "plate" in str(atom) and "cabinet" in str(atom)
                ) or (
                    str(atom).startswith("inside(") and "casserole" in str(atom) and "electric_refrigerator" in str(atom)
                ) or (
                    str(atom).startswith("inside(") and "apple" in str(atom) and "electric_refrigerator" in str(atom)
                ) or (
                    str(atom).startswith("inside(") and "rag" in str(atom) and "sink" in str(atom)
                ) or (
                    str(atom).startswith("nextto(") and "rag" in str(atom) and "sink" in str(atom)
                ) or (
                    str(atom).startswith("~dustyable(") and "cabinet" in str(atom)
                ) or (
                    str(atom).startswith("~stainable(") and "plate" in str(atom)
                )
            ])

        elif task_name == "MiniGrid-CollectMisplacedItems-16x16-N2-v0":
            return set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("onTop(") and "table" in str(atom) and (
                    "gym_shoe" in str(atom) or
                    "necklace" in str(atom) or
                    "notebook" in str(atom) or
                    "sock" in str(atom)
                ) and not str(atom).startswith("onTop(table") 
            ])
        
        elif task_name == "MiniGrid-InstallingAPrinter-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if str(atom).startswith("onTop(")]) | \
                set([atom for atom in ground_atoms_state if str(atom).startswith("toggleable(")])
        
        elif task_name == "MiniGrid-LayingWoodFloors-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if str(atom).startswith("nextto(")])
        
        elif task_name == "MiniGrid-MakingTea-16x16-N2-v0":
            return set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("sliceable(") and "lemon" in str(atom)
            ]) | set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("onTop(") and "teapot" in str(atom) and "stove" in str(atom)
            ]) | set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("atsamelocation(") and "tea_bag" in str(atom) and "teapot" in str(atom)
            ]) | set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("soakable(") and "teapot" in str(atom)
            ]) | set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("toggleable(") and "stove" in str(atom)
            ])

        elif task_name == "MiniGrid-MovingBoxesToStorage-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if str(atom).startswith("onTop(")])
        
        elif task_name == "MiniGrid-OrganizingFileCabinet-16x16-N2-v0":
            return set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("onTop(") and "marker" in str(atom) and "table" in str(atom)
        ]) | set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("inside(") and "document" in str(atom) and "cabinet" in str(atom)
        ]) | set([
            atom for atom in ground_atoms_state
            if str(atom).startswith("inside(") and "folder" in str(atom) and "cabinet" in str(atom)
        ])
        
        elif task_name == "MiniGrid-PreparingSalad-16x16-N2-v0":
            import ipdb; ipdb.set_trace()
            raise NotImplementedError("parse_goal not implemented for PreparingSalad")
        
        elif task_name == "MiniGrid-PuttingAwayDishesAfterCleaning-16x16-N2-v0":
            return set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("inside(") and "plate" in str(atom) and "cabinet" in str(atom)
            ])

        
        elif task_name == "MiniGrid-SettingUpCandles-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if str(atom).startswith("onTop(")])
        
        elif task_name == "MiniGrid-SortingBooks-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if str(atom).startswith("onTop(") and "shelf" in str(atom) and ("book" in str(atom) or "hardback" in str(atom))])
        
        elif task_name == "MiniGrid-StoringFood-16x16-N2-v0":
            return set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("inside(") and "cabinet" in str(atom) and (
                    "oatmeal" in str(atom) or "chip" in str(atom) or "vegetable_oil" in str(atom) or "sugar" in str(atom)
                )
            ])

        elif task_name == "MiniGrid-ThawingFrozenFood-16x16-N2-v0":
            return set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("nextto(") and (
                    ("date" in str(atom) and "fish" in str(atom)) or
                    ("fish" in str(atom) and "sink" in str(atom)) or
                    ("olive" in str(atom) and "sink" in str(atom))
                )
            ])
        
        elif task_name == "MiniGrid-ThrowingAwayLeftovers-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if str(atom).startswith("inside(") and "hamburger" in str(atom) and "ashcan" in str(atom)])
        
        elif task_name == "MiniGrid-WashingPotsAndPans-16x16-N2-v0":
            return set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("~stainable(") and (
                    "pan" in str(atom) or "kettle" in str(atom) or "teapot" in str(atom)
                )
            ]) | set([
                atom for atom in ground_atoms_state
                if str(atom).startswith("inside(") and "cabinet" in str(atom) and (
                    "pan" in str(atom) or "kettle" in str(atom) or "teapot" in str(atom)
                )
            ])

        elif task_name == "MiniGrid-WateringHouseplants-16x16-N2-v0":
            return set([atom for atom in ground_atoms_state if str(atom).startswith("soakable(") and "pot_plant" in str(atom)])
        else:
            import ipdb; ipdb.set_trace()


    def get_plan(self, state, seed):
        objs, _, _, ground_atoms_traj, all_atoms = parse_objs_preds_and_options(self.dataset[-1], train_task_idx=len(self.dataset))
        task = Task(State({}, None), self.goal)

        _, _, _, ground_atoms_traj, _ = parse_objs_preds_and_options(LowLevelTrajectory([state], [], _is_demo=True, _train_task_idx=0), train_task_idx=0, all_atoms=all_atoms)
        init_atoms = ground_atoms_traj[1][0]
        plan = self.plan(init_atoms, objs, self.preds, self.nsrts, task, seed)
        return plan

    def policy(self, obs, env):
        if self.actions is None:
            seed = self.seed_i
            self.seed_i += 1
            num_remove_pre = 0
            while self.actions is None or self.actions == []:
                try:
                    self.actions = self.get_plan(env.get_lifted_state(), seed)
                    break
                except _SkeletonSearchTimeout:
                    print("did not find skeleton - timeout")
                except PlanningFailure:
                    print("did not find skeleton - plan failure")
                self.actions = []
            #     num_remove_pre += 1
            #     new_nsrts = set()
            #     for nsrt in self.nsrts:
            #         pre = set()
            #         tot_pre = len(nsrt.op.preconditions) - num_remove_pre
            #         if tot_pre > 0:
            #             pre = random.sample(nsrt.op.preconditions, tot_pre)
            #         ignore_effects = nsrt.op.ignore_effects
            #         del_effs = nsrt.op.delete_effects
            #         # if num_remove_pre > 10:
            #         #     ignore_effects = set()
            #         #     del_effs = set()
            #         new_nsrts.add(
            #             nsrt.op.copy_with(preconditions=pre,
            #                               ignore_effects=ignore_effects,
            #                               delete_effects=del_effs).make_nsrt(
            #                 nsrt.option,
            #                 [],  # dummy sampler
            #                 lambda s, g, rng, o: np.zeros(1, dtype=np.float32)))
            #     self.nsrts = new_nsrts
            # with open("test_saved.NSRTs.txt", "w") as file:
            #     for nsrt in self.nsrts:
            #         if nsrt.op.add_effects != set():
            #             file.write(str(nsrt)+"\n")

        self.i += 1
        if self.i-1 < len(self.actions):
            return env.key_to_action[self.actions[self.i-1]]
        else:
            self.actions = None
            self.i = 0
            return env.key_to_action["0"]
    
    def clean_action_plan(self, action_plan):
        plan = []
        for step in action_plan:
            name = step[0]
            objs = step[1]
            if len(objs) > 0:
                obj_name = objs[0].name
                if name.startswith("Move"):
                    plan.append(f"moveto-{obj_name}")
                else:
                    for opname, key in opname_to_key.items():
                        if opname in name:
                            plan.append(key)
                            break
        return plan
    
    def plan(self, init_atoms, objects, predicates, nsrts, task, seed):
        ground_nsrts, reachable_atoms = task_plan_grounding(init_atoms, objects, nsrts, allow_noops=True)
        heuristic = utils.create_task_planning_heuristic("hadd", init_atoms,
                                                        task.goal, ground_nsrts,
                                                        predicates, objects)
        task_plan_generator = task_plan(init_atoms,
                                        task.goal,
                                        ground_nsrts,
                                        reachable_atoms,
                                        heuristic,
                                        timeout=1,
                                        seed=seed,
                                        max_skeletons_optimized=3)
        skeleton, _, _ = next(task_plan_generator)

        action_plan = []
        for step in skeleton:
            action_plan.append((step.option.name, step.objects))
        return self.clean_action_plan(action_plan)
    
    def get_data(self, task_name):
        for demo_file in demo_files:
            if task_name in demo_file:
                demo_traj = get_demo_traj(demo_file=demo_file, verbose=False)

                if self.short_task:
                    if task_name == 'MiniGrid-SortingBooks-16x16-N2-v0':
                        demo_traj = LowLevelTrajectory(demo_traj.states[:5], demo_traj.actions[:4], _is_demo=True, _train_task_idx=0)
                    elif task_name == 'MiniGrid-WateringHouseplants-16x16-N2-v0':
                        demo_traj = LowLevelTrajectory(demo_traj.states[:7], demo_traj.actions[:6], _is_demo=True, _train_task_idx=0)
                    elif task_name == 'MiniGrid-ThrowingAwayLeftovers-16x16-N2-v0':
                        demo_traj = LowLevelTrajectory(demo_traj.states[:5], demo_traj.actions[:4], _is_demo=True, _train_task_idx=0)
                    elif task_name == 'MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0':
                        demo_traj = LowLevelTrajectory(demo_traj.states[:5], demo_traj.actions[:4], _is_demo=True, _train_task_idx=0) 

                idx = len(self.dataset)
                demo_traj = LowLevelTrajectory(demo_traj.states, demo_traj.actions, _is_demo=True, _train_task_idx=idx)

                self.dataset.append(demo_traj)
                new_objs, new_preds, new_options, self.ground_atoms_traj, _ = parse_objs_preds_and_options(demo_traj, train_task_idx=idx)
                self.objs |= new_objs
                self.preds |= new_preds
                self.options |= new_options
                self.ground_atom_dataset.append(self.ground_atoms_traj)
                goal = self.parse_goal(task_name, self.ground_atoms_traj[1][-1])
                self.tasks.append(Task(State({}, None), goal))
                # if len(self.dataset) >= self.num_demos:
                #     break
        # assert len(self.dataset) == self.num_demos  
        return self.dataset, self.tasks, self.preds, self.options, self.action_space, self.ground_atom_dataset

    def learn_nsrts(self, task_name):
        dataset, tasks, preds, options, action_space, ground_atom_dataset = self.get_data(task_name)
        nsrts, _, _ = learn_nsrts_from_data(dataset,
                                            tasks,
                                            preds,
                                            options,
                                            action_space,
                                            ground_atom_dataset,
                                            sampler_learner="neural",
                                            annotations=None)  
        with open("test_saved.NSRTs.txt", "w") as file:
            for nsrt in nsrts:
                if nsrt.op.add_effects != set():
                    file.write(str(nsrt)+"\n")
        return nsrts
 
class DummyAgent(OperatorLearningAgent):
    def __init__(self, name="dummy", strips_learner="dummy"):
        super().__init__(name=name, strips_learner=strips_learner, single_grounding=True)

    def learn_nsrts(self, task_name):
        dataset, tasks, preds, options, action_space, ground_atom_dataset = self.get_data(task_name)
        goal = self.parse_goal(task_name, self.ground_atoms_traj[1][-1])
        obj_to_var = {obj:obj.type("?" + obj.name) for obj in self.objs}
        lifted_goal = {atom.lift(obj_to_var) for atom in goal}

        nsrts = set()
        name_i = 0
        for option in options:
            op = STRIPSOperator(
                name="Dummy" + str(name_i),
                parameters=[],
                preconditions=set(),
                add_effects=set(),
                delete_effects=set(),
                ignore_effects=set()
            )
            dummy_nsrt = op.make_nsrt(
                option,
                [],  # dummy sampler
                lambda s, g, rng, o: np.zeros(1, dtype=np.float32))
            nsrts.add(dummy_nsrt)
            name_i += 1
            
            params = []
            for sublist in [lifted_atom.variables for lifted_atom in lifted_goal]:
                params += sublist
            params = [x for x in set(params)]
            op = STRIPSOperator(
                name="Dummy" + str(name_i),
                parameters=params,
                preconditions=set(),
                add_effects=lifted_goal,
                delete_effects=set(),
                ignore_effects=set()
            )
            dummy_nsrt = op.make_nsrt(
                option,
                [],  # dummy sampler
                lambda s, g, rng, o: np.zeros(1, dtype=np.float32))
            nsrts.add(dummy_nsrt)
            name_i += 1
        return nsrts
    
class GroundTruthAgent(OperatorLearningAgent):
    def __init__(self, name):
        super().__init__(name=name, strips_learner="NONE")
        self.name = name
        self.ground_truth_trajs = {}
        self.i = 0
        self.actions = None
    
    def reset(self, task_name):
        self.dataset = []
        self.ground_atom_dataset = []
        self.tasks = []
        self.action_space = Box(0, 7, (1,))
        self.objs = set()
        self.preds = set()
        self.options = set()
        self.ground_atoms_traj = []

        dataset, tasks, preds, options, action_space, ground_atom_dataset = self.get_data(task_name)

        self.goal = self.parse_goal(task_name, self.ground_atoms_traj[1][-1])
        self.actions = None
        self.i = 0

        action_plan = []
        for i, step in enumerate(self.ground_atoms_traj[0].actions):
            curr_state = self.ground_atoms_traj[1][i]
            next_state = self.ground_atoms_traj[1][i+1]
            def count_object_occurrences(atom_set):
                counter = Counter()
                for atom in atom_set:
                    for obj in atom.objects:
                        if not atom.predicate.name.startswith("~inreachofrobot"):
                            counter[obj] += 1
                return counter
            counter = count_object_occurrences(next_state - curr_state)
            def get_max_count_object(counter, exclude_types=("table", "shelf")):
                max_count = max(counter.values())
                candidates = [
                    obj for obj, count in counter.items()
                    if count == max_count and all(ex_type not in str(obj) for ex_type in exclude_types)
                ]

                if candidates:
                    return candidates[0]
                else:
                    return None
                
            try:
                if get_max_count_object(counter) is None:
                    objs = [max(counter, key=counter.get)]
                else:
                    objs = [get_max_count_object(counter)]
            except:
                objs = random.sample(self.objs, 1)
            action_plan.append((step._option.name, objs))
        self.ground_truth_trajs[task_name] = self.clean_action_plan(action_plan)

    def policy(self, obs, env):
        #print(env.get_lifted_state())
        try:
            assert env.env_id in self.ground_truth_trajs.keys()
        except:
            import ipdb; ipdb.set_trace()
        self.i += 1
        if self.i-1 < len(self.ground_truth_trajs[env.env_id]):
            return env.key_to_action[self.ground_truth_trajs[env.env_id][self.i-1]]
        else:
            return env.key_to_action["0"]
        
    def learn_nsrts(self, task_name):
        return None

# tasks = ["MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0",
#             "MiniGrid-CollectMisplacedItems-16x16-N2-v0",
#             "MiniGrid-ThrowingAwayLeftovers-16x16-N2-v0",
#             "MiniGrid-OpeningPackages-16x16-N2-v0",
#             "MiniGrid-WateringHouseplants-16x16-N2-v0",
#             "MiniGrid-PuttingAwayDishesAfterCleaning-16x16-N2-v0"]

# agents = [RandomAgent("random"), GroundTruthAgent("ground-truth")]
# results = evaluation(agents, tasks, num_iterations=3)
# df = structure_results(results)
# plot_lifelong_success(df)

#####

# tasks = ["MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0"]
# agents = [GroundTruthAgent("ground-truth")]
# results = evaluation(agents, tasks, num_iterations=1)
# df = structure_results(results)
# #plot_lifelong_success(df)
# import ipdb; ipdb.set_trace()

#####

############ 
# tasks = ["MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0"]
# agents = [
#             DummyAgent("dummy", strips_learner="dummy"),
#             OperatorLearningAgent("cluster-intersect", strips_learner="cluster_and_intersect"),
#             OperatorLearningAgent("backchaining", strips_learner="backchaining"),
#             OperatorLearningAgent("hill-climbing", strips_learner="pnad_search"),
#             # OperatorLearningAgent("llm", strips_learner="llm")
#         ]
# results = evaluation(agents, tasks, num_iterations=1)
# df = structure_results(results)
# #plot_lifelong_success(df)
# for agent in agents:
#     print(agent.name, len(agent.nsrts), agent.actions)
#     print()
# import ipdb; ipdb.set_trace()

# Note: grounding should only be for operators based on the goal....
# Maybe LLM can help with grounding too

# TODO Finally - Collect Demos, Increment Num_Demos

# TODO Try Run 3-5 Env Eval on all 5 Baselines (Dummy, CI, Pnad_Search, Back_Chaining)

# TODO Fix LLM Agents

# TODO Make BC+FS+LLM Agent
# see other code

# TODO Make Version-Space Agent
# Note: from CI it should fall back to BC+FS+LLM then to BC then to Dummy

# TODO Try Run Full Eval on all 7 Agents

# tasks = [
#     'MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0',
#     'MiniGrid-CleaningACar-16x16-N2-v0',
#     'MiniGrid-CleaningShoes-16x16-N2-v0', #1
#     'MiniGrid-CleaningUpTheKitchenOnly-16x16-N2-v0',
#     'MiniGrid-CollectMisplacedItems-16x16-N2-v0',
#     'MiniGrid-InstallingAPrinter-16x16-N2-v0',
#     'MiniGrid-LayingWoodFloors-16x16-N2-v0',
#     'MiniGrid-MakingTea-16x16-N2-v0',
#     'MiniGrid-MovingBoxesToStorage-16x16-N2-v0',
#     'MiniGrid-OpeningPackages-16x16-N2-v0',
#     'MiniGrid-OrganizingFileCabinet-16x16-N2-v0',
#     #[DEBUG]'MiniGrid-PreparingSalad-16x16-N2-v0',
#     'MiniGrid-PuttingAwayDishesAfterCleaning-16x16-N2-v0',
#     'MiniGrid-SettingUpCandles-16x16-N2-v0', #1
#     'MiniGrid-SortingBooks-16x16-N2-v0',
#     'MiniGrid-StoringFood-16x16-N2-v0',
#     #[DEBUG]'MiniGrid-ThawingFrozenFood-16x16-N2-v0',
#     'MiniGrid-ThrowingAwayLeftovers-16x16-N2-v0',
#     'MiniGrid-WashingPotsAndPans-16x16-N2-v0',
#     'MiniGrid-WateringHouseplants-16x16-N2-v0'
# ]

# task_info = {}

# for demo_file in demo_files:
#     traj = get_demo_traj(demo_file, verbose=False)
#     add_count = 0
#     for i, action in enumerate(traj.actions):
#         curr_state = set(traj.states[i])
#         next_state = set(traj.states[i+1])
#         del_effs = curr_state - next_state
#         add_effs = next_state - curr_state
#         # print(action)
#         # print("DEL:", del_effs)
#         # print("ADD:", add_effs)
#         # print()
#         add_count += len(add_effs)
#         assert len(add_effs) != 0 or str(action) == "Move"
#     task_name = demo_file.split("/")[-1].split("_")[0]
#     agent = OperatorLearningAgent("cluster-intersect", strips_learner="cluster_and_intersect")
#     agent.get_data(task_name=task_name)
#     goal = agent.parse_goal(task_name=task_name, ground_atoms_state=agent.ground_atoms_traj[1][-1])
#     task_info[demo_file.split("/")[-1]] = (len(traj.actions), len(goal), add_count)

# i = 0
# curriculum = []
# for k,v in sorted([(k,v) for k,v in task_info.items()], key=lambda x: x[1][2]): # by add effects
#     i+=1
#     print("|", v[0], "| goal length:",  v[1], "| add count:", v[2], "|", k.split("_")[0], i)
#     curriculum.append(k.split("_")[0])

import time
start_time = time.time()
tasks = ['MiniGrid-OpeningPackages-16x16-N2-v0',
         'MiniGrid-InstallingAPrinter-16x16-N2-v0',
         'MiniGrid-MovingBoxesToStorage-16x16-N2-v0',
         'MiniGrid-SortingBooks-16x16-N2-v0',#
         'MiniGrid-WateringHouseplants-16x16-N2-v0',#
         #'MiniGrid-ThrowingAwayLeftovers-16x16-N2-v0',#
         'MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0'#
         ]
# tasks = curriculum
# tasks.remove('MiniGrid-LayingWoodFloors-16x16-N2-v0')
# tasks.remove('MiniGrid-CleaningACar-16x16-N2-v0')
# tasks.remove('MiniGrid-SortingBooks-16x16-N2-v0')
# tasks.remove('MiniGrid-CleaningUpTheKitchenOnly-16x16-N2-v0')
print("#"*30)
print(tasks)

all_agents = [
                #GroundTruthAgent("ground-truth"),
                #DummyAgent("dummy", strips_learner="dummy"),
                #OperatorLearningAgent("cluster-intersect", strips_learner="cluster_and_intersect"),
                #OperatorLearningAgent("backchaining", strips_learner="backchaining"),
                #OperatorLearningAgent("hill-climbing", strips_learner="pnad_search"),
                #OperatorLearningAgent("llm", strips_learner="llm"),
                #OperatorLearningAgent("backward-forward", strips_learner="backward-forward"),
            ]

for agent in all_agents:
    with open("test_saved.NSRTs.txt", "w") as file:
        file.write("""NSRT-Move0:
    Parameters: [?x0:obj_type]
    Preconditions: []
    Add Effects: [inreachofrobot(?x0:obj_type)]
    Delete Effects: [~inreachofrobot(?x0:obj_type)]
    Ignore Effects: [inreachofrobot, ~inreachofrobot]
    Option Spec: Move()""")
    results = evaluation([agent], tasks, num_iterations=1, start_seed=100)
    df = structure_results(results)
    plot_lifelong_success(df)
    end_time = time.time()
    print("time elasped", end_time - start_time)
    df.to_csv('test_results/' + agent.name + '_output.csv')

    # results = evaluation([agent], tasks, num_iterations=1, start_seed=100, short_task=False)
    # df2 = structure_results(results)
    # plot_lifelong_success(df2)
    # end_time = time.time()
    # print("time elasped", end_time - start_time)
    # df2.to_csv('test_results/' + agent.name + '_long_output.csv')

