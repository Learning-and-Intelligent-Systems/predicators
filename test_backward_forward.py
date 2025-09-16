import random
from typing import List, Set, Tuple
import numpy as np
from collections import deque, defaultdict

np.random.seed(0)
random.seed(0)

# Parameters
NUM_PREDICATES = 10
NUM_OPERATORS = 10
TRAJ_MAX = 10
TRAJ_LEN = 5
NUM_TRAJS = 50 # Fewer for readability
ACTION_SPACE = list(range(NUM_OPERATORS))

# --- Operator Representation ---
class Operator:
    def __init__(self, pre: Set[int], add: Set[int], delete: Set[int], action: int):
        self.pre = pre
        self.add = add
        self.delete = delete
        self.action = action

    def is_applicable(self, state: Set[int]) -> bool:
        return self.pre.issubset(state)

    def apply(self, state: Set[int]) -> Set[int]:
        if not self.is_applicable(state):
            return state  # no-op if not applicable
        new_state = state.copy()
        new_state.difference_update(self.delete)
        new_state.update(self.add)
        return new_state

    def __repr__(self):
        return f"Op(action={self.action}, pre={self.pre}, add={self.add}, del={self.delete})"


def plan(start: Set[int], goal: Set[int], operators: List[Operator], max_depth=10):
    visited = set()
    queue = deque()
    queue.append((start.copy(), []))

    while queue:
        state, path = queue.popleft()
        state_key = frozenset(state)
        if state_key in visited:
            continue
        visited.add(state_key)

        if goal.issubset(state):
            return path

        if len(path) >= max_depth:
            continue

        for op in operators:
            if op.is_applicable(state):
                next_state = op.apply(state)
                if next_state != state:
                    queue.append((next_state, path + [(state.copy(), op.action, next_state.copy())]))

    return None


# --- Generate Random Operators ---
def generate_random_operator(pred_pool: List[int], action_id: int) -> Operator:
    pre = set(random.sample(pred_pool, random.randint(1, 3)))
    effects = list(set(pred_pool) - pre)
    add = set(random.sample(effects, random.randint(1, min(2, len(effects)))))
    delete = set(random.sample(list(pre), random.randint(0, len(pre))))
    return Operator(pre, add, delete, action_id)


def compute_reachable_states(init_state: Set[int], operators: List[Operator], max_iters: int = 100) -> List[Set[int]]:
    reached_states = set()
    reachable = []
    frontier = [init_state.copy()]

    for _ in range(max_iters):
        new_frontier = []

        for state in frontier:
            state_key = frozenset(state)
            if state_key in reached_states:
                continue

            reached_states.add(state_key)
            reachable.append(frozenset(state))

            for op in operators:
                if op.is_applicable(state):
                    next_state = op.apply(state)
                    next_key = frozenset(next_state)
                    if next_key not in reached_states:
                        new_frontier.append(next_state)

        if not new_frontier:
            break
        frontier = new_frontier

    return set(reachable)


# --- Generate Demo Data ---
def generate_planned_demo_trajectories(operators: List[Operator], num_trajs: int, max_depth: int) -> List[Tuple[List[Tuple[Set[int], int, Set[int]]], Set[int]]]:
    demos = []
    attempts = 0

    while len(demos) < num_trajs and attempts < 10000000:
        attempts += 1
        init_state = set(random.sample(range(NUM_PREDICATES), random.randint(2, NUM_PREDICATES)))
        reachable = compute_reachable_states(init_state, operators) - init_state

        if len(reachable) == 0:
            continue

        plan_traj = []
        goals = reachable
        while len(goals) > 0 and len(plan_traj) < TRAJ_LEN:
            goal_state = random.choice(list(goals))
            goal = goal_state - init_state

            if not goal:
                goals.remove(goal_state)
                continue

            plan_traj = plan(init_state, goal, operators, max_depth)
            if plan_traj is None or len(plan_traj) < TRAJ_LEN:
                goals.remove(goal_state)
                plan_traj = []

        if plan_traj and len(plan_traj) >= TRAJ_LEN:
            demos.append((plan_traj, goal))

    return demos


# --- Backwards-Forwards Operator Learning ---
def backward_infer_minimal_effects(demo_data, current_operators=None):
    candidate_ops = defaultdict(lambda: {'demos': []})
    op_index = {}
    if current_operators:
        op_index = {(op.action, frozenset(op.add)): op for op in current_operators}

    for traj, goal in sorted(demo_data, key=lambda x: len(x[0])): #sorted(demo_data, key=lambda x: len(x[0])*(1+len(x[1]))): # order by smallest demo
        current_goal = goal.copy()

        for (s, action, s_prime) in reversed(traj):
            effect = s_prime - s
            if len(effect) == 0:
                raise Exception("No effect")
            elif len(effect) == 1:
                necessary_effect = effect
            else:
                necessary_effect = effect & current_goal

            key = (action, frozenset(necessary_effect))
            candidate_ops[key]['demos'].append((s, action, s_prime))

            preconditions = set()
            if key in op_index:
                preconditions = op_index[key].pre
            current_goal = (current_goal - necessary_effect) | preconditions

    return candidate_ops


def refine_by_plan_divergence(demos, learned_operators):
    op_index = {(op.action, frozenset(op.add)): op for op in learned_operators}
    support_sets = {key: [] for key in op_index}

    for traj, goal in demos:
        current_goal = goal.copy()
        for s, a, s_prime in traj:
            effect = s_prime - s
            necessary_effect = effect if len(effect) == 1 else effect & current_goal
            key = (a, frozenset(necessary_effect))
            if key in support_sets:
                support_sets[key].append(s)

            preconditions = set()
            if key in op_index:
                preconditions = op_index[key].pre
            current_goal = (current_goal - necessary_effect) | preconditions

    for traj, goal in demos:
        state = traj[0][0]
        for (s_true, a_true, s_next_true) in traj:
            applicable = [op for op in learned_operators if op.is_applicable(state)]
            if not applicable:
                break
            op_planner = random.choice(applicable)

            key_true = (a_true, frozenset(s_next_true - s_true))
            op_true = op_index.get(key_true, None)
            if op_true is None:
                continue

            if op_planner is op_true:
                state = op_true.apply(state)
                continue

            key_planner = (op_planner.action, frozenset(op_planner.add))
            support = support_sets.get(key_planner, [])
            if not support:
                continue

            common_preds = set.intersection(*support)
            potential_preds_to_add = common_preds - state
            if not potential_preds_to_add:
                continue
            # if len(potential_preds_to_add - goal) > 0:
            #     preds_to_add = {random.choice(list(potential_preds_to_add - goal))}
            # else:
            #     preds_to_add = {random.choice(list(potential_preds_to_add))}
            preds_to_add = {random.choice(list(potential_preds_to_add))}
            op_planner.pre.update(preds_to_add)

            state = op_true.apply(state)

    return list(op_index.values())


def learn_operators_from_demos(demo_data, max_iters=100, verbose=True):
    learned_ops = []
    last_preconds = None

    for iteration in range(max_iters):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} (Backward + Forward) ---")

        candidate_ops = backward_infer_minimal_effects(demo_data, current_operators=learned_ops or None)

        if learned_ops == []:
            learned_ops = [
                Operator(pre=set(), add=set(effect_frozen), delete=set(), action=action)
                for (action, effect_frozen), entry in candidate_ops.items()
            ]
        else:
            op_index = {(op.action, frozenset(op.add)): op for op in learned_ops}
            learned_ops = []
            for (action, effect_frozen), entry in candidate_ops.items():
                if (action, effect_frozen) not in op_index:
                    learned_ops.append(Operator(pre=set(), add=set(effect_frozen), delete=set(), action=action))
                else:
                    learned_ops.append(Operator(
                        pre=op_index[(action, effect_frozen)].pre,
                        add=set(effect_frozen),
                        delete=set(),
                        action=action
                    ))

        if verbose:
            print("Backward Learned Operators:")
            for op in sorted(learned_ops, key=lambda x: x.action):
                print(op)

        learned_ops = refine_by_plan_divergence(demo_data, learned_ops)

        if verbose:
            print("Forward Learned Operators:")
            for op in sorted(learned_ops, key=lambda x: x.action):
                print(op)

    return learned_ops


# --- Run Learning ---
def run_operator_learning_trials(num_trials=10, verbose=True) -> int:
    invalid_count = 0
    valid_count = 0

    for _ in range(num_trials):
        pred_pool = [i for i in range(NUM_PREDICATES)]
        operators = [generate_random_operator(pred_pool, i) for i in range(NUM_OPERATORS)]
        demo_data = generate_planned_demo_trajectories(operators, NUM_TRAJS, max_depth=TRAJ_MAX)

        if verbose:
            print("\n--- Ground Truth Operators ---")
            for op in sorted(operators, key=lambda x: x.action):
                print(op)
            print()

            print("Demos:")
            for traj in demo_data:
                print("Goal:", traj[1], "Length:", len(traj[0]))

        refined_ops = learn_operators_from_demos(demo_data, max_iters=1000, verbose=verbose)

        if verbose:
            print("\n--- Final Learned Operators ---")
            for op in sorted(refined_ops, key=lambda x: x.action):
                print(op)
            print()

        # Add delete effects
        op_index = {(op.action, frozenset(op.add)): op for op in refined_ops}
        for (action, effect_frozen), entry in backward_infer_minimal_effects(demo_data, current_operators=refined_ops).items():
            delete = op_index[(action, effect_frozen)].pre & set.intersection(
                *[set(entry['demos'][i][0] - entry['demos'][i][2]) for i in range(len(entry['demos']))]
            )
            op_index[(action, effect_frozen)].delete = delete

        for op in op_index.values():
            for actual_op in operators:
                if op.action == actual_op.action:
                    if op.pre <= actual_op.pre and op.add <= actual_op.add and op.delete <= actual_op.delete:
                        valid_count += 1
                        if verbose:
                            print("VALID\n\tLEARNED |", op, "\n\tORIGINAL|", actual_op)
                    else:
                        invalid_count += 1
                        if verbose:
                            print("INVALID\n\tLEARNED |", op, "\n\tORIGINAL|", actual_op)

    return invalid_count, valid_count

def deduplicate_predicates_by_equivalence(demos, operators, num_preds):
    from collections import defaultdict

    # Step 1: Build truth vectors for each predicate
    pred_vectors = defaultdict(list)

    for traj, goal in demos:
        for s, _, s_prime in traj:
            for i in range(num_preds):
                pred_vectors[i].append(int(i in s))
                pred_vectors[i].append(int(i in s_prime))
        for i in range(num_preds):
            pred_vectors[i].append(int(i in goal))

    # Step 2: Group predicates with identical truth vectors
    vector_to_preds = defaultdict(list)
    for pred, vec in pred_vectors.items():
        vector_to_preds[tuple(vec)].append(pred)

    # Step 3: Build a mapping from redundant predicate -> representative
    replace_map = {}
    for group in vector_to_preds.values():
        representative = min(group)  # pick smallest index as canonical
        for pred in group:
            replace_map[pred] = representative

    # Step 4: Replace predicates in demos
    new_demos = []
    for traj, goal in demos:
        new_traj = []
        for s, a, s_prime in traj:
            s_new = {replace_map[p] for p in s}
            s_prime_new = {replace_map[p] for p in s_prime}
            new_traj.append((s_new, a, s_prime_new))
        new_goal = {replace_map[p] for p in goal}
        new_demos.append((new_traj, new_goal))

    # Step 5: Replace predicates in operators
    new_operators = []
    for op in operators:
        pre = {replace_map[p] for p in op.pre}
        add = {replace_map[p] for p in op.add}
        delete = {replace_map[p] for p in op.delete}
        new_operators.append(Operator(pre, add, delete, op.action))

    return new_demos, new_operators, replace_map

def augment_demos_with_missing_ground_truth_ops(demos, learned_ops, true_ops, num_preds, num_augments=1):
    from collections import defaultdict

    # Index learned ops by (action, add, delete, pre)
    learned_op_keys = set(
        (op.action, frozenset(op.add), frozenset(op.delete), frozenset(op.pre))
        for op in learned_ops
    )

    augmented = []

    for true_op in true_ops:
        key = (true_op.action, frozenset(true_op.add), frozenset(true_op.delete), frozenset(true_op.pre))
        if key in learned_op_keys:
            continue  # already learned correctly

        # Add demos for this missing operator
        for _ in range(num_augments):
            possible_goals = None
            while not possible_goals:
                base_state = set(random.sample(range(num_preds), random.randint(2, num_preds)))
                false_pre = set()
                for op in learned_ops:
                    if op.action == true_op.action and op.add == true_op.add:
                        false_pre |= op.pre - true_op.pre
                base_state -= false_pre # remove wrong precondition
                base_state |= true_op.pre  # ensure it's applicable
                next_state = true_op.apply(base_state)

                # Choose a goal that is newly added by the operator
                possible_goals = true_op.add - base_state

            goal = possible_goals
            demo = [(base_state.copy(), true_op.action, next_state.copy())]
            augmented.append((demo, goal))

    return demos + augmented

# invalids, valids = run_operator_learning_trials(num_trials=50, verbose=False)
# print(f"Number of invalid learned operators: {invalids} / {invalids+valids}")

#
results = {"tot_matches":[], "tot_soft_matches":[],"tot_exsoft_matches":[], "tot_num_ops":[]}
#
pred_pool = [i for i in range(NUM_PREDICATES)]
operators = [generate_random_operator(pred_pool, i) for i in range(NUM_OPERATORS)]
all_demo_data = generate_planned_demo_trajectories(operators, NUM_TRAJS, max_depth=TRAJ_MAX)

# all_demo_data, operators, pred_replace_map = deduplicate_predicates_by_equivalence(all_demo_data, operators, NUM_PREDICATES)
# print("Predicate replacement map:", pred_replace_map)

for num_trajs in range(1, 52, 10):

    print("\n--- Ground Truth Operators ---")
    for op in sorted(operators, key=lambda x: x.action):
        print(op)
    print()

    print("Demos:")
    # for traj in demo_data:
    #     print("Goal:", traj[1], "Length:", len(traj[0]))
    demo_data = all_demo_data[:num_trajs]
    print(len(demo_data))

    op_index = None

    potential_op_sets = {}
    for run_i in range(100):
        refined_ops = learn_operators_from_demos(demo_data, max_iters=1000, verbose=False)
        op_index = {(op.action, frozenset(op.add)): op for op in refined_ops}
        for (action, effect_frozen), entry in backward_infer_minimal_effects(demo_data, current_operators=refined_ops).items():
            delete = op_index[(action, effect_frozen)].pre & set.intersection(
                *[set(entry['demos'][i][0] - entry['demos'][i][2]) for i in range(len(entry['demos']))]
            )
            op_index[(action, effect_frozen)].delete = delete

        augmented_demo_data = augment_demos_with_missing_ground_truth_ops(
            demo_data, list(op_index.values()), operators, NUM_PREDICATES, num_augments=5
        )

        refined_ops = learn_operators_from_demos(augmented_demo_data, max_iters=1000, verbose=False)
        op_index = {(op.action, frozenset(op.add)): op for op in refined_ops}
        for (action, effect_frozen), entry in backward_infer_minimal_effects(augmented_demo_data, current_operators=refined_ops).items():
            delete = op_index[(action, effect_frozen)].pre & set.intersection(
                *[set(entry['demos'][i][0] - entry['demos'][i][2]) for i in range(len(entry['demos']))]
            )
            op_index[(action, effect_frozen)].delete = delete

        print("\n--- Final Learned Operators ---")
        for op in sorted(op_index.values(), key=lambda x: x.action):
            print(op)
        print()

        num_match = 0
        num_soft_match = 0
        num_exsoft_match = 0
        actions = set()
        for actual_op in operators:
            is_match = False
            is_soft_match = False
            is_exsoft_match = False
            for op in op_index.values():
                actions.add(op.action)
                if op.action == actual_op.action:
                    if len(op.pre - actual_op.pre) <= 2 and op.add == actual_op.add and op.delete == actual_op.delete:
                        is_exsoft_match = True
                        if len(op.pre - actual_op.pre) <= 1 and op.add == actual_op.add and op.delete == actual_op.delete:
                            is_soft_match = True
                            if op.pre == actual_op.pre and op.add == actual_op.add and op.delete == actual_op.delete:
                                is_match = True
            if is_exsoft_match:
                num_exsoft_match += 1
            if is_soft_match:
                num_soft_match += 1
            if is_match:
                num_match += 1
        num_actions = len(actions)
        actions = set()
        for traj in demo_data[:run_i]:
            for (s, a, s_prime) in traj[0]:
                actions.add(a)
        new_op_set_str = str([op for op in sorted(op_index.values(), key=lambda x: x.action)])
        if new_op_set_str in potential_op_sets:
            potential_op_sets[new_op_set_str] += 1
        else:
            potential_op_sets[new_op_set_str] = 0
        results["tot_matches"].append((num_trajs, run_i, num_match))
        results["tot_soft_matches"].append((num_trajs, run_i, num_soft_match))
        results["tot_exsoft_matches"].append((num_trajs, run_i, num_exsoft_match))
        results["tot_num_ops"].append((num_trajs, run_i, len(potential_op_sets.keys()), num_actions))
        print(num_trajs, run_i, num_match, num_soft_match, num_exsoft_match, len(potential_op_sets.keys()), num_actions)

    print("\n--- Ground Truth Operators ---")
    for op in sorted(operators, key=lambda x: x.action):
        print(op)
    print()

import pickle

filename = 'HITL_more_results.pkl'

# Open the file in binary write mode ('wb')
with open(filename, 'wb') as file:
    pickle.dump(results, file)

import ipdb; ipdb.set_trace()









quit()

for _ in range(10):
    demo_data = augment_demos_with_missing_ground_truth_ops(
        demo_data, list(op_index.values()), operators, NUM_PREDICATES, num_augments=1
    )

    refined_ops = learn_operators_from_demos(demo_data, max_iters=1000, verbose=False)
    op_index = {(op.action, frozenset(op.add)): op for op in refined_ops}
    for (action, effect_frozen), entry in backward_infer_minimal_effects(demo_data[:run_i], current_operators=refined_ops).items():
        delete = op_index[(action, effect_frozen)].pre & set.intersection(
            *[set(entry['demos'][i][0] - entry['demos'][i][2]) for i in range(len(entry['demos']))]
        )
        op_index[(action, effect_frozen)].delete = delete

print("\n--- Final Learned Operators ---")
for op in sorted(op_index.values(), key=lambda x: x.action):
    print(op)
print()

num_match = 0
actions = set()
for actual_op in operators:
    is_match = False
    for op in op_index.values():
        actions.add(op.action)
        if op.action == actual_op.action:
            if op.pre == actual_op.pre and op.add == actual_op.add and op.delete == actual_op.delete:
                is_match = True
    if is_match:
        num_match += 1
    else:
        print(actual_op)
num_actions = len(actions)
print("final", num_match, num_actions)


# TODO Fix Delete Effects

import ipdb; ipdb.set_trace()