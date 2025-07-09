# Cleaned-up and organized version of your operator learning code
# - Uses dataclasses
# - Removes duplication
# - Adds helpers
# - Keeps everything in one file

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict
import random
import numpy as np
from collections import deque, defaultdict

# --- Config ---

np.random.seed(1)
random.seed(1)

NUM_PREDICATES = 10
NUM_OPERATORS = 10
TRAJ_MAX = 10
TRAJ_LEN = 5
NUM_TRAJS = 100

# --- Operator Representation ---

@dataclass
class Operator:
    action: int
    pre: Set[int] = field(default_factory=set)
    add: Set[int] = field(default_factory=set)
    delete: Set[int] = field(default_factory=set)

    def is_applicable(self, state: Set[int]) -> bool:
        return self.pre.issubset(state)

    def apply(self, state: Set[int]) -> Set[int]:
        if not self.is_applicable(state):
            return state
        return (state - self.delete) | self.add

# Utility

def op_key(op: Operator) -> Tuple[int, frozenset]:
    return (op.action, frozenset(op.add))

def is_equivalent(op1: Operator, op2: Operator) -> bool:
    return op1.pre == op2.pre and op1.add == op2.add and op1.delete == op2.delete

def is_covered_by(op1: Operator, op2: Operator) -> bool:
    return op1.pre >= op2.pre and op1.add == op2.add and op1.delete >= op2.delete

# Planning

def plan(start: Set[int], goal: Set[int], operators: List[Operator], max_depth=10):
    visited, queue = set(), deque([(start.copy(), [])])
    
    while queue:
        state, path = queue.popleft()
        state_key = frozenset(state)
        if state_key in visited: continue
        visited.add(state_key)

        if goal.issubset(state): return path
        if len(path) >= max_depth: continue

        for op in operators:
            if op.is_applicable(state):
                next_state = op.apply(state)
                if next_state != state:
                    queue.append((next_state, path + [(state.copy(), op.action, next_state.copy())]))
    return None

# Random Generator

def generate_random_operator(pred_pool: List[int], action_id: int) -> Operator:
    pre = set(random.sample(pred_pool, random.randint(1, 3)))
    effects = list(set(pred_pool) - pre)
    add = set(random.sample(effects, random.randint(1, min(2, len(effects)))))
    delete = set(random.sample(list(pre), random.randint(0, len(pre))))
    return Operator(action=action_id, pre=pre, add=add, delete=delete)

# Reachability

def compute_reachable_states(init_state: Set[int], operators: List[Operator], max_iters=100) -> Set[frozenset]:
    reached_states, frontier = set(), [init_state.copy()]
    reachable = set()

    for _ in range(max_iters):
        new_frontier = []
        for state in frontier:
            key = frozenset(state)
            if key in reached_states: continue
            reached_states.add(key)
            reachable.add(key)

            for op in operators:
                if op.is_applicable(state):
                    next_state = op.apply(state)
                    if frozenset(next_state) not in reached_states:
                        new_frontier.append(next_state)

        if not new_frontier: break
        frontier = new_frontier

    return reachable

# Demo Generation

def generate_planned_demo_trajectories(operators: List[Operator], num_trajs: int, max_depth: int) -> List[Tuple[List[Tuple[Set[int], int, Set[int]]], Set[int]]]:
    demos, attempts = [], 0

    while len(demos) < num_trajs and attempts < 100000:
        attempts += 1
        init_state = set(random.sample(range(NUM_PREDICATES), random.randint(2, NUM_PREDICATES)))
        reachable = compute_reachable_states(init_state, operators) - {frozenset(init_state)}
        if not reachable: continue

        plan_traj = []
        goals = list(reachable)

        while goals and len(plan_traj) < TRAJ_LEN:
            goal_state = random.choice(goals)
            goal = set(goal_state) - init_state
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

# Backward Pass

def backward_infer_minimal_effects(demo_data, current_operators=None):
    candidate_ops = defaultdict(lambda: {'demos': []})
    op_index = {(op.action, frozenset(op.add)): op for op in current_operators} if current_operators else {}

    for traj, goal in sorted(demo_data, key=lambda x: len(x[0])):
        current_goal = goal.copy()

        for (s, action, s_prime) in reversed(traj):
            effect = s_prime - s
            if not effect:
                raise Exception("No effect")
            necessary_effect = effect if len(effect) == 1 else effect & current_goal
            key = (action, frozenset(necessary_effect))
            candidate_ops[key]['demos'].append((s, action, s_prime))

            preconditions = op_index.get(key, Operator(action)).pre
            current_goal = (current_goal - necessary_effect) | preconditions

    return candidate_ops

# Forward Refinement

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
            preconditions = op_index.get(key, Operator(a)).pre
            current_goal = (current_goal - necessary_effect) | preconditions

    for traj, _ in demos:
        state = traj[0][0]
        for (s_true, a_true, s_next_true) in traj:
            applicable = [op for op in learned_operators if op.is_applicable(state)]
            if not applicable: break
            op_planner = random.choice(applicable)

            key_true = (a_true, frozenset(s_next_true - s_true))
            op_true = op_index.get(key_true)
            if op_true is None: continue

            if op_planner is op_true:
                state = op_true.apply(state)
                continue

            key_planner = (op_planner.action, frozenset(op_planner.add))
            support = support_sets.get(key_planner, [])
            if not support: continue

            common_preds = set.intersection(*support)
            potential_preds = common_preds - state
            if not potential_preds: continue

            preds_to_add = set(random.sample(list(potential_preds), random.randint(1, len(potential_preds))))
            op_planner.pre.update(preds_to_add)
            state = op_true.apply(state)

    return list(op_index.values())

# Learning Loop

def learn_operators_from_demos(demo_data, max_iters=100, verbose=True):
    learned_ops = []

    for iteration in range(max_iters):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")

        # Backward pass
        candidate_ops = backward_infer_minimal_effects(demo_data, current_operators=learned_ops or None)

        # Create new operators from candidate effects
        op_index = {}
        for (action, effect_frozen), entry in candidate_ops.items():
            op = Operator(action=action, add=set(effect_frozen), pre=set(), delete=set())
            op_index[(action, frozenset(op.add))] = op
        learned_ops = list(op_index.values())

        # Assign each transition to at most one operator
        demo_assignments = defaultdict(list)
        assigned_transitions = {}
        for traj, _ in demo_data:
            for s, a, s_prime in traj:
                effect = s_prime - s
                matching_keys = [(key, op) for key, op in op_index.items() if key[0] == a and key[1] <= set(effect)]
                matching_vals = [len(set(effect) - key[1]) for key, op in op_index.items() if key[0] == a and key[1] <= set(effect)]
                if matching_keys:
                    best_key, _ = matching_keys[np.argmin(matching_vals)]  # choose the first match
                    demo_assignments[best_key].append((s, a, s_prime))
                    if (frozenset(s), a, frozenset(s_prime)) in assigned_transitions:
                        assigned_transitions[(frozenset(s), a, frozenset(s_prime))] += 1
                    else:
                        assigned_transitions[(frozenset(s), a, frozenset(s_prime))] = 1

        # Assert total assignments match demo transitions
        total_transitions = sum(len(traj) for traj, _ in demo_data)
        assert sum(assigned_transitions.values()) == total_transitions, (
            f"Assigned transitions ({len(assigned_transitions)}) != total demo transitions ({total_transitions})")
        used_keys = set(demo_assignments.keys())
        learned_ops = [op for key, op in op_index.items() if key in used_keys]

        if verbose:
            print("Backward Pass Result:")
            for op in sorted(learned_ops, key=lambda x: x.action):
                print(op)

        # Forward refinement
        learned_ops = refine_by_plan_divergence(demo_data, learned_ops)

        if verbose:
            print("Forward Pass Result:")
            for op in sorted(learned_ops, key=lambda x: x.action):
                print(op)

    return learned_ops

# Evaluation

def evaluate_learned_operators(learned_ops: List[Operator], true_ops: List[Operator], verbose=True, is_equal=True) -> Tuple[int, int]:
    valid, invalid = 0, 0
    for true_op in true_ops:
        match_found = False
        for learned_op in learned_ops:
            if is_equal:
                if learned_op.action == true_op.action and is_equivalent(true_op, learned_op):
                    match_found = True
                    break
            else:
                if learned_op.action == true_op.action and is_covered_by(true_op, learned_op):
                    match_found = True
                    break
        if match_found:
            valid += 1
            if verbose:
                print(f"VALID\n\tLEARNED | {learned_op}\n\tTRUE    | {true_op}")
        else:
            invalid += 1
            if verbose:
                print(f"INVALID\n\tLEARNED | MISSING\n\tTRUE    | {true_op}")
    return valid, invalid

def augment_demos_with_missing_ground_truth_ops(demos, learned_ops, true_ops, num_preds, num_augments=1):
    from collections import defaultdict

    learned_op_keys = set(
        (op.action, frozenset(op.add), frozenset(op.delete), frozenset(op.pre))
        for op in learned_ops
    )

    augmented = []

    for true_op in true_ops:
        key = (true_op.action, frozenset(true_op.add), frozenset(true_op.delete), frozenset(true_op.pre))
        if key in learned_op_keys:
            continue

        for _ in range(num_augments):
            possible_goals = None
            while not possible_goals:
                base_state = set(random.sample(range(num_preds), random.randint(2, num_preds)))
                false_pre = set()
                for op in learned_ops:
                    if op.action == true_op.action and op.add == true_op.add:
                        false_pre |= op.pre - true_op.pre
                base_state -= false_pre
                base_state |= true_op.pre
                next_state = true_op.apply(base_state)
                possible_goals = true_op.add - base_state

            goal = possible_goals
            demo = [(base_state.copy(), true_op.action, next_state.copy())]
            augmented.append((demo, goal))

    return demos + augmented

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
        new_operators.append(Operator(op.action, pre, add, delete))

    return new_demos, new_operators, replace_map

# Main Execution

def main():
    pred_pool = list(range(NUM_PREDICATES))
    operators = [generate_random_operator(pred_pool, i) for i in range(NUM_OPERATORS)]
    print("\n--- Ground Truth Operators ---")
    for op in sorted(operators, key=lambda x: x.action):
        print(op)

    demo_data = generate_planned_demo_trajectories(operators, NUM_TRAJS, max_depth=TRAJ_MAX)
    print(f"\nGenerated {len(demo_data)} demo trajectories.")

    op_nums = {i: 0 for i in range(NUM_OPERATORS)}
    for traj, goal in  demo_data:
        for t in traj:
            op_nums[t[1]] += 1

    print("\nOPERATOR DEMO COUNT:", op_nums,"\n")


    results = {}
    for op_set_idx in range(100):
        learned_ops = learn_operators_from_demos(demo_data, max_iters=10, verbose=False)
        op_index = {(op.action, frozenset(op.add)): op for op in learned_ops}
        for (action, effect_frozen), entry in backward_infer_minimal_effects(demo_data, current_operators=learned_ops).items():
            if (action, effect_frozen) in op_index:
                delete = op_index[(action, effect_frozen)].pre & set.intersection(
                    *[set(entry['demos'][i][0] - entry['demos'][i][2]) for i in range(len(entry['demos']))]
                )
                op_index[(action, effect_frozen)].delete = delete
        print(op_set_idx, sum([len(op.pre) for op in learned_ops]))
        valid, invalid = evaluate_learned_operators(learned_ops, operators, verbose=False)
        print(f"Summary: {valid} valid / {valid + invalid} total operators correctly learned.\n")
        val = sum([len(op.pre) for op in learned_ops])
        if val in results:
            results[val] += [float(valid) / float(valid + invalid)]
        else:
            results[val] = [float(valid) / float(valid + invalid)]

    print([(k, np.mean(v)) for k,v in sorted(results.items(), key=lambda x: np.mean(x[1]))])
    
    print("\n--- Final Learned Operators ---")
    for op in sorted(learned_ops, key=lambda x: x.action):
        print(op)

    valid, invalid = evaluate_learned_operators(learned_ops, operators, verbose=True)
    print(f"\nSummary: {valid} valid / {valid + invalid} total operators correctly learned.")

    valid, invalid = evaluate_learned_operators(learned_ops, operators, verbose=True, is_equal=False)
    print(f"\n(Coverage) Summary: {valid} valid / {valid + invalid} total operators correctly learned.")


    # Augment and re-evaluate
    for round in range(1, 10):
        demo_data = augment_demos_with_missing_ground_truth_ops(demo_data, learned_ops, operators, NUM_PREDICATES, num_augments=1)
        learned_ops = learn_operators_from_demos(demo_data, max_iters=5, verbose=False)
        op_index = {(op.action, frozenset(op.add)): op for op in learned_ops}
        for (action, effect_frozen), entry in backward_infer_minimal_effects(demo_data, current_operators=learned_ops).items():
            if (action, effect_frozen) in op_index:
                delete = op_index[(action, effect_frozen)].pre & set.intersection(
                    *[set(entry['demos'][i][0] - entry['demos'][i][2]) for i in range(len(entry['demos']))]
                )
                op_index[(action, effect_frozen)].delete = delete

        print(f"\n--- After Augmentation Round {round} ---")
        # for op in sorted(learned_ops, key=lambda x: x.action):
        #     print(op)

        valid, invalid = evaluate_learned_operators(learned_ops, operators, verbose=False)
        print(f"Round {round} Summary: {valid} valid / {valid + invalid} total operators correctly learned.")

        valid, invalid = evaluate_learned_operators(learned_ops, operators, verbose=False, is_equal=False)
        print(f"\n(Coverage) Summary: {valid} valid / {valid + invalid} total operators correctly learned.")

    print(f"\n--- After Augmentation Round {round} ---")
    for op in sorted(learned_ops, key=lambda x: x.action):
        print(op)

    ##################

    # PREDICATES = {
    #     "at_A": 0,
    #     "at_B": 1,
    #     "handempty": 2,
    #     "holding_block1": 3,
    #     "holding_block2": 4,
    #     "inside_block1": 7,
    #     "inside_block2": 8,
    # }

    # OPERATORS = [
    #     # move from B to A
    #     Operator(pre={PREDICATES["at_B"]}, add={PREDICATES["at_A"]}, delete={PREDICATES["at_B"]}, action=0),
    #     # move from A to B
    #     Operator(pre={PREDICATES["at_A"]}, add={PREDICATES["at_B"]}, delete={PREDICATES["at_A"]}, action=1),

    #     # pick block1
    #     Operator(pre={PREDICATES["at_A"], PREDICATES["handempty"]},
    #             add={PREDICATES["holding_block1"]},
    #             delete={PREDICATES["handempty"]},
    #             action=2),

    #     # pick block2
    #     Operator(pre={PREDICATES["at_A"], PREDICATES["handempty"]},
    #             add={PREDICATES["holding_block2"]},
    #             delete={PREDICATES["handempty"]},
    #             action=3),

    #     # place block1 in box (at B)
    #     Operator(pre={PREDICATES["at_B"], PREDICATES["holding_block1"]},
    #             add={PREDICATES["inside_block1"], PREDICATES["handempty"]},
    #             delete={PREDICATES["holding_block1"]},
    #             action=4),

    #     # place block2 in box (at B)
    #     Operator(pre={PREDICATES["at_B"], PREDICATES["holding_block2"]},
    #             add={PREDICATES["inside_block2"], PREDICATES["handempty"]},
    #             delete={PREDICATES["holding_block2"]},
    #             action=5),
    # ]


    # init_state = {
    #     PREDICATES["at_B"], PREDICATES["handempty"]
    # }

    # actions = [0, 2, 1, 4, 0, 3, 1, 5]  # move→pick→move→place (block1), move→pick→move→place (block2)

    # state = init_state.copy()
    # traj1 = []

    # for action_id in actions:
    #     op = OPERATORS[action_id]
    #     next_state = op.apply(state)
    #     traj1.append((state.copy(), action_id, next_state.copy()))
    #     state = next_state.copy()

    # goal1 = {PREDICATES["inside_block1"], PREDICATES["inside_block2"], PREDICATES["handempty"]}

    # actions = [0, 3, 1, 5, 0, 2, 1, 4]  # move→pick→move→place (block1), move→pick→move→place (block2)

    # state = init_state.copy()
    # traj2 = []

    # for action_id in actions:
    #     op = OPERATORS[action_id]
    #     next_state = op.apply(state)
    #     traj2.append((state.copy(), action_id, next_state.copy()))
    #     state = next_state.copy()

    # goal2 = {PREDICATES["inside_block1"], PREDICATES["inside_block2"], PREDICATES["handempty"]}
    # demo_data = [(traj1, goal1), (traj2, goal2)]

    # demo_data, operators, pred_replace_map = deduplicate_predicates_by_equivalence(demo_data, OPERATORS, NUM_PREDICATES)
    # print("Predicate replacement map:", pred_replace_map)

    # print("\n--- Ground Truth Operators ---")
    # for op in sorted(operators, key=lambda x: x.action):
    #     print(op)
    # print()

    # print("Demos:")
    # # for traj in demo_data:
    # #     print("Goal:", traj[1], "Length:", len(traj[0]))
    # print(len(demo_data))

    # learned_ops = learn_operators_from_demos(demo_data, max_iters=1000, verbose=True)
    # op_index = {(op.action, frozenset(op.add)): op for op in learned_ops}
    # for (action, effect_frozen), entry in backward_infer_minimal_effects(demo_data, current_operators=learned_ops).items():
    #     if (action, effect_frozen) in op_index:
    #         delete = op_index[(action, effect_frozen)].pre & set.intersection(
    #             *[set(entry['demos'][i][0] - entry['demos'][i][2]) for i in range(len(entry['demos']))]
    #         )
    #         op_index[(action, effect_frozen)].delete = delete
    # valid, invalid = evaluate_learned_operators(learned_ops, operators, verbose=False)
    # print(f"Summary: {valid} valid / {valid + invalid} total operators correctly learned.\n")


    # print("\n--- Final Learned Operators ---")
    # for op in sorted(learned_ops, key=lambda x: x.action):
    #     print(op)

    # valid, invalid = evaluate_learned_operators(learned_ops, operators, verbose=True)
    # print(f"\nSummary: {valid} valid / {valid + invalid} total operators correctly learned.")

    # # Augment and re-evaluate
    # demo_data = augment_demos_with_missing_ground_truth_ops(demo_data, learned_ops, operators, NUM_PREDICATES, num_augments=1)
    # learned_ops = learn_operators_from_demos(demo_data, max_iters=5, verbose=False)
    # op_index = {(op.action, frozenset(op.add)): op for op in learned_ops}
    # for (action, effect_frozen), entry in backward_infer_minimal_effects(demo_data, current_operators=learned_ops).items():
    #     if (action, effect_frozen) in op_index:
    #         delete = op_index[(action, effect_frozen)].pre & set.intersection(
    #             *[set(entry['demos'][i][0] - entry['demos'][i][2]) for i in range(len(entry['demos']))]
    #         )
    #         op_index[(action, effect_frozen)].delete = delete

    # print(f"\n--- After Augmentation Round {round} ---")
    # for op in sorted(learned_ops, key=lambda x: x.action):
    #     print(op)

    # valid, invalid = evaluate_learned_operators(learned_ops, operators, verbose=True)
    # print(f"HITL Summary: {valid} valid / {valid + invalid} total operators correctly learned.")

# Batch Evaluation Experiment

def main_experiment():
    results = {
        "num_actions": [],
        "equivalent": [], "covered": [], "overfit": [], "missed": [],
        "hitl_1_equivalent": [], "hitl_1_covered": [], "hitl_1_overfit": [], "hitl_1_missed": [],
        "hitl_5_equivalent": [], "hitl_5_covered": [], "hitl_5_overfit": [], "hitl_5_missed": []
    }

    for num_trajs in range(1, 102, 10):
        for run_i in range(100):
            pred_pool = list(range(NUM_PREDICATES))
            operators = [generate_random_operator(pred_pool, i) for i in range(NUM_OPERATORS)]
            all_demo_data = generate_planned_demo_trajectories(operators, 110, max_depth=TRAJ_MAX)
            # all_demo_data, operators, pred_replace_map = deduplicate_predicates_by_equivalence(all_demo_data, operators, NUM_PREDICATES)
            # print("Predicate replacement map:", pred_replace_map)
            
            demo_data = all_demo_data[:num_trajs]
            unique_actions = set()
            for traj, _ in demo_data:
                for (s, a, s_prime) in traj:
                    unique_actions.add(a)
            results["num_actions"].append((num_trajs, len(unique_actions)))

            refined_ops = learn_operators_from_demos(demo_data, max_iters=1000, verbose=False)
            op_index = {(op.action, frozenset(op.add)): op for op in refined_ops}
            for (action, effect_frozen), entry in backward_infer_minimal_effects(demo_data, current_operators=refined_ops).items():
                if (action, effect_frozen) in op_index:
                    delete = op_index[(action, effect_frozen)].pre & set.intersection(
                        *[set(entry['demos'][i][0] - entry['demos'][i][2]) for i in range(len(entry['demos']))]
                    )
                    op_index[(action, effect_frozen)].delete = delete

            refined_ops_hitl = augment_demos_with_missing_ground_truth_ops(
                demo_data, list(op_index.values()), operators, NUM_PREDICATES, num_augments=1
            )
            refined_ops_hitl5 = augment_demos_with_missing_ground_truth_ops(
                demo_data, list(op_index.values()), operators, NUM_PREDICATES, num_augments=5
            )

            def count_matches(learned_ops):
                eq, cov, ofit = 0, 0, 0
                done_ops = set()
                for actual_op in operators:
                    for op in learned_ops:
                        if op.action == actual_op.action:
                            if (actual_op.action, frozenset(actual_op.add)) not in done_ops:
                                if op.pre == actual_op.pre and op.add == actual_op.add and op.delete == actual_op.delete:
                                    eq += 1
                                    done_ops.add((actual_op.action, frozenset(actual_op.add)))
                                elif is_covered_by(actual_op, op):
                                    cov += 1
                                    done_ops.add((actual_op.action, frozenset(actual_op.add)))
                                elif len(op.pre - actual_op.pre) <= 2 and op.add == actual_op.add and op.delete == actual_op.delete:
                                    ofit += 1
                                    done_ops.add((actual_op.action, frozenset(actual_op.add)))
                return eq, cov, ofit

            eq, cov, ofit = count_matches(list(op_index.values()))
            miss = len(unique_actions) - (eq+cov+ofit)
            results["equivalent"].append((num_trajs, run_i, eq))
            results["covered"].append((num_trajs, run_i, cov))
            results["overfit"].append((num_trajs, run_i, ofit))
            results["missed"].append((num_trajs, run_i, miss))

            ops_hitl1 = learn_operators_from_demos(refined_ops_hitl, max_iters=1000, verbose=False)
            eq1, cov1, ofit1 = count_matches(ops_hitl1)
            miss1 = NUM_OPERATORS - (eq1+cov1+ofit1)
            results["hitl_1_equivalent"].append((num_trajs, run_i, eq1))
            results["hitl_1_covered"].append((num_trajs, run_i, cov1))
            results["hitl_1_overfit"].append((num_trajs, run_i, ofit1))
            results["hitl_1_missed"].append((num_trajs, run_i, miss1))

            ops_hitl5 = learn_operators_from_demos(refined_ops_hitl5, max_iters=1000, verbose=False)
            eq5, cov5, ofit5 = count_matches(ops_hitl5)
            miss5 = NUM_OPERATORS - (eq5+cov5+ofit5)
            results["hitl_5_equivalent"].append((num_trajs, run_i, eq5))
            results["hitl_5_covered"].append((num_trajs, run_i, cov5))
            results["hitl_5_overfit"].append((num_trajs, run_i, ofit5))
            results["hitl_5_missed"].append((num_trajs, run_i, miss5))

            print(f"Trajs: {num_trajs}, Actions: {len(unique_actions)}, Run: {run_i}, Eq: {eq}, Cov: {cov}, Ofit: {ofit}, Missed: {miss}, HITL1: Eq={eq1}, Cov={cov1}, Ofit={ofit1}, Missed={miss1}, HITL5: Eq={eq5}, Cov={cov5}, Ofit={ofit5}, Missed={miss5}")

    import pickle
    with open('HITL_experiment_results_random.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
    main_experiment()