"""Script to evaluate learned predicate classifiers on held-out test cases."""

import os
from typing import List, Optional, Set

import numpy as np

from predicators.src import utils
from predicators.src.approaches import create_approach
from predicators.src.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.src.envs import BaseEnv, create_new_env
from predicators.src.envs.cover import CoverEnv
from predicators.src.main import _generate_interaction_results, \
    _generate_or_load_offline_dataset
from predicators.src.settings import CFG
from predicators.src.structs import Action, Dataset, Predicate, Task
from predicators.src.teacher import Teacher


def _main() -> None:
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    # Create results directory.
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)
    # Create classes. Note that seeding happens inside the env and approach.
    env = create_new_env(CFG.env, do_cache=True)
    # The action space and options need to be seeded externally, because
    # env.action_space and env.options are often created during env __init__().
    env.action_space.seed(CFG.seed)
    for option in env.options:
        option.params_space.seed(CFG.seed)
    assert env.goal_predicates.issubset(env.predicates)
    preds, _ = utils.parse_config_excluded_predicates(env)
    # Create the train tasks.
    train_tasks = env.get_train_tasks()
    # If train tasks have goals that involve excluded predicates, strip those
    # predicate classifiers to prevent leaking information to the approaches.
    stripped_train_tasks = [
        utils.strip_task(task, preds) for task in train_tasks
    ]
    # Create the agent (approach).
    approach = create_approach(CFG.approach, preds, env.options, env.types,
                               env.action_space, stripped_train_tasks)
    assert approach.is_learning_based
    assert isinstance(approach, BilevelPlanningApproach)
    offline_dataset = _generate_or_load_offline_dataset(env, train_tasks)
    _run_pipeline(env, approach, stripped_train_tasks, offline_dataset)


def _run_pipeline(env: BaseEnv,
                  approach: BilevelPlanningApproach,
                  train_tasks: List[Task],
                  offline_dataset: Optional[Dataset] = None) -> None:
    assert offline_dataset is not None, "Missing offline dataset"
    total_num_transitions = sum(
        len(traj.actions) for traj in offline_dataset.trajectories)
    approach.load(online_learning_cycle=None)
    teacher = Teacher(train_tasks)
    for i in range(CFG.num_online_learning_cycles):
        print(f"\n\nONLINE LEARNING CYCLE {i}\n")
        if total_num_transitions > CFG.online_learning_max_transitions:
            break
        interaction_requests = approach.get_interaction_requests()
        if not interaction_requests:
            break  # agent doesn't want to learn anything more; terminate
        interaction_results, _ = _generate_interaction_results(
            env, teacher, interaction_requests, i)
        total_num_transitions += sum(
            len(result.actions) for result in interaction_results)
        approach.load(online_learning_cycle=i)
    # Only evaluate the last cycle
    _evaluate_preds(
        approach._get_current_predicates(),  # pylint: disable=protected-access
        env)


def _evaluate_preds(preds: Set[Predicate], env: BaseEnv) -> None:
    if CFG.env == "cover":
        assert isinstance(env, CoverEnv)
        return _evaluate_preds_cover(preds, env)
    raise NotImplementedError(
        f"Held out predicate test set not yet implemented for {CFG.env}")


def _evaluate_preds_cover(preds: Set[Predicate], env: CoverEnv) -> None:
    Holding = [p for p in preds if p.name == "Holding"][0]
    Covers = [p for p in preds if p.name == "Covers"][0]
    HoldingGT = [p for p in env.predicates if p.name == "Holding"][0]
    CoversGT = [p for p in env.predicates if p.name == "Covers"][0]
    # Create initial state
    task = env.get_test_tasks()[0]
    state = task.init
    block0 = [b for b in state if b.name == "block0"][0]
    block1 = [b for b in state if b.name == "block1"][0]
    target0 = [b for b in state if b.name == "target0"][0]
    target1 = [b for b in state if b.name == "target1"][0]
    robot = [b for b in state if b.name == "robby"][0]
    blocks = [block0, block1]
    targets = [target0, target1]
    block_poses = [0.15, 0.605]
    target_poses = [0.375, 0.815]
    for block, pose in zip(blocks, block_poses):
        # [is_block, is_target, width, pose, grasp]
        state.set(block, "pose", pose)
        # Make sure blocks are not held
        state.set(block, "grasp", -1)
    for target, pose in zip(targets, target_poses):
        # [is_block, is_target, width, pose]
        state.set(target, "pose", pose)
    state.set(robot, "hand", 0.0)
    # Test 1: no blocks overlap any targets, none are held
    atoms = utils.abstract(state, (Holding, Covers))
    atoms_gt = utils.abstract(state, (HoldingGT, CoversGT))
    print(f"False positives: {atoms - atoms_gt}\n"
          f"False negatives: {atoms_gt - atoms}")
    # Pick up block0 and place it over target0
    action = Action(np.array([0.15], dtype=np.float32))
    state = env.simulate(state, action)
    action = Action(np.array([0.375], dtype=np.float32))
    state = env.simulate(state, action)
    # Test 2: block0 covers target0
    atoms = utils.abstract(state, (Holding, Covers))
    atoms_gt = utils.abstract(state, (HoldingGT, CoversGT))
    print(f"False positives: {atoms - atoms_gt}\n"
          f"False negatives: {atoms_gt - atoms}")
    # Pick and place block1 so it partially overlaps target1
    action = Action(np.array([0.63], dtype=np.float32))
    state = env.simulate(state, action)
    action = Action(np.array([0.815], dtype=np.float32))
    state = env.simulate(state, action)
    # Test 3: block1 does not completely cover target1
    atoms = utils.abstract(state, (Holding, Covers))
    atoms_gt = utils.abstract(state, (HoldingGT, CoversGT))
    print(f"False positives: {atoms - atoms_gt}\n"
          f"False negatives: {atoms_gt - atoms}")


if __name__ == "__main__":
    _main()
