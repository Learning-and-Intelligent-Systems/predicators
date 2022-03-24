"""Procedurally generates PDDL problem strs."""
from functools import partial
from typing import Collection, List, Optional

import numpy as np

from predicators.src.structs import PDDLProblemGenerator


def create_blocks_pddl_generator(min_num_blocks: int, max_num_blocks: int,
                                 min_num_blocks_goal: int,
                                 max_num_blocks_goal: int,
                                 new_pile_prob: float) -> PDDLProblemGenerator:
    """Create a generator for blocks problems."""
    return partial(_generate_blocks_problems, min_num_blocks, max_num_blocks,
                   min_num_blocks_goal, max_num_blocks_goal, new_pile_prob)


def _generate_blocks_problems(min_num_blocks: int, max_num_blocks: int,
                              min_num_blocks_goal: int,
                              max_num_blocks_goal: int, new_pile_prob: float,
                              num_problems: int,
                              rng: np.random.Generator) -> List[str]:
    problems = []
    for _ in range(num_problems):
        num_blocks = rng.integers(min_num_blocks, max_num_blocks + 1)
        num_goal_blocks = rng.integers(min_num_blocks_goal,
                                       max_num_blocks_goal + 1)
        problem = _generate_blocks_problem(num_blocks, num_goal_blocks,
                                           new_pile_prob, rng)
        problems.append(problem)
    return problems


def _generate_blocks_problem(num_blocks: int, num_goal_blocks: int,
                             new_pile_prob: float,
                             rng: np.random.Generator) -> str:
    # Create blocks.
    blocks = [f"b{i}" for i in range(num_blocks)]
    goal_block_idxs = rng.choice(num_blocks,
                                 size=num_goal_blocks,
                                 replace=False)
    goal_blocks = [blocks[i] for i in goal_block_idxs]
    # Create piles for the initial state and goal.
    piles: List[List[str]] = []
    goal_piles: List[List[str]] = []
    for block_group, pile_group in ((blocks, piles), (goal_blocks,
                                                      goal_piles)):
        for block in block_group:
            if not pile_group or rng.uniform() < new_pile_prob:
                # Create a new pile.
                pile_group.append([])
            # Add the block to the pile.
            pile_group[-1].append(block)
    # Create strings from pile groups.
    init_str = _blocks_piles_to_str(piles)
    goal_str = _blocks_piles_to_str(goal_piles,
                                    excluded_predicates={"clear", "handempty"})
    # Finalize PDDL problem str.
    blocks_str = " ".join(blocks)
    problem_str = f"""(define (problem blocks-procgen)
    (:domain BLOCKS)
    (:objects {blocks_str} - block)
    (:init {init_str})
    (:goal (and {goal_str}))
)"""
    return problem_str


def _blocks_piles_to_str(
        piles: List[List[str]],
        excluded_predicates: Optional[Collection[str]] = None) -> str:
    if excluded_predicates is None:
        excluded_predicates = set()

    all_strs = []

    if "handempty" not in excluded_predicates:
        all_strs.append("(handempty)")

    for pile in piles:
        if "ontable" not in excluded_predicates:
            all_strs.append(f"(ontable {pile[0]})")
        if "clear" not in excluded_predicates:
            all_strs.append(f"(clear {pile[-1]})")
        if "on" not in excluded_predicates:
            for i in range(1, len(pile)):
                top = pile[i]
                bottom = pile[i - 1]
                all_strs.append(f"(on {top} {bottom})")

    return " ".join(all_strs)
