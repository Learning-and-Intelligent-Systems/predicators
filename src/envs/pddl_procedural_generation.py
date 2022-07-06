"""Procedurally generates PDDL problem strings."""

import functools
from typing import Collection, List, Optional, Set

import numpy as np

from predicators.src.structs import PDDLProblemGenerator


def create_blocks_pddl_generator(
        min_num_blocks: int,
        max_num_blocks: int,
        min_num_blocks_goal: int,
        max_num_blocks_goal: int,
        new_pile_prob: float,
        force_goal_not_achieved: bool = True) -> PDDLProblemGenerator:
    """Create a generator for blocks problems."""
    if force_goal_not_achieved:
        assert new_pile_prob < 1.0, ("Impossible to create an unsolved problem"
                                     " with new_pile_prob = 1.0.")
    return functools.partial(_generate_blocks_problems, min_num_blocks,
                             max_num_blocks, min_num_blocks_goal,
                             max_num_blocks_goal, new_pile_prob,
                             force_goal_not_achieved)


def _generate_blocks_problems(min_num_blocks: int, max_num_blocks: int,
                              min_num_blocks_goal: int,
                              max_num_blocks_goal: int, new_pile_prob: float,
                              force_goal_not_achieved: bool, num_problems: int,
                              rng: np.random.Generator) -> List[str]:
    assert max_num_blocks_goal <= min_num_blocks
    problems = []
    for _ in range(num_problems):
        num_blocks = rng.integers(min_num_blocks, max_num_blocks + 1)
        num_goal_blocks = rng.integers(min_num_blocks_goal,
                                       max_num_blocks_goal + 1)
        problem = _generate_blocks_problem(num_blocks, num_goal_blocks,
                                           new_pile_prob,
                                           force_goal_not_achieved, rng)
        problems.append(problem)
    return problems


def _generate_blocks_problem(num_blocks: int, num_goal_blocks: int,
                             new_pile_prob: float,
                             force_goal_not_achieved: bool,
                             rng: np.random.Generator) -> str:
    # Repeat until the goal does not hold in the initial state.
    while True:
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
                # Add the block to the most recently created pile.
                pile_group[-1].append(block)
        # Create strings from pile groups.
        init_strs = _blocks_piles_to_strs(piles)
        goal_strs = _blocks_piles_to_strs(
            goal_piles, excluded_predicates={"clear", "handempty"})
        if not force_goal_not_achieved or not goal_strs.issubset(init_strs):
            break
    # Finalize PDDL problem str.
    blocks_str = " ".join(blocks)
    init_str = " ".join(sorted(init_strs))
    goal_str = " ".join(sorted(goal_strs))
    problem_str = f"""(define (problem blocks-procgen)
    (:domain BLOCKS)
    (:objects {blocks_str} - block)
    (:init {init_str})
    (:goal (and {goal_str}))
)"""
    return problem_str


def _blocks_piles_to_strs(
        piles: List[List[str]],
        excluded_predicates: Optional[Collection[str]] = None) -> Set[str]:
    if excluded_predicates is None:
        excluded_predicates = set()

    all_strs = set()

    if "handempty" not in excluded_predicates:
        all_strs.add("(handempty)")

    for pile in piles:
        if "ontable" not in excluded_predicates:
            all_strs.add(f"(ontable {pile[0]})")
        if "clear" not in excluded_predicates:
            all_strs.add(f"(clear {pile[-1]})")
        if "on" not in excluded_predicates:
            for i in range(1, len(pile)):
                top = pile[i]
                bottom = pile[i - 1]
                all_strs.add(f"(on {top} {bottom})")

    return all_strs


def create_delivery_pddl_generator(
        min_num_locs: int, max_num_locs: int, min_num_want_locs: int,
        max_num_want_locs: int, min_num_extra_newspapers: int,
        max_num_extra_newspapers: int) -> PDDLProblemGenerator:
    """Create a generator for delivery problems."""
    return functools.partial(_generate_delivery_problems, min_num_locs,
                             max_num_locs, min_num_want_locs,
                             max_num_want_locs, min_num_extra_newspapers,
                             max_num_extra_newspapers)


def _generate_delivery_problems(min_num_locs: int, max_num_locs: int,
                                min_num_want_locs: int, max_num_want_locs: int,
                                min_num_extra_newspapers: int,
                                max_num_extra_newspapers: int,
                                num_problems: int,
                                rng: np.random.Generator) -> List[str]:
    problems = []
    for _ in range(num_problems):
        num_locs = rng.integers(min_num_locs, max_num_locs + 1)
        num_want_locs = rng.integers(min_num_want_locs, max_num_want_locs + 1)
        num_extra_newspapers = rng.integers(min_num_extra_newspapers,
                                            max_num_extra_newspapers + 1)
        num_newspapers = num_want_locs + num_extra_newspapers
        problem = _generate_delivery_problem(num_locs, num_want_locs,
                                             num_newspapers, rng)
        problems.append(problem)
    return problems


def _generate_delivery_problem(num_locs: int, num_want_locs: int,
                               num_newspapers: int,
                               rng: np.random.Generator) -> str:
    init_strs = set()
    goal_strs = set()

    # Create locations.
    locs = [f"loc-{i}" for i in range(num_locs)]
    # Randomize the home location.
    home_loc = locs[rng.choice(num_locs)]
    possible_targets = [l for l in locs if l != home_loc]
    target_locs = rng.choice(possible_targets, num_want_locs, replace=False)
    # Add the initial state and goal atoms about the locations.
    for loc in locs:
        if loc == home_loc:
            init_strs.add(f"(isHomeBase {loc})")
            init_strs.add(f"(at {loc})")
            init_strs.add(f"(safe {loc})")
            init_strs.add(f"(satisfied {loc})")
        if loc in target_locs:
            init_strs.add(f"(wantsPaper {loc})")
            init_strs.add(f"(safe {loc})")
            goal_strs.add(f"(satisfied {loc})")

    # Create papers.
    papers = [f"paper-{i}" for i in range(num_newspapers)]
    # Add the initial state atoms about the papers.
    for paper in papers:
        init_strs.add(f"(unpacked {paper})")

    # Finalize PDDL problem str.
    locs_str = "\n        ".join(locs)
    papers_str = "\n        ".join(papers)
    init_str = " ".join(sorted(init_strs))
    goal_str = " ".join(sorted(goal_strs))
    problem_str = f"""(define (problem delivery-procgen)
    (:domain delivery)
    (:objects
        {locs_str} - loc
        {papers_str} - paper
    )
    (:init {init_str})
    (:goal (and {goal_str}))
)"""

    return problem_str


def create_spanner_pddl_generator(min_num_nuts: int
        
        
                                 ) -> PDDLProblemGenerator:
    """Create a generator for  problems."""
    import ipdb; ipdb.set_trace()
    
    
def _generate_spanner_problem():
    import ipdb; ipdb.set_trace()
