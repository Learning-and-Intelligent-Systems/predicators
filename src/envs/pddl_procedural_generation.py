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


def create_forest_pddl_generator(min_size: int,
                                 max_size: int) -> PDDLProblemGenerator:
    """Create a generator for forest problems."""
    return functools.partial(_generate_forest_problems, min_size, max_size)


def _generate_forest_problems(min_size: int, max_size: int, num_problems: int,
                              rng: np.random.Generator) -> List[str]:
    problems = []
    for _ in range(num_problems):
        height = rng.integers(min_size, max_size + 1)
        width = rng.integers(min_size, max_size + 1)
        problem = _generate_forest_problem(height, width, rng)
        problems.append(problem)
    return problems


def _generate_random_forest_grid(grid_height, grid_width):
    I, G, W, P, X, H = range(6)
    
    I_row = np.random.randint(0, grid_height)
    I_col = np.random.randint(0, grid_width)

    G_row = np.random.randint(0, grid_height)
    G_col = np.random.randint(0, grid_width)
    while (G_row, G_col) == (I_row, I_col):
        G_row = np.random.randint(0, grid_height)
        G_col = np.random.randint(0, grid_width)

    random_path =  random_grid_walk((I_row, I_col), (G_row, G_col), set(), grid_height, grid_width, None)
    assert random_path

    remaining_coords = {(r, c) for r in range(grid_height) for c in range(grid_width)} - set(random_path)

    grid = [[None for c in range(grid_width)] for r in range(grid_height)]

    for non_path_coord in remaining_coords:
        loc_prob = np.random.uniform()
        if loc_prob <= 0.5:
            grid[non_path_coord[0]][non_path_coord[1]] = X
        else:
            grid[non_path_coord[0]][non_path_coord[1]] = W

    last_was_hill = False
    for i, path_coord in enumerate(random_path):
        loc_prob = np.random.uniform()
        if path_coord == (I_row, I_col):
            grid[path_coord[0]][path_coord[1]] = I
        elif path_coord == (G_row, G_col): 
            grid[path_coord[0]][path_coord[1]] = G
        elif i > 1 and not last_was_hill and loc_prob <= 0.2:
            grid[path_coord[0]][path_coord[1]] = H
            last_was_hill = True
        else:
            grid[path_coord[0]][path_coord[1]] = P
    
    #print("Random path", sorted(random_path))
    #print("Reamaining_coords", sorted(remaining_coords))

    for r in range(grid_height):
        for c in range(grid_width):
            assert grid[r][c] != None
    
    for r in grid:
        print(r)
    return grid


def random_grid_walk(currCoords, goalCoords, visited, grid_height, grid_width, previousCoords):
    if currCoords == goalCoords:
        return [currCoords]
    
    for delta in np.random.permutation([[0, 1], [1, 0], [0, -1], [-1, 0]]):
        new_coord = (currCoords[0] + delta[0], currCoords[1] + delta[1])
        if new_coord[0] < 0 or new_coord[0] >= grid_height or new_coord[1] < 0 or new_coord[1] >= grid_width: 
            #print("Out of bounds")
            continue

        if new_coord in visited:
            #print("Already visited")
            continue

        adjacent_excluding_previous = {(currCoords[0] + adj_delta[0], currCoords[1] + adj_delta[1]) for adj_delta in [[0, 1], [1, 0], [0, -1], [-1, 0]]} - {previousCoords}
        adjacent_hit = False
        for adjacent_coord in adjacent_excluding_previous:
            if adjacent_coord in visited:
                adjacent_hit = True
        if adjacent_hit:
            #print("Adjacent hit")
            continue

        if not reachable(new_coord, goalCoords, visited | {currCoords}, grid_height, grid_width):
            #print("Not reachable from here")
            continue
                        
        grid_walk_from_child = random_grid_walk(new_coord, goalCoords, visited | {currCoords}, grid_height, grid_width, currCoords)
        if grid_walk_from_child != None:
            return [currCoords] + grid_walk_from_child

    return None


def reachable(currCoords, goalCoords, prev_visited, grid_height, grid_width):
    queue = [(currCoords, prev_visited.copy())]
    coord_queue = [currCoords]
    visited = prev_visited.copy()

    while len(queue) > 0:
        curr, curr_visited = queue[0]
        del queue[0]
        del coord_queue[0]

        if curr == goalCoords:
            return True

        for delta in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
            newC = (curr[0] + delta[0], curr[1] + delta[1])
            if newC[0] < 0 or newC[0] >= grid_height or newC[1] < 0 or newC[1] >= grid_width: 
                #print("Out of bounds")
                continue

            if newC in visited or newC in coord_queue:
                #print("Already visited or in queue")
                continue

            adjacent_excluding_previous = {(newC[0] + adj_delta[0], newC[1] + adj_delta[1]) for adj_delta in [[0, 1], [1, 0], [0, -1], [-1, 0]]} - {curr}
            adjacent_hit = False
            for adjacent_coord in adjacent_excluding_previous:
                if adjacent_coord in curr_visited:
                    adjacent_hit = True
            if adjacent_hit:
                #print("Adjacent hit")
                continue

            queue.append((newC, curr_visited | {curr}))
            coord_queue.append(newC)

    return False


def grid_A_star(grid_weights, Irow, Icol, Grow, Gcol):
    for row in grid_weights:
        print(row)
    print(Irow, Icol, Grow, Gcol)

    queue = []
    visited = set()
    distances = {}
    heapq.heappush(queue, (0, (Irow, Icol)))
    predecessors = {}

    while heapq:
        dist, coords = heapq.heappop(queue)
        print(dist, coords)
        if coords == (Grow, Gcol):
            break
        if coords in visited:
            continue

        visited.add(coords)

        for delta in ([0, 1], [1, 0], [0, -1], [-1, 0]):
            new_coord = (coords[0] + delta[0], coords[1] + delta[1])
            if new_coord[0] < 0 or new_coord[0] >= len(grid_weights) or new_coord[1] < 0 or new_coord[1] >= len(grid_weights): 
                continue

            new_dist = dist + grid_weights[new_coord[0]][new_coord[1]]
            if new_coord not in distances.keys() or new_dist < distances[new_coord]:
                distances[new_coord] = new_dist
                predecessors[new_coord] = coords
                heapq.heappush(queue, (new_dist, new_coord))
    
    curr_coord = (Grow, Gcol)
    path = [curr_coord]
    while curr_coord != (Irow, Icol):
        path.append(predecessors[curr_coord])
        curr_coord = predecessors[curr_coord]
    
    return path



def _generate_forest_problem(height: int, width: int,
                             rng: np.random.Generator) -> str:
    grid = np.array(_generate_random_forest_grid(height, width))

    import ipdb
    ipdb.set_trace()
