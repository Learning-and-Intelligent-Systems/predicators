"""Procedurally generates PDDL problem strings."""

import functools
from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from predicators.structs import PDDLProblemGenerator

################################### Blocks ####################################


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


################################## Delivery ###################################


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


################################## Spanner ####################################


def create_spanner_pddl_generator(min_nuts: int, max_nuts: int,
                                  min_extra_span: int, max_extra_span: int,
                                  min_locs: int,
                                  max_locs: int) -> PDDLProblemGenerator:
    """Create a generator for spanner problems."""
    return functools.partial(_generate_spanner_problems, min_nuts, max_nuts,
                             min_extra_span, max_extra_span, min_locs,
                             max_locs)


def _generate_spanner_problems(min_nuts: int, max_nuts: int,
                               min_extra_span: int, max_extra_span: int,
                               min_locs: int, max_locs: int, num_problems: int,
                               rng: np.random.Generator) -> List[str]:
    problems = []
    for _ in range(num_problems):
        num_nuts = rng.integers(min_nuts, max_nuts + 1)
        num_extra_span = rng.integers(min_extra_span, max_extra_span + 1)
        num_spanners = num_nuts + num_extra_span
        num_locs = rng.integers(min_locs, max_locs + 1)
        problem = _generate_spanner_problem(num_nuts, num_spanners, num_locs,
                                            rng)
        problems.append(problem)
    return problems


def _generate_spanner_problem(num_nuts: int, num_spanners: int, num_locs: int,
                              rng: np.random.Generator) -> str:
    # Create objects.
    man = "bob"
    spanners = [f"spanner{i}" for i in range(num_spanners)]
    nuts = [f"nut{i}" for i in range(num_nuts)]
    locs = [f"location{i}" for i in range(num_locs)]
    shed = "shed"
    gate = "gate"

    # Create the initial state.
    init_strs = {f"(at {man} {shed})"}
    for spanner in spanners:
        loc = rng.choice(locs)
        init_strs.add(f"(at {spanner} {loc})")
        init_strs.add(f"(useable {spanner})")
    for nut in nuts:
        init_strs.add(f"(at {nut} {gate})")
        init_strs.add(f"(loose {nut})")
    init_strs.add(f"(link shed {locs[0]})")
    for i in range(num_locs - 1):
        init_strs.add(f"(link {locs[i]} {locs[i+1]})")
    init_strs.add(f"(link {locs[-1]} gate)")

    # Create the goal.
    goal_strs = {f"(tightened {nut})" for nut in nuts}

    # Finalize PDDL problem str.
    man_str = "\n        ".join([man])
    spanner_str = "\n        ".join(spanners)
    nuts_str = "\n        ".join(nuts)
    locs_str = "\n        ".join([shed, gate] + locs)
    init_str = " ".join(sorted(init_strs))
    goal_str = " ".join(sorted(goal_strs))
    problem_str = f"""(define (problem spanner-procgen)
    (:domain spanner)
    (:objects
        {man_str} - man
        {spanner_str} - spanner
        {nuts_str} - nut
        {locs_str} - location
    )
    (:init {init_str})
    (:goal (and {goal_str}))
    )"""

    return problem_str


################################### Forest ####################################

FOREST_I, FOREST_G, FOREST_W, FOREST_P, FOREST_X, FOREST_H = range(6)


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


def _generate_random_forest_grid(grid_height: int, grid_width: int,
                                 rng: np.random.Generator) -> List[List[int]]:

    I_row = rng.integers(0, grid_height)
    I_col = rng.integers(0, grid_width)

    while True:
        G_row = rng.integers(0, grid_height)
        G_col = rng.integers(0, grid_width)
        if (G_row, G_col) != (I_row, I_col):
            break

    random_path = _random_grid_walk((I_row, I_col), (G_row, G_col), set(),
                                    grid_height, grid_width, None, rng)
    assert random_path

    remaining_coords = {(r, c)
                        for r in range(grid_height)
                        for c in range(grid_width)} - set(random_path)

    grid = [[-1 for c in range(grid_width)] for r in range(grid_height)]

    for non_path_coord in remaining_coords:
        loc_prob = rng.uniform()
        if loc_prob <= 0.5:
            grid[non_path_coord[0]][non_path_coord[1]] = FOREST_X
        else:
            grid[non_path_coord[0]][non_path_coord[1]] = FOREST_W

    last_was_hill = False
    for i, path_coord in enumerate(random_path):
        loc_prob = rng.uniform()
        if path_coord == (I_row, I_col):
            grid[path_coord[0]][path_coord[1]] = FOREST_I
        elif path_coord == (G_row, G_col):
            grid[path_coord[0]][path_coord[1]] = FOREST_G
        elif i > 1 and not last_was_hill and loc_prob <= 0.2:
            grid[path_coord[0]][path_coord[1]] = FOREST_H
            last_was_hill = True
        else:
            grid[path_coord[0]][path_coord[1]] = FOREST_P

    for r in range(grid_height):
        for c in range(grid_width):
            assert grid[r][c] != -1

    return grid


def _random_grid_walk(
        curr_coords: Tuple[int, int], goal_coords: Tuple[int, int],
        visited: Set[Tuple[int, int]], grid_height: int, grid_width: int,
        previous_coords: Optional[Tuple[int, int]],
        rng: np.random.Generator) -> Optional[List[Tuple[int, int]]]:
    """Generates a random path through a grid.

    For aesthetic reasons, the grid is not allowed to self-intersect.
    """
    if curr_coords == goal_coords:
        return [curr_coords]

    for delta in rng.permutation([[0, 1], [1, 0], [0, -1], [-1, 0]]):
        new_coord = (curr_coords[0] + delta[0], curr_coords[1] + delta[1])
        # Out of bounds.
        if new_coord[0] < 0 or new_coord[0] >= grid_height or new_coord[
                1] < 0 or new_coord[1] >= grid_width:
            continue

        # Already visited.
        if new_coord in visited:
            continue

        # Prevent visiting coords that are adjacent to visited coords, except
        # for the most recent predecessor.
        adjacent_excluding_previous = {
            (curr_coords[0] + adj_delta[0], curr_coords[1] + adj_delta[1])
            for adj_delta in [[0, 1], [1, 0], [0, -1], [-1, 0]]
        } - {previous_coords}
        adjacent_hit = False
        for adjacent_coord in adjacent_excluding_previous:
            if adjacent_coord in visited:
                adjacent_hit = True
        if adjacent_hit:
            continue

        # Prevent visiting unreachable coordinates.
        if not _random_walk_reachable(new_coord, goal_coords,
                                      visited | {curr_coords}, grid_height,
                                      grid_width):
            continue

        # Successfully extended the path.
        grid_walk_from_child = _random_grid_walk(new_coord, goal_coords,
                                                 visited | {curr_coords},
                                                 grid_height, grid_width,
                                                 curr_coords, rng)
        if grid_walk_from_child is not None:
            return [curr_coords] + grid_walk_from_child

    return None


def _random_walk_reachable(curr_coords: Tuple[int,
                                              int], goal_coords: Tuple[int,
                                                                       int],
                           prev_visited: Set[Tuple[int, int]],
                           grid_height: int, grid_width: int) -> bool:
    """This helper for _random_grid_walk() checks whether some path to the goal
    still exists.

    This is used to rule out bad steps in the random walk that would
    never possibly reach the goal.
    """
    queue = [(curr_coords, prev_visited.copy())]
    coord_queue = [curr_coords]
    visited = prev_visited.copy()

    while len(queue) > 0:
        curr, curr_visited = queue[0]
        del queue[0]
        del coord_queue[0]

        if curr == goal_coords:
            return True

        for delta in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
            # Out of bounds.
            newC = (curr[0] + delta[0], curr[1] + delta[1])
            if newC[0] < 0 or newC[0] >= grid_height or newC[1] < 0 or newC[
                    1] >= grid_width:
                continue

            # Already visited or in queue.
            if newC in visited or newC in coord_queue:
                continue

            # Adjacent to already visited.
            adjacent_excluding_previous = {
                (newC[0] + adj_delta[0], newC[1] + adj_delta[1])
                for adj_delta in [[0, 1], [1, 0], [0, -1], [-1, 0]]
            } - {curr}
            adjacent_hit = False
            for adjacent_coord in adjacent_excluding_previous:
                if adjacent_coord in curr_visited:
                    adjacent_hit = True
            if adjacent_hit:
                continue

            queue.append((newC, curr_visited | {curr}))
            coord_queue.append(newC)

    return False


def _generate_forest_problem(height: int, width: int,
                             rng: np.random.Generator) -> str:
    grid = np.array(_generate_random_forest_grid(height, width, rng))

    init_strs = set()
    goal_strs = set()

    # Create location objects.
    objects = set()
    grid_locs = np.empty(grid.shape, dtype=object)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            obj = f"r{r}_c{c}"
            objects.add(obj)
            grid_locs[r, c] = obj

    # Add at, IsWater, and isHill to init_strs.
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            obj = grid_locs[r, c]
            if grid[r, c] == FOREST_I:
                init_strs.add(f"(at {obj})")

            if grid[r, c] != FOREST_W:
                init_strs.add(f"(isNotWater {obj})")

            if grid[r, c] == FOREST_H:
                init_strs.add(f"(isHill {obj})")
            else:
                init_strs.add(f"(isNotHill {obj})")

    # Add adjacent to init_strs.
    def get_neighbors(r: int, c: int) -> Iterator[Tuple[int, int]]:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                yield (nr, nc)

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            obj = grid_locs[r, c]
            for (nr, nc) in get_neighbors(r, c):
                nobj = grid_locs[nr, nc]
                init_strs.add(f"(adjacent {obj} {nobj})")

    # Add onTrail to init_strs.

    # Construct the entire path from the initial location to the goal while
    # staying on then trail.
    trail_path = []
    r, c = np.argwhere(grid == FOREST_I)[0]
    while True:
        trail_path.append((r, c))
        if grid[r, c] == FOREST_G:
            break
        for (nr, nc) in get_neighbors(r, c):
            if (nr, nc) in trail_path:
                continue
            if grid[nr, nc] in [FOREST_P, FOREST_G, FOREST_H]:
                r, c = nr, nc
                break
        else:  # pragma: no cover
            raise Exception("Should not happen")

    for (r, c), (nr, nc) in zip(trail_path[:-1], trail_path[1:]):
        obj = grid_locs[r, c]
        nobj = grid_locs[nr, nc]
        init_strs.add(f"(onTrail {obj} {nobj})")

    # Create goal str.
    goal_rcs = np.argwhere(grid == FOREST_G)
    assert len(goal_rcs) == 1
    goal_r, goal_c = goal_rcs[0]
    goal_obj = grid_locs[goal_r, goal_c]
    goal_strs.add(f"(at {goal_obj})")

    # Finalize PDDL problem str.
    locs_str = "\n        ".join(objects)
    init_str = " ".join(sorted(init_strs))
    goal_str = " ".join(sorted(goal_strs))
    problem_str = f"""(define (problem forest-procgen)
    (:domain forest)
    (:objects
        {locs_str} - loc
    )
    (:init {init_str})
    (:goal (and {goal_str}))
)"""

    return problem_str


################################### Gripper ####################################


def create_gripper_pddl_generator(min_num_rooms: int,
                                  max_num_rooms: int,
                                  min_num_balls: int,
                                  max_num_balls: int,
                                  prefix: str = "") -> PDDLProblemGenerator:
    """Create a generator for gripper problems."""
    return functools.partial(_generate_gripper_problems, min_num_rooms,
                             max_num_rooms, min_num_balls, max_num_balls,
                             prefix)


def _generate_gripper_problems(
    min_num_rooms: int,
    max_num_rooms: int,
    min_num_balls: int,
    max_num_balls: int,
    prefix: str,
    num_problems: int,
    rng: np.random.Generator,
) -> List[str]:
    problems = []
    for _ in range(num_problems):
        num_rooms = rng.integers(min_num_rooms, max_num_rooms + 1)
        num_balls = rng.integers(min_num_balls, max_num_balls + 1)
        problem = _generate_gripper_problem(num_rooms, num_balls, prefix, rng)
        problems.append(problem)
    return problems


def _generate_gripper_problem(
    num_rooms: int,
    num_balls: int,
    prefix: str,
    rng: np.random.Generator,
) -> str:

    init_strs = set()
    goal_strs = set()

    # Create objects and add typing predicates.
    room_objects = set()
    for r in range(num_rooms):
        obj = f"room{r}"
        room_objects.add(obj)
        init_strs.add(f"({prefix}room {obj})")

    ball_objects = set()
    for ball_id in range(num_balls):
        obj = f"ball{ball_id}"
        ball_objects.add(obj)
        init_strs.add(f"({prefix}ball {obj})")

    gripper_objects = set()
    num_grippers = 2
    for gripper_id in range(num_grippers):
        obj = f"gripper{gripper_id}"
        gripper_objects.add(obj)
        init_strs.add(f"({prefix}gripper {obj})")

    # Add free and at ground literals
    for gripper_object in gripper_objects:
        init_strs.add(f"({prefix}free {gripper_object})")

    initial_ball_rooms = {}
    for ball_object in ball_objects:
        initial_ball_rooms[ball_object] = rng.integers(num_rooms)
        init_strs.add(
            f"({prefix}at {ball_object} room{initial_ball_rooms[ball_object]})"
        )

    # Always start robby at room0
    init_strs.add(f"({prefix}at-robby room0)")

    # Create goal str.
    num_goal_balls = rng.integers(1, num_balls + 1)
    goal_balls = rng.choice(sorted(list(ball_objects)),
                            size=num_goal_balls,
                            replace=False)
    possible_goal_rooms = list(range(num_rooms))
    for goal_ball in goal_balls:
        possible_goal_rooms.remove(initial_ball_rooms[goal_ball])
        goal_room = rng.choice(possible_goal_rooms)
        goal_strs.add(f"({prefix}at {goal_ball} room{goal_room})")
        possible_goal_rooms.append(initial_ball_rooms[goal_ball])

    # Finalize PDDL problem str.
    all_objects = room_objects | ball_objects | gripper_objects
    objects_str = "\n        ".join(all_objects)
    init_str = " ".join(sorted(init_strs))
    goal_str = " ".join(sorted(goal_strs))
    problem_str = f"""(define (problem gripper-procgen)
    (:domain {prefix}gripper)
    (:objects
        {objects_str} - object
    )
    (:init {init_str})
    (:goal (and {goal_str}))
)"""
    return problem_str


################################### Ferry #####################################


def create_ferry_pddl_generator(min_locs: int, max_locs: int, min_cars: int,
                                max_cars: int) -> PDDLProblemGenerator:
    """Create a generator for ferry problems."""
    return functools.partial(_generate_ferry_problems, min_locs, max_locs,
                             min_cars, max_cars)


def _generate_ferry_problems(
    min_locs: int,
    max_locs: int,
    min_cars: int,
    max_cars: int,
    num_problems: int,
    rng: np.random.Generator,
) -> List[str]:
    problems = []
    for _ in range(num_problems):
        num_locs = rng.integers(min_locs, max_locs + 1)
        num_cars = rng.integers(min_cars, max_cars + 1)
        problem = _generate_ferry_problem(num_locs, num_cars, rng)
        problems.append(problem)
    return problems


def _generate_ferry_problem(
    num_locs: int,
    num_cars: int,
    rng: np.random.Generator,
) -> str:

    init_strs = set()
    goal_strs = set()

    # Create objects and add typing predicates.
    loc_objects = []
    for i in range(num_locs):
        obj = f"l{i}"
        loc_objects.append(obj)
        init_strs.add(f"(location {obj})")
    car_objects = []
    for i in range(num_cars):
        obj = f"c{i}"
        car_objects.append(obj)
        init_strs.add(f"(car {obj})")

    # Add not-eq predicates for locations.
    for loc1 in loc_objects:
        for loc2 in loc_objects:
            if loc1 != loc2:
                init_strs.add(f"(not-eq {loc1} {loc2})")

    # Add empty-ferry predicate.
    init_strs.add("(empty-ferry)")

    # Create car origins and destinations.
    for i, car in enumerate(car_objects):
        car_origin = rng.choice(loc_objects)
        init_strs.add(f"(at {car} {car_origin})")
        # Prevent trivial problems by forcing the first origin and dest to
        # differ.
        if i == 0:
            remaining_locs = [l for l in loc_objects if l != car_origin]
        else:
            remaining_locs = loc_objects
        car_dest = rng.choice(remaining_locs)
        goal_strs.add(f"(at {car} {car_dest})")

    # Create the ferry origin.
    ferry_origin = rng.choice(loc_objects)
    init_strs.add(f"(at-ferry {ferry_origin})")

    # Finalize PDDL problem str.
    all_objects = car_objects + loc_objects
    objects_str = "\n        ".join(all_objects)
    init_str = " ".join(sorted(init_strs))
    goal_str = " ".join(sorted(goal_strs))
    problem_str = f"""(define (problem ferry-procgen)
    (:domain ferry)
    (:objects
        {objects_str} - object
    )
    (:init {init_str})
    (:goal (and {goal_str}))
)"""

    return problem_str


################################## Miconic ####################################


def create_miconic_pddl_generator(
    min_num_buildings: int,
    max_num_buildings: int,
    min_num_floors: int,
    max_num_floors: int,
    min_num_passengers: int,
    max_num_passengers: int,
) -> PDDLProblemGenerator:
    """Create a generator for miconic problems."""
    return functools.partial(_generate_miconic_problems, min_num_buildings,
                             max_num_buildings, min_num_floors, max_num_floors,
                             min_num_passengers, max_num_passengers)


def _generate_miconic_problems(
    min_num_buildings: int,
    max_num_buildings: int,
    min_num_floors: int,
    max_num_floors: int,
    min_num_passengers: int,
    max_num_passengers: int,
    num_problems: int,
    rng: np.random.Generator,
) -> List[str]:
    problems = []
    for _ in range(num_problems):
        num_buildings = rng.integers(min_num_buildings, max_num_buildings + 1)
        num_floors = rng.integers(min_num_floors, max_num_floors + 1)
        num_passengers = rng.integers(min_num_passengers,
                                      max_num_passengers + 1)
        problem = _generate_miconic_problem(num_buildings, num_floors,
                                            num_passengers, rng)
        problems.append(problem)
    return problems


def _generate_miconic_problem(
    num_buildings: int,
    num_floors: int,
    num_passengers: int,
    rng: np.random.Generator,
) -> str:

    init_strs = set()
    goal_strs = set()

    # Create floors and passengers per building.
    buildings = list(range(num_buildings))
    building_to_floors: Dict[int, List[str]] = {b: [] for b in buildings}
    building_to_passengers: Dict[int, List[str]] = {b: [] for b in buildings}
    for b in buildings:
        # Create floors.
        for i in range(num_floors):
            floor = f"f{i}_b{b}"
            building_to_floors[b].append(floor)
        # Create passengers.
        for i in range(num_passengers):
            passenger = f"p{i}_b{b}"
            building_to_passengers[b].append(passenger)

    # Create above atoms.
    for b in buildings:
        building_floors = building_to_floors[b]
        for i, below_floor in enumerate(building_floors[:-1]):
            for above_floor in building_floors[i + 1:]:
                init_strs.add(f"(above {below_floor} {above_floor})")

    # Create origin and destination atoms.
    for b in buildings:
        building_passengers = building_to_passengers[b]
        free_floors = list(building_to_floors[b])
        for passenger in building_passengers:
            # Only allow one passenger origin or destination per floor.
            origin = rng.choice(free_floors)
            free_floors.remove(origin)
            destination = rng.choice(free_floors)
            init_strs.add(f"(origin {passenger} {origin})")
            init_strs.add(f"(destin {passenger} {destination})")

    # Create lift origins.
    for b in buildings:
        building_floors = building_to_floors[b]
        lift_origin = rng.choice(building_floors)
        init_strs.add(f"(lift-at {lift_origin})")

    # Create goal atoms.
    for b in buildings:
        building_passengers = building_to_passengers[b]
        for passenger in building_passengers:
            goal_strs.add(f"(served {passenger})")

    # Finalize PDDL problem str.
    all_floors = [f for fs in building_to_floors.values() for f in fs]
    all_passengers = [p for ps in building_to_passengers.values() for p in ps]
    floors_str = " ".join(sorted(all_floors))
    passengers_str = " ".join(sorted(all_passengers))
    init_str = " ".join(sorted(init_strs))
    goal_str = " ".join(sorted(goal_strs))
    problem_str = f"""(define (problem miconic-procgen)
    (:domain miconic)
    (:objects
        {floors_str} - floor
        {passengers_str} - passenger
    )
    (:init {init_str})
    (:goal (and {goal_str}))
)"""

    return problem_str

####### DETYPED MICONIC ########
def create_detypedmiconic_pddl_generator(
    min_num_buildings: int,
    max_num_buildings: int,
    min_num_floors: int,
    max_num_floors: int,
    min_num_passengers: int,
    max_num_passengers: int,
) -> PDDLProblemGenerator:
    """Create a generator for miconic problems."""
    return functools.partial(_generate_detypedmiconic_problems, min_num_buildings,
                             max_num_buildings, min_num_floors, max_num_floors,
                             min_num_passengers, max_num_passengers)


def _generate_detypedmiconic_problems(
    min_num_buildings: int,
    max_num_buildings: int,
    min_num_floors: int,
    max_num_floors: int,
    min_num_passengers: int,
    max_num_passengers: int,
    num_problems: int,
    rng: np.random.Generator,
) -> List[str]:
    problems = []
    for _ in range(num_problems):
        num_buildings = rng.integers(min_num_buildings, max_num_buildings + 1)
        num_floors = rng.integers(min_num_floors, max_num_floors + 1)
        num_passengers = rng.integers(min_num_passengers,
                                      max_num_passengers + 1)
        problem = _generate_detypedmiconic_problem(num_buildings, num_floors,
                                            num_passengers, rng)
        problems.append(problem)
    return problems


def _generate_detypedmiconic_problem(
    num_buildings: int,
    num_floors: int,
    num_passengers: int,
    rng: np.random.Generator,
) -> str:

    init_strs = set()
    goal_strs = set()

    # Create floors and passengers per building.
    buildings = list(range(num_buildings))
    building_to_floors: Dict[int, List[str]] = {b: [] for b in buildings}
    building_to_passengers: Dict[int, List[str]] = {b: [] for b in buildings}
    for b in buildings:
        # Create floors.
        for i in range(num_floors):
            floor = f"f{i}_b{b}"
            building_to_floors[b].append(floor)
            init_strs.add(f"(floor {floor})")
        # Create passengers.
        for i in range(num_passengers):
            passenger = f"p{i}_b{b}"
            building_to_passengers[b].append(passenger)
            init_strs.add(f"(passenger {passenger})")

    # Create above atoms.
    for b in buildings:
        building_floors = building_to_floors[b]
        for i, below_floor in enumerate(building_floors[:-1]):
            for above_floor in building_floors[i + 1:]:
                init_strs.add(f"(above {below_floor} {above_floor})")

    # Create origin and destination atoms.
    for b in buildings:
        building_passengers = building_to_passengers[b]
        free_floors = list(building_to_floors[b])
        for passenger in building_passengers:
            # Only allow one passenger origin or destination per floor.
            origin = rng.choice(free_floors)
            free_floors.remove(origin)
            destination = rng.choice(free_floors)
            init_strs.add(f"(origin {passenger} {origin})")
            init_strs.add(f"(destin {passenger} {destination})")

    # Create lift origins.
    for b in buildings:
        building_floors = building_to_floors[b]
        lift_origin = rng.choice(building_floors)
        init_strs.add(f"(lift-at {lift_origin})")

    # Create goal atoms.
    for b in buildings:
        building_passengers = building_to_passengers[b]
        for passenger in building_passengers:
            goal_strs.add(f"(served {passenger})")

    # Finalize PDDL problem str.
    all_floors = [f for fs in building_to_floors.values() for f in fs]
    all_passengers = [p for ps in building_to_passengers.values() for p in ps]
    floors_str = " ".join(sorted(all_floors))
    passengers_str = " ".join(sorted(all_passengers))
    init_str = " ".join(sorted(init_strs))
    goal_str = " ".join(sorted(goal_strs))
    problem_str = f"""(define (problem miconic-procgen)
    (:domain detypedmiconic)
    (:objects
        {floors_str} 
        {passengers_str} 
    )
    (:init {init_str})
    (:goal (and {goal_str}))
)"""

    return problem_str

################################## DETYPED Delivery ###################################


def create_detypeddelivery_pddl_generator(
        min_num_locs: int, max_num_locs: int, min_num_want_locs: int,
        max_num_want_locs: int, min_num_extra_newspapers: int,
        max_num_extra_newspapers: int) -> PDDLProblemGenerator:
    """Create a generator for delivery problems."""
    return functools.partial(_generate_detypeddelivery_problems, min_num_locs,
                             max_num_locs, min_num_want_locs,
                             max_num_want_locs, min_num_extra_newspapers,
                             max_num_extra_newspapers)


def _generate_detypeddelivery_problems(min_num_locs: int, max_num_locs: int,
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
        problem = _generate_detypeddelivery_problem(num_locs, num_want_locs,
                                             num_newspapers, rng)
        problems.append(problem)
    return problems


def _generate_detypeddelivery_problem(num_locs: int, num_want_locs: int,
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
        init_strs.add(f"(loc {loc})")

    # Create papers.
    papers = [f"paper-{i}" for i in range(num_newspapers)]
    # Add the initial state atoms about the papers.
    for paper in papers:
        init_strs.add(f"(unpacked {paper})")
        init_strs.add(f"(paper {paper})")

    # Finalize PDDL problem str.
    locs_str = "\n        ".join(locs)
    papers_str = "\n        ".join(papers)
    init_str = " ".join(sorted(init_strs))
    goal_str = " ".join(sorted(goal_strs))
    problem_str = f"""(define (problem delivery-procgen)
    (:domain delivery)
    (:objects
        {locs_str}
        {papers_str}
    )
    (:init {init_str})
    (:goal (and {goal_str}))
)"""

    return problem_str

################################## DETYPED Forest ###################################

def create_detypedforest_pddl_generator(min_size: int,
                                 max_size: int) -> PDDLProblemGenerator:
    """Create a generator for forest problems."""
    return functools.partial(_generate_detypedforest_problems, min_size, max_size)


def _generate_detypedforest_problems(min_size: int, max_size: int, num_problems: int,
                              rng: np.random.Generator) -> List[str]:
    problems = []
    for _ in range(num_problems):
        height = rng.integers(min_size, max_size + 1)
        width = rng.integers(min_size, max_size + 1)
        problem = _generate_detypedforest_problem(height, width, rng)
        problems.append(problem)
    return problems


def _generate_detypedforest_problem(height: int, width: int,
                             rng: np.random.Generator) -> str:
    grid = np.array(_generate_random_forest_grid(height, width, rng))

    init_strs = set()
    goal_strs = set()

    # Create location objects.
    objects = set()
    grid_locs = np.empty(grid.shape, dtype=object)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            obj = f"r{r}_c{c}"
            objects.add(obj)
            grid_locs[r, c] = obj

    # Add at, IsWater, and isHill to init_strs.
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            obj = grid_locs[r, c]
            if grid[r, c] == FOREST_I:
                init_strs.add(f"(at {obj})")

            if grid[r, c] != FOREST_W:
                init_strs.add(f"(isNotWater {obj})")

            if grid[r, c] == FOREST_H:
                init_strs.add(f"(isHill {obj})")
            else:
                init_strs.add(f"(isNotHill {obj})")
            
            init_strs.add(f"(loc {obj})")

    # Add adjacent to init_strs.
    def get_neighbors(r: int, c: int) -> Iterator[Tuple[int, int]]:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                yield (nr, nc)

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            obj = grid_locs[r, c]
            for (nr, nc) in get_neighbors(r, c):
                nobj = grid_locs[nr, nc]
                init_strs.add(f"(adjacent {obj} {nobj})")

    # Add onTrail to init_strs.

    # Construct the entire path from the initial location to the goal while
    # staying on then trail.
    trail_path = []
    r, c = np.argwhere(grid == FOREST_I)[0]
    while True:
        trail_path.append((r, c))
        if grid[r, c] == FOREST_G:
            break
        for (nr, nc) in get_neighbors(r, c):
            if (nr, nc) in trail_path:
                continue
            if grid[nr, nc] in [FOREST_P, FOREST_G, FOREST_H]:
                r, c = nr, nc
                break
        else:  # pragma: no cover
            raise Exception("Should not happen")

    for (r, c), (nr, nc) in zip(trail_path[:-1], trail_path[1:]):
        obj = grid_locs[r, c]
        nobj = grid_locs[nr, nc]
        init_strs.add(f"(onTrail {obj} {nobj})")

    # Create goal str.
    goal_rcs = np.argwhere(grid == FOREST_G)
    assert len(goal_rcs) == 1
    goal_r, goal_c = goal_rcs[0]
    goal_obj = grid_locs[goal_r, goal_c]
    goal_strs.add(f"(at {goal_obj})")

    # Finalize PDDL problem str.
    locs_str = "\n        ".join(objects)
    init_str = " ".join(sorted(init_strs))
    goal_str = " ".join(sorted(goal_strs))
    problem_str = f"""(define (problem forest-procgen)
    (:domain forest)
    (:objects
        {locs_str}
    )
    (:init {init_str})
    (:goal (and {goal_str}))
)"""

    return problem_str
