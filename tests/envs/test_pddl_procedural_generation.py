"""Tests for PDDL procedural generation."""

import numpy as np

from predicators.envs.pddl_procedural_generation import \
    create_blocks_pddl_generator, create_delivery_pddl_generator, \
    create_forest_pddl_generator, create_spanner_pddl_generator


def _split_pddl_problem_str(problem_str):
    """A hacky helper function for splitting up a PDDL problem string.

    Not reliable, definitely don't use anywhere outside of this test
    file.
    """
    def _parse_helper(start_marker, end_marker=None):
        str_start = problem_str.index(start_marker)
        start = str_start + len(start_marker) + 1
        if end_marker is None:
            end = len(problem_str)
        else:
            str_end = problem_str.index(end_marker)
            end = str_end - len(end_marker) - 1
        return problem_str[start:end + 1]

    object_str = _parse_helper("(:objects", "(:init")
    init_str = _parse_helper("(:init", "(:goal")
    goal_str = _parse_helper("(:goal")

    return object_str, init_str, goal_str


def test_create_blocks_pddl_generator():
    """Tests for create_blocks_pddl_generator()."""

    # Test with new_pile_prob = 0.0, meaning that all problems should have one
    # pile in the initial state and goal.
    gen = create_blocks_pddl_generator(min_num_blocks=3,
                                       max_num_blocks=3,
                                       min_num_blocks_goal=2,
                                       max_num_blocks_goal=2,
                                       new_pile_prob=0.0)
    rng = np.random.default_rng(123)
    problem_strs = gen(10, rng)
    for problem_str in problem_strs:
        obj_str, init_str, goal_str = _split_pddl_problem_str(problem_str)
        # There should be exactly 3 blocks.
        for i in range(3):
            assert f"b{i}" in obj_str
        assert "b4" not in obj_str
        assert " - block" in obj_str
        # One ontable in init and one in goal.
        assert init_str.count("ontable ") == 1
        assert goal_str.count("ontable ") == 1
        # The goal should have exactly two objects, so one on.
        assert goal_str.count("on ") == 1

    # Test with new_pile_prob = 1.0, meaning that all problems should have no
    # "on" predicates in the initial state or goal.
    gen = create_blocks_pddl_generator(min_num_blocks=3,
                                       max_num_blocks=3,
                                       min_num_blocks_goal=2,
                                       max_num_blocks_goal=2,
                                       new_pile_prob=1.0,
                                       force_goal_not_achieved=False)
    rng = np.random.default_rng(123)
    problem_strs = gen(10, rng)
    for problem_str in problem_strs:
        _, init_str, goal_str = _split_pddl_problem_str(problem_str)
        assert init_str.count("ontable ") == 3
        assert goal_str.count("ontable ") == 2
        assert init_str.count("on ") == 0
        assert goal_str.count("on ") == 0


def test_create_delivery_pddl_generator():
    """Tests for create_delivery_pddl_generator()."""
    gen = create_delivery_pddl_generator(min_num_locs=2,
                                         max_num_locs=2,
                                         min_num_want_locs=1,
                                         max_num_want_locs=1,
                                         min_num_extra_newspapers=1,
                                         max_num_extra_newspapers=1)
    rng = np.random.default_rng(123)
    problem_strs = gen(2, rng)
    for problem_str in problem_strs:
        obj_str, init_str, goal_str = _split_pddl_problem_str(problem_str)
        # There should be exactly 2 locations.
        for i in range(2):
            assert f"loc-{i}" in obj_str
        assert "loc-2" not in obj_str
        assert " - loc" in obj_str
        # There should be exactly 3 papers.
        for i in range(2):
            assert f"paper-{i}" in obj_str
        assert "paper-3" not in obj_str
        assert " - paper" in obj_str
        # One at in init.
        assert init_str.count("at ") == 1
        # The goal should have exactly one satisfied.
        assert goal_str.count("satisfied ") == 1


def test_create_spanner_pddl_generator():
    """Tests for create_spanner_pddl_generator()."""
    gen = create_spanner_pddl_generator(min_nuts=2,
                                        max_nuts=2,
                                        min_extra_span=1,
                                        max_extra_span=1,
                                        min_locs=2,
                                        max_locs=2)

    rng = np.random.default_rng(123)
    problem_strs = gen(2, rng)
    for problem_str in problem_strs:
        obj_str, init_str, goal_str = _split_pddl_problem_str(problem_str)
        # There should be exactly 2 nuts.
        for i in range(2):
            assert f"nut{i}" in obj_str
        assert "nut2" not in obj_str
        assert " - nut" in obj_str
        # There should be exactly 2 locations.
        for i in range(2):
            assert f"location{i}" in obj_str
        assert "location2" not in obj_str
        assert " - location" in obj_str
        # There should be exactly 3 spanners.
        for i in range(2):
            assert f"spanner{i}" in obj_str
        assert "spanner3" not in obj_str
        assert " - spanner" in obj_str
        # One at bob in init.
        assert init_str.count("at bob ") == 1
        # The goal should have exactly two tightened.
        assert goal_str.count("tightened ") == 2


def test_create_forest_pddl_generator():
    """Tests for create_forest_pddl_generator()."""
    gen = create_forest_pddl_generator(min_size=5, max_size=5)
    rng = np.random.default_rng(123)
    problem_strs = gen(2, rng)
    for problem_str in problem_strs:
        obj_str, init_str, goal_str = _split_pddl_problem_str(problem_str)
        # There should be exactly 25 locations.
        for r in range(5):
            for c in range(5):
                assert f"r{r}_c{c}" in obj_str
        assert "r5_5" not in obj_str
        assert " - loc" in obj_str
        # One at in init.
        assert init_str.count("at ") == 1
        # The goal should have exactly one at.
        assert goal_str.count("at ") == 1
