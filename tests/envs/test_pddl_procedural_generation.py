"""Tests for PDDL procedural generation."""

import numpy as np

from predicators.src.envs.pddl_procedural_generation import \
    create_blocks_pddl_generator


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
