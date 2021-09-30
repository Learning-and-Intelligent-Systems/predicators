"""Test cases for the random actions approach class.
"""
from gym.spaces import Box  # type: ignore
from predicators.src.approaches import RandomActionsApproach
from predicators.src.structs import State, Type, Predicate, Task


def test_random_actions_approach():
    """Tests for RandomActionsApproach class.
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    pred1 = Predicate("On", [cup_type, plate_type], _classifier=None)
    pred2 = Predicate("Is", [cup_type, plate_type, plate_type],
                      _classifier=None)
    predicates = {pred1, pred2}
    cup = cup_type("cup")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup: [0.5], plate1: [1.0, 1.2], plate2: [-9.0, 1.0]})
    goal = {pred1([cup, plate1])}
    task = Task(state, goal)
    action_space = Box(0, 0.5, (1,))
    approach = RandomActionsApproach(lambda s, a: s, predicates, set(),
                                     action_space)
    approach.seed(123)
    policy = approach.solve(task, 500)
    for _ in range(10):
        act = policy(state)
        assert action_space.contains(act)
