"""Test cases for utils."""

import os
import time
from typing import Iterator, Tuple
import pytest
import numpy as np
from gym.spaces import Box
from predicators.src.structs import State, Type, ParameterizedOption, \
    Predicate, NSRT, Action, GroundAtom, DummyOption, STRIPSOperator, \
    LowLevelTrajectory
from predicators.src.approaches.oracle_approach import get_gt_nsrts
from predicators.src.envs import CoverEnv
from predicators.src.settings import CFG
from predicators.src import utils
from predicators.src.utils import _TaskPlanningHeuristic, \
    _PyperplanHeuristicWrapper


def test_aabb_volume():
    """Tests for get_aabb_volume."""
    lo = np.array([1.0, 1.5, -1.0])
    hi = np.array([2.0, 2.5, 0.0])
    assert utils.get_aabb_volume(lo, hi) == 1.0


def test_aabb_closest_point():
    """Tests for get_closest_point_on_aabb."""
    xyz = [1.5, 3.0, -2.5]
    lo = np.array([1.0, 1.5, -1.0])
    hi = np.array([2.0, 2.5, 0.0])
    assert utils.get_closest_point_on_aabb(xyz, lo, hi) == [1.5, 2.5, -1.0]


def test_intersects():
    """Tests for intersects()."""
    p1, p2 = (2, 5), (7, 6)
    p3, p4 = (2.5, 7.1), (7.4, 5.3)
    assert utils.intersects(p1, p2, p3, p4)

    p1, p2 = (1, 3), (5, 3)
    p3, p4 = (3, 7), (3, 2)
    assert utils.intersects(p1, p2, p3, p4)

    p1, p2 = (2, 5), (7, 6)
    p3, p4 = (2, 6), (7, 7)
    assert not utils.intersects(p1, p2, p3, p4)

    p1, p2 = (1, 1), (3, 3)
    p3, p4 = (2, 2), (4, 4)
    assert not utils.intersects(p1, p2, p3, p4)

    p1, p2 = (1, 1), (3, 3)
    p3, p4 = (1, 1), (6.7, 7.4)
    assert not utils.intersects(p1, p2, p3, p4)


def test_overlap():
    """Tests for overlap()."""
    l1, r1 = (1, 7), (3, 1)
    l2, r2 = (2, 10), (7, 3)
    assert utils.overlap(l1, r1, l2, r2)

    l1, r1 = (1, 7), (3, 1)
    l2, r2 = (1, 8), (6, 1)
    assert utils.overlap(l1, r1, l2, r2)

    l1, r1 = (1, 7), (5, 1)
    l2, r2 = (2, 4), (4, 2)
    assert utils.overlap(l1, r1, l2, r2)

    l1, r1 = (1, 4), (5, 1)
    l2, r2 = (2, 5), (4, 3)
    assert utils.overlap(l1, r1, l2, r2)

    l1, r1 = (1, 7), (3, 1)
    l2, r2 = (3, 5), (5, 3)
    assert not utils.overlap(l1, r1, l2, r2)

    l1, r1 = (1, 4), (3, 1)
    l2, r2 = (5, 8), (7, 6)
    assert not utils.overlap(l1, r1, l2, r2)

    l1, r1 = (1, 4), (6, 1)
    l2, r2 = (2, 7), (5, 5)
    assert not utils.overlap(l1, r1, l2, r2)


def test_option_to_trajectory():
    """Tests for option_to_trajectory()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0, 1.2]})

    def _simulator(s, a):
        ns = s.copy()
        assert a.arr.shape == (1, )
        ns[cup][0] += a.arr.item()
        return ns

    params_space = Box(0, 1, (1, ))

    def _policy(_1, _2, _3, p):
        return Action(p)

    def _initiable(_1, _2, _3, p):
        return p > 0.25

    def _terminal(s, _1, _2, _3):
        return s[cup][0] > 9.9

    parameterized_option = ParameterizedOption("Move", [], params_space,
                                               _policy, _initiable, _terminal)
    params = [0.1]
    option = parameterized_option.ground([], params)
    with pytest.raises(AssertionError):
        # option is not initiable from start state
        utils.option_to_trajectory(state, _simulator, option, max_num_steps=5)
    params = [0.5]
    option = parameterized_option.ground([], params)
    traj = utils.option_to_trajectory(state,
                                      _simulator,
                                      option,
                                      max_num_steps=100)
    assert len(traj.actions) == len(traj.states) - 1 == 19
    traj = utils.option_to_trajectory(state,
                                      _simulator,
                                      option,
                                      max_num_steps=10)
    assert len(traj.actions) == len(traj.states) - 1 == 10


def test_option_plan_to_policy():
    """Tests for option_plan_to_policy()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0, 1.2]})

    def _simulator(s, a):
        ns = s.copy()
        assert a.arr.shape == (1, )
        ns[cup][0] += a.arr.item()
        return ns

    params_space = Box(0, 1, (1, ))

    def _policy(_1, _2, _3, p):
        return Action(p)

    def _initiable(_1, _2, _3, p):
        return p > 0.25

    def _terminal(s, _1, _2, _3):
        return s[cup][0] > 9.9

    parameterized_option = ParameterizedOption("Move", [], params_space,
                                               _policy, _initiable, _terminal)
    params = [0.1]
    option = parameterized_option.ground([], params)
    plan = [option]
    policy = utils.option_plan_to_policy(plan)
    with pytest.raises(AssertionError):
        # option is not initiable from start state
        policy(state)
    params = [0.5]
    option = parameterized_option.ground([], params)
    plan = [option]
    policy = utils.option_plan_to_policy(plan)
    traj = utils.option_to_trajectory(state,
                                      _simulator,
                                      option,
                                      max_num_steps=100)
    assert len(traj.actions) == len(traj.states) - 1 == 19
    for t in range(19):
        assert not option.terminal(state)
        assert state.allclose(traj.states[t])
        action = policy(state)
        assert np.allclose(action.arr, traj.actions[t].arr)
        state = _simulator(state, action)
    assert option.terminal(state)
    with pytest.raises(utils.OptionPlanExhausted):
        # Ran out of options
        policy(state)


def test_strip_predicate():
    """Test for strip_predicate()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])

    def _classifier1(state, objects):
        cup, plate = objects
        return state[cup][0] + state[plate][0] < 2

    pred = Predicate("On", [cup_type, plate_type], _classifier1)
    cup = cup_type("cup")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup: [0.5], plate1: [1.0, 1.2], plate2: [-9.0, 1.0]})
    pred_stripped = utils.strip_predicate(pred)
    assert pred.name == pred_stripped.name
    assert pred.types == pred_stripped.types
    assert pred.holds(state, (cup, plate1))
    assert pred.holds(state, (cup, plate2))
    assert not pred_stripped.holds(state, (cup, plate1))
    assert not pred_stripped.holds(state, (cup, plate2))


def test_abstract():
    """Tests for abstract() and wrap_atom_predicates()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])

    def _classifier1(state, objects):
        cup, plate = objects
        return state[cup][0] + state[plate][0] < 2

    pred1 = Predicate("On", [cup_type, plate_type], _classifier1)

    def _classifier2(state, objects):
        cup, _, plate = objects
        return state[cup][0] + state[plate][0] < -1

    pred2 = Predicate("Is", [cup_type, plate_type, plate_type], _classifier2)
    cup = cup_type("cup")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup: [0.5], plate1: [1.0, 1.2], plate2: [-9.0, 1.0]})
    atoms = utils.abstract(state, {pred1, pred2})
    wrapped = utils.wrap_atom_predicates(atoms, "TEST-PREFIX-G-")
    assert len(wrapped) == len(atoms)
    for atom in wrapped:
        assert atom.predicate.name.startswith("TEST-PREFIX-G-")
    lifted_atoms = {pred1([cup_type("?cup"), plate_type("?plate")])}
    wrapped = utils.wrap_atom_predicates(lifted_atoms, "TEST-PREFIX-L-")
    assert len(wrapped) == len(lifted_atoms)
    for atom in wrapped:
        assert atom.predicate.name.startswith("TEST-PREFIX-L-")
    assert len(atoms) == 4
    assert atoms == {
        pred1([cup, plate1]),
        pred1([cup, plate2]),
        pred2([cup, plate1, plate2]),
        pred2([cup, plate2, plate2])
    }


def test_powerset():
    """Tests for powerset()."""
    lst = [3, 1, 2]
    pwr = list(utils.powerset(lst, exclude_empty=False))
    assert len(pwr) == len(set(pwr)) == 8
    assert tuple(lst) in pwr
    assert tuple() in pwr
    pwr = list(utils.powerset(lst, exclude_empty=True))
    assert len(pwr) == len(set(pwr)) == 7
    assert tuple(lst) in pwr
    assert tuple() not in pwr
    for s in utils.powerset(lst, exclude_empty=False):
        assert set(s).issubset(set(lst))
    assert not list(utils.powerset([], exclude_empty=True))
    assert list(utils.powerset([], exclude_empty=False)) == [tuple()]


def test_unify_lifted_to_ground():
    """Tests for unify() when lifted atoms are the first argument and ground
    atoms are the second argument."""
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    var0 = cup_type("?var0")
    var1 = cup_type("?var1")
    var2 = cup_type("?var2")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: True)
    pred1 = Predicate("Pred1", [cup_type, cup_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: True)

    kb0 = frozenset({pred0([cup0])})
    q0 = frozenset({pred0([var0])})
    found, assignment = utils.unify(kb0, q0)
    assert found
    assert assignment == {cup0: var0}

    q1 = frozenset({pred0([var0]), pred0([var1])})
    found, assignment = utils.unify(kb0, q1)
    assert not found
    assert assignment == {}

    kb1 = frozenset({pred0([cup0]), pred0([cup1])})
    found, assignment = utils.unify(kb1, q0)
    assert not found  # different number of predicates/objects
    assert assignment == {}

    kb2 = frozenset({pred0([cup0]), pred2([cup2])})
    q2 = frozenset({pred0([var0]), pred2([var2])})
    found, assignment = utils.unify(kb2, q2)
    assert found
    assert assignment == {cup0: var0, cup2: var2}

    kb3 = frozenset({pred0([cup0])})
    q3 = frozenset({pred0([var0]), pred2([var2])})
    found, assignment = utils.unify(kb3, q3)
    assert not found
    assert assignment == {}

    kb4 = frozenset({pred1([cup0, cup1]), pred1([cup1, cup2])})
    q4 = frozenset({pred1([var0, var1])})
    found, assignment = utils.unify(kb4, q4)
    assert not found  # different number of predicates
    assert assignment == {}

    kb5 = frozenset({pred0([cup2]), pred1([cup0, cup1]), pred1([cup1, cup2])})
    q5 = frozenset({pred1([var0, var1]), pred0([var1]), pred0([var0])})
    found, assignment = utils.unify(kb5, q5)
    assert not found
    assert assignment == {}

    kb6 = frozenset({
        pred0([cup0]),
        pred2([cup1]),
        pred1([cup0, cup2]),
        pred1([cup2, cup1])
    })
    q6 = frozenset({pred0([var0]), pred2([var1]), pred1([var0, var1])})
    found, assignment = utils.unify(kb6, q6)
    assert not found
    assert assignment == {}

    kb7 = frozenset({pred0([cup0]), pred2([cup1])})
    q7 = frozenset({pred0([var0]), pred2([var0])})
    found, assignment = utils.unify(kb7, q7)
    assert not found  # different number of objects
    assert assignment == {}

    kb8 = frozenset({pred0([cup0]), pred2([cup0])})
    q8 = frozenset({pred0([var0]), pred2([var0])})
    found, assignment = utils.unify(kb8, q8)
    assert found
    assert assignment == {cup0: var0}

    kb9 = frozenset({pred1([cup0, cup1]), pred1([cup1, cup2]), pred2([cup0])})
    q9 = frozenset({pred1([var0, var1]), pred1([var2, var0]), pred2([var0])})
    found, assignment = utils.unify(kb9, q9)
    assert not found
    assert assignment == {}


def test_unify_other_liftedground_combinations():
    """Tests for unify() with other combinations of ground/lifted atoms."""
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup2 = cup_type("cup2")
    var0 = cup_type("?var0")
    var1 = cup_type("?var1")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: True)

    kb0 = frozenset({pred0([var0])})
    q0 = frozenset({pred0([cup0])})
    found, assignment = utils.unify(kb0, q0)
    assert found
    assert assignment == {var0: cup0}

    kb1 = frozenset({pred0([var0])})
    q1 = frozenset({pred0([var1])})
    found, assignment = utils.unify(kb1, q1)
    assert found
    assert assignment == {var0: var1}

    kb2 = frozenset({pred0([cup0])})
    q2 = frozenset({pred0([cup2])})
    found, assignment = utils.unify(kb2, q2)
    assert found
    assert assignment == {cup0: cup2}


def test_unify_preconds_effects_options():
    """Tests for unify_preconds_effects_options()."""
    # The following test checks edge cases of unification with respect to
    # the split between effects and option variables.
    # The case is basically this:
    # Add set 1: P(a, b)
    # Option 1: A(b, c)
    # Add set 2: P(w, x)
    # Option 2: A(y, z)
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    w = cup_type("?w")
    x = cup_type("?x")
    y = cup_type("?y")
    z = cup_type("?z")
    pred0 = Predicate("Pred0", [cup_type, cup_type], lambda s, o: False)
    param_option0 = ParameterizedOption("dummy0", [cup_type],
                                        Box(0.1, 1, (1, )),
                                        lambda s, m, o, p: Action(p),
                                        lambda s, m, o, p: False,
                                        lambda s, m, o, p: False)
    # Option0(cup0, cup1)
    ground_option_args = (cup0, cup1)
    # Pred0(cup1, cup2) true
    ground_add_effects = frozenset({pred0([cup1, cup2])})
    ground_delete_effects = frozenset()
    # Option0(w, x)
    lifted_option_args = (w, x)
    # Pred0(y, z) True
    lifted_add_effects = frozenset({pred0([y, z])})
    lifted_delete_effects = frozenset()
    suc, sub = utils.unify_preconds_effects_options(
        frozenset(), frozenset(), ground_add_effects, lifted_add_effects,
        ground_delete_effects, lifted_delete_effects, param_option0,
        param_option0, ground_option_args, lifted_option_args)
    assert not suc
    assert not sub
    # The following test is for an edge case where everything is identical
    # except for the name of the parameterized option. We do not want to
    # unify in this case.
    # First, a unify that should succeed.
    suc, sub = utils.unify_preconds_effects_options(frozenset(), frozenset(),
                                                    frozenset(), frozenset(),
                                                    frozenset(), frozenset(),
                                                    param_option0,
                                                    param_option0,
                                                    (cup0, cup1), (cup0, cup1))
    assert suc
    assert sub == {cup0: cup0, cup1: cup1}
    # Now, a unify that should fail because of different parameterized options.
    param_option1 = ParameterizedOption("dummy1", [cup_type],
                                        Box(0.1, 1, (1, )),
                                        lambda s, m, o, p: Action(p),
                                        lambda s, m, o, p: False,
                                        lambda s, m, o, p: False)
    suc, sub = utils.unify_preconds_effects_options(frozenset(), frozenset(),
                                                    frozenset(), frozenset(),
                                                    frozenset(), frozenset(),
                                                    param_option0,
                                                    param_option1,
                                                    (cup0, cup1), (cup0, cup1))
    assert not suc
    assert not sub


def test_get_random_object_combination():
    """Tests for get_random_object_combination()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat2"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate0 = plate_type("plate0")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    rng = np.random.default_rng(123)
    objs = utils.get_random_object_combination({cup0, cup1, cup2},
                                               [cup_type, cup_type], rng)
    assert all(obj.type == cup_type for obj in objs)
    objs = utils.get_random_object_combination(
        {cup0, cup1, cup2, plate0, plate1, plate2}, [cup_type, plate_type],
        rng)
    assert [obj.type for obj in objs] == [cup_type, plate_type]
    objs = utils.get_random_object_combination({cup0},
                                               [cup_type, cup_type, cup_type],
                                               rng)
    assert len(objs) == 3
    assert len(set(objs)) == 1
    with pytest.raises(ValueError):
        objs = utils.get_random_object_combination({cup0}, [plate_type], rng)


def test_get_all_groundings():
    """Tests for get_all_groundings()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat2"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    cup_var = cup_type("?cup")
    plate0 = plate_type("plate0")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    plate3 = plate_type("plate3")
    plate_var1 = plate_type("?plate1")
    plate_var2 = plate_type("?plate2")
    plate_var3 = plate_type("?plate3")
    pred1 = Predicate("Pred1", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type, plate_type, plate_type],
                      lambda s, o: True)
    lifted_atoms = frozenset({
        pred1([cup_var, plate_var1]),
        pred2([cup_var, plate_var1, plate_var2])
    })
    objs = frozenset({cup0, cup1, cup2, plate0, plate1, plate2, plate3})
    start_time = time.time()
    for _ in range(10000):
        all_groundings = list(utils.get_all_groundings(lifted_atoms, objs))
    assert time.time() - start_time < 1, "Should be fast due to caching"
    # For pred1, there are 12 groundings (3 cups * 4 plates).
    # Pred2 adds on 4 options for plate_var2, bringing the total to 48.
    assert len(all_groundings) == 48
    for grounding, sub in all_groundings:
        assert len(grounding) == len(lifted_atoms)
        assert len(sub) == 3  # three variables
    lifted_atoms = frozenset({
        pred1([cup_var, plate_var1]),
        pred2([cup_var, plate_var2, plate_var3])
    })
    objs = frozenset({cup0, cup1, cup2, plate0, plate1, plate2, plate3})
    start_time = time.time()
    for _ in range(10000):
        all_groundings = list(utils.get_all_groundings(lifted_atoms, objs))
    assert time.time() - start_time < 1, "Should be fast due to caching"
    # For pred1, there are 12 groundings (3 cups * 4 plates).
    # Pred2 adds on 4*4 options, bringing the total to 12*16.
    assert len(all_groundings) == 12 * 16
    for grounding, sub in all_groundings:
        assert len(grounding) == len(lifted_atoms)
        assert len(sub) == 4  # four variables


def test_find_substitution():
    """Tests for find_substitution()."""
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    var0 = cup_type("?var0")
    var1 = cup_type("?var1")
    var2 = cup_type("?var2")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: True)
    pred1 = Predicate("Pred1", [cup_type, cup_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: True)

    kb0 = [pred0([cup0])]
    q0 = [pred0([var0])]
    found, assignment = utils.find_substitution(kb0, q0)
    assert found
    assert assignment == {var0: cup0}

    q1 = [pred0([var0]), pred0([var1])]
    found, assignment = utils.find_substitution(kb0, q1)
    assert not found
    assert assignment == {}

    q1 = [pred0([var0]), pred0([var1])]
    found, assignment = utils.find_substitution(kb0, q1, allow_redundant=True)
    assert found
    assert assignment == {var0: cup0, var1: cup0}

    kb1 = [pred0([cup0]), pred0([cup1])]
    found, assignment = utils.find_substitution(kb1, q0)
    assert found
    assert assignment == {var0: cup0}

    kb2 = [pred0([cup0]), pred2([cup2])]
    q2 = [pred0([var0]), pred2([var2])]
    found, assignment = utils.find_substitution(kb2, q2)
    assert found
    assert assignment == {var0: cup0, var2: cup2}

    kb3 = [pred0([cup0])]
    q3 = [pred0([var0]), pred2([var2])]
    found, assignment = utils.find_substitution(kb3, q3)
    assert not found
    assert assignment == {}

    kb4 = [pred1([cup0, cup1]), pred1([cup1, cup2])]
    q4 = [pred1([var0, var1])]
    found, assignment = utils.find_substitution(kb4, q4)
    assert found
    assert assignment == {var0: cup0, var1: cup1}

    kb5 = [pred0([cup2]), pred1([cup0, cup1]), pred1([cup1, cup2])]
    q5 = [pred1([var0, var1]), pred0([var1]), pred0([var0])]
    found, assignment = utils.find_substitution(kb5, q5)
    assert not found
    assert assignment == {}

    kb6 = [
        pred0([cup0]),
        pred2([cup1]),
        pred1([cup0, cup2]),
        pred1([cup2, cup1])
    ]
    q6 = [pred0([var0]), pred2([var1]), pred1([var0, var1])]
    found, assignment = utils.find_substitution(kb6, q6)
    assert not found
    assert assignment == {}

    kb7 = [pred1([cup0, cup0])]
    q7 = [pred1([var0, var0])]
    found, assignment = utils.find_substitution(kb7, q7)
    assert found
    assert assignment == {var0: cup0}

    kb8 = [pred1([cup0, cup0])]
    q8 = [pred1([var0, var1])]
    found, assignment = utils.find_substitution(kb8, q8)
    assert not found
    assert assignment == {}

    found, assignment = utils.find_substitution(kb8, q8, allow_redundant=True)
    assert found
    assert assignment == {var0: cup0, var1: cup0}

    kb9 = [pred1([cup0, cup1])]
    q9 = [pred1([var0, var0])]
    found, assignment = utils.find_substitution(kb9, q9)
    assert not found
    assert assignment == {}

    found, assignment = utils.find_substitution(kb9, q9, allow_redundant=True)
    assert not found
    assert assignment == {}

    kb10 = [pred1([cup0, cup1]), pred1([cup1, cup0])]
    q10 = [pred1([var0, var1]), pred1([var0, var2])]
    found, assignment = utils.find_substitution(kb10, q10)
    assert not found
    assert assignment == {}

    kb11 = [pred1([cup0, cup1]), pred1([cup1, cup0])]
    q11 = [pred1([var0, var1]), pred1([var1, var0])]
    found, assignment = utils.find_substitution(kb11, q11)
    assert found
    assert assignment == {var0: cup0, var1: cup1}

    plate_type = Type("plate_type", ["feat1"])
    plate0 = plate_type("plate0")
    var3 = plate_type("?var3")
    pred4 = Predicate("Pred4", [plate_type], lambda s, o: True)
    pred5 = Predicate("Pred5", [plate_type, cup_type], lambda s, o: True)

    kb12 = [pred4([plate0])]
    q12 = [pred0([var0])]
    found, assignment = utils.find_substitution(kb12, q12)
    assert not found
    assert assignment == {}

    kb13 = [pred4([plate0]), pred5([plate0, cup0])]
    q13 = [pred4([var3]), pred5([var3, var0])]
    found, assignment = utils.find_substitution(kb13, q13)
    assert found
    assert assignment == {var3: plate0, var0: cup0}


def test_nsrt_methods():
    """Tests for all_ground_nsrts(), extract_preds_and_types()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate1_var = plate_type("?plate1")
    plate2_var = plate_type("?plate1")
    parameters = [cup_var, plate1_var, plate2_var]
    preconditions = {not_on([cup_var, plate1_var])}
    add_effects = {on([cup_var, plate1_var])}
    delete_effects = {not_on([cup_var, plate1_var])}
    params_space = Box(-10, 10, (2, ))
    parameterized_option = ParameterizedOption("Pick", [cup_type],
                                               params_space,
                                               lambda s, m, o, p: 2 * p,
                                               lambda s, m, o, p: True,
                                               lambda s, m, o, p: True)
    nsrt = NSRT("PickNSRT",
                parameters,
                preconditions,
                add_effects,
                delete_effects,
                set(),
                parameterized_option, [parameters[0]],
                _sampler=None)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_nsrts = sorted(utils.all_ground_nsrts(nsrt, objects))
    assert len(ground_nsrts) == 8
    all_obj = [nsrt.objects for nsrt in ground_nsrts]
    assert [cup1, plate1, plate1] in all_obj
    assert [cup1, plate2, plate1] in all_obj
    assert [cup2, plate1, plate1] in all_obj
    assert [cup2, plate2, plate1] in all_obj
    assert [cup1, plate1, plate2] in all_obj
    assert [cup1, plate2, plate2] in all_obj
    assert [cup2, plate1, plate2] in all_obj
    assert [cup2, plate2, plate2] in all_obj
    preds, types = utils.extract_preds_and_types({nsrt})
    assert preds == {"NotOn": not_on, "On": on}
    assert types == {"plate_type": plate_type, "cup_type": cup_type}


def test_all_ground_operators():
    """Tests for all_ground_operators()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate1_var = plate_type("?plate1")
    plate2_var = plate_type("?plate2")
    parameters = [cup_var, plate1_var, plate2_var]
    preconditions = {not_on([cup_var, plate1_var])}
    add_effects = {on([cup_var, plate1_var])}
    delete_effects = {not_on([cup_var, plate1_var])}
    op = STRIPSOperator("Pick", parameters, preconditions, add_effects,
                        delete_effects, set())
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_ops = sorted(utils.all_ground_operators(op, objects))
    assert len(ground_ops) == 8
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate1, plate1] in all_obj
    assert [cup1, plate2, plate1] in all_obj
    assert [cup2, plate1, plate1] in all_obj
    assert [cup2, plate2, plate1] in all_obj
    assert [cup1, plate1, plate2] in all_obj
    assert [cup1, plate2, plate2] in all_obj
    assert [cup2, plate1, plate2] in all_obj
    assert [cup2, plate2, plate2] in all_obj
    preds, types = utils.extract_preds_and_types({op})
    assert preds == {"NotOn": not_on, "On": on}
    assert types == {"plate_type": plate_type, "cup_type": cup_type}


def test_all_ground_operators_given_partial():
    """Tests for all_ground_operators_given_partial()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate1_var = plate_type("?plate1")
    plate2_var = plate_type("?plate2")
    parameters = [cup_var, plate1_var, plate2_var]
    preconditions = {not_on([cup_var, plate1_var])}
    add_effects = {on([cup_var, plate1_var])}
    delete_effects = {not_on([cup_var, plate1_var])}
    op = STRIPSOperator("Pick", parameters, preconditions, add_effects,
                        delete_effects, set())
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    # First test empty partial sub.
    ground_ops = sorted(
        utils.all_ground_operators_given_partial(op, objects, {}))
    assert ground_ops == sorted(utils.all_ground_operators(op, objects))
    # Test with one partial sub.
    sub = {plate1_var: plate1}
    ground_ops = sorted(
        utils.all_ground_operators_given_partial(op, objects, sub))
    assert len(ground_ops) == 4
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate1, plate1] in all_obj
    assert [cup2, plate1, plate1] in all_obj
    assert [cup1, plate1, plate2] in all_obj
    assert [cup2, plate1, plate2] in all_obj
    preds, types = utils.extract_preds_and_types({op})
    assert preds == {"NotOn": not_on, "On": on}
    assert types == {"plate_type": plate_type, "cup_type": cup_type}
    # Test another single partial sub.
    sub = {plate1_var: plate2}
    ground_ops = sorted(
        utils.all_ground_operators_given_partial(op, objects, sub))
    assert len(ground_ops) == 4
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate2, plate1] in all_obj
    assert [cup2, plate2, plate1] in all_obj
    assert [cup1, plate2, plate2] in all_obj
    assert [cup2, plate2, plate2] in all_obj
    # Test multiple partial subs.
    sub = {plate1_var: plate1, plate2_var: plate2}
    ground_ops = sorted(
        utils.all_ground_operators_given_partial(op, objects, sub))
    assert len(ground_ops) == 2
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate1, plate2] in all_obj
    assert [cup2, plate1, plate2] in all_obj
    sub = {plate1_var: plate2, plate2_var: plate1, cup_var: cup1}
    ground_ops = sorted(
        utils.all_ground_operators_given_partial(op, objects, sub))
    assert len(ground_ops) == 1
    all_obj = [op.objects for op in ground_ops]
    assert [cup1, plate2, plate1] in all_obj


def test_prune_ground_atom_dataset():
    """Tests for prune_ground_atom_dataset()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: False)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: False)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup1: [0.5], cup2: [0.1], plate1: [1.0], plate2: [1.2]})
    on_ground = {
        GroundAtom(on, [cup1, plate1]),
        GroundAtom(on, [cup2, plate2])
    }
    not_on_ground = {
        GroundAtom(not_on, [cup1, plate2]),
        GroundAtom(not_on, [cup2, plate1])
    }
    all_atoms = on_ground | not_on_ground
    ground_atom_dataset = [(LowLevelTrajectory([state], []), [all_atoms])]
    pruned_dataset1 = utils.prune_ground_atom_dataset(ground_atom_dataset,
                                                      {on})
    assert pruned_dataset1[0][1][0] == on_ground
    pruned_dataset2 = utils.prune_ground_atom_dataset(ground_atom_dataset,
                                                      {not_on})
    assert pruned_dataset2[0][1][0] == not_on_ground
    pruned_dataset3 = utils.prune_ground_atom_dataset(ground_atom_dataset,
                                                      {on, not_on})
    assert pruned_dataset3[0][1][0] == all_atoms
    pruned_dataset4 = utils.prune_ground_atom_dataset(ground_atom_dataset,
                                                      set())
    assert pruned_dataset4[0][1][0] == set()


def test_ground_atom_methods():
    """Tests for all_ground_predicates(), all_possible_ground_atoms()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: False)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: False)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    state = State({cup1: [0.5], cup2: [0.1], plate1: [1.0], plate2: [1.2]})
    on_ground = {
        GroundAtom(on, [cup1, plate1]),
        GroundAtom(on, [cup1, plate2]),
        GroundAtom(on, [cup2, plate1]),
        GroundAtom(on, [cup2, plate2])
    }
    not_on_ground = {
        GroundAtom(not_on, [cup1, plate1]),
        GroundAtom(not_on, [cup1, plate2]),
        GroundAtom(not_on, [cup2, plate1]),
        GroundAtom(not_on, [cup2, plate2])
    }
    ground_atoms = sorted(on_ground | not_on_ground)
    assert utils.all_ground_predicates(on, objects) == on_ground
    assert utils.all_ground_predicates(not_on, objects) == not_on_ground
    assert utils.all_possible_ground_atoms(state, {on, not_on}) == ground_atoms
    assert not utils.abstract(state, {on, not_on})


def test_create_ground_atom_dataset():
    """Tests for create_ground_atom_dataset()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type],
                   lambda s, o: s.get(o[0], "feat1") > s.get(o[1], "feat1"))
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    states = [
        State({
            cup1: [0.5],
            cup2: [0.1],
            plate1: [1.0],
            plate2: [1.2]
        }),
        State({
            cup1: [1.1],
            cup2: [0.1],
            plate1: [1.0],
            plate2: [1.2]
        })
    ]
    actions = [DummyOption]
    dataset = [LowLevelTrajectory(states, actions)]
    ground_atom_dataset = utils.create_ground_atom_dataset(dataset, {on})
    assert len(ground_atom_dataset) == 1
    assert len(ground_atom_dataset[0]) == 2
    assert len(ground_atom_dataset[0][0].states) == len(states)
    assert all(gs.allclose(s) for gs, s in \
               zip(ground_atom_dataset[0][0].states, states))
    assert len(ground_atom_dataset[0][0].actions) == len(actions)
    assert all(ga == a
               for ga, a in zip(ground_atom_dataset[0][0].actions, actions))
    assert len(ground_atom_dataset[0][1]) == len(states) == 2
    assert ground_atom_dataset[0][1][0] == set()
    assert ground_atom_dataset[0][1][1] == {GroundAtom(on, [cup1, plate1])}


def test_get_reachable_atoms():
    """Tests for get_reachable_atoms()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred1 = Predicate("Pred1", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type, plate_type], lambda s, o: True)
    pred3 = Predicate("Pred3", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    # pred3 is unreachable
    preconditions1 = {pred1([cup_var, plate_var])}
    add_effects1 = {pred2([cup_var, plate_var])}
    delete_effects1 = {}
    preconditions2 = {pred1([cup_var, plate_var])}
    add_effects2 = {}
    delete_effects2 = {pred3([cup_var, plate_var])}
    nsrt1 = NSRT("Pick",
                 parameters,
                 preconditions1,
                 add_effects1,
                 delete_effects1,
                 side_predicates=set(),
                 option=None,
                 option_vars=[],
                 _sampler=None)
    nsrt2 = NSRT("Place",
                 parameters,
                 preconditions2,
                 add_effects2,
                 delete_effects2,
                 side_predicates=set(),
                 option=None,
                 option_vars=[],
                 _sampler=None)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_nsrts = (set(utils.all_ground_nsrts(nsrt1, objects))
                    | set(utils.all_ground_nsrts(nsrt2, objects)))
    assert len(ground_nsrts) == 8
    atoms = {pred1([cup1, plate1]), pred1([cup1, plate2])}
    reachable_atoms = utils.get_reachable_atoms(ground_nsrts, atoms)
    assert reachable_atoms == {
        pred1([cup1, plate1]),
        pred1([cup1, plate2]),
        pred2([cup1, plate1]),
        pred2([cup1, plate2])
    }


def test_nsrt_application():
    """Tests for get_applicable_operators() and apply_operator() with a
    _GroundNSRT."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred1 = Predicate("Pred1", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type, plate_type], lambda s, o: True)
    pred3 = Predicate("Pred3", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions1 = {pred1([cup_var, plate_var])}
    add_effects1 = {pred2([cup_var, plate_var])}
    delete_effects1 = {}
    preconditions2 = {pred1([cup_var, plate_var])}
    add_effects2 = {}
    delete_effects2 = {pred3([cup_var, plate_var])}
    nsrt1 = NSRT("Pick",
                 parameters,
                 preconditions1,
                 add_effects1,
                 delete_effects1,
                 side_predicates=set(),
                 option=None,
                 option_vars=[],
                 _sampler=None)
    nsrt2 = NSRT("Place",
                 parameters,
                 preconditions2,
                 add_effects2,
                 delete_effects2,
                 side_predicates=set(),
                 option=None,
                 option_vars=[],
                 _sampler=None)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_nsrts = (set(utils.all_ground_nsrts(nsrt1, objects))
                    | set(utils.all_ground_nsrts(nsrt2, objects)))
    assert len(ground_nsrts) == 8
    applicable = list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup1, plate1])}))
    assert len(applicable) == 2
    all_obj = [(nsrt.name, nsrt.objects) for nsrt in applicable]
    assert ("Pick", [cup1, plate1]) in all_obj
    assert ("Place", [cup1, plate1]) in all_obj
    next_atoms = [
        utils.apply_operator(nsrt, {pred1([cup1, plate1])})
        for nsrt in applicable
    ]
    assert {pred1([cup1, plate1])} in next_atoms
    assert {pred1([cup1, plate1]), pred2([cup1, plate1])} in next_atoms
    assert list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup1, plate2])}))
    assert list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup2, plate1])}))
    assert list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup2, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred2([cup1, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred2([cup1, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred2([cup2, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred2([cup2, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred3([cup1, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred3([cup1, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred3([cup2, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_nsrts, {pred3([cup2, plate2])}))
    # Tests with side predicates.
    side_predicates = {pred2}
    nsrt3 = NSRT("Pick",
                 parameters,
                 preconditions1,
                 add_effects1,
                 delete_effects1,
                 side_predicates=side_predicates,
                 option=None,
                 option_vars=[],
                 _sampler=None)
    ground_nsrts = sorted(utils.all_ground_nsrts(nsrt3, objects))
    applicable = list(
        utils.get_applicable_operators(ground_nsrts, {pred1([cup1, plate1])}))
    assert len(applicable) == 1
    ground_nsrt = applicable[0]
    atoms = {pred1([cup1, plate1]), pred2([cup2, plate2])}
    next_atoms = utils.apply_operator(ground_nsrt, atoms)
    assert next_atoms == {pred1([cup1, plate1]), pred2([cup1, plate1])}


def test_operator_application():
    """Tests for get_applicable_operators(), apply_operator(), and
    get_successors_from_ground_ops() with a _GroundSTRIPSOperator."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    pred1 = Predicate("Pred1", [cup_type, plate_type], lambda s, o: True)
    pred2 = Predicate("Pred2", [cup_type, plate_type], lambda s, o: True)
    pred3 = Predicate("Pred3", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions1 = {pred1([cup_var, plate_var])}
    add_effects1 = {pred2([cup_var, plate_var])}
    delete_effects1 = {}
    preconditions2 = {pred1([cup_var, plate_var])}
    add_effects2 = {}
    delete_effects2 = {pred3([cup_var, plate_var])}
    op1 = STRIPSOperator("Pick", parameters, preconditions1, add_effects1,
                         delete_effects1, set())
    op2 = STRIPSOperator("Place", parameters, preconditions2, add_effects2,
                         delete_effects2, set())
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    objects = {cup1, cup2, plate1, plate2}
    ground_ops = (set(utils.all_ground_operators(op1, objects))
                  | set(utils.all_ground_operators(op2, objects)))
    assert len(ground_ops) == 8
    applicable = list(
        utils.get_applicable_operators(ground_ops, {pred1([cup1, plate1])}))
    assert len(applicable) == 2
    all_obj = [(op.name, op.objects) for op in applicable]
    assert ("Pick", [cup1, plate1]) in all_obj
    assert ("Place", [cup1, plate1]) in all_obj
    next_atoms = [
        utils.apply_operator(op, {pred1([cup1, plate1])}) for op in applicable
    ]
    assert {pred1([cup1, plate1])} in next_atoms
    assert {pred1([cup1, plate1]), pred2([cup1, plate1])} in next_atoms
    assert list(
        utils.get_applicable_operators(ground_ops, {pred1([cup1, plate2])}))
    assert list(
        utils.get_applicable_operators(ground_ops, {pred1([cup2, plate1])}))
    assert list(
        utils.get_applicable_operators(ground_ops, {pred1([cup2, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred2([cup1, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred2([cup1, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred2([cup2, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred2([cup2, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred3([cup1, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred3([cup1, plate2])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred3([cup2, plate1])}))
    assert not list(
        utils.get_applicable_operators(ground_ops, {pred3([cup2, plate2])}))
    # Test for get_successors_from_ground_ops().
    # Make sure uniqueness is handled properly.
    op3 = STRIPSOperator("Pick", parameters, preconditions1, add_effects1,
                         delete_effects1, set())
    preconditions3 = {pred2([cup_var, plate_var])}
    op4 = STRIPSOperator("Place", parameters, preconditions3, add_effects2,
                         delete_effects2, set())
    op5 = STRIPSOperator("Pick2", parameters, preconditions1, add_effects1,
                         delete_effects1, set())
    ground_ops = (set(utils.all_ground_operators(op3, objects))
                  | set(utils.all_ground_operators(op4, objects))
                  | set(utils.all_ground_operators(op5, objects)))
    successors = list(
        utils.get_successors_from_ground_ops({pred1([cup1, plate1])},
                                             ground_ops))
    assert len(successors) == 1
    assert successors[0] == {pred1([cup1, plate1]), pred2([cup1, plate1])}
    successors = list(
        utils.get_successors_from_ground_ops({pred1([cup1, plate1])},
                                             ground_ops,
                                             unique=False))
    assert len(successors) == 2
    assert successors[0] == successors[1]
    assert not list(
        utils.get_successors_from_ground_ops({pred3([cup2, plate2])},
                                             ground_ops))
    # Tests with side predicates.
    side_predicates = {pred2}
    op3 = STRIPSOperator("Pick",
                         parameters,
                         preconditions1,
                         add_effects1,
                         delete_effects1,
                         side_predicates=side_predicates)
    ground_ops = sorted(utils.all_ground_operators(op3, objects))
    applicable = list(
        utils.get_applicable_operators(ground_ops, {pred1([cup1, plate1])}))
    assert len(applicable) == 1
    ground_op = applicable[0]
    atoms = {pred1([cup1, plate1]), pred2([cup2, plate2])}
    next_atoms = utils.apply_operator(ground_op, atoms)
    assert next_atoms == {pred1([cup1, plate1]), pred2([cup1, plate1])}


def test_create_task_planning_heuristic():
    """Tests for create_task_planning_heuristic()."""
    hadd_heuristic = utils.create_task_planning_heuristic(
        "hadd", set(), set(), set(), set(), set())
    assert isinstance(hadd_heuristic, _PyperplanHeuristicWrapper)
    assert hadd_heuristic.name == "hadd"
    hmax_heuristic = utils.create_task_planning_heuristic(
        "hmax", set(), set(), set(), set(), set())
    assert hmax_heuristic.name == "hmax"
    assert isinstance(hmax_heuristic, _PyperplanHeuristicWrapper)
    hff_heuristic = utils.create_task_planning_heuristic(
        "hff", set(), set(), set(), set(), set())
    assert isinstance(hff_heuristic, _PyperplanHeuristicWrapper)
    assert hff_heuristic.name == "hff"
    hsa_heuristic = utils.create_task_planning_heuristic(
        "hsa", set(), set(), set(), set(), set())
    assert hsa_heuristic.name == "hsa"
    assert isinstance(hsa_heuristic, _PyperplanHeuristicWrapper)
    lmcut_heuristic = utils.create_task_planning_heuristic(
        "lmcut", set(), set(), set(), set(), set())
    assert isinstance(lmcut_heuristic, _PyperplanHeuristicWrapper)
    assert lmcut_heuristic.name == "lmcut"
    with pytest.raises(ValueError):
        utils.create_task_planning_heuristic("not a real heuristic", set(),
                                             set(), set(), set(), set())
    # Cover _TaskPlanningHeuristic base class.
    base_heuristic = _TaskPlanningHeuristic("base", set(), set(), set())
    with pytest.raises(NotImplementedError):
        base_heuristic(set())


def test_create_pddl():
    """Tests for create_pddl_domain() and create_pddl_problem()."""
    utils.update_config({"env": "cover"})
    # All predicates and options
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    env.seed(123)
    train_task = next(env.train_tasks_generator())[0]
    state = train_task.init
    objects = list(state)
    init_atoms = utils.abstract(state, env.predicates)
    goal = train_task.goal
    domain_str = utils.create_pddl_domain(nsrts, env.predicates, env.types,
                                          "cover")
    problem_str = utils.create_pddl_problem(objects, init_atoms, goal, "cover",
                                            "cover-problem0")
    assert domain_str == """(define (domain cover)
  (:requirements :typing)
  (:types block robot target)

  (:predicates
    (Covers ?x0 - block ?x1 - target)
    (HandEmpty)
    (Holding ?x0 - block)
    (IsBlock ?x0 - block)
    (IsTarget ?x0 - target)
  )

  (:action Pick
    :parameters (?block - block)
    :precondition (and (HandEmpty)
        (IsBlock ?block))
    :effect (and (Holding ?block)
        (not (HandEmpty)))
  )

  (:action Place
    :parameters (?block - block ?target - target)
    :precondition (and (Holding ?block)
        (IsBlock ?block)
        (IsTarget ?target))
    :effect (and (Covers ?block ?target)
        (HandEmpty)
        (not (Holding ?block)))
  )
)"""

    assert problem_str == """(define (problem cover-problem0) (:domain cover)
  (:objects
    block0 - block
    block1 - block
    robby - robot
    target0 - target
    target1 - target
  )
  (:init
    (HandEmpty)
    (IsBlock block0)
    (IsBlock block1)
    (IsTarget target0)
    (IsTarget target1)
  )
  (:goal (and (Covers block0 target0)))
)
"""


def test_save_video():
    """Tests for save_video()."""
    dirname = "_fake_tmp_video_dir"
    filename = "video.mp4"
    utils.update_config({"video_dir": dirname})
    rng = np.random.default_rng(123)
    video = [rng.integers(255, size=(3, 3), dtype=np.uint8) for _ in range(3)]
    utils.save_video(filename, video)
    os.remove(os.path.join(dirname, filename))
    os.rmdir(dirname)


def test_get_config_path_str():
    """Tests for get_config_path_str()."""
    utils.update_config({
        "env": "dummyenv",
        "approach": "dummyapproach",
        "seed": 321,
        "excluded_predicates": "all",
    })
    s = utils.get_config_path_str()
    assert s == "dummyenv__dummyapproach__321__all"


def test_get_save_path_str():
    """Tests for get_save_path_str()."""
    dirname = "_fake_tmp_save_dir"
    old_save_dir = CFG.save_dir
    utils.update_config({
        "env": "test_env",
        "approach": "test_approach",
        "seed": 123,
        "save_dir": dirname,
        "excluded_predicates": "test_pred1,test_pred2"
    })
    save_path = utils.get_save_path_str()
    assert save_path == dirname + ("/test_env__test_approach__123__"
                                   "test_pred1,test_pred2.saved")
    utils.update_config({
        "env": "test_env",
        "approach": "test_approach",
        "seed": 123,
        "save_dir": dirname,
        "excluded_predicates": ""
    })
    save_path = utils.get_save_path_str()
    assert save_path == dirname + "/test_env__test_approach__123__.saved"
    os.rmdir(dirname)
    utils.update_config({"save_dir": old_save_dir})


def test_update_config():
    """Tests for update_config()."""
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "seed": 123,
    })
    assert CFG.env == "cover"
    assert CFG.approach == "random_actions"
    assert CFG.seed == 123
    utils.update_config({
        "env": "dummyenv",
        "approach": "dummyapproach",
        "seed": 321,
    })
    assert CFG.env == "dummyenv"
    assert CFG.approach == "dummyapproach"
    assert CFG.seed == 321
    with pytest.raises(ValueError):
        utils.update_config({"not a real setting name": 0})


def test_run_gbfs():
    """Tests for run_gbfs()."""
    S = Tuple[int, int]  # grid (row, col)
    A = str  # up, down, left, right

    def _grid_successor_fn(state: S) -> Iterator[Tuple[A, S, float]]:
        arrival_costs = np.array([
            [1, 1, 8, 1, 1],
            [1, 8, 1, 1, 1],
            [1, 8, 1, 1, 1],
            [1, 1, 1, 8, 1],
            [1, 1, 2, 1, 1],
        ],
                                 dtype=float)

        act_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        r, c = state

        for act in sorted(act_to_delta):
            dr, dc = act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            # Check if in bounds
            if not (0 <= new_r < arrival_costs.shape[0] and \
                    0 <= new_c < arrival_costs.shape[1]):
                continue
            # Valid action
            yield (act, (new_r, new_c), arrival_costs[new_r, new_c])

    def _grid_check_goal_fn(state: S) -> bool:
        # Bottom right corner of grid
        return state == (4, 4)

    def _grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    initial_state = (0, 0)
    state_sequence, action_sequence = utils.run_gbfs(initial_state,
                                                     _grid_check_goal_fn,
                                                     _grid_successor_fn,
                                                     _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'
    ]

    # Same, but actually reaching the goal is impossible.
    state_sequence, action_sequence = utils.run_gbfs(initial_state,
                                                     lambda s: False,
                                                     _grid_successor_fn,
                                                     _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'
    ]

    # Test with an infinite branching factor.
    def _inf_grid_successor_fn(state: S) -> Iterator[Tuple[A, S, float]]:
        # Change all costs to 1.
        for (a, ns, _) in _grid_successor_fn(state):
            yield (a, ns, 1.)
        # Yield unnecessary and costly noops.
        # These lines should not be covered, and that's the point!
        i = 0  # pragma: no cover
        while True:  # pragma: no cover
            action = f"noop{i}"  # pragma: no cover
            yield (action, state, 100.)  # pragma: no cover
            i += 1  # pragma: no cover

    state_sequence, action_sequence = utils.run_gbfs(initial_state,
                                                     _grid_check_goal_fn,
                                                     _inf_grid_successor_fn,
                                                     _grid_heuristic_fn,
                                                     lazy_expansion=True)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'
    ]
    # Test limit on max evals.
    state_sequence, action_sequence = utils.run_gbfs(
        initial_state,
        _grid_check_goal_fn,
        _inf_grid_successor_fn,
        _grid_heuristic_fn,
        max_evals=2)  # note: need lazy_expansion to be False here
    assert state_sequence == [(0, 0), (1, 0)]
    assert action_sequence == ['down']


def test_run_hill_climbing():
    """Tests for run_hill_climbing()."""
    S = Tuple[int, int]  # grid (row, col)
    A = str  # up, down, left, right

    def _grid_successor_fn(state: S) -> Iterator[Tuple[A, S, float]]:
        arrival_costs = np.array([
            [1, 1, 8, 1, 1],
            [1, 8, 1, 1, 1],
            [1, 8, 1, 1, 1],
            [1, 1, 1, 8, 1],
            [1, 1, 2, 1, 1],
        ],
                                 dtype=float)

        act_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        r, c = state

        for act in sorted(act_to_delta):
            dr, dc = act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            # Check if in bounds
            if not (0 <= new_r < arrival_costs.shape[0] and \
                    0 <= new_c < arrival_costs.shape[1]):
                continue
            # Valid action
            yield (act, (new_r, new_c), arrival_costs[new_r, new_c])

    def _grid_check_goal_fn(state: S) -> bool:
        # Bottom right corner of grid
        return state == (4, 4)

    def _grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    initial_state = (0, 0)
    state_sequence, action_sequence = utils.run_hill_climbing(
        initial_state, _grid_check_goal_fn, _grid_successor_fn,
        _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        "down", "down", "down", "down", "right", "right", "right", "right"
    ]

    # Same, but actually reaching the goal is impossible.
    state_sequence, action_sequence = utils.run_hill_climbing(
        initial_state, lambda s: False, _grid_successor_fn, _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        "down", "down", "down", "down", "right", "right", "right", "right"
    ]

    # Search with no successors
    def _no_successor_fn(state: S) -> Iterator[Tuple[A, S, float]]:
        if state == initial_state:
            yield "dummy_action", (2, 2), 1.0

    state_sequence, action_sequence = utils.run_hill_climbing(
        initial_state, lambda s: False, _no_successor_fn, _grid_heuristic_fn)
    assert state_sequence == [(0, 0), (2, 2)]
    assert action_sequence == ["dummy_action"]

    # Tests showing the benefit of enforced hill climbing.
    def _local_minimum_grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        if state in [(1, 0), (0, 1)]:
            return float("inf")
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    # With enforced_depth 0, search fails.
    state_sequence, action_sequence = utils.run_hill_climbing(
        initial_state, _grid_check_goal_fn, _grid_successor_fn,
        _local_minimum_grid_heuristic_fn)
    assert state_sequence == [(0, 0)]
    assert not action_sequence

    # With enforced_depth 1, search succeeds.
    state_sequence, action_sequence = utils.run_hill_climbing(
        initial_state,
        _grid_check_goal_fn,
        _grid_successor_fn,
        _local_minimum_grid_heuristic_fn,
        enforced_depth=1)
    # Note that hill-climbing does not care about costs.
    assert state_sequence == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1),
                              (4, 2), (4, 3), (4, 4)]
    assert action_sequence == [
        "down", "down", "down", "down", "right", "right", "right", "right"
    ]


def test_ops_and_specs_to_dummy_nsrts():
    """Tests fo ops_and_specs_to_dummy_nsrts()."""
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    not_on = Predicate("NotOn", [cup_type, plate_type], lambda s, o: True)
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions = {not_on([cup_var, plate_var])}
    add_effects = {on([cup_var, plate_var])}
    delete_effects = {not_on([cup_var, plate_var])}
    params_space = Box(-10, 10, (2, ))
    parameterized_option = ParameterizedOption("Pick", [], params_space,
                                               lambda s, m, o, p: 2 * p,
                                               lambda s, m, o, p: True,
                                               lambda s, m, o, p: True)
    strips_operator = STRIPSOperator("Pick", parameters, preconditions,
                                     add_effects, delete_effects, set())
    nsrts = utils.ops_and_specs_to_dummy_nsrts([strips_operator],
                                               [(parameterized_option, [])])
    assert len(nsrts) == 1
    nsrt = next(iter(nsrts))
    assert nsrt.parameters == parameters
    assert nsrt.preconditions == preconditions
    assert nsrt.add_effects == add_effects
    assert nsrt.delete_effects == delete_effects
    assert nsrt.option == parameterized_option
    assert not nsrt.option_vars
