"""Test cases for the grammar search invention approach.
"""

import pytest
import numpy as np
from predicators.src.approaches.grammar_search_invention_approach import \
    _PredicateGrammar, _DataBasedPredicateGrammar, \
    _SingleFeatureInequalitiesPredicateGrammar, _count_positives_for_ops, \
    _create_grammar, _halving_constant_generator, _ForallClassifier, \
    _UnaryFreeForallClassifier, _create_heuristic, _PredicateSearchHeuristic, \
    _OperatorLearningBasedHeuristic, _HAddBasedHeuristic, _HAddMatchHeuristic, \
    _PredictionErrorHeuristic, _HAddLookaheadHeuristic, \
    _BranchingFactorHeuristic, _TaskPlanningHeuristic
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv, BlocksEnv, PaintingEnv
from predicators.src.structs import Type, Predicate, STRIPSOperator, State, \
    Action, ParameterizedOption, Box, LowLevelTrajectory
from predicators.src.nsrt_learning import segment_trajectory
from predicators.src import utils


def test_predicate_grammar():
    """Tests for _PredicateGrammar class.
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    train_task = next(env.train_tasks_generator())[0]
    state = train_task.init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    state.set(robby, "hand", 0.5)
    other_state.set(robby, "hand", 0.8)
    dataset = [LowLevelTrajectory(
        [state, other_state], [np.zeros(1, dtype=np.float32)])]
    base_grammar = _PredicateGrammar()
    with pytest.raises(NotImplementedError):
        base_grammar.generate(max_num=1)
    data_based_grammar = _DataBasedPredicateGrammar(dataset)
    assert data_based_grammar.types == env.types
    with pytest.raises(NotImplementedError):
        data_based_grammar.generate(max_num=1)
    with pytest.raises(NotImplementedError):
        _create_grammar("not a real grammar name", dataset, set())
    env = CoverEnv()
    holding_dummy_grammar = _create_grammar("holding_dummy", dataset,
                                            env.predicates)
    assert len(holding_dummy_grammar.generate(max_num=1)) == 1
    assert len(holding_dummy_grammar.generate(max_num=3)) == 2
    single_ineq_grammar = _SingleFeatureInequalitiesPredicateGrammar(dataset)
    assert len(single_ineq_grammar.generate(max_num=1)) == 1
    feature_ranges = single_ineq_grammar._get_feature_ranges()  # pylint: disable=protected-access
    assert feature_ranges[robby.type]["hand"] == (0.5, 0.8)
    neg_sfi_grammar = _create_grammar("single_feat_ineqs", dataset,
                                      env.predicates)
    candidates = neg_sfi_grammar.generate(max_num=4)
    assert str(sorted(candidates)) == \
        ("[((0:block).pose<=2.33), ((0:block).width<=19.0), "
         "NOT-((0:block).pose<=2.33), NOT-((0:block).width<=19.0)]")
    forall_grammar = _create_grammar("forall_single_feat_ineqs", dataset,
                                     env.predicates)
    assert len(forall_grammar.generate(max_num=100)) == 100


def test_count_positives_for_ops():
    """Tests for _count_positives_for_ops().
    """
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
    strips_operator = STRIPSOperator("Pick", parameters, preconditions,
                                     add_effects, delete_effects)
    cup = cup_type("cup")
    plate = plate_type("plate")
    parameterized_option = ParameterizedOption(
        "Dummy", [], Box(0, 1, (1,)),
        lambda s, m, o, p: Action(np.array([0.0])),
        lambda s, m, o, p: True, lambda s, m, o, p: True)
    option = parameterized_option.ground([], np.array([0.0]))
    state = State({cup: [0.5], plate: [1.0]})
    action = Action(np.zeros(1, dtype=np.float32))
    action.set_option(option)
    states = [state, state]
    actions = [action]
    strips_ops = [strips_operator]
    option_specs = [(parameterized_option, [])]
    pruned_atom_data = [
        # Test empty sequence.
        (LowLevelTrajectory([state], []), [{on([cup, plate])}]),
        # Test not positive.
        (LowLevelTrajectory(states, actions), [{on([cup, plate])}, set()]),
        # Test true positive.
        (LowLevelTrajectory(states, actions), [{not_on([cup, plate])},
                                               {on([cup, plate])}]),
        # Test false positive.
        (LowLevelTrajectory(states, actions), [{not_on([cup, plate])}, set()]),
    ]
    segments = [seg for traj in pruned_atom_data
                for seg in segment_trajectory(traj)]

    num_true, num_false, _, _ = _count_positives_for_ops(strips_ops,
         option_specs, segments)
    assert num_true == 1
    assert num_false == 1


def test_halving_constant_generator():
    """Tests for _halving_constant_generator().
    """
    expected_sequence = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]
    generator = _halving_constant_generator(0., 1.)
    for i, x in zip(range(len(expected_sequence)), generator):
        assert abs(expected_sequence[i] - x) < 1e-6


def test_forall_classifier():
    """Tests for _ForallClassifier().
    """
    cup_type = Type("cup_type", ["feat1"])
    pred = Predicate("Pred", [cup_type],
        lambda s, o: s.get(o[0], "feat1") > 0.5)
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    state0 = State({cup1: [0.], cup2: [0.]})
    state1 = State({cup1: [0.], cup2: [1.]})
    state2 = State({cup1: [1.], cup2: [1.]})
    classifier = _ForallClassifier(pred)
    assert not classifier(state0, [])
    assert not classifier(state1, [])
    assert classifier(state2, [])
    assert str(classifier) == "Forall[0:cup_type].[Pred(0)]"


def test_unary_free_forall_classifier():
    """Tests for _UnaryFreeForallClassifier().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1"])
    on = Predicate("On", [cup_type, plate_type], lambda s, o: True)
    cup0 = cup_type("cup0")
    plate0 = plate_type("plate0")
    state0 = State({cup0: [0.], plate0: [0.]})
    classifier0 = _UnaryFreeForallClassifier(on, 0)
    assert classifier0(state0, [cup0])
    assert str(classifier0) == "Forall[1:plate_type].[On(0,1)]"
    classifier1 = _UnaryFreeForallClassifier(on, 1)
    assert classifier1(state0, [plate0])
    assert str(classifier1) == "Forall[0:cup_type].[On(0,1)]"


def test_create_heuristic():
    """Tests for _create_heuristic().
    """
    utils.update_config({"grammar_search_heuristic": "prediction_error"})
    heuristic = _create_heuristic(set(), [], [], {})
    assert isinstance(heuristic, _PredictionErrorHeuristic)
    utils.update_config({"grammar_search_heuristic": "hadd_match"})
    heuristic = _create_heuristic(set(), [], [], {})
    assert isinstance(heuristic, _HAddMatchHeuristic)
    utils.update_config({"grammar_search_heuristic": "branching_factor"})
    heuristic = _create_heuristic(set(), [], [], {})
    assert isinstance(heuristic, _BranchingFactorHeuristic)
    utils.update_config({"grammar_search_heuristic": "hadd_lookahead_match"})
    heuristic = _create_heuristic(set(), [], [], {})
    assert isinstance(heuristic, _HAddLookaheadHeuristic)
    utils.update_config({"grammar_search_heuristic": "task_planning"})
    heuristic = _create_heuristic(set(), [], [], {})
    assert isinstance(heuristic, _TaskPlanningHeuristic)
    utils.update_config({"grammar_search_heuristic": "not a real heuristic"})
    with pytest.raises(NotImplementedError):
        _create_heuristic(set(), [], [], {})


def test_predicate_search_heuristic_base_classes():
    """Cover the abstract methods for _PredicateSearchHeuristic and subclasses
    """
    pred_search_heuristic = _PredicateSearchHeuristic(set(), [], [], {})
    with pytest.raises(NotImplementedError):
        pred_search_heuristic.evaluate(set())
    op_learning_heuristic = _OperatorLearningBasedHeuristic(set(), [], [], {})
    with pytest.raises(NotImplementedError):
        op_learning_heuristic.evaluate(set())
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    train_tasks = next(env.train_tasks_generator())
    state = train_tasks[0].init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    state.set(robby, "hand", 0.5)
    other_state.set(robby, "hand", 0.8)
    parameterized_option = ParameterizedOption(
        "Dummy", [], Box(0, 1, (1,)),
        lambda s, m, o, p: Action(np.array([0.0])),
        lambda s, m, o, p: True, lambda s, m, o, p: True)
    option = parameterized_option.ground([], np.array([0.0]))
    action = Action(np.zeros(1, dtype=np.float32))
    action.set_option(option)
    dataset = [LowLevelTrajectory(
        [state, other_state], [action], set())]
    atom_dataset = utils.create_ground_atom_dataset(dataset, set())
    hadd_heuristic = _HAddBasedHeuristic(set(), atom_dataset, train_tasks, {})
    with pytest.raises(NotImplementedError):
        hadd_heuristic.evaluate(set())


def test_prediction_error_heuristic():
    """Tests for _PredictionErrorHeuristic().
    """
    # Tests for CoverEnv.
    utils.update_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = CoverEnv()
    ablated = {"HandEmpty", "Holding"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    heuristic = _PredictionErrorHeuristic(initial_predicates, atom_dataset,
                                          train_tasks, candidates)
    all_included_h = heuristic.evaluate(set(candidates))
    handempty_included_h = heuristic.evaluate({name_to_pred["HandEmpty"]})
    holding_included_h = heuristic.evaluate({name_to_pred["Holding"]})
    none_included_h = heuristic.evaluate(set())
    assert all_included_h < holding_included_h < none_included_h
    assert all_included_h < handempty_included_h  # not better than none

    # Tests for BlocksEnv.
    utils.flush_cache()
    utils.update_config({
        "env": "blocks",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = BlocksEnv()
    ablated = {"Holding", "Clear", "GripperOpen"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    heuristic = _PredictionErrorHeuristic(initial_predicates, atom_dataset,
                                          train_tasks, candidates)
    all_included_h = heuristic.evaluate(set(candidates))
    holding_included_h = heuristic.evaluate({name_to_pred["Holding"]})
    clear_included_h = heuristic.evaluate({name_to_pred["Clear"]})
    gripper_open_included_h = heuristic.evaluate({name_to_pred["GripperOpen"]})
    none_included_h = heuristic.evaluate(set())
    assert all_included_h < holding_included_h < none_included_h
    assert all_included_h < clear_included_h < none_included_h
    assert all_included_h < gripper_open_included_h < none_included_h

    # This example shows why this heuristic is bad.
    utils.flush_cache()
    utils.update_config({
        "env": "painting",
        "offline_data_method": "demo+replay",
        "seed": 0,
        "painting_train_families": ["box_and_shelf"],
    })
    env = PaintingEnv()
    ablated = {"IsWet", "IsDry"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    heuristic = _PredictionErrorHeuristic(initial_predicates, atom_dataset,
                                          train_tasks, candidates)
    all_included_h = heuristic.evaluate(set(candidates))
    none_included_h = heuristic.evaluate(set())
    assert all_included_h > none_included_h  # this is very bad!


def test_hadd_match_heuristic():
    """Tests for _HAddMatchHeuristic().
    """
    # We know that this heuristic is bad, and this test shows why.
    utils.update_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = CoverEnv()
    ablated = {"HandEmpty"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    heuristic = _HAddMatchHeuristic(initial_predicates, atom_dataset,
                                    train_tasks, candidates)
    handempty_included_h = heuristic.evaluate({name_to_pred["HandEmpty"]})
    none_included_h = heuristic.evaluate(set())
    assert handempty_included_h > none_included_h # this is very bad!


def test_hadd_lookahead_heuristic():
    """Tests for _HAddLookaheadHeuristic().
    """
    # Tests for CoverEnv.
    utils.update_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = CoverEnv()
    ablated = {"HandEmpty", "Holding"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    heuristic = _HAddLookaheadHeuristic(initial_predicates, atom_dataset,
                                        train_tasks, candidates)
    all_included_h = heuristic.evaluate(set(candidates))
    handempty_included_h = heuristic.evaluate({name_to_pred["HandEmpty"]})
    holding_included_h = heuristic.evaluate({name_to_pred["Holding"]})
    none_included_h = heuristic.evaluate(set())
    assert all_included_h < holding_included_h < none_included_h
    assert all_included_h < handempty_included_h  # not better than none

    # Test that the score is inf when the operators make the data impossible.
    ablated = {"Covers"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    # Reuse dataset from above.
    heuristic = _HAddLookaheadHeuristic(initial_predicates, atom_dataset,
                                        train_tasks, candidates)
    assert heuristic.evaluate(set()) == float("inf")

    # Tests for BlocksEnv.
    utils.flush_cache()
    utils.update_config({
        "env": "blocks",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = BlocksEnv()
    ablated = {"Holding", "Clear", "GripperOpen"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    heuristic = _HAddLookaheadHeuristic(initial_predicates, atom_dataset,
                                        train_tasks, candidates)
    all_included_h = heuristic.evaluate(set(candidates))
    none_included_h = heuristic.evaluate(set())
    # Note: the values for Holding alone, Clear alone, and GripperOpen alone
    # are not in between the bounds. Here are all the values:
    # ipdb> all_included_h
    # 11411.369394796297
    # ipdb> none_included_h
    # 17640.461089410717
    # ipdb> holding_included_h
    # 11240.23793807844
    # ipdb> clear_included_h
    # 21144.93016115656
    # ipdb> gripper_open_included_h
    # 17641.505279500798
    # This is  peculiar. But we do see that learning works well in the end.
    assert all_included_h < none_included_h

    # Tests for PaintEnv.
    utils.flush_cache()
    utils.update_config({
        "env": "painting",
        "offline_data_method": "demo+replay",
        "seed": 0,
        "painting_train_families": ["box_and_shelf"],
    })
    env = PaintingEnv()
    ablated = {"IsWet", "IsDry"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset, env.predicates)
    heuristic = _HAddLookaheadHeuristic(initial_predicates, atom_dataset,
                                        train_tasks, candidates)
    all_included_h = heuristic.evaluate(set(candidates))
    none_included_h = heuristic.evaluate(set())
    assert all_included_h < none_included_h  # hooray!


def test_branching_factor_heuristic():
    """Tests for _BranchingFactorHeuristic().
    """
    # We know that this heuristic is bad, because it prefers predicates that
    # make segmentation collapse demo actions into one.
    utils.update_config({
        "env": "cover",
    })
    utils.update_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = CoverEnv()

    name_to_pred = {p.name : p for p in env.predicates}
    Covers = name_to_pred["Covers"]
    Holding = name_to_pred["Holding"]

    forall_not_covers0 = Predicate(
        "Forall[0:block].[NOT-Covers(0,1)]",
        [Covers.types[1]],
        _UnaryFreeForallClassifier(Covers.get_negation(), 1)
    )

    forall_not_covers1 = Predicate(
        "Forall[1:target].[NOT-Covers(0,1)]",
        [Covers.types[0]],
        _UnaryFreeForallClassifier(Covers.get_negation(), 0)
    )

    candidates = {
        forall_not_covers0: 1.0,
        forall_not_covers1: 1.0,
        Holding: 1.0,
    }
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset,
        env.goal_predicates | set(candidates))
    heuristic = _BranchingFactorHeuristic(env.goal_predicates, atom_dataset,
                                          train_tasks, candidates)
    holding_h = heuristic.evaluate({Holding})
    forall_not_covers_h = heuristic.evaluate({forall_not_covers0,
                                              forall_not_covers1})
    # This is just to illustrate that the heuristic for these two bad predicates
    # is lower than we would like. These are actually the predicates that get
    # returned by running the grammar search on covers with branching factor.
    assert forall_not_covers_h < holding_h


def test_task_planning_heuristic():
    """Tests for _TaskPlanningHeuristic().
    """
    # We know that this heuristic is bad, because it's way too optimistic: it
    # thinks that any valid sequence of operators can be refined into a plan.
    # This unit test illustrates that pitfall.
    utils.update_config({
        "env": "cover",
    })
    utils.update_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "seed": 0,
    })
    env = CoverEnv()

    name_to_pred = {p.name : p for p in env.predicates}
    Holding = name_to_pred["Holding"]
    HandEmpty = name_to_pred["HandEmpty"]

    candidates = {
        Holding: 1.0,
        HandEmpty: 1.0,
    }
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    atom_dataset = utils.create_ground_atom_dataset(dataset,
        env.goal_predicates | set(candidates))
    heuristic = _TaskPlanningHeuristic(env.goal_predicates, atom_dataset,
                                       train_tasks, candidates)
    all_included_h = heuristic.evaluate({Holding, HandEmpty})
    none_included_h = heuristic.evaluate(set())
    # This is terrible!
    assert none_included_h < all_included_h
    # Test cases where operators cannot plan to goal.
    utils.update_config({
        "min_data_for_nsrt": 10000,
    })
    assert heuristic.evaluate(set()) == len(train_tasks) * 1e7
