"""Tests for PredicateSearchScoreFunction classes."""

from typing import Callable, FrozenSet, List, Set

import numpy as np
import pytest
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches.grammar_search_invention_approach import \
    _UnaryFreeForallClassifier
from predicators.src.datasets import create_dataset
from predicators.src.envs.blocks import BlocksEnv
from predicators.src.envs.cover import CoverEnv
from predicators.src.nsrt_learning.segmentation import segment_trajectory
from predicators.src.predicate_search_score_functions import \
    _BranchingFactorScoreFunction, _ExactHeuristicCountBasedScoreFunction, \
    _ExactHeuristicEnergyBasedScoreFunction, _ExpectedNodesScoreFunction, \
    _HeuristicBasedScoreFunction, _OperatorLearningBasedScoreFunction, \
    _PredicateSearchScoreFunction, _PredictionErrorScoreFunction, \
    _RelaxationHeuristicBasedScoreFunction, \
    _RelaxationHeuristicCountBasedScoreFunction, \
    _RelaxationHeuristicEnergyBasedScoreFunction, \
    _RelaxationHeuristicMatchBasedScoreFunction, _TaskPlanningScoreFunction, \
    create_score_function
from predicators.src.settings import CFG
from predicators.src.structs import Action, GroundAtom, LowLevelTrajectory, \
    OptionSpec, Predicate, STRIPSOperator, _GroundSTRIPSOperator


def test_create_score_function():
    """Tests for create_score_function()."""
    score_func = create_score_function("prediction_error", set(), [], {}, [])
    assert isinstance(score_func, _PredictionErrorScoreFunction)
    score_func = create_score_function("hadd_match", set(), [], {}, [])
    assert isinstance(score_func, _RelaxationHeuristicMatchBasedScoreFunction)
    assert score_func.heuristic_names == ["hadd"]
    score_func = create_score_function("branching_factor", set(), [], {}, [])
    assert isinstance(score_func, _BranchingFactorScoreFunction)
    score_func = create_score_function("hadd_energy_lookaheaddepth0", set(),
                                       [], {}, [])
    assert isinstance(score_func, _RelaxationHeuristicEnergyBasedScoreFunction)
    assert score_func.lookahead_depth == 0
    assert score_func.heuristic_names == ["hadd"]
    score_func = create_score_function("hmax_energy_lookaheaddepth0", set(),
                                       [], {}, [])
    assert isinstance(score_func, _RelaxationHeuristicEnergyBasedScoreFunction)
    assert score_func.lookahead_depth == 0
    assert score_func.heuristic_names == ["hmax"]
    score_func = create_score_function("hsa_energy_lookaheaddepth0", set(), [],
                                       {}, [])
    assert isinstance(score_func, _RelaxationHeuristicEnergyBasedScoreFunction)
    assert score_func.lookahead_depth == 0
    assert score_func.heuristic_names == ["hsa"]
    score_func = create_score_function("lmcut_energy_lookaheaddepth0", set(),
                                       [], {}, [])
    assert isinstance(score_func, _RelaxationHeuristicEnergyBasedScoreFunction)
    assert score_func.lookahead_depth == 0
    assert score_func.heuristic_names == ["lmcut"]
    score_func = create_score_function("hadd_energy_lookaheaddepth1", set(),
                                       [], {}, [])
    assert score_func.lookahead_depth == 1
    score_func = create_score_function("hadd_energy_lookaheaddepth2", set(),
                                       [], {}, [])
    assert score_func.lookahead_depth == 2
    score_func = create_score_function("hff_energy_lookaheaddepth0", set(), [],
                                       {}, [])
    assert isinstance(score_func, _RelaxationHeuristicEnergyBasedScoreFunction)
    assert score_func.heuristic_names == ["hff"]
    score_func = create_score_function("lmcut,hff_energy_lookaheaddepth0",
                                       set(), [], {}, [])
    assert isinstance(score_func, _RelaxationHeuristicEnergyBasedScoreFunction)
    assert score_func.lookahead_depth == 0
    assert score_func.heuristic_names == ["lmcut", "hff"]
    score_func = create_score_function("exact_energy", set(), [], {}, [])
    assert isinstance(score_func, _ExactHeuristicEnergyBasedScoreFunction)
    score_func = create_score_function("task_planning", set(), [], {}, [])
    assert isinstance(score_func, _TaskPlanningScoreFunction)
    score_func = create_score_function("expected_nodes_created", set(), [], {},
                                       [])
    assert isinstance(score_func, _ExpectedNodesScoreFunction)
    score_func = create_score_function("expected_nodes_expanded", set(), [],
                                       {}, [])
    assert isinstance(score_func, _ExpectedNodesScoreFunction)
    score_func = create_score_function("lmcut_count_lookaheaddepth0", set(),
                                       [], {}, [])
    assert isinstance(score_func, _RelaxationHeuristicCountBasedScoreFunction)
    score_func = create_score_function("exact_count", set(), [], {}, [])
    assert isinstance(score_func, _ExactHeuristicCountBasedScoreFunction)
    with pytest.raises(NotImplementedError):
        create_score_function("not a real score function", set(), [], {}, [])


def test_predicate_search_heuristic_base_classes():
    """Cover the abstract methods for _PredicateSearchScoreFunction &
    subclasses."""
    pred_search_score_function = _PredicateSearchScoreFunction(
        set(), [], {}, [])
    with pytest.raises(NotImplementedError):
        pred_search_score_function.evaluate(set())
    op_learning_score_function = _OperatorLearningBasedScoreFunction(
        set(), [], {}, [])
    with pytest.raises(NotImplementedError):
        op_learning_score_function.evaluate(set())
    utils.reset_config({"env": "cover", "cover_initial_holding_prob": 0.0})
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    state = train_tasks[0].init
    other_state = state.copy()
    robby = [o for o in state if o.type.name == "robot"][0]
    state.set(robby, "hand", 0.5)
    other_state.set(robby, "hand", 0.8)
    parameterized_option = utils.SingletonParameterizedOption(
        "Dummy",
        lambda s, m, o, p: Action(np.array([0.0])),
        params_space=Box(0, 1, (1, )))
    option = parameterized_option.ground([], np.array([0.0]))
    assert option.initiable(state)  # set memory
    action = Action(np.zeros(1, dtype=np.float32))
    action.set_option(option)
    trajectories = [
        LowLevelTrajectory([state, other_state], [action],
                           _is_demo=True,
                           _train_task_idx=0)
    ]
    atom_dataset = utils.create_ground_atom_dataset(trajectories, set())
    heuristic_score_fn = _HeuristicBasedScoreFunction(set(), atom_dataset, {},
                                                      train_tasks, ["hadd"])
    with pytest.raises(NotImplementedError):
        heuristic_score_fn.evaluate(set())
    hadd_score_fn = _RelaxationHeuristicBasedScoreFunction(
        set(), atom_dataset, {}, train_tasks, ["hadd"])
    with pytest.raises(NotImplementedError):
        hadd_score_fn.evaluate(set())


def test_prediction_error_score_function():
    """Tests for _PredictionErrorScoreFunction()."""
    # Tests for CoverEnv.
    utils.reset_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "num_train_tasks": 5,
        "cover_initial_holding_prob": 0.0,
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
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    atom_dataset = utils.create_ground_atom_dataset(dataset.trajectories,
                                                    env.predicates)
    score_function = _PredictionErrorScoreFunction(initial_predicates,
                                                   atom_dataset, candidates,
                                                   train_tasks)
    all_included_s = score_function.evaluate(set(candidates))
    handempty_included_s = score_function.evaluate({name_to_pred["HandEmpty"]})
    holding_included_s = score_function.evaluate({name_to_pred["Holding"]})
    none_included_s = score_function.evaluate(set())
    assert all_included_s < holding_included_s < none_included_s
    assert all_included_s < handempty_included_s  # not better than none


def test_hadd_match_score_function():
    """Tests for _RelaxationHeuristicMatchBasedScoreFunction() with hAdd.."""
    # We know that this score function is bad, and this test shows why.
    utils.reset_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "num_train_tasks": 5,
        "cover_initial_holding_prob": 0.0,
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
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    atom_dataset = utils.create_ground_atom_dataset(dataset.trajectories,
                                                    env.predicates)
    score_function = _RelaxationHeuristicMatchBasedScoreFunction(
        initial_predicates, atom_dataset, candidates, train_tasks, ["hadd"])
    handempty_included_s = score_function.evaluate({name_to_pred["HandEmpty"]})
    none_included_s = score_function.evaluate(set())
    assert handempty_included_s > none_included_s  # this is very bad!


def test_relaxation_energy_score_function():
    """Tests for _RelaxationHeuristicEnergyBasedScoreFunction()."""
    # Tests for CoverEnv.
    utils.reset_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "num_train_tasks": 5,
        "cover_initial_holding_prob": 0.0,
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
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    atom_dataset = utils.create_ground_atom_dataset(dataset.trajectories,
                                                    env.predicates)
    score_function = _RelaxationHeuristicEnergyBasedScoreFunction(
        initial_predicates,
        atom_dataset,
        candidates,
        train_tasks, ["hadd"],
        lookahead_depth=1)
    all_included_s = score_function.evaluate(set(candidates))
    handempty_included_s = score_function.evaluate({name_to_pred["HandEmpty"]})
    holding_included_s = score_function.evaluate({name_to_pred["Holding"]})
    none_included_s = score_function.evaluate(set())
    assert all_included_s < holding_included_s < none_included_s
    assert all_included_s < handempty_included_s  # not better than none

    # Test that the score is inf when the operators make the data impossible.
    # Sanity check this for all heuristic choices.
    class _MockEnergy(_RelaxationHeuristicEnergyBasedScoreFunction):
        """Mock class."""

        def evaluate(self,
                     candidate_predicates: FrozenSet[Predicate]) -> float:
            pruned_atom_data = utils.prune_ground_atom_dataset(
                self._atom_dataset,
                candidate_predicates | self._initial_predicates)
            segmented_trajs = [
                segment_trajectory(traj) for traj in pruned_atom_data
            ]
            low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
            # This is the part that we are overriding, to force no successors.
            strips_ops: List[STRIPSOperator] = []
            option_specs: List[OptionSpec] = []
            return self.evaluate_with_operators(candidate_predicates,
                                                low_level_trajs,
                                                segmented_trajs, strips_ops,
                                                option_specs)

    candidates = {p: 1.0 for p in name_to_pred.values()}
    for heuristic_name in ["hadd", "hmax", "hff", "hsa", "lmcut"]:
        # Reuse dataset from above.
        score_function = _MockEnergy(initial_predicates, atom_dataset,
                                     candidates, train_tasks, [heuristic_name])
        assert score_function.evaluate(set()) == float("inf")

    # Cover edge case where there are no successors.
    # The below is kind of a lot to get one line of coverage (the line is
    # if not successor_hs: return float("inf")) but I can't figure out any
    # simpler way. One tricky part is that if there are no ground operators,
    # the heuristic will never get called (see evaluate_atom_trajectory).
    class _MockHAddEnergy(_RelaxationHeuristicEnergyBasedScoreFunction):
        """Mock class."""

        def evaluate(self,
                     candidate_predicates: FrozenSet[Predicate]) -> float:
            pruned_atom_data = utils.prune_ground_atom_dataset(
                self._atom_dataset,
                candidate_predicates | self._initial_predicates)
            segmented_trajs = [
                segment_trajectory(traj) for traj in pruned_atom_data
            ]
            low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
            # This is the part that we are overriding, to force no successors.
            strips_ops: List[STRIPSOperator] = []
            option_specs: List[OptionSpec] = []
            return self.evaluate_with_operators(candidate_predicates,
                                                low_level_trajs,
                                                segmented_trajs, strips_ops,
                                                option_specs)

        def _evaluate_atom_trajectory(self,
                                      atoms_sequence: List[Set[GroundAtom]],
                                      heuristic_fn: Callable[[Set[GroundAtom]],
                                                             float],
                                      ground_ops: Set[_GroundSTRIPSOperator],
                                      demo_atom_sets: Set[
                                          FrozenSet[GroundAtom]],
                                      is_demo: bool) -> float:
            # We also need to override this to get coverage.
            return heuristic_fn(atoms_sequence[0])

    score_function = _MockHAddEnergy(initial_predicates,
                                     atom_dataset,
                                     candidates,
                                     train_tasks, ["hadd"],
                                     lookahead_depth=1)
    assert score_function.evaluate(set(candidates)) == float("inf")


def test_exact_energy_score_function():
    """Tests for _ExactHeuristicEnergyBasedScoreFunction()."""
    # Just test this on BlocksEnv, since that's a known problem case
    # for hadd_energy_lookaheaddepth*.
    utils.flush_cache()
    utils.reset_config({
        "env": "blocks",
        "offline_data_method": "demo+replay",
        "num_train_tasks": 2,
        "blocks_num_blocks_train": [3],
        "blocks_num_blocks_test": [4],
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
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    atom_dataset = utils.create_ground_atom_dataset(dataset.trajectories,
                                                    env.predicates)
    score_function = _ExactHeuristicEnergyBasedScoreFunction(
        initial_predicates, atom_dataset, candidates, train_tasks)
    all_included_s = score_function.evaluate(set(candidates))
    none_included_s = score_function.evaluate(set())
    gripperopen_excluded_s = score_function.evaluate(
        {name_to_pred["Holding"], name_to_pred["Clear"]})
    assert all_included_s < none_included_s  # good!
    assert all_included_s < gripperopen_excluded_s  # good!
    # Test that the score is inf when the operators make the data impossible.
    # Note: this test will crash pyperplan's implementation of LM-Cut, because
    #       there is a predicate (On) named in the goal that doesn't appear in
    #       any of the reachable facts. So, we'll use HAdd.
    old_heur = CFG.sesame_task_planning_heuristic
    utils.update_config({"sesame_task_planning_heuristic": "hadd"})
    ablated = {"On"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    # Reuse dataset from above.
    score_function = _ExactHeuristicEnergyBasedScoreFunction(
        initial_predicates, atom_dataset, candidates, train_tasks)
    assert score_function.evaluate(set()) == float("inf")
    utils.update_config({"sesame_task_planning_heuristic": old_heur})
    old_hbmd = CFG.grammar_search_max_demos
    utils.update_config({"grammar_search_max_demos": 0})
    assert score_function.evaluate(set()) == 0.0
    utils.update_config({"grammar_search_max_demos": old_hbmd})


def test_count_score_functions():
    """Tests for _RelaxationHeuristicCountBasedScoreFunction() and
    _ExactHeuristicCountBasedScoreFunction."""

    utils.flush_cache()
    utils.reset_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "num_train_tasks": 5,
        "offline_data_num_replays": 50,
        "min_data_for_nsrt": 0,
        "grammar_search_max_demos": 4,
        "grammar_search_max_nondemos": 40,
        "cover_initial_holding_prob": 0.0,
    })
    env = CoverEnv()
    ablated = {"Holding", "HandEmpty"}
    initial_predicates = set()
    name_to_pred = {}
    for p in env.predicates:
        if p.name in ablated:
            name_to_pred[p.name] = p
        else:
            initial_predicates.add(p)
    candidates = {p: 1.0 for p in name_to_pred.values()}
    NotHandEmpty = name_to_pred["HandEmpty"].get_negation()
    candidates[NotHandEmpty] = 1.0
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    atom_dataset = utils.create_ground_atom_dataset(dataset.trajectories,
                                                    env.predicates)
    for name in ["exact_count", "lmcut_count_lookaheaddepth0"]:
        score_function = create_score_function(name, initial_predicates,
                                               atom_dataset, candidates,
                                               train_tasks)
        all_included_s = score_function.evaluate(set(candidates))
        # Cover bad case 1: transition is optimal and sequence is not a demo.
        not_handempty_s = score_function.evaluate({NotHandEmpty})
        assert not_handempty_s > all_included_s
        # Cover bad case 2: transition is not optimal and sequence is a demo.
        none_included_s = score_function.evaluate(set())
        assert all_included_s < none_included_s  # good!
        # Cover bad case 3: there is a "suspicious" optimal state.
        score_function.evaluate({name_to_pred["Holding"]})


def test_branching_factor_score_function():
    """Tests for _BranchingFactorScoreFunction()."""
    # We know that this score function is bad, because it prefers predicates
    # that make segmentation collapse demo actions into one.
    utils.reset_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "num_train_tasks": 2,
        "offline_data_num_replays": 500,
        "min_data_for_nsrt": 3,
        "cover_initial_holding_prob": 0.0,
    })
    env = CoverEnv()

    name_to_pred = {p.name: p for p in env.predicates}
    Covers = name_to_pred["Covers"]
    Holding = name_to_pred["Holding"]

    forall_not_covers0 = Predicate(
        "Forall[0:block].[NOT-Covers(0,1)]", [Covers.types[1]],
        _UnaryFreeForallClassifier(Covers.get_negation(), 1))

    forall_not_covers1 = Predicate(
        "Forall[1:target].[NOT-Covers(0,1)]", [Covers.types[0]],
        _UnaryFreeForallClassifier(Covers.get_negation(), 0))

    candidates = {
        forall_not_covers0: 1.0,
        forall_not_covers1: 1.0,
        Holding: 1.0,
    }
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.goal_predicates | set(candidates))
    score_function = _BranchingFactorScoreFunction(env.goal_predicates,
                                                   atom_dataset, candidates,
                                                   train_tasks)
    holding_s = score_function.evaluate({Holding})
    forall_not_covers_s = score_function.evaluate(
        {forall_not_covers0, forall_not_covers1})
    assert forall_not_covers_s > holding_s


def test_task_planning_score_function():
    """Tests for _TaskPlanningScoreFunction()."""
    # We know that this score function is bad, because it's way too
    # optimistic: it thinks that any valid sequence of operators can
    # be refined into a plan. This unit test illustrates that pitfall.
    utils.reset_config({
        "env": "cover",
        "offline_data_method": "demo+replay",
        "num_train_tasks": 5,
        "cover_initial_holding_prob": 0.0,
    })
    env = CoverEnv()

    name_to_pred = {p.name: p for p in env.predicates}
    Holding = name_to_pred["Holding"]
    HandEmpty = name_to_pred["HandEmpty"]

    candidates = {
        Holding: 1.0,
        HandEmpty: 1.0,
    }
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.goal_predicates | set(candidates))
    score_function = _TaskPlanningScoreFunction(env.goal_predicates,
                                                atom_dataset, candidates,
                                                train_tasks)
    all_included_s = score_function.evaluate({Holding, HandEmpty})
    none_included_s = score_function.evaluate(set())
    # This is terrible!
    assert none_included_s < all_included_s
    # Test cases where operators cannot plan to goal.
    utils.update_config({
        "min_data_for_nsrt": 10000,
    })
    assert score_function.evaluate(set()) == len(train_tasks) * 1e7
    assert score_function.evaluate({Holding, HandEmpty}) == \
        2 * CFG.grammar_search_pred_complexity_weight + len(train_tasks) * 1e7


def test_expected_nodes_score_function():
    """Tests for _ExpectedNodesScoreFunction()."""
    # Cover cases where the number of training tasks is less than or greater
    # than the max number of demos.
    max_num_demos = 5
    utils.reset_config({
        "env": "cover",
        "grammar_search_max_demos": max_num_demos,
        "cover_initial_holding_prob": 0.0,
    })
    assert CFG.segmenter == "option_changes"
    for num_train_tasks in [2, 15]:
        utils.update_config({
            "offline_data_method": "demo+replay",
            "num_train_tasks": num_train_tasks,
            "min_data_for_nsrt": 0,
        })
        env = CoverEnv()
        name_to_pred = {p.name: p for p in env.predicates}
        Holding = name_to_pred["Holding"]
        HandEmpty = name_to_pred["HandEmpty"]
        candidates = {
            Holding: 1.0,
            HandEmpty: 1.0,
        }
        train_tasks = env.get_train_tasks()
        dataset = create_dataset(env, train_tasks, env.options)
        atom_dataset = utils.create_ground_atom_dataset(
            dataset.trajectories, env.goal_predicates | set(candidates))
        score_function = _ExpectedNodesScoreFunction(
            env.goal_predicates,
            atom_dataset,
            candidates,
            train_tasks,
            metric_name="num_nodes_created")
        all_included_s = score_function.evaluate({Holding, HandEmpty})
        none_included_s = score_function.evaluate(set())
        ub = CFG.grammar_search_expected_nodes_upper_bound
        assert all_included_s < none_included_s
        assert all_included_s < ub * min(num_train_tasks, max_num_demos)
        # Test cases where operators cannot plan to goal.
        utils.update_config({
            "min_data_for_nsrt": 10000,
        })
        all_included_s = score_function.evaluate({Holding, HandEmpty})
        assert all_included_s >= ub * min(num_train_tasks, max_num_demos)
    # Test case where max skeletons is specified separately. With a lower
    # max skeletons, the score should increase slightly.
    # First, get a baseline score to compare against.
    utils.update_config({
        "num_train_tasks": 5,
        "sesame_max_skeletons_optimized": 8,
        "grammar_search_expected_nodes_max_skeletons": -1,
        "offline_data_method": "demo",
        "min_data_for_nsrt": 0,
    })
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.goal_predicates | set(candidates))
    score_function = _ExpectedNodesScoreFunction(
        env.goal_predicates,
        atom_dataset,
        candidates,
        train_tasks,
        metric_name="num_nodes_created")
    more_skeletons_score = score_function.evaluate({Holding, HandEmpty})
    # Repeat but with max skeletons 1.
    utils.update_config({
        "grammar_search_expected_nodes_max_skeletons": 1,
    })
    one_skeleton_score = score_function.evaluate({Holding, HandEmpty})
    assert one_skeleton_score > more_skeletons_score
    # Repeat but with max skeletons 10 (should crash).
    utils.update_config({
        "grammar_search_expected_nodes_max_skeletons": 10,
    })
    with pytest.raises(AssertionError):
        score_function.evaluate({Holding, HandEmpty})
