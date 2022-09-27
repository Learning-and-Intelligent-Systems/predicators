"""Tests for general-to-specific STRIPS operator learning."""

import itertools

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.nsrt_learning.nsrt_learning_main import learn_nsrts_from_data
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning.gen_to_spec_learner import \
    BackchainingSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    PartialNSRTAndDatastore, Predicate, Segment, State, STRIPSOperator, Task, \
    Type

longrun = pytest.mark.skipif("not config.getoption('longrun')")


class _MockBackchainingSTRIPSLearner(BackchainingSTRIPSLearner):
    """Mock class that exposes private methods for testing."""

    def spawn_new_pnad(self, necessary_add_effects, segment):
        """Exposed for testing."""
        segment.necessary_add_effects = necessary_add_effects
        return self._spawn_new_pnad(segment)

    def recompute_datastores_from_segments(self, pnads):
        """Exposed for testing."""
        return self._recompute_datastores_from_segments(pnads)

    def find_unification(self,
                         necessary_add_effects,
                         pnad,
                         segment,
                         check_only_preconditions=True):
        """Exposed for testing."""
        segment.necessary_add_effects = necessary_add_effects
        objects = list(segment.states[0])
        best_pnad, best_sub = self._find_best_matching_pnad_and_sub(
            segment,
            objects, [pnad],
            check_only_preconditions=check_only_preconditions)
        assert best_pnad is not None
        assert best_sub is not None
        ground_best_pnad = best_pnad.op.ground(
            tuple(best_sub[var] for var in best_pnad.op.parameters))
        return best_pnad, ground_best_pnad

    def reset_all_segment_add_effs(self):
        """Exposed for testing."""
        return self._reset_all_segment_add_effs()


def test_backchaining_strips_learner():
    """Test the BackchainingSTRIPSLearner."""
    utils.reset_config({"backchaining_check_intermediate_harmlessness": True})
    # Set up the structs.
    human_type = Type("human_type", ["feat1", "feat2"])
    Asleep = Predicate("Asleep", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    Sad = Predicate("Sad", [human_type], lambda s, o: s[o[0]][1] < 0.5)
    opt_name_to_opt = {}
    for opt_name in ["Cry", "Eat"]:
        opt = utils.SingletonParameterizedOption(opt_name,
                                                 lambda s, m, o, p: None)
        opt_name_to_opt[opt_name] = opt
    # Set up the data.
    bob = human_type("bob")
    state_awake_and_sad = State({bob: [0.0, 0.0]})
    state_awake_and_happy = State({bob: [0.0, 1.0]})
    state_asleep_and_sad = State({bob: [1.0, 0.0]})
    assert not Asleep.holds(state_awake_and_sad, [bob])
    assert Sad.holds(state_awake_and_sad, [bob])
    assert not Asleep.holds(state_awake_and_happy, [bob])
    assert not Sad.holds(state_awake_and_happy, [bob])
    assert Asleep.holds(state_asleep_and_sad, [bob])
    assert Sad.holds(state_asleep_and_sad, [bob])
    Cry = opt_name_to_opt["Cry"].ground([], [])
    Eat = opt_name_to_opt["Eat"].ground([], [])
    goal1 = {Asleep([bob])}
    act1 = Action([], Cry)
    traj1 = LowLevelTrajectory([state_awake_and_sad, state_asleep_and_sad],
                               [act1], True, 0)
    task1 = Task(state_awake_and_sad, goal1)
    segment1 = Segment(traj1, set(), goal1, Cry)
    goal2 = set()
    act2 = Action([], Eat)
    traj2 = LowLevelTrajectory([state_awake_and_sad, state_awake_and_sad],
                               [act2], True, 1)
    task2 = Task(state_awake_and_sad, set())
    segment2 = Segment(traj2, set(), goal2, Eat)
    learner = _MockBackchainingSTRIPSLearner([traj1, traj2], [task1, task2],
                                             {Asleep},
                                             [[segment1], [segment2]],
                                             verify_harmlessness=True)
    pnads = learner.learn()
    # Verify the results are as expected.
    expected_strs = [
        """STRIPS-Cry0:
    Parameters: [?x0:human_type]
    Preconditions: []
    Add Effects: [Asleep(?x0:human_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Cry()""", """STRIPS-Eat0:
    Parameters: []
    Preconditions: []
    Add Effects: []
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Eat()"""
    ]
    for pnad, exp_str in zip(sorted(pnads, key=lambda pnad: pnad.op.name),
                             expected_strs):
        assert str(pnad) == repr(pnad) == exp_str

    # Test sidelining where an existing operator needs to be spawned.
    goal3 = {Asleep([bob])}
    act3 = Action([], Cry)
    traj3 = LowLevelTrajectory([state_awake_and_happy, state_asleep_and_sad],
                               [act3], True, 0)
    goal4 = {Asleep([bob]), Sad([bob])}
    traj4 = LowLevelTrajectory([state_awake_and_happy, state_asleep_and_sad],
                               [act3], True, 1)
    task3 = Task(state_awake_and_sad, goal3)
    task4 = Task(state_awake_and_sad, goal4)
    segment3 = Segment(traj3, set(), {Asleep([bob]), Sad([bob])}, Cry)
    segment4 = Segment(traj4, set(), {Asleep([bob]), Sad([bob])}, Cry)
    # Create and run the sidelining approach.
    learner = _MockBackchainingSTRIPSLearner([traj3, traj4], [task3, task4],
                                             {Asleep, Sad},
                                             [[segment3], [segment4]],
                                             verify_harmlessness=True)
    pnads = learner.learn()
    assert len(pnads) == 1
    expected_str = """STRIPS-Cry0:
    Parameters: [?x0:human_type]
    Preconditions: []
    Add Effects: [Asleep(?x0:human_type), Sad(?x0:human_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Cry()"""
    assert str(pnads[0]) == repr(pnads[0]) == expected_str


def test_backchaining_strips_learner_order_dependence():
    """Test that the BackchainingSTRIPSLearner is invariant to order of
    traversal through trajectories."""
    utils.reset_config({"backchaining_check_intermediate_harmlessness": True})
    # Set up the types and predicates.
    light_type = Type("light_type", ["brightness", "color"])
    LightOn = Predicate("LightOn", [light_type], lambda s, o: s[o[0]][0] > 0.5)
    NotLightOn = Predicate("NotLightOn", [light_type],
                           lambda s, o: s[o[0]][0] <= 0.5)
    LightColorBlue = Predicate("LightColorBlue", [light_type],
                               lambda s, o: s[o[0]][1] > 0.5)
    LightColorRed = Predicate("LightColorRed", [light_type],
                              lambda s, o: s[o[0]][1] <= 0.5)
    fridge_type = Type("fridge_type", ["x", "y"])
    robot_type = Type("robot_type", ["x", "y"])
    RobotAt = Predicate(
        "RobotAt", [robot_type, fridge_type], lambda s, o: abs(s[o[0]][0] - s[
            o[1]][0]) < 0.05 and abs(s[o[0]][1] - s[o[1]][1]) < 0.05)
    light = light_type("light")
    fridge = fridge_type("fridge")
    robby = robot_type("robby")
    # Create states to be used as part of trajectories.
    not_on_light_red = State({
        light: [0.0, 0.0],
        robby: [0.0, 0.0],
        fridge: [1.03, 1.03]
    })
    at_fridge_not_on_light_red = State({
        light: [0.0, 0.0],
        robby: [1.0, 1.0],
        fridge: [1.03, 1.03]
    })
    on_light_blue = State({
        light: [1.0, 1.0],
        robby: [0.0, 0.0],
        fridge: [1.03, 1.03]
    })
    at_fridge_on_light_red = State({
        light: [1.0, 0.0],
        robby: [1.0, 1.0],
        fridge: [1.03, 1.03]
    })
    not_on_light_blue = State({
        light: [0.0, 1.0],
        robby: [0.0, 0.0],
        fridge: [1.03, 1.03]
    })
    at_fridge_on_light_blue = State({
        light: [1.0, 0.0],
        robby: [1.0, 1.0],
        fridge: [1.03, 1.03]
    })
    # Create the single necessary option and action.
    move_and_mess_with_lights = utils.SingletonParameterizedOption(
        "MoveAndMessWithLights", lambda s, m, o, p: None)
    MoveAndMessWithLights = move_and_mess_with_lights.ground([], [])
    act = Action([], MoveAndMessWithLights)
    # Now create the trajectories, goals and tasks.
    traj1 = LowLevelTrajectory([not_on_light_red, at_fridge_not_on_light_red],
                               [act], True, 0)
    goal1 = {
        RobotAt([robby, fridge]),
    }
    traj2 = LowLevelTrajectory([on_light_blue, at_fridge_on_light_red], [act],
                               True, 1)
    traj3 = LowLevelTrajectory([not_on_light_blue, at_fridge_on_light_blue],
                               [act], True, 2)
    goal2 = {RobotAt([robby, fridge]), LightOn([light])}
    task1 = Task(not_on_light_red, goal1)
    task2 = Task(on_light_blue, goal2)
    task3 = Task(not_on_light_blue, goal2)
    # Define the 3 demos to backchain over.
    segment1 = Segment(
        traj1,
        {NotLightOn([light]), LightColorRed([light])},
        goal1 | {NotLightOn([light])}, MoveAndMessWithLights)
    segment2 = Segment(
        traj2, {LightOn([light]), LightColorBlue([light])}, goal2,
        MoveAndMessWithLights)
    segment3 = Segment(
        traj3,
        {NotLightOn([light]), LightColorBlue([light])}, goal2,
        MoveAndMessWithLights)

    # Create and run the learner with the 3 demos in the natural order.
    learner = _MockBackchainingSTRIPSLearner(
        [traj1, traj2, traj3], [task1, task2, task3],
        {RobotAt, LightOn, NotLightOn, LightColorBlue, LightColorRed},
        [[segment1], [segment2], [segment3]],
        verify_harmlessness=True)
    natural_order_pnads = learner.learn()
    # Now, create and run the learner with the 3 demos in the reverse order.
    learner = _MockBackchainingSTRIPSLearner(
        [traj3, traj2, traj1], [task1, task2, task3],
        {RobotAt, LightOn, NotLightOn, LightColorBlue, LightColorRed},
        [[segment3], [segment2], [segment1]],
        verify_harmlessness=True)
    # Be sure to reset the segment add effects before doing this.
    learner.reset_all_segment_add_effs()
    reverse_order_pnads = learner.learn()

    # First, check that the two sets of PNADs have the same number of PNADs.
    assert len(natural_order_pnads) == len(reverse_order_pnads) == 2

    correct_pnads = {
        """STRIPS-MoveAndMessWithLights:
    Parameters: [?x0:fridge_type, ?x1:light_type, ?x2:robot_type]
    Preconditions: [LightColorBlue(?x1:light_type), NotLightOn(?x1:light_type)]
    Add Effects: [LightOn(?x1:light_type), """ +
        """RobotAt(?x2:robot_type, ?x0:fridge_type)]
    Delete Effects: [LightColorBlue(?x1:light_type), """ +
        """NotLightOn(?x1:light_type)]
    Ignore Effects: []
    Option Spec: MoveAndMessWithLights()""", """STRIPS-MoveAndMessWithLights:
    Parameters: [?x0:fridge_type, ?x1:robot_type, ?x2:light_type]
    Preconditions: [LightColorBlue(?x2:light_type), NotLightOn(?x2:light_type)]
    Add Effects: [LightOn(?x2:light_type), """ +
        """RobotAt(?x1:robot_type, ?x0:fridge_type)]
    Delete Effects: [LightColorBlue(?x2:light_type), """ +
        """NotLightOn(?x2:light_type)]
    Ignore Effects: []
    Option Spec: MoveAndMessWithLights()""", """STRIPS-MoveAndMessWithLights:
    Parameters: [?x0:fridge_type, ?x1:robot_type]
    Preconditions: []
    Add Effects: [RobotAt(?x1:robot_type, ?x0:fridge_type)]
    Delete Effects: []
    Ignore Effects: [LightColorBlue, LightColorRed]
    Option Spec: MoveAndMessWithLights()"""
    }
    # Edit the names of all the returned PNADs to match the correct ones for
    # easy checking.
    for i in range(len(natural_order_pnads)):
        natural_order_pnads[i].op = natural_order_pnads[i].op.copy_with(
            name="MoveAndMessWithLights")
        reverse_order_pnads[i].op = reverse_order_pnads[i].op.copy_with(
            name="MoveAndMessWithLights")

        # Check that the two sets of PNADs are both correct.
        assert str(natural_order_pnads[i]) in correct_pnads
        assert str(reverse_order_pnads[i]) in correct_pnads

    # Weird Case: This case shows that our algorithm is not data order
    # invariant!
    utils.reset_config({
        "approach": "nsrt_learning",
        "strips_learner": "backchaining",
        # Following are necessary to solve this case.
        "data_orderings_to_search": 10,
        "enable_harmless_op_pruning": True,
        "disable_harmlessness_check": False
    })
    # Agent features are loc: 0, 1, 2, 3 [start, shelf1, shelf2, far away];
    # holding: True or False whether an object is in hand
    agent_type = Type("agent_type", ["loc", "holding"])
    agent = agent_type("agent")
    # Hardback features are loc: -1, 0, 1, 2, 3 [in_hand, start, shelf1,
    # shelf2, far away]
    hardback_type = Type(
        "hardback_type",
        ["loc"])  # loc: -1, 0, 1, 2 [in_hand, start, shelf1, shelf2]
    hardback1 = hardback_type("hardback1")
    hardback2 = hardback_type("hardback2")
    book1 = hardback_type("book1")
    # Shelf features are loc: 2 (only a shelf at location two)
    shelf_type = Type("shelf_type", ["loc"])
    shelf = shelf_type("shelf")
    # Predicates
    NextTo = Predicate(
        "NextTo", [hardback_type], lambda s, o: s[o[0]][0] == s[agent][0] or
        (s[o[0]][0] in [1, 2] and s[agent][0] in [1, 2]))
    NextToShelf = Predicate("NextToShelf", [shelf_type],
                            lambda s, o: s[agent][0] == 2)
    HandEmpty = Predicate("HandEmpty", [], lambda s, o: not s[agent][1])
    Holding = Predicate("Holding", [hardback_type],
                        lambda s, o: s[o[0]][0] == -1)
    OnTop = Predicate("OnTop", [hardback_type, shelf_type],
                      lambda s, o: s[o[0]][0] == s[o[1]][0])
    preds = {NextTo, NextToShelf, HandEmpty, Holding, OnTop}

    # Agent not holding anything at location 0, hardbacks at loaction 1,
    # and shelf at location 2
    state1 = State({
        agent: [0, False],
        book1: [3],
        hardback1: [1],
        hardback2: [1],
        shelf: [2]
    })
    # Agent moves to location 1
    state2 = State({
        agent: [1, False],
        book1: [3],
        hardback1: [1],
        hardback2: [1],
        shelf: [2]
    })
    # Agent grabs hardback1
    state3 = State({
        agent: [1, True],
        book1: [3],
        hardback1: [-1],
        hardback2: [1],
        shelf: [2]
    })
    # Agent moves to location 2
    state4 = State({
        agent: [2, True],
        book1: [3],
        hardback1: [-1],
        hardback2: [1],
        shelf: [2]
    })
    # Agent places hardback1
    state5 = State({
        agent: [2, False],
        book1: [3],
        hardback1: [2],
        hardback2: [1],
        shelf: [2]
    })
    # Agent moves to location 1
    state6 = State({
        agent: [1, False],
        book1: [3],
        hardback1: [2],
        hardback2: [1],
        shelf: [2]
    })
    # Agent grabs hardback2
    state7 = State({
        agent: [1, True],
        book1: [3],
        hardback1: [2],
        hardback2: [-1],
        shelf: [2]
    })
    # Agent moves to location 2
    state8 = State({
        agent: [2, True],
        book1: [3],
        hardback1: [2],
        hardback2: [-1],
        shelf: [2]
    })
    # Agent places hardback2
    state9 = State({
        agent: [2, False],
        book1: [3],
        hardback1: [2],
        hardback2: [2],
        shelf: [2]
    })

    # States for second trajectory
    # Agent moves to location 3
    state10 = State({
        agent: [3, False],
        book1: [3],
        hardback1: [1],
        hardback2: [1],
        shelf: [2]
    })
    # Agent grabs book1 at location 3
    state11 = State({
        agent: [3, True],
        book1: [-1],
        hardback1: [1],
        hardback2: [1],
        shelf: [2]
    })
    # Agent moves to location 2
    state12 = State({
        agent: [2, True],
        book1: [-1],
        hardback1: [1],
        hardback2: [1],
        shelf: [2]
    })
    # Agent places hardback1
    state13 = State({
        agent: [2, False],
        book1: [2],
        hardback1: [1],
        hardback2: [1],
        shelf: [2]
    })

    moveto_param_option = utils.ParameterizedOption(
        "MoveTo", [hardback_type],
        policy=lambda s, m, o, p: Action(p),
        params_space=Box(0.1, 1, (1, )),
        initiable=lambda _1, _2, _3, _4: True,
        terminal=lambda _1, _2, _3, _4: True)
    moveto_option = moveto_param_option.ground([hardback1], np.array([0.5]))
    assert moveto_option.initiable(state1)
    moveto_hard2 = moveto_option.policy(state1)
    moveto_hard2.set_option(moveto_option)
    pick_param_option = utils.ParameterizedOption(
        "Pick", [hardback_type],
        policy=lambda s, m, o, p: Action(p),
        params_space=Box(0.1, 1, (1, )),
        initiable=lambda _1, _2, _3, _4: True,
        terminal=lambda _1, _2, _3, _4: True)
    pick_option = pick_param_option.ground([hardback1], np.array([0.5]))
    assert pick_option.initiable(state2)
    pick_hard2 = pick_option.policy(state2)
    pick_hard2.set_option(pick_option)
    movetoshelf_param_option = utils.ParameterizedOption(
        "MoveToShelf", [shelf_type],
        policy=lambda s, m, o, p: Action(p),
        params_space=Box(0.1, 1, (1, )),
        initiable=lambda _1, _2, _3, _4: True,
        terminal=lambda _1, _2, _3, _4: True)
    movetoshelf_option = movetoshelf_param_option.ground([shelf],
                                                         np.array([0.5]))
    assert movetoshelf_option.initiable(state3)
    movetoshelf1 = movetoshelf_option.policy(state3)
    movetoshelf1.set_option(movetoshelf_option)
    place_param_option = utils.ParameterizedOption(
        "Place", [hardback_type, shelf_type],
        policy=lambda s, m, o, p: Action(p),
        params_space=Box(0.1, 1, (1, )),
        initiable=lambda _1, _2, _3, _4: True,
        terminal=lambda _1, _2, _3, _4: True)
    place_option = place_param_option.ground([hardback1, shelf],
                                             np.array([0.5]))
    assert place_option.initiable(state4)
    place_hard2 = place_option.policy(state4)
    place_hard2.set_option(place_option)
    moveto_option_2 = moveto_param_option.ground([hardback2], np.array([0.5]))
    assert moveto_option_2.initiable(state5)
    moveto_hard1 = moveto_option_2.policy(state5)
    moveto_hard1.set_option(moveto_option_2)
    pick_option_2 = pick_param_option.ground([hardback2], np.array([0.5]))
    assert pick_option_2.initiable(state6)
    pick_hard1 = pick_option_2.policy(state6)
    pick_hard1.set_option(pick_option_2)
    movetoshelf_option_2 = movetoshelf_param_option.ground([shelf],
                                                           np.array([0.5]))
    assert movetoshelf_option_2.initiable(state7)
    movetoshelf2 = movetoshelf_option_2.policy(state7)
    movetoshelf2.set_option(movetoshelf_option_2)
    place_option_2 = place_param_option.ground([hardback2, shelf],
                                               np.array([0.5]))
    assert place_option_2.initiable(state8)
    place_hard1 = place_option_2.policy(state8)
    place_hard1.set_option(place_option_2)

    moveto_option_3 = moveto_param_option.ground([book1], np.array([0.5]))
    assert moveto_option_3.initiable(state1)
    moveto_book1 = moveto_option_3.policy(state1)
    moveto_book1.set_option(moveto_option_3)
    pick_option_3 = pick_param_option.ground([book1], np.array([0.5]))
    assert pick_option_3.initiable(state2)
    pick_book1 = pick_option_3.policy(state2)
    pick_book1.set_option(pick_option_3)
    movetoshelf_option_3 = movetoshelf_param_option.ground([shelf],
                                                           np.array([0.5]))
    assert movetoshelf_option_3.initiable(state10)
    movetoshelf2 = movetoshelf_option_3.policy(state10)
    movetoshelf2.set_option(movetoshelf_option_3)
    place_option_3 = place_param_option.ground([book1, shelf], np.array([0.5]))
    assert place_option_3.initiable(state11)
    place_book1 = place_option_3.policy(state11)
    place_book1.set_option(place_option_3)

    # Two Tasks: (1) place both close books on top shelf and (2) place
    # book1 on top of shelf
    goal1 = set([
        GroundAtom(OnTop, [hardback1, shelf]),
        GroundAtom(OnTop, [hardback2, shelf])
    ])
    goal2 = set([GroundAtom(OnTop, [book1, shelf])])
    task1 = Task(init=state1, goal=goal1)
    task2 = Task(init=state1, goal=goal2)
    traj1 = LowLevelTrajectory([
        state1, state2, state3, state4, state5, state6, state7, state8, state9
    ], [
        moveto_hard2, pick_hard2, movetoshelf1, place_hard2, moveto_hard1,
        pick_hard1, movetoshelf2, place_hard1
    ], True, 0)
    traj2 = LowLevelTrajectory(
        [state1, state10, state11, state12, state13],
        [moveto_book1, pick_book1, movetoshelf1, place_book1], True, 1)
    ground_atom_trajs = utils.create_ground_atom_dataset([traj1, traj2], preds)
    segmented_trajs = [segment_trajectory(traj) for traj in ground_atom_trajs]
    # Now, run the learner on the demo.
    learner = _MockBackchainingSTRIPSLearner([traj1, traj2], [task1, task2],
                                             preds,
                                             segmented_trajs,
                                             verify_harmlessness=True)
    natural_order_pnads = learner.learn()
    action_space = Box(0, 1, (1, ))
    dataset = [traj1, traj2]
    train_tasks = [task1, task2]
    options = {
        moveto_param_option, pick_param_option, movetoshelf_param_option,
        place_param_option
    }
    ground_atom_dataset = utils.create_ground_atom_dataset(dataset, preds)
    natural_order_nsrts, _, _ = learn_nsrts_from_data(dataset,
                                                      train_tasks,
                                                      preds,
                                                      options,
                                                      action_space,
                                                      ground_atom_dataset,
                                                      sampler_learner="random")

    traj1 = LowLevelTrajectory([
        state1, state2, state3, state4, state5, state6, state7, state8, state9
    ], [
        moveto_hard2, pick_hard2, movetoshelf1, place_hard2, moveto_hard1,
        pick_hard1, movetoshelf2, place_hard1
    ], True, 1)
    traj2 = LowLevelTrajectory(
        [state1, state10, state11, state12, state13],
        [moveto_book1, pick_book1, movetoshelf1, place_book1], True, 0)
    ground_atom_trajs = utils.create_ground_atom_dataset([traj2, traj1], preds)
    segmented_trajs = [segment_trajectory(traj) for traj in ground_atom_trajs]
    # Now, create and run the learner with the 3 demos in the reverse order.
    learner = _MockBackchainingSTRIPSLearner([traj2, traj1], [task2, task1],
                                             preds,
                                             segmented_trajs,
                                             verify_harmlessness=True)
    # Be sure to reset the segment add effects before doing this.
    learner.reset_all_segment_add_effs()
    reverse_order_pnads = learner.learn()
    action_space = Box(0, 1, (1, ))
    dataset = [traj2, traj1]
    train_tasks = [task2, task1]
    options = {
        moveto_param_option, pick_param_option, movetoshelf_param_option,
        place_param_option
    }
    ground_atom_dataset = utils.create_ground_atom_dataset(dataset, preds)
    reverse_order_nsrts, _, _ = learn_nsrts_from_data(dataset,
                                                      train_tasks,
                                                      preds,
                                                      options,
                                                      action_space,
                                                      ground_atom_dataset,
                                                      sampler_learner="random")

    # First, check that the two sets of PNADs have the same number of PNADs.
    # Uh oh, they don't
    assert len(natural_order_pnads) != len(reverse_order_pnads)
    # Second, check that the two sets of NSRTs have the same number of NSRTs.
    # They do! Because our NSRTs were learned with dataset reordering and
    # harmless operator pruning, as opposed to our PNADs which were learned
    # with our _MockBackchainingSTRIPSLearner that does not have these
    # additions.
    assert len(natural_order_nsrts) == len(reverse_order_nsrts)


def test_spawn_new_pnad():
    """Test the spawn_new_pnad() method in the BackchainingSTRIPSLearner.

    Also, test the finding of a unification necessary for specializing,
    which involves calling the _find_best_matching_pnad_and_sub method
    of the BaseSTRIPSLearner.
    """
    utils.reset_config({"backchaining_check_intermediate_harmlessness": True})
    human_type = Type("human_type", ["feat"])
    Asleep = Predicate("Asleep", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    Happy = Predicate("Happy", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    opt = utils.SingletonParameterizedOption("Move", lambda s, m, o, p: None)
    human_var = human_type("?x0")
    params = [human_var]
    add_effects = {Asleep([human_var])}
    op = STRIPSOperator("MoveOp", params, set(), add_effects, set(),
                        set([Asleep, Happy]))
    pnad = PartialNSRTAndDatastore(op, [], (opt, []))
    bob = human_type("bob")
    state = State({bob: [0.0]})
    task = Task(state, set())
    Move = opt.ground([], [])
    traj = LowLevelTrajectory([state], [])
    segment = Segment(traj, {Happy([bob])},
                      {Asleep([bob]), Happy([bob])}, Move)
    # Create the sidelining approach.
    learner = _MockBackchainingSTRIPSLearner([traj], [task], {Asleep, Happy},
                                             [[segment]],
                                             verify_harmlessness=True)
    # Normal usage: the PNAD add effects can capture a subset of
    # the necessary_add_effects.
    _, ground_op = learner.find_unification(
        {Asleep([bob]), Happy([bob])}, pnad,
        Segment(traj, set(), {Asleep([bob]), Happy([bob])}, Move))
    assert ground_op is not None
    assert str(ground_op) == repr(ground_op) == """GroundSTRIPS-MoveOp:
    Parameters: [bob:human_type]
    Preconditions: []
    Add Effects: [Asleep(bob:human_type)]
    Delete Effects: []
    Ignore Effects: [Asleep, Happy]"""
    # Make the preconditions be satisfiable in the segment's init_atoms.
    # Now, we are back to normal usage.
    _, ground_op = learner.find_unification(
        {Asleep([bob])}, pnad,
        Segment(traj, {Happy([bob])},
                {Happy([bob]), Asleep([bob])}, Move))
    assert ground_op is not None
    assert str(ground_op) == repr(ground_op) == """GroundSTRIPS-MoveOp:
    Parameters: [bob:human_type]
    Preconditions: []
    Add Effects: [Asleep(bob:human_type)]
    Delete Effects: []
    Ignore Effects: [Asleep, Happy]"""
    new_pnad = learner.spawn_new_pnad({Asleep([bob])},
                                      Segment(traj, {Happy([bob])},
                                              {Asleep([bob])}, Move))

    learner.recompute_datastores_from_segments([new_pnad])
    assert len(new_pnad.datastore) == 1


def test_keep_effect_data_partitioning():
    """Test that the BackchainingSTRIPSLearner is able to correctly induce
    operators with keep effects in a case where a naive procedure that does not
    keep the original operators (without keep effects) would fail.

    There are two demonstrations: [Configure, TurnOn, Run] and [TurnOn,
    Configure, Run]. TurnOn always just turns on a machine, while
    Configure makes it configured but may turn on/off arbitrary other
    machines. You are allowed to Configure when the machine is either
    off or on (but MachineConfigurableWhileOff must be true in order to
    Configure when it's off). Our algorithm will say that MachineOn is a
    keep effect of Configure, but it's important to keep around the
    original (non-KEEP) operator for Configure, otherwise we will be
    harmful to the second demonstration, where the machine was off when
    it was configured.
    """
    utils.reset_config({
        "segmenter": "atom_changes",
        "backchaining_check_intermediate_harmlessness": True
    })
    # Set up the types and predicates.
    machine_type = Type(
        "machine_type",
        ["on", "configuration", "run", "configurable_while_off"])
    MachineOn = Predicate("MachineOn", [machine_type],
                          lambda s, o: s[o[0]][0] > 0.5)
    MachineConfigurableWhileOff = Predicate("MachineConfigurableWhileOff",
                                            [machine_type],
                                            lambda s, o: s[o[0]][3] > 0.5)
    MachineConfigured = Predicate("MachineConfigured", [machine_type],
                                  lambda s, o: s[o[0]][1] > 0.5)
    MachineRun = Predicate("MachineRun", [machine_type],
                           lambda s, o: s[o[0]][2] > 0.5)
    predicates = {
        MachineOn, MachineConfigurableWhileOff, MachineConfigured, MachineRun
    }

    m1 = machine_type("m1")
    m2 = machine_type("m2")
    m3 = machine_type("m3")

    # Create states to be used as part of trajectories.
    all_off_not_configed = State({
        m1: [0.0, 0.0, 0.0, 1.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m1_off_configed_m2_on = State({
        m1: [0.0, 1.0, 0.0, 1.0],
        m2: [1.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m1_on_configed_m2_on = State({
        m1: [1.0, 1.0, 0.0, 1.0],
        m2: [1.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m1_on_configed_run_m2_on = State({
        m1: [1.0, 1.0, 1.0, 1.0],
        m2: [1.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m3_on = State({
        m1: [0.0, 0.0, 0.0, 0.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [1.0, 0.0, 0.0, 0.0],
    })
    m1_on_m3_on = State({
        m1: [1.0, 0.0, 0.0, 0.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [1.0, 0.0, 0.0, 0.0],
    })
    m1_on_configed = State({
        m1: [1.0, 1.0, 0.0, 0.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0],
    })
    m1_on_configed_run = State({
        m1: [1.0, 1.0, 1.0, 0.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0],
    })

    # Create the necessary options and actions.
    turn_on = utils.SingletonParameterizedOption("TurnOn",
                                                 lambda s, m, o, p: None)
    TurnOn = turn_on.ground([], [])
    turn_on_act = Action([], TurnOn)
    configure = utils.SingletonParameterizedOption("Configure",
                                                   lambda s, m, o, p: None)
    Configure = configure.ground([], [])
    configure_act = Action([], Configure)
    run = utils.SingletonParameterizedOption("Run", lambda s, m, o, p: None)
    Run = run.ground([], [])
    run_act = Action([], Run)

    # Create the trajectories, goals, and tasks.
    # The first trajectory is: [Configure, TurnOn, Run].
    # The second trajectory is: [TurnOn, Configure, Run].
    traj1 = LowLevelTrajectory([
        all_off_not_configed, m1_off_configed_m2_on, m1_on_configed_m2_on,
        m1_on_configed_run_m2_on
    ], [configure_act, turn_on_act, run_act], True, 0)
    traj2 = LowLevelTrajectory(
        [m3_on, m1_on_m3_on, m1_on_configed, m1_on_configed_run],
        [turn_on_act, configure_act, run_act], True, 1)
    goal = {
        MachineRun([m1]),
    }
    task1 = Task(all_off_not_configed, goal)
    task2 = Task(m3_on, goal)

    ground_atom_trajs = utils.create_ground_atom_dataset([traj1, traj2],
                                                         predicates)
    segmented_trajs = [segment_trajectory(traj) for traj in ground_atom_trajs]

    # Now, run the learner on the two demos.
    learner = _MockBackchainingSTRIPSLearner([traj1, traj2], [task1, task2],
                                             predicates,
                                             segmented_trajs,
                                             verify_harmlessness=True)
    output_pnads = learner.learn()
    # There should be exactly 4 output PNADs: 2 for Configure, and 1 for
    # each of TurnOn and Run. One of the Configure operators should have
    # a keep effect, while the other shouldn't.
    assert len(output_pnads) == 4
    correct_pnads = set([
        """STRIPS-Run:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineConfigured(?x0:machine_type), """ + \
        """MachineOn(?x0:machine_type)]
    Add Effects: [MachineRun(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Run()""", """STRIPS-TurnOn:
    Parameters: [?x0:machine_type]
    Preconditions: []
    Add Effects: [MachineOn(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: TurnOn()""", """STRIPS-Configure:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineConfigurableWhileOff(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: [MachineOn]
    Option Spec: Configure()""", """STRIPS-Configure:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineOn(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), """ + \
        """MachineOn(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: [MachineOn]
    Option Spec: Configure()"""
    ])

    # Verify that all the output PNADs are correct.
    for pnad in output_pnads:
        # Rename the output PNADs to standardize naming
        # and make comparison easier.
        pnad.op = pnad.op.copy_with(name=pnad.option_spec[0].name)
        assert str(pnad) in correct_pnads


def test_combinatorial_keep_effect_data_partitioning():
    """Test that the BackchainingSTRIPSLearner is able to correctly induce
    operators with keep effects in a case where a naive procedure that always
    induces potential keep effects would fail.

    The domain here is identical to the domain in the above test, except that
    there is no MachineConfigurableWhileOff predicate and thus the Configure
    action can be run on any machine regardless of if it is on or off.
    There are four demonstrations here:
    1. Fix, Configure, Turn On, Run
    2. Fix, Turn On, Configure, Run
    3. Configure, Turn On, Fix, Run
    4. Turn On, Configure, Fix, Run
    The goal is always to run machine 1, which requires it being working,
    on, and configured. The main idea of this test is that configuring a
    machine may turn off other machines or render them not working. Thus,
    given these four demos, the learner should induce 4 different Configure
    PNADs with keep effects for MachineWorking, MachineOn, both and neither
    in order to preserve harmlessness. Additionally, if demo 4 is removed, then
    the learner no longer needs a Configure PNAD with a keep effect for
    MachineOn and thus only needs 3 Configure PNADs.
    """
    utils.reset_config({
        "segmenter": "atom_changes",
        "backchaining_check_intermediate_harmlessness": True
    })
    # Set up the types and predicates.
    machine_type = Type("machine_type",
                        ["on", "configuration", "run", "working"])
    MachineOn = Predicate("MachineOn", [machine_type],
                          lambda s, o: s[o[0]][0] > 0.5)
    MachineConfigured = Predicate("MachineConfigured", [machine_type],
                                  lambda s, o: s[o[0]][1] > 0.5)
    MachineRun = Predicate("MachineRun", [machine_type],
                           lambda s, o: s[o[0]][2] > 0.5)
    MachineWorking = Predicate("MachineWorking", [machine_type],
                               lambda s, o: s[o[0]][3] > 0.5)
    predicates = {MachineOn, MachineConfigured, MachineRun, MachineWorking}

    m1 = machine_type("m1")
    m2 = machine_type("m2")
    m3 = machine_type("m3")

    # Create states to be used as part of trajectories.
    all_off_not_configed = State({
        m1: [0.0, 0.0, 0.0, 0.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m1_off_configed_m2_on = State({
        m1: [0.0, 1.0, 0.0, 0.0],
        m2: [1.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m1_on_configed_m2_on = State({
        m1: [1.0, 1.0, 0.0, 0.0],
        m2: [1.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m1_on_configed = State({
        m1: [1.0, 1.0, 0.0, 0.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0],
    })
    m1_fix = State({
        m1: [0.0, 0.0, 0.0, 1.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m1_fix_m1_off_configed_m2_on = State({
        m1: [0.0, 1.0, 0.0, 1.0],
        m2: [1.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m1_fix_m1_on_configed_m2_on = State({
        m1: [1.0, 1.0, 0.0, 1.0],
        m2: [1.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m1_fix_m1_on_configed_run_m2_on = State({
        m1: [1.0, 1.0, 1.0, 1.0],
        m2: [1.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0]
    })
    m3_fix_m3_on = State({
        m1: [0.0, 0.0, 0.0, 0.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [1.0, 0.0, 0.0, 1.0],
    })
    m1_on_m3_fix_m3_on = State({
        m1: [1.0, 0.0, 0.0, 0.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [1.0, 0.0, 0.0, 1.0],
    })
    m1_fix_m1_on_configed = State({
        m1: [1.0, 1.0, 0.0, 1.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0],
    })
    m1_fix_m1_on_configed_run = State({
        m1: [1.0, 1.0, 1.0, 1.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0, 0.0],
    })
    m1_fix_m3_fix_m3_on = State({
        m1: [0.0, 0.0, 0.0, 1.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [1.0, 0.0, 0.0, 1.0],
    })
    m1_fix_m1_on_m3_fix_m3_on = State({
        m1: [1.0, 0.0, 0.0, 1.0],
        m2: [0.0, 0.0, 0.0, 0.0],
        m3: [1.0, 0.0, 0.0, 1.0],
    })

    # Create the necessary options and actions.
    turn_on = utils.SingletonParameterizedOption("TurnOn",
                                                 lambda s, m, o, p: None)
    TurnOn = turn_on.ground([], [])
    turn_on_act = Action([], TurnOn)
    configure = utils.SingletonParameterizedOption("Configure",
                                                   lambda s, m, o, p: None)
    Configure = configure.ground([], [])
    configure_act = Action([], Configure)
    run = utils.SingletonParameterizedOption("Run", lambda s, m, o, p: None)
    Run = run.ground([], [])
    run_act = Action([], Run)
    fix = utils.SingletonParameterizedOption("Fix", lambda s, m, o, p: None)
    Fix = fix.ground([], [])
    fix_act = Action([], Fix)

    # Create the trajectories, goals, and tasks.
    traj1 = LowLevelTrajectory([
        all_off_not_configed, m1_fix, m1_fix_m1_off_configed_m2_on,
        m1_fix_m1_on_configed_m2_on, m1_fix_m1_on_configed_run_m2_on
    ], [fix_act, configure_act, turn_on_act, run_act], True, 0)
    traj2 = LowLevelTrajectory([
        m3_fix_m3_on, m1_fix_m3_fix_m3_on, m1_fix_m1_on_m3_fix_m3_on,
        m1_fix_m1_on_configed, m1_fix_m1_on_configed_run
    ], [fix_act, turn_on_act, configure_act, run_act], True, 1)
    traj3 = LowLevelTrajectory([
        all_off_not_configed, m1_off_configed_m2_on, m1_on_configed_m2_on,
        m1_fix_m1_on_configed_m2_on, m1_fix_m1_on_configed_run_m2_on
    ], [configure_act, turn_on_act, fix_act, run_act], True, 2)
    traj4 = LowLevelTrajectory([
        m3_fix_m3_on, m1_on_m3_fix_m3_on, m1_on_configed,
        m1_fix_m1_on_configed, m1_fix_m1_on_configed_run
    ], [turn_on_act, configure_act, fix_act, run_act], True, 3)
    goal = {
        MachineRun([m1]),
    }
    task1 = Task(all_off_not_configed, goal)
    task2 = Task(m3_fix_m3_on, goal)
    task3 = Task(all_off_not_configed, goal)
    task4 = Task(m3_fix_m3_on, goal)

    ground_atom_trajs = utils.create_ground_atom_dataset(
        [traj1, traj2, traj3, traj4], predicates)
    segmented_trajs = [segment_trajectory(traj) for traj in ground_atom_trajs]

    # Now, run the learner on the four demos.
    learner = _MockBackchainingSTRIPSLearner([traj1, traj2, traj3, traj4],
                                             [task1, task2, task3, task4],
                                             predicates,
                                             segmented_trajs,
                                             verify_harmlessness=True)
    output_pnads = learner.learn()
    # We need 7 PNADs: 4 for configure, and 1 each for turn on, run, and fix.
    assert len(output_pnads) == 7
    correct_pnads = set([
        """STRIPS-Run:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineConfigured(?x0:machine_type), """ +
        """MachineOn(?x0:machine_type), MachineWorking(?x0:machine_type)]
    Add Effects: [MachineRun(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Run()""", """STRIPS-TurnOn:
    Parameters: [?x0:machine_type]
    Preconditions: []
    Add Effects: [MachineOn(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: TurnOn()""", """STRIPS-Fix:
    Parameters: [?x0:machine_type]
    Preconditions: []
    Add Effects: [MachineWorking(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Fix()""", """STRIPS-Configure:
    Parameters: [?x0:machine_type]
    Preconditions: []
    Add Effects: [MachineConfigured(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: [MachineOn]
    Option Spec: Configure()""", """STRIPS-Configure:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineWorking(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), """ +
        """MachineWorking(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: [MachineOn]
    Option Spec: Configure()""", """STRIPS-Configure:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineOn(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), """ +
        """MachineOn(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: [MachineOn, MachineWorking]
    Option Spec: Configure()""", """STRIPS-Configure:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineOn(?x0:machine_type), """ +
        """MachineWorking(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), """ +
        """MachineOn(?x0:machine_type), MachineWorking(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: [MachineOn, MachineWorking]
    Option Spec: Configure()"""
    ])

    # Verify that all the output PNADs are correct.
    for pnad in output_pnads:
        # Rename the output PNADs to standardize naming
        # and make comparison easier.
        pnad.op = pnad.op.copy_with(name=pnad.option_spec[0].name)
        assert str(pnad) in correct_pnads

    # Now, run the learner on 3/4 of the demos and verify that it produces only
    # 3 PNADs for the Configure action.
    learner = _MockBackchainingSTRIPSLearner([traj1, traj2, traj3],
                                             [task1, task2, task3],
                                             predicates,
                                             segmented_trajs[:-1],
                                             verify_harmlessness=True)
    learner.reset_all_segment_add_effs()
    output_pnads = learner.learn()
    assert len(output_pnads) == 6

    correct_pnads = correct_pnads - set([
        """STRIPS-Configure:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineOn(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), """ +
        """MachineOn(?x0:machine_type)]
    Delete Effects: []
    Ignore Effects: [MachineOn, MachineWorking]
    Option Spec: Configure()"""
    ])

    # Verify that all the output PNADs are correct.
    for pnad in output_pnads:
        # Rename the output PNADs to standardize naming
        # and make comparison easier.
        pnad.op = pnad.op.copy_with(name=pnad.option_spec[0].name)
        assert str(pnad) in correct_pnads


def test_keep_effect_adding_new_variables():
    """Test that the BackchainingSTRIPSLearner is able to correctly induce
    operators when the keep effects must create new variables to ensure
    harmlessness."""
    utils.reset_config({
        "segmenter": "atom_changes",
        "backchaining_check_intermediate_harmlessness": True
    })
    # Set up the types and predicates.
    button_type = Type("button_type", ["pressed"])
    potato_type = Type("potato_type", ["held", "intact"])
    ButtonPressed = Predicate("ButtonPressed", [button_type],
                              lambda s, o: s[o[0]][0] > 0.5)
    PotatoHeld = Predicate("PotatoHeld", [potato_type],
                           lambda s, o: s[o[0]][0] > 0.5)
    PotatoIntact = Predicate("PotatoIntact", [potato_type],
                             lambda s, o: s[o[0]][1] > 0.5)
    predicates = {ButtonPressed, PotatoHeld, PotatoIntact}

    button = button_type("button")
    potato1 = potato_type("potato1")
    potato2 = potato_type("potato2")
    potato3 = potato_type("potato3")

    # Create states to be used as part of the trajectory.
    s0 = State({
        button: [0.0],
        potato1: [0.0, 1.0],
        potato2: [0.0, 1.0],
        potato3: [0.0, 1.0],
    })
    s1 = State({
        button: [1.0],
        potato1: [0.0, 1.0],
        potato2: [0.0, 0.0],
        potato3: [0.0, 1.0],
    })
    s2 = State({
        button: [1.0],
        potato1: [0.0, 1.0],
        potato2: [0.0, 0.0],
        potato3: [1.0, 1.0],
    })

    # Create the necessary options and actions.
    press = utils.SingletonParameterizedOption("Press",
                                               lambda s, m, o, p: None,
                                               types=[button_type])
    Press = press.ground([button], [])
    press_act = Action([], Press)
    pick = utils.SingletonParameterizedOption("Pick",
                                              lambda s, m, o, p: None,
                                              types=[potato_type])
    Pick = pick.ground([potato3], [])
    pick_act = Action([], Pick)

    # Create the trajectory, goal, and task.
    traj = LowLevelTrajectory([s0, s1, s2], [press_act, pick_act], True, 0)
    goal = {ButtonPressed([button]), PotatoHeld([potato3])}
    task = Task(s0, goal)

    ground_atom_traj = utils.create_ground_atom_dataset([traj], predicates)[0]
    segmented_traj = segment_trajectory(ground_atom_traj)

    # Now, run the learner on the demo.
    learner = _MockBackchainingSTRIPSLearner([traj], [task],
                                             predicates, [segmented_traj],
                                             verify_harmlessness=True)
    output_pnads = learner.learn()

    # Verify that all the output PNADs are correct. The PNAD for Press should
    # have a keep effect that keeps potato3 intact (in the datastore's sub).
    assert len(output_pnads) == 2
    correct_pnads = set([
        """STRIPS-Press:
    Parameters: [?x0:button_type, ?x1:potato_type]
    Preconditions: [PotatoIntact(?x1:potato_type)]
    Add Effects: [ButtonPressed(?x0:button_type), PotatoIntact(?x1:potato_type)]
    Delete Effects: []
    Ignore Effects: [PotatoIntact]
    Option Spec: Press(?x0:button_type)""", """STRIPS-Pick:
    Parameters: [?x0:potato_type]
    Preconditions: [PotatoIntact(?x0:potato_type)]
    Add Effects: [PotatoHeld(?x0:potato_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Pick(?x0:potato_type)"""
    ])

    button_x0 = button_type("?x0")
    potato_x0 = potato_type("?x0")
    potato_x1 = potato_type("?x1")
    for pnad in output_pnads:
        # Rename the output PNADs to standardize naming
        # and make comparison easier.
        pnad.op = pnad.op.copy_with(name=pnad.option_spec[0].name)
        assert str(pnad) in correct_pnads
        if pnad.option_spec[0].name == "Press":
            # We need that potato3 in particular is left intact during the
            # Press, because it needs to be intact for the subsequent Pick.
            assert len(pnad.datastore) == 1
            seg, sub = pnad.datastore[0]
            assert seg is segmented_traj[0]
            assert sub == {button_x0: button, potato_x1: potato3}
        else:
            assert pnad.option_spec[0].name == "Pick"
            # The demonstrator Picked potato3.
            assert len(pnad.datastore) == 1
            seg, sub = pnad.datastore[0]
            assert seg is segmented_traj[1]
            assert sub == {potato_x0: potato3}


@pytest.mark.parametrize("val", [0.0, 1.0])
def test_multi_pass_backchaining(val):
    """Test that the BackchainingSTRIPSLearner does multiple passes of
    backchaining, which is needed to ensure harmlessness."""
    utils.reset_config({
        "segmenter": "atom_changes",
        "backchaining_check_intermediate_harmlessness": True
    })
    # Set up the types, objects, and, predicates.
    dummy_type = Type("dummy_type",
                      ["feat1", "feat2", "feat3", "feat4", "feat5"])
    dummy = dummy_type("dummy")
    A = Predicate("A", [], lambda s, o: s[dummy][0] > 0.5)
    B = Predicate("B", [], lambda s, o: s[dummy][1] > 0.5)
    C = Predicate("C", [], lambda s, o: s[dummy][2] > 0.5)
    D = Predicate("D", [], lambda s, o: s[dummy][3] > 0.5)
    E = Predicate("E", [], lambda s, o: s[dummy][4] > 0.5)
    predicates = {A, B, C, D, E}

    # Create the necessary options and actions.
    Pick = utils.SingletonParameterizedOption("Pick",
                                              lambda s, m, o, p: None,
                                              types=[]).ground([], [])
    pick_act = Action([], Pick)
    Place = utils.SingletonParameterizedOption("Place",
                                               lambda s, m, o, p: None,
                                               types=[]).ground([], [])
    place_act = Action([], Place)

    # Create trajectories.
    s10 = State({dummy: [1.0, 0.0, 0.0, 0.0, 0.0]})
    s11 = State({dummy: [1.0, 1.0, 1.0, 0.0, 0.0]})
    s12 = State({dummy: [1.0, 0.0, 1.0, 1.0, 1.0]})
    traj1 = LowLevelTrajectory([s10, s11, s12], [pick_act, place_act], True, 0)
    goal1 = {GroundAtom(D, [])}
    task1 = Task(s10, goal1)

    s20 = State({dummy: [1.0, 1.0, 0.0, 0.0, val]})
    s21 = State({dummy: [1.0, 0.0, 0.0, 1.0, 1.0]})
    traj2 = LowLevelTrajectory([s20, s21], [place_act], True, 1)
    goal2 = {GroundAtom(D, []), GroundAtom(E, [])}
    task2 = Task(s20, goal2)

    s30 = State({dummy: [1.0, 1.0, val, 0.0, 0.0]})
    s31 = State({dummy: [1.0, 0.0, 1.0, 1.0, 1.0]})
    traj3 = LowLevelTrajectory([s30, s31], [place_act], True, 2)
    goal3 = {GroundAtom(C, []), GroundAtom(D, [])}
    task3 = Task(s30, goal3)

    ground_atom_trajs = utils.create_ground_atom_dataset([traj1, traj2, traj3],
                                                         predicates)
    segmented_trajs = [segment_trajectory(traj) for traj in ground_atom_trajs]

    # Now, run the learner on the three demos.
    learner = _MockBackchainingSTRIPSLearner([traj1, traj2, traj3],
                                             [task1, task2, task3],
                                             predicates,
                                             segmented_trajs,
                                             verify_harmlessness=True)
    # Running this automatically checks that harmlessness passes.
    learned_pnads = learner.learn()
    assert len(learned_pnads) == 3
    if val == 0.0:
        correct_pnads = [
            """STRIPS-Pick:
    Parameters: []
    Preconditions: [A()]
    Add Effects: [B()]
    Delete Effects: []
    Ignore Effects: [C]
    Option Spec: Pick()""", """STRIPS-Place:
    Parameters: []
    Preconditions: [A(), B()]
    Add Effects: [D(), E()]
    Delete Effects: [B()]
    Ignore Effects: []
    Option Spec: Place()""", """STRIPS-Place:
    Parameters: []
    Preconditions: [A(), B()]
    Add Effects: [C(), D()]
    Delete Effects: [B()]
    Ignore Effects: [E]
    Option Spec: Place()"""
        ]
    else:
        correct_pnads = [
            """STRIPS-Pick:
    Parameters: []
    Preconditions: [A()]
    Add Effects: [B(), C()]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Pick()""", """STRIPS-Place:
    Parameters: []
    Preconditions: [A(), B(), C()]
    Add Effects: [D()]
    Delete Effects: [B()]
    Ignore Effects: [E]
    Option Spec: Place()""", """STRIPS-Place:
    Parameters: []
    Preconditions: [A(), B(), E()]
    Add Effects: [D(), E()]
    Delete Effects: [B()]
    Ignore Effects: []
    Option Spec: Place()"""
        ]

    for pnad in learned_pnads:
        # Rename the output PNADs to standardize naming
        # and make comparison easier.
        pnad.op = pnad.op.copy_with(name=pnad.option_spec[0].name)
        assert str(pnad) in correct_pnads


def test_backchaining_segment_not_in_datastore():
    """Test the BackchainingSTRIPSLearner on a case where it can cover a
    particular segment using an operator that doesn't have that segment in its
    datastore.

    This will lead to the intermediate harmlessness check failing if not
    handled correctly.
    """
    utils.reset_config({
        "segmenter": "atom_changes",
        "backchaining_check_intermediate_harmlessness": True
    })
    # Set up the types, objects, and, predicates.
    dummy_type = Type("dummy_type",
                      ["feat1", "feat2", "feat3", "feat4", "feat5"])
    dummy = dummy_type("dummy")
    A = Predicate("A", [], lambda s, o: s[dummy][0] > 0.5)
    B = Predicate("B", [], lambda s, o: s[dummy][1] > 0.5)
    C = Predicate("C", [], lambda s, o: s[dummy][2] > 0.5)
    D = Predicate("D", [], lambda s, o: s[dummy][3] > 0.5)
    E = Predicate("E", [], lambda s, o: s[dummy][4] > 0.5)
    predicates = {A, B, C, D, E}
    # Create the necessary options and actions.
    Pick = utils.SingletonParameterizedOption("Pick",
                                              lambda s, m, o, p: None,
                                              types=[]).ground([], [])
    act = Action([], Pick)
    # Set up the first trajectory.
    goal0 = {GroundAtom(B, [])}
    s00 = State({dummy: [1.0, 1.0, 0.0, 1.0, 0.0]})
    s01 = State({dummy: [0.0, 0.0, 1.0, 0.0, 1.0]})
    s02 = State({dummy: [1.0, 1.0, 1.0, 1.0, 1.0]})
    traj0 = LowLevelTrajectory([s00, s01, s02], [act, act], True, 0)
    task0 = Task(s00, goal0)
    # Set up the second trajectory.
    goal1 = {GroundAtom(B, [])}
    s10 = State({dummy: [1.0, 0.0, 0.0, 1.0, 0.0]})
    s11 = State({dummy: [1.0, 1.0, 0.0, 0.0, 1.0]})
    traj1 = LowLevelTrajectory([s10, s11], [act], True, 1)
    task1 = Task(s10, goal1)
    # Set up the third trajectory.
    goal2 = {GroundAtom(A, []), GroundAtom(C, [])}
    s20 = State({dummy: [0.0, 1.0, 0.0, 1.0, 1.0]})
    s21 = State({dummy: [1.0, 1.0, 1.0, 1.0, 1.0]})
    traj2 = LowLevelTrajectory([s20, s21], [act], True, 2)
    task2 = Task(s20, goal2)
    # Set up the fourth trajectory.
    goal3 = {GroundAtom(A, []), GroundAtom(D, [])}
    s30 = State({dummy: [0.0, 1.0, 1.0, 1.0, 1.0]})
    s31 = State({dummy: [1.0, 0.0, 1.0, 1.0, 0.0]})
    s32 = State({dummy: [1.0, 1.0, 0.0, 1.0, 1.0]})
    traj3 = LowLevelTrajectory([s30, s31, s32], [act, act], True, 3)
    task3 = Task(s30, goal3)
    # Ground and segment these trajectories.
    trajs = [traj0, traj1, traj2, traj3]
    ground_atom_trajs = utils.create_ground_atom_dataset(trajs, predicates)
    segmented_trajs = [segment_trajectory(traj) for traj in ground_atom_trajs]
    # Now, run the learner on the demos.
    learner = _MockBackchainingSTRIPSLearner(trajs,
                                             [task0, task1, task2, task3],
                                             predicates,
                                             segmented_trajs,
                                             verify_harmlessness=True)
    # Running this automatically checks that harmlessness passes.
    learned_pnads = learner.learn()
    correct_pnads = [
        """STRIPS-Pick:
    Parameters: []
    Preconditions: [C(), E()]
    Add Effects: [B()]
    Delete Effects: []
    Ignore Effects: [A, D]
    Option Spec: Pick()""", """STRIPS-Pick:
    Parameters: []
    Preconditions: [B(), D(), E()]
    Add Effects: [A(), C()]
    Delete Effects: [B(), E()]
    Ignore Effects: [B, E]
    Option Spec: Pick()""", """STRIPS-Pick:
    Parameters: []
    Preconditions: [A(), C(), D()]
    Add Effects: [A(), B(), D()]
    Delete Effects: [C()]
    Ignore Effects: [E]
    Option Spec: Pick()""", """STRIPS-Pick:
    Parameters: []
    Preconditions: [A(), D()]
    Add Effects: [A(), B()]
    Delete Effects: [D()]
    Ignore Effects: [E]
    Option Spec: Pick()""", """STRIPS-Pick:
    Parameters: []
    Preconditions: [A(), B(), D()]
    Add Effects: [C(), E()]
    Delete Effects: [A(), B(), D()]
    Ignore Effects: []
    Option Spec: Pick()"""
    ]
    for pnad in learned_pnads:
        # Rename the output PNADs to standardize naming
        # and make comparison easier.
        pnad.op = pnad.op.copy_with(name=pnad.option_spec[0].name)
        assert str(pnad) in correct_pnads


@longrun
@pytest.mark.parametrize("use_single_option,num_demos,seed_offset",
                         itertools.product([True, False], [1, 2, 3, 4],
                                           range(250)))
def test_backchaining_randomly_generated(use_single_option, num_demos,
                                         seed_offset):
    """Test the BackchainingSTRIPSLearner on randomly generated test cases."""
    utils.reset_config({
        "segmenter": "atom_changes",
        "backchaining_check_intermediate_harmlessness": True
    })
    rng = np.random.default_rng(CFG.seed + seed_offset)
    # Set up the types, objects, and, predicates.
    dummy_type = Type("dummy_type",
                      ["feat1", "feat2", "feat3", "feat4", "feat5"])
    dummy = dummy_type("dummy")
    A = Predicate("A", [], lambda s, o: s[dummy][0] > 0.5)
    B = Predicate("B", [], lambda s, o: s[dummy][1] > 0.5)
    C = Predicate("C", [], lambda s, o: s[dummy][2] > 0.5)
    D = Predicate("D", [], lambda s, o: s[dummy][3] > 0.5)
    E = Predicate("E", [], lambda s, o: s[dummy][4] > 0.5)
    predicates = {A, B, C, D, E}
    pred_to_feat = {A: "feat1", B: "feat2", C: "feat3", D: "feat4", E: "feat5"}

    # Create the necessary options and actions.
    Pick = utils.SingletonParameterizedOption("Pick",
                                              lambda s, m, o, p: None,
                                              types=[]).ground([], [])
    act1 = Action([], Pick)
    Place = utils.SingletonParameterizedOption("Place",
                                               lambda s, m, o, p: None,
                                               types=[]).ground([], [])
    act2 = Action([], Place)

    # Create trajectories.

    # Create trajectory 1 (length 3).
    # Sample a goal.
    goal1 = {
        GroundAtom(pred, [])
        for pred in rng.permutation([A, B, C, D, E])[:rng.integers(1, 5)]
    }
    s10 = State({dummy: list(rng.choice([0.0, 1.0], size=5))})
    while True:
        # Sample s11 until it is different from s10.
        s11 = State({dummy: list(rng.choice([0.0, 1.0], size=5))})
        if s11[dummy] != s10[dummy]:
            break
    while True:
        # Sample s12 until it is different from s11.
        s12 = State({dummy: list(rng.choice([0.0, 1.0], size=5))})
        # Force the goal to be achieved.
        for atom in goal1:
            s12.set(dummy, pred_to_feat[atom.predicate], 1.0)
        if s12[dummy] != s11[dummy]:
            break
    if use_single_option:
        acts = [act1, act1]
    else:
        poss_acts = [[act1, act1], [act1, act2], [act2, act1], [act2, act2]]
        acts = poss_acts[rng.integers(len(poss_acts))]
    traj1 = LowLevelTrajectory([s10, s11, s12], acts, True, 0)
    task1 = Task(s10, goal1)

    # Create trajectory 2 (length 2).
    # Sample a goal.
    goal2 = {
        GroundAtom(pred, [])
        for pred in rng.permutation([A, B, C, D, E])[:rng.integers(1, 5)]
    }
    s20 = State({dummy: list(rng.choice([0.0, 1.0], size=5))})
    while True:
        # Sample s21 until it is different from s20.
        s21 = State({dummy: list(rng.choice([0.0, 1.0], size=5))})
        # Force the goal to be achieved.
        for atom in goal2:
            s21.set(dummy, pred_to_feat[atom.predicate], 1.0)
        if s21[dummy] != s20[dummy]:
            break
    if use_single_option:
        acts = [act1]
    else:
        poss_acts = [[act1], [act2]]
        acts = poss_acts[rng.integers(len(poss_acts))]
    traj2 = LowLevelTrajectory([s20, s21], acts, True, 1)
    task2 = Task(s20, goal2)

    # Create trajectory 3 (length 2).
    # Sample a goal.
    goal3 = {
        GroundAtom(pred, [])
        for pred in rng.permutation([A, B, C, D, E])[:rng.integers(1, 5)]
    }
    s30 = State({dummy: list(rng.choice([0.0, 1.0], size=5))})
    while True:
        # Sample s31 until it is different from s30.
        s31 = State({dummy: list(rng.choice([0.0, 1.0], size=5))})
        # Force the goal to be achieved.
        for atom in goal3:
            s31.set(dummy, pred_to_feat[atom.predicate], 1.0)
        if s31[dummy] != s30[dummy]:
            break
    if use_single_option:
        acts = [act1]
    else:
        poss_acts = [[act1], [act2]]
        acts = poss_acts[rng.integers(len(poss_acts))]
    traj3 = LowLevelTrajectory([s30, s31], acts, True, 2)
    task3 = Task(s30, goal3)

    # Create trajectory 4 (length 3).
    # Sample a goal.
    goal4 = {
        GroundAtom(pred, [])
        for pred in rng.permutation([A, B, C, D, E])[:rng.integers(1, 5)]
    }
    s40 = State({dummy: list(rng.choice([0.0, 1.0], size=5))})
    while True:
        # Sample s41 until it is different from s40.
        s41 = State({dummy: list(rng.choice([0.0, 1.0], size=5))})
        if s41[dummy] != s40[dummy]:
            break
    while True:
        # Sample s42 until it is different from s41.
        s42 = State({dummy: list(rng.choice([0.0, 1.0], size=5))})
        # Force the goal to be achieved.
        for atom in goal4:
            s42.set(dummy, pred_to_feat[atom.predicate], 1.0)
        if s42[dummy] != s41[dummy]:
            break
    if use_single_option:
        acts = [act1, act1]
    else:
        poss_acts = [[act1, act1], [act1, act2], [act2, act1], [act2, act2]]
        acts = poss_acts[rng.integers(len(poss_acts))]
    traj4 = LowLevelTrajectory([s40, s41, s42], acts, True, 3)
    task4 = Task(s40, goal4)

    trajs = [traj1, traj2, traj3, traj4][:num_demos]
    tasks = [task1, task2, task3, task4][:num_demos]

    ground_atom_trajs = utils.create_ground_atom_dataset(trajs, predicates)
    segmented_trajs = [segment_trajectory(traj) for traj in ground_atom_trajs]

    # Now, run the learner on the demos.
    learner = _MockBackchainingSTRIPSLearner(trajs,
                                             tasks,
                                             predicates,
                                             segmented_trajs,
                                             verify_harmlessness=True)
    # Running this automatically checks that harmlessness passes.
    learner.learn()
