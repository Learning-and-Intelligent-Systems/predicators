"""Tests for general-to-specific STRIPS operator learning."""

import pytest

from predicators import utils
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning.effect_search_learner import \
    EffectSearchSTRIPSLearner
from predicators.structs import Action, LowLevelTrajectory, \
    Predicate, Segment, State, Task, Type

longrun = pytest.mark.skipif("not config.getoption('longrun')")


def test_effect_search_strips_learner():
    """Test the EffectSearchSTRIPSLearner."""
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
    learner = EffectSearchSTRIPSLearner([traj1, traj2], [task1, task2],
                                        {Asleep}, [[segment1], [segment2]],
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
    learner = EffectSearchSTRIPSLearner([traj3, traj4], [task3, task4],
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


def test_effect_search_strips_learner_order_dependence():
    """Test that the EffectSearchSTRIPSLearner is invariant to order of
    traversal through trajectories."""
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
    learner = EffectSearchSTRIPSLearner(
        [traj1, traj2, traj3], [task1, task2, task3],
        {RobotAt, LightOn, NotLightOn, LightColorBlue, LightColorRed},
        [[segment1], [segment2], [segment3]],
        verify_harmlessness=True)
    natural_order_pnads = learner.learn()
    # Now, create and run the learner with the 3 demos in the reverse order.
    learner = EffectSearchSTRIPSLearner(
        [traj3, traj2, traj1], [task1, task2, task3],
        {RobotAt, LightOn, NotLightOn, LightColorBlue, LightColorRed},
        [[segment3], [segment2], [segment1]],
        verify_harmlessness=True)
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
    learner = EffectSearchSTRIPSLearner([traj1, traj2], [task1, task2],
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
    Ignore Effects: []
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
