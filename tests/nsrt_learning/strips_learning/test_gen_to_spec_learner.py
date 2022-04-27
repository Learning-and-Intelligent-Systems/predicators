"""Tests for general-to-specific STRIPS operator learning."""

from predicators.src.nsrt_learning.strips_learning.gen_to_spec_learner import \
    BackchainingSTRIPSLearner
from predicators.src.structs import Action, LowLevelTrajectory, \
    PartialNSRTAndDatastore, Predicate, Segment, State, STRIPSOperator, Task, \
    Type
from predicators.src import utils
from predicators.src.nsrt_learning.segmentation import segment_trajectory

class _MockBackchainingSTRIPSLearner(BackchainingSTRIPSLearner):
    """Mock class that exposes private methods for testing."""

    def try_specializing_pnad(self, necessary_add_effects, pnad, segment):
        """Exposed for testing."""
        return self._try_specializing_pnad(necessary_add_effects, pnad,
                                           segment)

    def recompute_datastores_from_segments(self, pnads):
        """Exposed for testing."""
        return self._recompute_datastores_from_segments(pnads)

    @staticmethod
    def find_unification(necessary_add_effects,
                         pnad,
                         segment,
                         find_partial_grounding=True):
        """Exposed for testing."""
        return (BackchainingSTRIPSLearner._find_unification(
            necessary_add_effects, pnad, segment, find_partial_grounding))


def test_backchaining_strips_learner():
    """Test the BackchainingSTRIPSLearner."""
    # Set up the structs.
    human_type = Type("human_type", ["feat1", "feat2"])
    Asleep = Predicate("Asleep", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    Sad = Predicate("Sad", [human_type], lambda s, o: s[o[0]][1] < 0.5)
    opt_name_to_opt = {}
    for opt_name in ["Cry", "Eat"]:
        opt = utils.SingletonParameterizedOption(opt_name, lambda s, m, o, p: None)
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
                                             [[segment1], [segment2]])
    pnads = learner.learn()
    # Verify the results are as expected.
    expected_strs = [
        """STRIPS-Cry0:
    Parameters: [?x0:human_type]
    Preconditions: []
    Add Effects: [Asleep(?x0:human_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Cry()""", """STRIPS-Eat0:
    Parameters: []
    Preconditions: []
    Add Effects: []
    Delete Effects: []
    Side Predicates: []
    Option Spec: Eat()"""
    ]
    for pnad, exp_str in zip(sorted(pnads, key=lambda pnad: pnad.op.name),
                             expected_strs):
        assert str(pnad) == repr(pnad) == exp_str

    # Test sidelining where an existing operator needs to
    # be specialized.
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
                                             [[segment3], [segment4]])
    pnads = learner.learn()
    assert len(pnads) == 1
    expected_str = """STRIPS-Cry0:
    Parameters: [?x0:human_type]
    Preconditions: []
    Add Effects: [Asleep(?x0:human_type), Sad(?x0:human_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Cry()"""
    assert str(pnads[0]) == repr(pnads[0]) == expected_str


def test_backchaining_strips_learner_order_dependence():
    """Test that the BackchainingSTRIPSLearner is invariant to order of
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
    learner = _MockBackchainingSTRIPSLearner(
        [traj1, traj2, traj3], [task1, task2, task3],
        {RobotAt, LightOn, NotLightOn, LightColorBlue, LightColorRed},
        [[segment1], [segment2], [segment3]])
    natural_order_pnads = learner.learn()
    # Now, create and run the learner with the 3 demos in the reverse order.
    learner = _MockBackchainingSTRIPSLearner(
        [traj3, traj2, traj1], [task1, task2, task3],
        {RobotAt, LightOn, NotLightOn, LightColorBlue, LightColorRed},
        [[segment3], [segment2], [segment1]])
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
    Side Predicates: []
    Option Spec: MoveAndMessWithLights()""", """STRIPS-MoveAndMessWithLights:
    Parameters: [?x0:fridge_type, ?x1:robot_type, ?x2:light_type]
    Preconditions: [LightColorBlue(?x2:light_type), NotLightOn(?x2:light_type)]
    Add Effects: [LightOn(?x2:light_type), """ +
        """RobotAt(?x1:robot_type, ?x0:fridge_type)]
    Delete Effects: [LightColorBlue(?x2:light_type), """ +
        """NotLightOn(?x2:light_type)]
    Side Predicates: []
    Option Spec: MoveAndMessWithLights()""", """STRIPS-MoveAndMessWithLights:
    Parameters: [?x0:fridge_type, ?x1:robot_type]
    Preconditions: []
    Add Effects: [RobotAt(?x1:robot_type, ?x0:fridge_type)]
    Delete Effects: []
    Side Predicates: [LightColorBlue, LightColorRed]
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


def test_find_unification_and_try_specializing_pnad():
    """Test the find_unification() and try_specializing_pnad() methods in the
    BackchainingSTRIPSLearner."""
    human_type = Type("human_type", ["feat"])
    Asleep = Predicate("Asleep", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    Happy = Predicate("Happy", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    opt = utils.SingletonParameterizedOption("Move", lambda s, m, o, p: None)
    human_var = human_type("?x0")
    params = [human_var]
    add_effects = {Asleep([human_var])}
    op = STRIPSOperator("MoveOp", params, set(), add_effects, set(), set())
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
                                             [[segment]])
    # Normal usage: the PNAD add effects can capture a subset of
    # the necessary_add_effects.
    ground_op = learner.find_unification(
        {Asleep([bob]), Happy([bob])}, pnad, Segment(traj, set(), set(), Move))
    assert ground_op is not None
    assert str(ground_op) == repr(ground_op) == """GroundSTRIPS-MoveOp:
    Parameters: [bob:human_type]
    Preconditions: []
    Add Effects: [Asleep(bob:human_type)]
    Delete Effects: []
    Side Predicates: []"""
    # The necessary_add_effects is empty, but the PNAD has an add effect,
    # so no grounding is possible.
    ground_op = learner.find_unification(set(), pnad,
                                         Segment(traj, set(), set(), Move))
    assert ground_op is None
    ground_op = learner.find_unification(set(), pnad,
                                         Segment(traj, set(), set(), Move),
                                         False)
    assert ground_op is None
    # Change the PNAD to have non-trivial preconditions.
    pnad.op = pnad.op.copy_with(preconditions={Happy([human_var])})
    # The new preconditions are not satisfiable in the segment's init_atoms,
    # so no grounding is possible.
    ground_op = learner.find_unification(
        set(), pnad, Segment(traj, {Asleep([bob])}, set(), Move))
    assert ground_op is None
    new_pnad = learner.try_specializing_pnad(
        set(), pnad, Segment(traj, {Asleep([bob])}, set(), Move))
    assert new_pnad is None
    # Make the preconditions be satisfiable in the segment's init_atoms.
    # Now, we are back to normal usage.
    ground_op = learner.find_unification({Asleep([bob])}, pnad,
                                         Segment(traj, {Happy([bob])}, set(),
                                                 Move))
    assert ground_op is not None
    assert str(ground_op) == repr(ground_op) == """GroundSTRIPS-MoveOp:
    Parameters: [bob:human_type]
    Preconditions: [Happy(bob:human_type)]
    Add Effects: [Asleep(bob:human_type)]
    Delete Effects: []
    Side Predicates: []"""
    new_pnad = learner.try_specializing_pnad({Asleep([bob])}, pnad,
                                             Segment(traj, {Happy([bob])},
                                                     set(), Move))

    learner.recompute_datastores_from_segments([new_pnad])
    assert len(new_pnad.datastore) == 1


def test_keep_effect_data_partitioning():
    """Test that the BackchainingSTRIPSLearner is able to correctly induce
    operators with keep effects in a case where a naive procedure that always
    induces potential keep effects would fail."""

    utils.reset_config({"segmenter": "option_changes"})
    # Set up the types and predicates.
    machine_type = Type("machine_type", ["on", "configuration", "run"])
    MachineOn = Predicate("MachineOn", [machine_type],
                          lambda s, o: s[o[0]][0] > 0.5)
    NotMachineOn = Predicate("NotMachineOn", [machine_type],
                             lambda s, o: s[o[0]][0] <= 0.5)
    MachineConfigured = Predicate("MachineConfigured", [machine_type],
                                  lambda s, o: s[o[0]][1] > 0.5)
    MachineRun = Predicate("MachineRun", [machine_type],
                           lambda s, o: s[o[0]][2] > 0.5)
    predicates = set([MachineOn, NotMachineOn, MachineConfigured, MachineRun])

    m1 = machine_type("m1")
    m2 = machine_type("m2")
    m3 = machine_type("m3")

    # Create states to be used as part of trajectories.
    all_off_not_configed = State({
        m1: [0.0, 0.0, 0.0],
        m2: [0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0]
    })
    m1_off_configed_m2_on = State({
        m1: [0.0, 1.0, 0.0],
        m2: [1.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0]
    })
    m1_on_configed_m2_on = State({
        m1: [1.0, 1.0, 0.0],
        m2: [1.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0]
    })
    m1_on_configed_run_m2_on = State({
        m1: [1.0, 1.0, 1.0],
        m2: [1.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0]
    })
    m3_on = State({
        m1: [0.0, 0.0, 0.0],
        m2: [0.0, 0.0, 0.0],
        m3: [1.0, 0.0, 0.0],
    })
    m1_on_m3_on = State({
        m1: [1.0, 0.0, 0.0],
        m2: [0.0, 0.0, 0.0],
        m3: [1.0, 0.0, 0.0],
    })
    m1_on_configed = State({
        m1: [1.0, 1.0, 0.0],
        m2: [0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0],
    })
    m1_on_configed_run = State({
        m1: [1.0, 1.0, 1.0],
        m2: [0.0, 0.0, 0.0],
        m3: [0.0, 0.0, 0.0],
    })

    # Create the necessary options and actions.
    turn_on = utils.SingletonParameterizedOption("TurnOn", lambda s, m, o, p: None)
    TurnOn = turn_on.ground([], [])
    turn_on_act = Action([], TurnOn)
    configure = utils.SingletonParameterizedOption("Configure",
                                             lambda s, m, o, p: None)
    Configure = configure.ground([], [])
    configure_act = Action([], Configure)
    run = utils.SingletonParameterizedOption("Run", lambda s, m, o, p: None)
    Run = run.ground([], [])
    run_act = Action([], Run)

    # Now create the trajectories, goals and tasks.
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

    ground_atom_trajs = utils.create_ground_atom_dataset([traj1, traj2], predicates)
    segmented_trajs = [
        segment_trajectory(traj) for traj in ground_atom_trajs
    ]
    import ipdb; ipdb.set_trace()

    # Define the 2 demos (each with 3 segments) to backchain over.
    segment1_1 = Segment(
        traj1, {NotMachineOn([m1]),
                NotMachineOn([m2]),
                NotMachineOn([m3])}, {
                    NotMachineOn([m1]),
                    MachineConfigured([m1]),
                    MachineOn([m2]),
                    NotMachineOn([m3])
                }, configure, goal)
    segment1_2 = Segment(
        traj1, {
            NotMachineOn([m1]),
            MachineConfigured([m1]),
            MachineOn([m2]),
            NotMachineOn([m3])
        }, {
            MachineOn([m1]),
            MachineConfigured([m1]),
            MachineOn([m2]),
            NotMachineOn([m3])
        }, turn_on, goal)
    segment1_3 = Segment(
        traj1, {
            MachineOn([m1]),
            MachineConfigured([m1]),
            MachineOn([m2]),
            NotMachineOn([m3])
        }, {
            MachineOn([m1]),
            MachineConfigured([m1]),
            MachineRun([m1]),
            MachineOn([m2]),
            NotMachineOn([m3])
        }, run, goal)

    segment2_1 = Segment(
        traj2, {NotMachineOn([m1]),
                NotMachineOn([m2]),
                MachineOn([m3])},
        {MachineOn([m1]), NotMachineOn([m2]),
         MachineOn([m3])}, turn_on, goal)
    segment2_2 = Segment(
        traj2,
        {MachineOn([m1]), NotMachineOn([m2]),
         MachineOn([m3])}, {
             MachineOn([m1]),
             MachineConfigured([m1]),
             NotMachineOn([m2]),
             NotMachineOn([m3])
         }, configure, goal)
    segment2_3 = Segment(
        traj2, {
            MachineOn([m1]),
            MachineConfigured([m1]),
            NotMachineOn([m2]),
            NotMachineOn([m3])
        }, {
            MachineOn([m1]),
            MachineConfigured([m1]),
            MachineRun([m1]),
            NotMachineOn([m2]),
            NotMachineOn([m3])
        }, run, goal)
    # Now, run the learner on the two demos and make sure verify_harmlessness
    # is set to True.
    learner = _MockBackchainingSTRIPSLearner(
        [traj1, traj2], [task1, task2],
        set([MachineOn, NotMachineOn, MachineConfigured,
             MachineRun]), [[segment1_1, segment1_2, segment1_3],
                            [segment2_1, segment2_2, segment2_3]], verify_harmlessness=True)
    output_pnads = learner.learn()
    # There should be exactly 4 output PNADs: 2 for Configuring, and 1 for
    # each of TurningOn and Running.
    assert len(output_pnads) == 4

    correct_pnads = set(["""STRIPS-Run:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineOn(?x0:machine_type), MachineConfigured(?x0:machine_type)]
    Add Effects: [MachineRun(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Run()""",
    """STRIPS-TurnOn:
    Parameters: [?x0:machine_type]
    Preconditions: []
    Add Effects: [MachineOn(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: TuenOn()""",
    """STRIPS-Configure0:
    Parameters: [?x0:machine_type]
    Preconditions: [NotMachineOn(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: [MachineOn, NotMachineOn]
    Option Spec: Configure()""",
    """STRIPS-Configure1:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineOn(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), MachineOn(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: [MachineOn, NotMachineOn]
    Option Spec: Configure()"""
    ])

    # Verify that all the output PNADs are correct.
    for pnad in output_pnads:
        assert pnad in correct_pnads
