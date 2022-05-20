"""Tests for general-to-specific STRIPS operator learning."""

from predicators.src import utils
from predicators.src.nsrt_learning.segmentation import segment_trajectory
from predicators.src.nsrt_learning.strips_learning.gen_to_spec_learner import \
    BackchainingSTRIPSLearner
from predicators.src.structs import Action, LowLevelTrajectory, \
    PartialNSRTAndDatastore, Predicate, Segment, State, STRIPSOperator, Task, \
    Type


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
                                             [[segment3], [segment4]],
                                             verify_harmlessness=True)
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
        [[segment1], [segment2], [segment3]],
        verify_harmlessness=True)
    natural_order_pnads = learner.learn()
    # Now, create and run the learner with the 3 demos in the reverse order.
    learner = _MockBackchainingSTRIPSLearner(
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
                                             [[segment]],
                                             verify_harmlessness=True)
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

    utils.reset_config({"segmenter": "atom_changes"})
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
        """STRIPS-Run0:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineConfigured(?x0:machine_type), """ + \
        """MachineOn(?x0:machine_type)]
    Add Effects: [MachineRun(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Run()""", """STRIPS-TurnOn0:
    Parameters: [?x0:machine_type]
    Preconditions: []
    Add Effects: [MachineOn(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: TurnOn()""", """STRIPS-Configure0:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineConfigurableWhileOff(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: [MachineOn]
    Option Spec: Configure()""", """STRIPS-Configure0-KEEP0:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineOn(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), """ + \
        """MachineOn(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: [MachineOn]
    Option Spec: Configure()"""
    ])

    # Verify that all the output PNADs are correct.
    for pnad in output_pnads:
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

    utils.reset_config({"segmenter": "atom_changes"})
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
        """STRIPS-Run0:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineConfigured(?x0:machine_type), """ +
        """MachineOn(?x0:machine_type), MachineWorking(?x0:machine_type)]
    Add Effects: [MachineRun(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Run()""", """STRIPS-TurnOn0:
    Parameters: [?x0:machine_type]
    Preconditions: []
    Add Effects: [MachineOn(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: TurnOn()""", """STRIPS-Fix0:
    Parameters: [?x0:machine_type]
    Preconditions: []
    Add Effects: [MachineWorking(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Fix()""", """STRIPS-Configure0:
    Parameters: [?x0:machine_type]
    Preconditions: []
    Add Effects: [MachineConfigured(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: [MachineOn, MachineWorking]
    Option Spec: Configure()""", """STRIPS-Configure0-KEEP0:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineWorking(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), """ +
        """MachineWorking(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: [MachineOn, MachineWorking]
    Option Spec: Configure()""", """STRIPS-Configure0-KEEP1:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineOn(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), """ +
        """MachineOn(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: [MachineOn, MachineWorking]
    Option Spec: Configure()""", """STRIPS-Configure0-KEEP2:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineOn(?x0:machine_type), """ +
        """MachineWorking(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), """ +
        """MachineOn(?x0:machine_type), MachineWorking(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: [MachineOn, MachineWorking]
    Option Spec: Configure()"""
    ])

    # Verify that all the output PNADs are correct.
    for pnad in output_pnads:
        assert str(pnad) in correct_pnads

    # Now, run the learner on 3/4 of the demos and verify that it produces only
    # 3 PNADs for the Configure action.
    learner = _MockBackchainingSTRIPSLearner([traj1, traj2, traj3],
                                             [task1, task2, task3],
                                             predicates,
                                             segmented_trajs[:-1],
                                             verify_harmlessness=True)
    output_pnads = learner.learn()
    assert len(output_pnads) == 6

    correct_pnads = correct_pnads - set([
        """STRIPS-Configure0-KEEP1:
    Parameters: [?x0:machine_type]
    Preconditions: [MachineOn(?x0:machine_type)]
    Add Effects: [MachineConfigured(?x0:machine_type), """ +
        """MachineOn(?x0:machine_type)]
    Delete Effects: []
    Side Predicates: [MachineOn, MachineWorking]
    Option Spec: Configure()"""
    ])

    # Verify that all the output PNADs are correct.
    for pnad in output_pnads:
        assert str(pnad) in correct_pnads


def test_keep_effect_adding_new_variables():
    """Test that the BackchainingSTRIPSLearner is able to correctly induce
    operators when the keep effects must create new variables to ensure
    harmlessness."""
    utils.reset_config({"segmenter": "atom_changes"})
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
        """STRIPS-Press0-KEEP0:
    Parameters: [?x0:button_type, ?x1:potato_type]
    Preconditions: [PotatoIntact(?x1:potato_type)]
    Add Effects: [ButtonPressed(?x0:button_type), PotatoIntact(?x1:potato_type)]
    Delete Effects: []
    Side Predicates: [PotatoIntact]
    Option Spec: Press(?x0:button_type)""", """STRIPS-Pick0:
    Parameters: [?x0:potato_type]
    Preconditions: [PotatoIntact(?x0:potato_type)]
    Add Effects: [PotatoHeld(?x0:potato_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Pick(?x0:potato_type)"""
    ])

    button_x0 = button_type("?x0")
    potato_x0 = potato_type("?x0")
    potato_x1 = potato_type("?x1")
    for pnad in output_pnads:
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
