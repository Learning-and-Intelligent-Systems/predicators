"""Ground-truth processes for the boil environments."""
from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.envs.pybullet_boil import PyBulletBoilEnv
from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, DelayDistribution, \
    EndogenousProcess, ExogenousProcess, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler

_BOIL_DROP_Z = 0.49  # table_height (0.4) + jug_handle_height (0.09)


def _pick_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    del state, goal, objs
    return np.array([rng.uniform(0.0, 0.02)], dtype=np.float32)


def _push_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    if not CFG.boil_use_skill_factories:
        return np.array([], dtype=np.float32)
    del state, goal, rng, objs
    return np.array([0.05, 0.1], dtype=np.float32)


def _place_on_burner_sampler(state: State, goal: Set[GroundAtom],
                             rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
    if not CFG.boil_use_skill_factories:
        return np.array([], dtype=np.float32)
    del goal, rng
    # objs = [robot, jug, burner]
    burner = objs[2]
    x = state.get(burner, "x")
    y = state.get(burner, "y") - PyBulletBoilEnv.jug_handle_offset
    return np.array([x, y, _BOIL_DROP_Z, -1.57], dtype=np.float32)


def _place_under_faucet_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
    if not CFG.boil_use_skill_factories:
        return np.array([], dtype=np.float32)
    del goal, rng
    # objs = [robot, jug, faucet]
    faucet = objs[2]
    x = state.get(faucet, "x")
    y = (state.get(faucet, "y") - PyBulletBoilEnv.jug_handle_offset -
         PyBulletBoilEnv.faucet_x_len)
    return np.array([x, y, _BOIL_DROP_Z, -1.57], dtype=np.float32)


def _place_outside_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
    if not CFG.boil_use_skill_factories:
        return np.array([], dtype=np.float32)
    del state, goal, rng, objs
    # Drop the idle jug at a single tuned spot in the open table region between
    # the burner (south) and the faucet (east), clear of the table edges. This
    # deterministic point is reachable and collision-free for both the fixed
    # and mobile bases. (A per-sample randomized spread was tried for
    # mobile_fetch but measured strictly worse -- 8/10 vs 10/10 on the 2-jug
    # tasks -- because the spread occasionally lands near the burner or past the
    # arm's reach, so it was removed.)
    x = PyBulletBoilEnv.x_mid - 0.15
    y = PyBulletBoilEnv.y_mid + 0.10
    z = _BOIL_DROP_Z
    return np.array([x, y, z, 0.0], dtype=np.float32)


class PyBulletBoilGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the boil environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_boil"}

    @staticmethod
    def get_processes(
            env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        burner_type = types["burner"]
        faucet_type = types["faucet"]
        human_type = types["human"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        JugAtBurner = predicates["JugAtBurner"]
        JugAtFaucet = predicates["JugAtFaucet"]
        NoJugAtFaucet = predicates["NoJugAtFaucet"]
        NoJugAtBurner = predicates["NoJugAtBurner"]
        JugNotAtBurnerOrFaucet = predicates["JugNotAtBurnerOrFaucet"]
        if CFG.boil_add_jug_reached_capacity_predicate:
            NoJugAtFaucetOrAtFaucetAndReachedCapacity = predicates[
                "NoJugAtFaucetOrAtFaucetAndReachedCapacity"]
            JugAtCapacity = predicates["JugAtCapacity"]
        else:
            NoJugAtFaucetOrJugAtFaucetAndFilled = predicates[
                "NoJugAtFaucetOrAtFaucetAndFilled"]
        JugFilled = predicates["JugFilled"]
        # JugNotFilled = predicates["JugNotFilled"]
        # WaterSpilled = predicates["WaterSpilled"]
        NoWaterSpilled = predicates["NoWaterSpilled"]
        FaucetOn = predicates["FaucetOn"]
        FaucetOff = predicates["FaucetOff"]
        BurnerOn = predicates["BurnerOn"]
        BurnerOff = predicates["BurnerOff"]
        WaterBoiled = predicates["WaterBoiled"]
        if CFG.boil_goal == "human_happy":
            HumanHappy = predicates["HumanHappy"]
        elif CFG.boil_goal == "task_completed":
            TaskCompleted = predicates["TaskCompleted"]

        # Options
        PickJug = options["PickJug"]
        if CFG.boil_use_skill_factories:
            Place = options["Place"]
        else:
            # Legacy options expose object-keyed place options instead of a
            # generic Place; the samplers already return empty params for the
            # legacy path, so each place process just selects the right one.
            PlaceUnderFaucetOpt = options["PlaceUnderFaucet"]
            PlaceOnBurnerOpt = options["PlaceOnBurner"]
            PlaceOutsideOpt = options["PlaceOutsideBurnerAndFaucet"]
        # Having swtich for each because of the type
        SwitchFaucetOn = options["SwitchFaucetOn"]
        SwitchFaucetOff = options["SwitchFaucetOff"]
        SwitchBurnerOn = options["SwitchBurnerOn"]
        SwitchBurnerOff = options["SwitchBurnerOff"]
        Wait = options["Wait"]
        if CFG.boil_goal == "task_completed":
            DeclareComplete = options["DeclareComplete"]

        processes: Set[CausalProcess] = set()

        # --- Endogenous Processes / Durative Actions ---
        # PickJugFromFaucet
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, jug, faucet]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtFaucet, [jug, faucet]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtFaucet, [faucet]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtFaucet, [jug, faucet]),
        }
        delay_distribution: DelayDistribution = DiscreteGaussianDelay(
            mu=torch.tensor(4.0), sigma=torch.tensor(0.1))
        pick_jug_from_faucet_process = EndogenousProcess(
            "PickJugFromFaucet", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _pick_sampler)
        processes.add(pick_jug_from_faucet_process)

        # PickJugFromBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, jug, burner]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtBurner, [jug, burner]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtBurner, [burner]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtBurner, [jug, burner]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                                   sigma=torch.tensor(0.1))
        pick_jug_from_burner_process = EndogenousProcess(
            "PickJugFromBurner", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _pick_sampler)
        processes.add(pick_jug_from_burner_process)

        # PickJugFromOutsideFaucetAndBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugNotAtBurnerOrFaucet, [jug]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugNotAtBurnerOrFaucet, [jug]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        pick_jug_outside_faucet_burner_process = EndogenousProcess(
            "PickJugFromOutsideFaucetAndBurner", parameters,
            condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _pick_sampler)
        processes.add(pick_jug_outside_faucet_burner_process)

        # PlaceOnBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, jug, burner]
        if CFG.boil_use_skill_factories:
            option_vars = [robot]
            option = Place
        else:
            option_vars = [robot, burner]
            option = PlaceOnBurnerOpt
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtBurner, [burner]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtBurner, [jug, burner]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtBurner, [burner]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(5.0),
                                                   sigma=torch.tensor(0.1))
        place_on_burner_process = EndogenousProcess(
            "PlaceOnBurner", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _place_on_burner_sampler)
        processes.add(place_on_burner_process)

        # PlaceUnderFaucet
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, jug, faucet]
        if CFG.boil_use_skill_factories:
            option_vars = [robot]
            option = Place
        else:
            option_vars = [robot, faucet]
            option = PlaceUnderFaucetOpt
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtFaucet, [faucet]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtFaucet, [jug, faucet]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtFaucet, [faucet]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        place_under_faucet_process = EndogenousProcess(
            "PlaceUnderFaucet", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars,
            _place_under_faucet_sampler)
        processes.add(place_under_faucet_process)

        # PlaceAtOutsideFaucetAndBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        if CFG.boil_use_skill_factories:
            option_vars = [robot]
            option = Place
        else:
            option_vars = [robot]
            option = PlaceOutsideOpt
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugNotAtBurnerOrFaucet, [jug]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        place_at_outside_faucet_burner_process = EndogenousProcess(
            "PlaceOutsideFaucetAndBurner", parameters, condition_at_start,
            set(), set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _place_outside_sampler)
        processes.add(place_at_outside_faucet_burner_process)

        # SwitchFaucetOn
        robot = Variable("?robot", robot_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, faucet]
        option_vars = [robot, faucet]
        option = SwitchFaucetOn
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(FaucetOff, [faucet]),
        }
        add_effects = {
            LiftedAtom(FaucetOn, [faucet]),
        }
        delete_effects = {
            LiftedAtom(FaucetOff, [faucet]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        switch_faucet_on_process = EndogenousProcess(
            "SwitchFaucetOn", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _push_sampler)
        processes.add(switch_faucet_on_process)

        # SwitchFaucetOff
        robot = Variable("?robot", robot_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, faucet]
        option_vars = [robot, faucet]
        option = SwitchFaucetOff
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(FaucetOn, [faucet]),
        }
        add_effects = {
            LiftedAtom(FaucetOff, [faucet]),
        }
        delete_effects = {
            LiftedAtom(FaucetOn, [faucet]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        switch_faucet_off_process = EndogenousProcess(
            "SwitchFaucetOff", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _push_sampler)
        processes.add(switch_faucet_off_process)

        # SwitchBurnerOn
        robot = Variable("?robot", robot_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, burner]
        option_vars = [robot, burner]
        option = SwitchBurnerOn
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(BurnerOff, [burner]),
        }
        add_effects = {
            LiftedAtom(BurnerOn, [burner]),
        }
        delete_effects = {
            LiftedAtom(BurnerOff, [burner]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        switch_burner_on_process = EndogenousProcess(
            "SwitchBurnerOn", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _push_sampler)
        processes.add(switch_burner_on_process)

        # SwitchBurnerOff
        robot = Variable("?robot", robot_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, burner]
        option_vars = [robot, burner]
        option = SwitchBurnerOff
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(BurnerOn, [burner]),
        }
        add_effects = {
            LiftedAtom(BurnerOff, [burner]),
        }
        delete_effects = {
            LiftedAtom(BurnerOn, [burner]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        switch_burner_off_process = EndogenousProcess(
            "SwitchBurnerOff", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _push_sampler)
        processes.add(switch_burner_off_process)

        # Wait
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = Wait
        delay_distribution = ConstantDelay(1)
        wait_process = EndogenousProcess("Wait", parameters, set(), set(),
                                         set(), set(),
                                         set(), delay_distribution,
                                         torch.tensor(1.0), option,
                                         option_vars, null_sampler)
        processes.add(wait_process)

        if CFG.boil_goal == "task_completed":
            # DeclareComplete
            robot = Variable("?robot", robot_type)
            parameters = [robot, jug, burner]
            option_vars = [robot]
            option = DeclareComplete
            condition_at_start = {
                LiftedAtom(NoWaterSpilled, []),
                LiftedAtom(WaterBoiled, [jug]),
                LiftedAtom(JugFilled, [jug]),
                LiftedAtom(BurnerOff, [burner]),
            }
            add_effects = {LiftedAtom(TaskCompleted, [])}
            delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                       sigma=torch.tensor(0.1))
            declare_complete_process = EndogenousProcess(
                "DeclareComplete", parameters, condition_at_start,
                set(), set(), add_effects, set(), delay_distribution,
                torch.tensor(1.0), option, option_vars, null_sampler)
            processes.add(declare_complete_process)

        # --- Exogenous Processes ---
        # FillJug
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [jug, faucet]
        option_vars = [jug]
        condition_at_start = {
            LiftedAtom(JugAtFaucet, [jug, faucet]),
            LiftedAtom(FaucetOn, [faucet]),
            # LiftedAtom(JugNotFilled, [jug]),
        }
        condition_overall = {
            LiftedAtom(JugAtFaucet, [jug, faucet]),
            LiftedAtom(FaucetOn, [faucet]),
        }
        add_effects = {
            LiftedAtom(JugFilled, [jug]),
        }
        delete_effects = set()
        # delete_effects = {
        #     LiftedAtom(JugNotFilled, [jug]),
        # }
        # Legacy options take fewer low-level steps per option than the
        # skill-factory options, so the jug does not physically reach the
        # fill threshold within the SwitchFaucetOn(1)+SwitchBurnerOn(3)+
        # SwitchFaucetOff(1) window the skill-factory timing was calibrated
        # for. Use a longer symbolic fill delay for the legacy options so
        # the planner emits an explicit Wait (which terminates exactly on
        # JugFilled), filling robustly regardless of option duration. Keep
        # the original delay for the skill-factory options, whose longer
        # rollouts already fill within the window and would overfill / spill
        # if the faucet kept running through an added Wait.
        _fill_mu = 5.0 if CFG.boil_use_skill_factories else 8.0
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(_fill_mu),
                                                   sigma=torch.tensor(0.1))
        fill_jug_process = ExogenousProcess("FillJug", parameters,
                                            condition_at_start,
                                            condition_overall, set(),
                                            add_effects, delete_effects,
                                            delay_distribution,
                                            torch.tensor(1.0))
        processes.add(fill_jug_process)

        if CFG.boil_add_jug_reached_capacity_predicate:
            # ReachCapacity
            jug = Variable("?jug", jug_type)
            faucet = Variable("?faucet", faucet_type)
            parameters = [jug, faucet]
            condition_at_start = {
                LiftedAtom(JugFilled, [jug]),
                LiftedAtom(JugAtFaucet, [jug, faucet]),
                LiftedAtom(FaucetOn, [faucet]),
            }
            condition_overall = condition_at_start.copy()
            add_effects = {
                LiftedAtom(JugAtCapacity, [jug]),
            }
            delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(2.0),
                                                       sigma=torch.tensor(0.1))
            reach_capacity_process = ExogenousProcess("ReachJugCapacity",
                                                      parameters,
                                                      condition_at_start,
                                                      condition_overall, set(),
                                                      add_effects, set(),
                                                      delay_distribution,
                                                      torch.tensor(1.0))
            processes.add(reach_capacity_process)

        # OverfillJug
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [jug, faucet]
        condition_at_start = {
            LiftedAtom(FaucetOn, [faucet]),
        }
        if CFG.boil_use_derived_predicates:
            if CFG.boil_add_jug_reached_capacity_predicate:
                condition_at_start.add(
                    LiftedAtom(NoJugAtFaucetOrAtFaucetAndReachedCapacity,
                               [jug, faucet]))
            else:
                condition_at_start.add(
                    LiftedAtom(NoJugAtFaucetOrJugAtFaucetAndFilled,
                               [jug, faucet]))
        else:
            condition_at_start.add(LiftedAtom(JugAtFaucet, [jug, faucet]))
            condition_at_start.add(LiftedAtom(JugFilled, [jug]))
        condition_overall = condition_at_start.copy()
        add_effects = set()
        delete_effects = {
            LiftedAtom(NoWaterSpilled, []),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        overfill_jug_process = ExogenousProcess("OverfillJug", parameters,
                                                condition_at_start,
                                                condition_overall, set(),
                                                add_effects, delete_effects,
                                                delay_distribution,
                                                torch.tensor(1.0))
        processes.add(overfill_jug_process)

        # Spill
        if not CFG.boil_use_derived_predicates:
            faucet = Variable("?faucet", faucet_type)
            parameters = [faucet]
            condition_at_start = {
                LiftedAtom(NoJugAtFaucet, [faucet]),
                LiftedAtom(FaucetOn, [faucet]),
            }
            # add_effects = {
            #     LiftedAtom(WaterSpilled, []),
            # }
            add_effects = set()
            delete_effects = {
                LiftedAtom(NoWaterSpilled, []),
            }
            delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                       sigma=torch.tensor(0.1))
            spill_process = ExogenousProcess("Spill",
                                             parameters, condition_at_start,
                                             set(), set(), add_effects,
                                             delete_effects,
                                             delay_distribution,
                                             torch.tensor(1.0))
            processes.add(spill_process)

        # Boil
        burner = Variable("?burner", burner_type)
        jug = Variable("?jug", jug_type)
        parameters = [burner, jug]
        condition_at_start = {
            LiftedAtom(JugAtBurner, [jug, burner]),
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(BurnerOn, [burner]),
        }
        condition_overall = {
            LiftedAtom(JugAtBurner, [jug, burner]),
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(BurnerOn, [burner]),
        }
        add_effects = {
            LiftedAtom(WaterBoiled, [jug]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(10.0),
                                                   sigma=torch.tensor(0.1))
        boil_process = ExogenousProcess("Boil", parameters, condition_at_start,
                                        condition_overall, set(), add_effects,
                                        set(), delay_distribution,
                                        torch.tensor(1.0))
        processes.add(boil_process)

        if CFG.boil_goal == "human_happy":
            # HumanHappyProcess
            jug = Variable("?jug", jug_type)
            burner = Variable("?burner", burner_type)
            human = Variable("?human", human_type)
            parameters = [jug, burner, human]
            condition_at_start = {
                LiftedAtom(JugFilled, [jug]),
            }
            if not CFG.boil_goal_simple_human_happy:
                condition_at_start |= {
                    LiftedAtom(NoWaterSpilled, []),
                    LiftedAtom(WaterBoiled, [jug]),
                }
                if CFG.boil_goal_require_burner_off:
                    condition_at_start.add(LiftedAtom(BurnerOff, [burner]))
            add_effects = {LiftedAtom(HumanHappy, [human, jug, burner])}
            delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                       sigma=torch.tensor(0.1))
            human_happy_process = ExogenousProcess("HumanHappy", parameters,
                                                   condition_at_start, set(),
                                                   set(), add_effects, set(),
                                                   delay_distribution,
                                                   torch.tensor(1.0))
            processes.add(human_happy_process)

        return processes
