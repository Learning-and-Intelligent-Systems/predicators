"""Ground-truth processes for the coffee environments."""

from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, DelayDistribution, \
    EndogenousProcess, ExogenousProcess, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler

_COFFEE_DROP_Z = 0.5  # z_lb (0.4) + jug_handle_height (0.1)


def _pick_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    del state, goal, rng, objs
    return np.array([0.0], dtype=np.float32)


def _push_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Push params for TurnMachineOn (button press)."""
    if not CFG.coffee_use_skill_factories:
        return np.array([], dtype=np.float32)
    del state, goal, rng, objs
    return np.array([0.0, 0.01], dtype=np.float32)


def _place_jug_in_machine_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    if not CFG.coffee_use_skill_factories:
        return np.array([], dtype=np.float32)
    del state, goal, rng, objs
    # objs = [robot, jug, machine]
    return np.array(
        [
            PyBulletCoffeeEnv.dispense_area_x,
            PyBulletCoffeeEnv.dispense_area_y - .1, _COFFEE_DROP_Z,
            PyBulletCoffeeEnv.robot_init_wrist
        ],  # 0.98, 1.4, 0.5, -1.57
        dtype=np.float32)


def _pour_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, _objs: Sequence[Object]) -> Array:
    """Return empty pour params (all offsets are now fixed constants)."""
    del goal, rng, state
    return np.array([], dtype=np.float64)


class PyBulletCoffeeGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the coffee environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_coffee"}

    @classmethod
    def get_processes(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        cup_type = types["cup"]
        machine_type = types["coffee_machine"]
        plug_type = types["plug"]

        # Predicates
        CupFilled = predicates["CupFilled"]
        Holding = predicates["Holding"]
        JugInMachine = predicates["JugInMachine"]
        MachineOn = predicates["MachineOn"]
        OnTable = predicates["OnTable"]
        HandEmpty = predicates["HandEmpty"]
        JugFilled = predicates["JugFilled"]
        JugAboveCup = predicates["JugAboveCup"]
        NotAboveCup = predicates["NotAboveCup"]
        Twisting = predicates["Twisting"]
        if CFG.coffee_jug_pickable_pred:
            JugPickable = predicates["JugPickable"]
        if CFG.coffee_machine_has_plug:
            PluggedIn = predicates["PluggedIn"]

        # Options
        PickJug = options["PickJug"]
        PlaceJugInMachine = options["PlaceJugInMachine"]
        TurnMachineOn = options["TurnMachineOn"]
        Pour = options["Pour"]
        Wait = options["Wait"]

        processes: Set[CausalProcess] = set()

        # --- Endogenous Processes / Durative Actions ---

        if CFG.coffee_machine_has_plug:
            # PlugIn
            robot = Variable("?robot", robot_type)
            plug = Variable("?plug", plug_type)
            jug = Variable("?jug", jug_type)
            parameters = [robot, plug, jug]
            option_vars = [robot, plug]
            option = options["PlugIn"]
            condition_at_start = {
                LiftedAtom(HandEmpty, [robot]),
                LiftedAtom(OnTable, [jug]),
            }
            add_effects = {
                LiftedAtom(PluggedIn, [plug]),
            }
            delete_effects: Set[LiftedAtom] = set()
            delay_distribution: DelayDistribution = DiscreteGaussianDelay(
                mu=torch.tensor(2.0), sigma=torch.tensor(0.1))
            plug_in_process = EndogenousProcess("PlugIn",
                                                parameters, condition_at_start,
                                                set(), set(), add_effects,
                                                delete_effects,
                                                delay_distribution,
                                                torch.tensor(1.0), option,
                                                option_vars, null_sampler)
            processes.add(plug_in_process)

        if not CFG.coffee_use_pixelated_jug:
            if CFG.coffee_combined_move_and_twist_policy:
                # Twist (combined move and twist)
                robot = Variable("?robot", robot_type)
                jug = Variable("?jug", jug_type)
                parameters = [robot, jug]
                option_vars = [robot, jug]
                option = options["Twist"]
                condition_at_start = {
                    LiftedAtom(OnTable, [jug]),
                    LiftedAtom(HandEmpty, [robot]),
                }
                add_effects_twist: Set[LiftedAtom] = set()
                delete_effects_twist: Set[LiftedAtom] = set()
                if CFG.coffee_jug_pickable_pred:
                    add_effects_twist.add(LiftedAtom(JugPickable, [jug]))
                delay_distribution = DiscreteGaussianDelay(
                    mu=torch.tensor(4.0), sigma=torch.tensor(0.1))
                twist_process = EndogenousProcess("Twist", parameters,
                                                  condition_at_start, set(),
                                                  set(), add_effects_twist,
                                                  delete_effects_twist,
                                                  delay_distribution,
                                                  torch.tensor(1.0), option,
                                                  option_vars, null_sampler)
                processes.add(twist_process)
            else:
                # MoveToTwistJug
                robot = Variable("?robot", robot_type)
                jug = Variable("?jug", jug_type)
                parameters = [robot, jug]
                option_vars = [robot, jug]
                option = options["MoveToTwistJug"]
                condition_at_start = {
                    LiftedAtom(OnTable, [jug]),
                    LiftedAtom(HandEmpty, [robot]),
                }
                add_effects = {
                    LiftedAtom(Twisting, [robot, jug]),
                }
                delete_effects = {
                    LiftedAtom(HandEmpty, [robot]),
                }
                delay_distribution = DiscreteGaussianDelay(
                    mu=torch.tensor(2.0), sigma=torch.tensor(0.1))
                move_to_twist_jug_process = EndogenousProcess(
                    "MoveToTwistJug", parameters, condition_at_start, set(),
                    set(), add_effects, delete_effects, delay_distribution,
                    torch.tensor(1.0), option, option_vars, null_sampler)
                processes.add(move_to_twist_jug_process)

                # TwistJug
                robot = Variable("?robot", robot_type)
                jug = Variable("?jug", jug_type)
                parameters = [robot, jug]
                option_vars = [robot, jug]
                option = options["TwistJug"]
                condition_at_start = {
                    LiftedAtom(OnTable, [jug]),
                    LiftedAtom(Twisting, [robot, jug]),
                }
                add_effects = {
                    LiftedAtom(HandEmpty, [robot]),
                }
                if CFG.coffee_jug_pickable_pred:
                    add_effects.add(LiftedAtom(JugPickable, [jug]))
                delete_effects = {
                    LiftedAtom(Twisting, [robot, jug]),
                }
                delay_distribution = DiscreteGaussianDelay(
                    mu=torch.tensor(3.0), sigma=torch.tensor(0.1))
                twist_jug_process = EndogenousProcess(
                    "TwistJug", parameters, condition_at_start, set(), set(),
                    add_effects, delete_effects, delay_distribution,
                    torch.tensor(1.0), option, option_vars, null_sampler)
                processes.add(twist_jug_process)

        # PickJugFromTable
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(OnTable, [jug]),
            LiftedAtom(HandEmpty, [robot]),
        }
        if CFG.coffee_jug_pickable_pred:
            condition_at_start.add(LiftedAtom(JugPickable, [jug]))
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(OnTable, [jug]),
            LiftedAtom(HandEmpty, [robot])
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        pick_jug_from_table_process = EndogenousProcess(
            "PickJugFromTable", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _pick_sampler)
        processes.add(pick_jug_from_table_process)

        # PlaceJugInMachine
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        machine = Variable("?machine", machine_type)
        parameters = [robot, jug, machine]
        option_vars = [robot, jug, machine]
        option = PlaceJugInMachine
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NotAboveCup, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugInMachine, [jug, machine]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                                   sigma=torch.tensor(0.1))
        place_jug_in_machine_process = EndogenousProcess(
            "PlaceJugInMachine", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars,
            _place_jug_in_machine_sampler)
        processes.add(place_jug_in_machine_process)

        # TurnMachineOn
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        machine = Variable("?machine", machine_type)
        parameters = [robot, jug, machine]
        option_vars = [robot, machine]
        option = TurnMachineOn
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugInMachine, [jug, machine]),
        }
        if CFG.coffee_machine_has_plug:
            plug = Variable("?plug", plug_type)
            parameters.append(plug)
            condition_at_start.add(LiftedAtom(PluggedIn, [plug]))
        add_effects = {
            LiftedAtom(MachineOn, [machine]),
        }
        delete_effects = set()
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(2.0),
                                                   sigma=torch.tensor(0.1))
        turn_machine_on_process = EndogenousProcess(
            "TurnMachineOn", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _push_sampler)
        processes.add(turn_machine_on_process)

        # PickJugFromMachine
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        machine = Variable("?machine", machine_type)
        parameters = [robot, jug, machine]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugInMachine, [jug, machine]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugInMachine, [jug, machine]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        pick_jug_from_machine_process = EndogenousProcess(
            "PickJugFromMachine", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _pick_sampler)
        processes.add(pick_jug_from_machine_process)

        # Pour from not-above-cup
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        cup = Variable("?cup", cup_type)
        parameters = [robot, jug, cup]
        option_vars = [robot, jug, cup]
        option = Pour
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NotAboveCup, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(JugAboveCup, [jug, cup]),
        }
        delete_effects = {
            LiftedAtom(NotAboveCup, [robot, jug]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(2.0),
                                                   sigma=torch.tensor(0.1))
        pourFromNotAboveCup_process = EndogenousProcess(
            "PourFromNotAboveCup", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _pour_sampler)
        processes.add(pourFromNotAboveCup_process)

        # Pour from above-cup
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        from_cup = Variable("?from_cup", cup_type)
        to_cup = Variable("?to_cup", cup_type)
        parameters = [robot, jug, to_cup, from_cup]
        option_vars = [robot, jug, to_cup]
        option = Pour
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(JugAboveCup, [jug, from_cup]),
        }
        add_effects = {
            LiftedAtom(JugAboveCup, [jug, to_cup]),
        }
        delete_effects = {
            LiftedAtom(NotAboveCup, [robot, jug]),
            LiftedAtom(JugAboveCup, [jug, from_cup]),
        }
        ignore_effects = {NotAboveCup, JugAboveCup}
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(2.0),
                                                   sigma=torch.tensor(0.1))
        pourFromNotAboveCup_process = EndogenousProcess(
            "PourFromCup", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _pour_sampler,
            ignore_effects)
        processes.add(pourFromNotAboveCup_process)

        # Wait
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = Wait
        delay_distribution = ConstantDelay(1)
        ignore_effects = {NotAboveCup, JugAboveCup}
        wait_process = EndogenousProcess("Wait", parameters, set(), set(),
                                         set(), set(),
                                         set(), delay_distribution,
                                         torch.tensor(1.0), option,
                                         option_vars, null_sampler,
                                         ignore_effects)
        processes.add(wait_process)

        # --- Exogenous Processes ---
        # MakeCoffee (Exogenous)
        jug = Variable("?jug", jug_type)
        machine = Variable("?machine", machine_type)
        parameters = [jug, machine]
        condition_at_start = {
            LiftedAtom(JugInMachine, [jug, machine]),
            LiftedAtom(MachineOn, [machine]),
        }
        condition_overall = {
            LiftedAtom(JugInMachine, [jug, machine]),
            LiftedAtom(MachineOn, [machine]),
        }
        add_effects = {
            LiftedAtom(JugFilled, [jug]),
        }
        delete_effects_make_coffee: Set[LiftedAtom] = set()
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(5.0),
                                                   sigma=torch.tensor(0.1))
        make_coffee_process = ExogenousProcess(
            "MakeCoffee", parameters, condition_at_start, condition_overall,
            set(), add_effects, delete_effects_make_coffee, delay_distribution,
            torch.tensor(1.0))
        processes.add(make_coffee_process)

        # FillCup (Exogenous)
        jug = Variable("?jug", jug_type)
        cup = Variable("?cup", cup_type)
        parameters = [jug, cup]
        condition_at_start = {
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(JugAboveCup, [jug, cup]),
        }
        condition_overall = {
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(JugAboveCup, [jug, cup]),
        }
        add_effects = {
            LiftedAtom(CupFilled, [cup]),
        }
        delete_effects_fill_cup: Set[LiftedAtom] = set()
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(5.0),
                                                   sigma=torch.tensor(0.1))
        fill_cup_process = ExogenousProcess(
            "FillCup", parameters, condition_at_start, condition_overall,
            set(), add_effects, delete_effects_fill_cup, delay_distribution,
            torch.tensor(1.0))
        processes.add(fill_cup_process)

        return processes
