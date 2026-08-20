"""Ground-truth processes for the grow environments."""

from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, EndogenousProcess, \
    ExogenousProcess, GroundAtom, LiftedAtom, Object, ParameterizedOption, \
    Predicate, State, Type, Variable
from predicators.utils import DiscreteGaussianDelay, null_sampler

_GROW_DROP_Z = 0.55  # approximate table_height + jug_handle_height


def _pick_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return fixed grasp_z_offset for grow pick."""
    del state, goal, rng, objs
    return np.array([0.0], dtype=np.float32)


def _place_sampler(state: State, goal: Set[GroundAtom],
                   rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return placement params for grow place (jug init_x, init_y)."""
    if not CFG.grow_use_skill_factories:
        return np.array([], dtype=np.float32)
    del state, goal, rng
    _robot, jug = objs[:2]
    return np.array([jug.init_x, jug.init_y, _GROW_DROP_Z, -np.pi / 2],
                    dtype=np.float32)


def _pour_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return empty pour params (all offsets are now fixed constants)."""
    del goal, rng, state, objs
    return np.array([], dtype=np.float64)


class PyBulletGrowGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the grow environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_grow"}

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

        # Predicates
        Grown = predicates["Grown"]
        Holding = predicates["Holding"]
        HandEmpty = predicates["HandEmpty"]
        JugOnTable = predicates["JugOnTable"]
        SameColor = predicates["SameColor"]
        JugAboveCup = predicates["JugAboveCup"]
        NotAboveCup = predicates["NotAboveCup"]

        # Options
        PickJug = options["PickJug"]
        Pour = options["Pour"]
        Place = options["Place"]
        Wait = options["Wait"]

        processes: Set[CausalProcess] = set()

        # --- Endogenous Processes / Durative Actions ---

        # PickJugFromTable
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(JugOnTable, [jug]),
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(JugOnTable, [jug]),
            LiftedAtom(HandEmpty, [robot])
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        pick_jug_from_table_process = EndogenousProcess(
            "PickJugFromTable", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _pick_sampler)
        processes.add(pick_jug_from_table_process)

        # PlaceJugOnTableFromAboveCup
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        cup = Variable("?cup", cup_type)
        parameters = [robot, jug, cup]
        option_vars = [robot, jug]
        option = Place
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(JugAboveCup, [jug, cup]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugOnTable, [jug]),
            LiftedAtom(NotAboveCup, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(JugAboveCup, [jug, cup]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        ignore_effects = {
            Holding, HandEmpty, JugOnTable, JugAboveCup, NotAboveCup
        }
        place_jug_on_table_process = EndogenousProcess(
            "PlaceJugOnTable", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _place_sampler,
            ignore_effects)
        processes.add(place_jug_on_table_process)

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
        ignore_effects = {NotAboveCup, JugAboveCup}
        pour_from_not_above_cup_process = EndogenousProcess(
            "PourFromNotAboveCup", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _pour_sampler,
            ignore_effects)
        processes.add(pour_from_not_above_cup_process)

        # Pour from above-cup
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        from_cup = Variable("?from_cup", cup_type)
        to_cup = Variable("?to_cup", cup_type)
        parameters = [robot, jug, from_cup, to_cup]
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
            LiftedAtom(JugAboveCup, [jug, from_cup]),
        }
        ignore_effects = {NotAboveCup, JugAboveCup}
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(2.0),
                                                   sigma=torch.tensor(0.1))
        pour_from_not_above_cup_process = EndogenousProcess(
            "PourFromAboveCup", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _pour_sampler,
            ignore_effects)
        processes.add(pour_from_not_above_cup_process)

        # Wait
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = Wait
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        ignore_effects = {JugAboveCup, NotAboveCup}
        wait_process = EndogenousProcess("Wait", parameters, set(), set(),
                                         set(), set(),
                                         set(), delay_distribution,
                                         torch.tensor(1.0), option,
                                         option_vars, null_sampler)
        processes.add(wait_process)

        # --- Exogenous Processes ---

        # GrowPlant (Exogenous) - similar to CupFilled in coffee
        jug = Variable("?jug", jug_type)
        cup = Variable("?cup", cup_type)
        parameters = [jug, cup]
        condition_at_start = {
            LiftedAtom(JugAboveCup, [jug, cup]),
            LiftedAtom(SameColor, [cup, jug]),
        }
        condition_overall = {
            LiftedAtom(JugAboveCup, [jug, cup]),
            LiftedAtom(SameColor, [cup, jug]),
        }
        add_effects = {
            LiftedAtom(Grown, [cup]),
        }
        delete_effects_grow_plant: Set[LiftedAtom] = set()
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(5.0),
                                                   sigma=torch.tensor(0.1))
        grow_plant_process = ExogenousProcess(
            "GrowPlant", parameters, condition_at_start, condition_overall,
            set(), add_effects, delete_effects_grow_plant, delay_distribution,
            torch.tensor(1.0))
        processes.add(grow_plant_process)

        return processes
