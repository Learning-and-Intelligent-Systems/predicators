"""Ground-truth NSRTs for the exit garage environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class ExitGarageGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the exit garage environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"exit_garage"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        car_type = types["car"]
        robot_type = types["robot"]
        obstacle_type = types["obstacle"]

        # Predicates
        CarHasExited = predicates["CarHasExited"]
        CarryingObstacle = predicates["CarryingObstacle"]
        NotCarryingObstacle = predicates["NotCarryingObstacle"]
        ObstacleCleared = predicates["ObstacleCleared"]
        ObstacleNotCleared = predicates["ObstacleNotCleared"]

        # Options
        DriveCarToExit = options["DriveCarToExit"]
        PickupObstacle = options["PickupObstacle"]
        StoreObstacle = options["StoreObstacle"]

        nsrts = set()

        def random_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Note: just return a random value from 0 to 1
            return np.array([rng.uniform()], dtype=np.float32)

        # DriveCarToExit
        car = Variable("?car", car_type)
        parameters = [car]
        option_vars = [car]
        option = DriveCarToExit
        preconditions: Set[LiftedAtom] = set()
        add_effects: Set[LiftedAtom] = {
            LiftedAtom(CarHasExited, [car]),
        }
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects: Set[Predicate] = set()
        drive_car_to_exit_nsrt = NSRT("DriveCarToExit", parameters,
                                      preconditions, add_effects,
                                      delete_effects, ignore_effects, option,
                                      option_vars, random_sampler)
        nsrts.add(drive_car_to_exit_nsrt)

        # PickupObstacle
        robot = Variable("?robot", robot_type)
        obstacle = Variable("?obstacle", obstacle_type)
        parameters = [robot, obstacle]
        option_vars = [robot, obstacle]
        option = PickupObstacle
        preconditions = {
            LiftedAtom(NotCarryingObstacle, [robot]),
            LiftedAtom(ObstacleNotCleared, [obstacle]),
        }
        add_effects = {
            LiftedAtom(CarryingObstacle, [robot, obstacle]),
        }
        delete_effects = {
            LiftedAtom(NotCarryingObstacle, [robot]),
            LiftedAtom(ObstacleNotCleared, [obstacle]),
        }
        ignore_effects = set()
        pickup_obstacle_nsrt = NSRT("PickupObstacle", parameters,
                                    preconditions, add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    random_sampler)
        nsrts.add(pickup_obstacle_nsrt)

        # StoreObstacle
        robot = Variable("?robot", robot_type)
        obstacle = Variable("?obstacle", obstacle_type)
        parameters = [robot, obstacle]
        option_vars = [robot, obstacle]
        option = StoreObstacle
        preconditions = {
            LiftedAtom(CarryingObstacle, [robot, obstacle]),
        }
        add_effects = {
            LiftedAtom(NotCarryingObstacle, [robot]),
            LiftedAtom(ObstacleCleared, [obstacle]),
        }
        delete_effects = {
            LiftedAtom(CarryingObstacle, [robot, obstacle]),
        }
        ignore_effects = set()
        store_obstacle_nsrt = NSRT("StoreObstacle", parameters, preconditions,
                                   add_effects, delete_effects, ignore_effects,
                                   option, option_vars, random_sampler)
        nsrts.add(store_obstacle_nsrt)

        return nsrts
