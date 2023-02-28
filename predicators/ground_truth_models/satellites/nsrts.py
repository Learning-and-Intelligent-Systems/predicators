"""Ground-truth NSRTs for the satellites environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.satellites import SatellitesEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class SatellitesGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the satellites environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"satellites", "satellites_simple"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        sat_type = types["satellite"]
        obj_type = types["object"]

        # Predicates
        Sees = predicates["Sees"]
        CalibrationTarget = predicates["CalibrationTarget"]
        IsCalibrated = predicates["IsCalibrated"]
        HasCamera = predicates["HasCamera"]
        HasInfrared = predicates["HasInfrared"]
        HasGeiger = predicates["HasGeiger"]
        ShootsChemX = predicates["ShootsChemX"]
        ShootsChemY = predicates["ShootsChemY"]
        HasChemX = predicates["HasChemX"]
        HasChemY = predicates["HasChemY"]
        CameraReadingTaken = predicates["CameraReadingTaken"]
        InfraredReadingTaken = predicates["InfraredReadingTaken"]
        GeigerReadingTaken = predicates["GeigerReadingTaken"]

        # Options
        MoveTo = options["MoveTo"]
        Calibrate = options["Calibrate"]
        ShootChemX = options["ShootChemX"]
        ShootChemY = options["ShootChemY"]
        UseInstrument = options["UseInstrument"]

        nsrts = set()

        # MoveTo
        sat = Variable("?sat", sat_type)
        obj = Variable("?obj", obj_type)
        parameters = [sat, obj]
        option_vars = [sat, obj]
        option = MoveTo
        preconditions: Set[LiftedAtom] = set()
        add_effects = {
            LiftedAtom(Sees, [sat, obj]),
        }
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {Sees}

        def moveto_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
            del goal  # unused
            _, obj = objs
            obj_x = state.get(obj, "x")
            obj_y = state.get(obj, "y")
            min_dist = SatellitesEnv.radius * 4
            max_dist = SatellitesEnv.fov_dist - SatellitesEnv.radius * 2
            dist = rng.uniform(min_dist, max_dist)
            angle = rng.uniform(-np.pi, np.pi)
            x = obj_x + dist * np.cos(angle)
            y = obj_y + dist * np.sin(angle)
            return np.array([x, y], dtype=np.float32)

        moveto_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           moveto_sampler)
        nsrts.add(moveto_nsrt)

        # Calibrate
        sat = Variable("?sat", sat_type)
        obj = Variable("?obj", obj_type)
        parameters = [sat, obj]
        option_vars = [sat, obj]
        option = Calibrate
        preconditions = {
            LiftedAtom(Sees, [sat, obj]),
            LiftedAtom(CalibrationTarget, [sat, obj]),
        }
        add_effects = {
            LiftedAtom(IsCalibrated, [sat]),
        }
        delete_effects = set()
        ignore_effects = set()
        calibrate_nsrt = NSRT("Calibrate", parameters, preconditions,
                              add_effects, delete_effects, ignore_effects,
                              option, option_vars, null_sampler)
        nsrts.add(calibrate_nsrt)

        # ShootChemX
        sat = Variable("?sat", sat_type)
        obj = Variable("?obj", obj_type)
        parameters = [sat, obj]
        option_vars = [sat, obj]
        option = ShootChemX
        preconditions = {
            LiftedAtom(Sees, [sat, obj]),
            LiftedAtom(ShootsChemX, [sat]),
        }
        add_effects = {
            LiftedAtom(HasChemX, [obj]),
        }
        delete_effects = set()
        ignore_effects = set()
        shoot_chem_x_nsrt = NSRT("ShootChemX", parameters, preconditions,
                                 add_effects, delete_effects, ignore_effects,
                                 option, option_vars, null_sampler)
        nsrts.add(shoot_chem_x_nsrt)

        # ShootChemY
        sat = Variable("?sat", sat_type)
        obj = Variable("?obj", obj_type)
        parameters = [sat, obj]
        option_vars = [sat, obj]
        option = ShootChemY
        preconditions = {
            LiftedAtom(Sees, [sat, obj]),
            LiftedAtom(ShootsChemY, [sat]),
        }
        add_effects = {
            LiftedAtom(HasChemY, [obj]),
        }
        delete_effects = set()
        ignore_effects = set()
        shoot_chem_y_nsrt = NSRT("ShootChemY", parameters, preconditions,
                                 add_effects, delete_effects, ignore_effects,
                                 option, option_vars, null_sampler)
        nsrts.add(shoot_chem_y_nsrt)

        # TakeCameraReading
        sat = Variable("?sat", sat_type)
        obj = Variable("?obj", obj_type)
        parameters = [sat, obj]
        option_vars = [sat, obj]
        option = UseInstrument
        preconditions = {
            LiftedAtom(Sees, [sat, obj]),
            LiftedAtom(IsCalibrated, [sat]),
            LiftedAtom(HasCamera, [sat]),
            # taking a camera reading requires Chemical X
            LiftedAtom(HasChemX, [obj]),
        }
        add_effects = {
            LiftedAtom(CameraReadingTaken, [sat, obj]),
        }
        delete_effects = set()
        ignore_effects = set()
        take_camera_reading_nsrt = NSRT("TakeCameraReading", parameters,
                                        preconditions, add_effects,
                                        delete_effects, ignore_effects, option,
                                        option_vars, null_sampler)
        nsrts.add(take_camera_reading_nsrt)

        # TakeInfraredReading
        sat = Variable("?sat", sat_type)
        obj = Variable("?obj", obj_type)
        parameters = [sat, obj]
        option_vars = [sat, obj]
        option = UseInstrument
        preconditions = {
            LiftedAtom(Sees, [sat, obj]),
            LiftedAtom(IsCalibrated, [sat]),
            LiftedAtom(HasInfrared, [sat]),
            # taking an infrared reading requires Chemical Y
            LiftedAtom(HasChemY, [obj]),
        }
        add_effects = {
            LiftedAtom(InfraredReadingTaken, [sat, obj]),
        }
        delete_effects = set()
        ignore_effects = set()
        take_infrared_reading_nsrt = NSRT("TakeInfraredReading", parameters,
                                          preconditions, add_effects,
                                          delete_effects, ignore_effects,
                                          option, option_vars, null_sampler)
        nsrts.add(take_infrared_reading_nsrt)

        # TakeGeigerReading
        sat = Variable("?sat", sat_type)
        obj = Variable("?obj", obj_type)
        parameters = [sat, obj]
        option_vars = [sat, obj]
        option = UseInstrument
        preconditions = {
            LiftedAtom(Sees, [sat, obj]),
            LiftedAtom(IsCalibrated, [sat]),
            LiftedAtom(HasGeiger, [sat]),
            # taking a Geiger reading doesn't require any chemical
        }
        add_effects = {
            LiftedAtom(GeigerReadingTaken, [sat, obj]),
        }
        delete_effects = set()
        ignore_effects = set()
        take_geiger_reading_nsrt = NSRT("TakeGeigerReading", parameters,
                                        preconditions, add_effects,
                                        delete_effects, ignore_effects, option,
                                        option_vars, null_sampler)
        nsrts.add(take_geiger_reading_nsrt)

        return nsrts
