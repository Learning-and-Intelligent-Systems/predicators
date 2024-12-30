"""Ground-truth NSRTs for the coffee environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable, DummyParameterizedOption
from predicators.utils import null_sampler


class PyBulletCircuitGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the circuit environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_circuit"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        connector_type = types["connector"]
        light_type = types["light"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        Connected = predicates["Connected"]
        LightOn = predicates["LightOn"]
        CircuitClosed = predicates["CircuitClosed"]

        # Options
        PickConnector = options["PickConnector"]
        ConnectConnector = options["ConnectConnector"]

        nsrts = set()
        # PickConnector
        robot = Variable("?robot", robot_type)
        connector = Variable("?connector", connector_type)
        parameters = [robot, connector]
        option_vars = [robot, connector]
        option = PickConnector
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, connector]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        pick_connector_nsrt = NSRT("PickConnector", parameters,
                                      preconditions, add_effects,
                                      delete_effects, set(), option,
                                      option_vars, null_sampler)
        nsrts.add(pick_connector_nsrt)

        # ConnectConnector. Connect connector 1 with connector 2 and 3.
        connector1 = Variable("?connector1", connector_type)
        connector2 = Variable("?connector2", connector_type)
        connector3 = Variable("?connector3", connector_type)
        option = ConnectConnector
        preconditions = {
            LiftedAtom(Holding, [robot, connector1]),
            # Should add one that says the distance between the terminals are
            # close enough
        }
        add_effects = {
            LiftedAtom(Connected, [connector1, connector2, connector3]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, connector]),
        }
        connect_wire_with_wires_nsrt = NSRT("ConnectWireWithWires", parameters,
                                            preconditions, add_effects,
                                            delete_effects, set(), option,
                                            option_vars, null_sampler)
        nsrts.add(connect_wire_with_wires_nsrt)

        # Done.
        parameters = [Variable("?light", light_type)]
        light = Variable("?light", light_type)
        preconditions = {
            LiftedAtom(CircuitClosed, [])
        }
        add_effects = {
            LiftedAtom(LightOn, [light])
        }
        done_nsrt = NSRT("Done", [light], preconditions, add_effects, set(),
                         set(), DummyParameterizedOption, [], null_sampler)
        nsrts.add(done_nsrt)

        return nsrts
