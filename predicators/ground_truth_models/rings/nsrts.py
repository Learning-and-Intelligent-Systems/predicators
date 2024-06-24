"""Ground-truth NSRTs for the blocks environment."""
import logging
from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class RingsGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the ring stack environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"ring_stack", "pybullet_ring_stack"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        ring_type = types["ring"]
        pole_type = types["pole"]
        robot_type = types["robot"]

        # Predicates
        #On = predicates["On"]
        OnTable = predicates["OnTable"]
        GripperOpen = predicates["GripperOpen"]
        Holding = predicates["Holding"]
        Around = predicates["Around"]

        # Options
        Pick = options["Pick"]
        #Stack = options["Stack"]
        PutOnTable = options["PutOnTable"]
        PutOnTableAroundPole = options["PutOnTableAroundPole"]
        logging.info(options["PutOnTableAroundPole"].params_space)
        nsrts = set()

        # PickFromTable
        ring = Variable("?ring", ring_type)
        robot = Variable("?robot", robot_type)
        parameters = [ring, robot]
        option_vars = [robot, ring]
        option = Pick
        preconditions = {
            LiftedAtom(OnTable, [ring]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(Holding, [ring])}
        delete_effects = {
            LiftedAtom(OnTable, [ring]),
            LiftedAtom(GripperOpen, [robot])
        }

        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, null_sampler)
        nsrts.add(pickfromtable_nsrt)

        # Unstack
        # ring = Variable("?block", ring_type)
        # otherring = Variable("?otherring", ring_type)
        # robot = Variable("?robot", robot_type)
        # parameters = [ring, otherring, robot]
        # option_vars = [robot, ring]
        # option = Pick
        # preconditions = {
        #     LiftedAtom(On, [ring, otherring]),
        #     LiftedAtom(GripperOpen, [robot])
        # }
        # add_effects = {
        #     LiftedAtom(Holding, [ring]),
        # }
        # delete_effects = {
        #     LiftedAtom(On, [ring, otherring]),
        #     LiftedAtom(GripperOpen, [robot])
        # }
        # unstack_nsrt = NSRT("Unstack", parameters, preconditions, add_effects,
        #                     delete_effects, set(), option, option_vars,
        #                     null_sampler)
        # nsrts.add(unstack_nsrt)
        #
        # # Stack
        # ring = Variable("?ring", ring_type)
        # otherring = Variable("?otherring", ring_type)
        # robot = Variable("?robot", robot_type)
        # parameters = [ring, otherring, robot]
        # option_vars = [robot, otherring]
        # option = Stack
        # preconditions = {
        #     LiftedAtom(Holding, [ring]),
        # }
        # add_effects = {
        #     LiftedAtom(On, [ring, otherring]),
        #     LiftedAtom(GripperOpen, [robot])
        # }
        # delete_effects = {
        #     LiftedAtom(Holding, [ring]),
        # }
        #
        # stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
        #                   delete_effects, set(), option, option_vars,
        #                   null_sampler)
        # nsrts.add(stack_nsrt)

        # PutOnTable
        ring = Variable("?ring", ring_type)
        robot = Variable("?robot", robot_type)
        parameters = [ring, robot]
        option_vars = [robot]
        option = PutOnTable
        preconditions = {LiftedAtom(Holding, [ring])}
        add_effects = {
            LiftedAtom(OnTable, [ring]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {LiftedAtom(Holding, [ring])}

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Note: normalized coordinates w.r.t. workspace.
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, putontable_sampler)
        nsrts.add(putontable_nsrt)

        # PutOnTableAroundPole
        ring = Variable("?ring", ring_type)
        robot = Variable("?robot", robot_type)
        pole = Variable("?pole", pole_type)
        parameters = [ring, robot, pole]
        option_vars = [robot, pole]
        option = PutOnTableAroundPole
        preconditions = {LiftedAtom(Holding, [ring])}
        add_effects = {
            LiftedAtom(OnTable, [ring]),
            LiftedAtom(GripperOpen, [robot]),
            LiftedAtom(Around, [ring, pole])
        }
        delete_effects = {LiftedAtom(Holding, [ring])}

        putonpole_nsrt = NSRT("PutOnTableAroundPole", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          null_sampler)
        nsrts.add(putonpole_nsrt)

        for nsrt in nsrts:
            logging.info(f'NSRT CREATION: {nsrt.name}:{nsrt.option.params_space}')

        return nsrts
