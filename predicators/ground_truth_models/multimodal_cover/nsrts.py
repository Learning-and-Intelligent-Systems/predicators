"""Ground-truth NSRTs for the blocks environment."""
import logging
from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class MultiModalCoverGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the ring stack environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"multimodal_cover", "pybullet_multimodal_cover"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        block_type = types["block"]
        robot_type = types["robot"]
        zone_goal_type = types["dummy_zone_goal"]
        logging.info(env_name)

        # Predicates
        OnTable = predicates["OnTable"]
        GripperOpen = predicates["GripperOpen"]
        Holding = predicates["Holding"]
        OnZone = predicates["OnZone"]

        # Options
        Pick = options["Pick"]
        PutOnTable = options["PutOnTable"]
        nsrts = set()

        # PickFromTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, robot]
        option_vars = [robot, block]
        option = Pick
        preconditions = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(Holding, [block])}
        delete_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, null_sampler)
        nsrts.add(pickfromtable_nsrt)

        # PutOnTableInZone
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        zone_goal = Variable("?zone_goal", zone_goal_type)
        parameters = [block, robot, zone_goal]
        option_vars = [robot, block]
        option = PutOnTable
        preconditions = {LiftedAtom(Holding, [block])}
        add_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(OnZone, [block, zone_goal]),
            LiftedAtom(GripperOpen, [robot])
        }

        delete_effects = {LiftedAtom(Holding, [block])}

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Note: normalized coordinates w.r.t. workspace.
            x = rng.uniform()
            y = rng.uniform()
            return np.array([x, y], dtype=np.float32)

        putontablezone_nsrt = NSRT("PutOnTableInZone", parameters, preconditions,
                                   add_effects, delete_effects, set(), option,
                                   option_vars, putontable_sampler)
        nsrts.add(putontablezone_nsrt)

        # PutOnTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        zone_goal = Variable("?zone_goal", zone_goal_type)
        parameters = [block, robot]
        option_vars = [robot, block]
        option = PutOnTable
        preconditions = {LiftedAtom(Holding, [block])}
        add_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(GripperOpen, [robot])}

        delete_effects = {
            LiftedAtom(Holding, [block])
        }

        putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                                 add_effects, delete_effects, set(), option,
                                 option_vars, putontable_sampler)
        nsrts.add(putontable_nsrt)

        for nsrt in nsrts:
            logging.info(f'NSRT CREATION: {nsrt.name}:{nsrt.option.params_space}')

        return nsrts
