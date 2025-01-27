"""Ground-truth NSRTs for the ants environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class PyBulletAntsGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the ants environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_ants"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        block_type = types["food"]
        robot_type = types["robot"]

        # Predicates
        On = predicates["On"]
        OnTable = predicates["OnTable"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        Clear = predicates["Clear"]
        InSortedRegion = predicates["InSortedRegion"]

        # Options
        Pick = options["Pick"]
        Stack = options["Stack"]
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
            LiftedAtom(Clear, [block]),
            LiftedAtom(HandEmpty, [robot])
        }
        add_effects = {LiftedAtom(Holding, [robot, block])}
        delete_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(HandEmpty, [robot])
        }

        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, null_sampler)
        nsrts.add(pickfromtable_nsrt)

        # Unstack
        block = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, otherblock, robot]
        option_vars = [robot, block]
        option = Pick
        preconditions = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(HandEmpty, [robot])
        }
        add_effects = {
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(Clear, [otherblock])
        }
        delete_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(HandEmpty, [robot])
        }
        unstack_nsrt = NSRT("Unstack", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            null_sampler)
        nsrts.add(unstack_nsrt)

        # Stack
        block = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, otherblock, robot]
        option_vars = [robot, otherblock]
        option = Stack
        preconditions = {
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(Clear, [otherblock])
        }
        add_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(HandEmpty, [robot])
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, block]),
            LiftedAtom(Clear, [otherblock])
        }

        stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          null_sampler)
        nsrts.add(stack_nsrt)

        # PutOnTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, robot]
        option_vars = [robot]
        option = PutOnTable
        preconditions = {LiftedAtom(Holding, [robot, block])}
        add_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(InSortedRegion, [block])
        }
        delete_effects = {LiftedAtom(Holding, [robot, block])}

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del goal # unused
            block = objs[0]
            attractive = state.get(block, "attractive")
            if attractive:
                x_range = [2/3, 1]
            else:
                x_range = [0, 1/3]
            y_range = [1/4, 3/4]
            # Note: normalized coordinates w.r.t. workspace.
            x = rng.uniform(*x_range)
            y = rng.uniform(*y_range)
            return np.array([x, y], dtype=np.float32)

        putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, putontable_sampler)
        nsrts.add(putontable_nsrt)

        return nsrts
