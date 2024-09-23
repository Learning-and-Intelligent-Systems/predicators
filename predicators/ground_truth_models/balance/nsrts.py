"""Ground-truth NSRTs for the blocks environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.balance import BalanceEnv
from predicators.envs.pybullet_balance import PyBulletBalanceEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class BalanceGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the blocks environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"blocks", "pybullet_balance", "blocks_clear"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        if env_name == "pybullet_balance":
            env_cls = PyBulletBalanceEnv
        else:
            env_cls = BalanceEnv
            
        # Types
        block_type = types["block"]
        robot_type = types["robot"]
        machine_type = types["machine"]
        plate_type = types["plate"]

        # Predicates
        On = predicates["DirectlyOn"]
        OnPlate = predicates["DirectlyOnPlate"]
        GripperOpen = predicates["GripperOpen"]
        Holding = predicates["Holding"]
        Clear = predicates["Clear"]
        Balanced = predicates["Balanced"]
        MachineOn = predicates["MachineOn"]
        ClearPlate = predicates["ClearPlate"]

        # Options
        Pick = options["Pick"]
        Stack = options["Stack"]
        PutOnPlate = options["PutOnPlate"]
        TurnMachineOn = options["TurnMachineOn"]

        nsrts = set()

        # PickFromTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        plate = Variable("?plate", plate_type)
        parameters = [block, robot, plate]
        option_vars = [robot, block]
        option = Pick
        preconditions = {
            LiftedAtom(OnPlate, [block, plate]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(Holding, [block])}
        delete_effects = {
            LiftedAtom(OnPlate, [block, plate]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }

        pickfromplate_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, null_sampler)
        nsrts.add(pickfromplate_nsrt)

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
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }
        delete_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
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
        # option_vars = [block, otherblock, robot]
        option_vars = [robot, otherblock]
        option = Stack
        preconditions = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }
        add_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }

        stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          null_sampler)
        nsrts.add(stack_nsrt)

        # PutOnPlate
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        plate = Variable("?plate", plate_type)
        parameters = [block, robot, plate]
        # option_vars = [block, robot]
        option_vars = [robot, plate]
        option = PutOnPlate
        preconditions = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(ClearPlate, [plate])
            }
        add_effects = {
            LiftedAtom(OnPlate, [block, plate]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(ClearPlate, [plate])
            }

        def putonplate_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal # unused
            block, robot, plate = objs
            # Note: normalized coordinates w.r.t. workspace.
            # x = rng.uniform()
            x = 0.09
            if plate.name == "plate1":
                # y = rng.uniform(0, 
                #     (env_cls.y_plate1_ub - env_cls.y_lb) /
                #     (env_cls.y_ub - env_cls.y_lb))
                y = 0.06
            elif plate.name == "plate3":
                # y = rng.uniform(
                #     (env_cls.y_plate3_lb - env_cls.y_lb) /
                #     (env_cls.y_ub - env_cls.y_lb)
                #     , 1)
                y = 0.84
            else:
                raise ValueError(f"Unknown plate: {plate.name}")
            return np.array([x, y], dtype=np.float32)

        putonplate_nsrt = NSRT("PutOnPlate", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, putonplate_sampler)
        nsrts.add(putonplate_nsrt)

        # TurnMachineOn
        machine = Variable("?machine", machine_type)
        robot = Variable("?robot", robot_type)
        plate1 = Variable("?plate1", plate_type)
        plate2 = Variable("?plate2", plate_type)
        parameters = [robot, machine, plate1, plate2]
        option_vars = [robot, plate1, plate2]
        option = TurnMachineOn
        preconditions = {LiftedAtom(Balanced, [plate1, plate2]),
                         LiftedAtom(GripperOpen, [robot])}
        add_effects = {LiftedAtom(MachineOn, [machine])}
        delete_effects = set()

        turn_machine_on_nsrt = NSRT("TurnMachineOn", parameters, preconditions,
                                  add_effects, delete_effects, set(), option,
                                  option_vars, null_sampler)
        nsrts.add(turn_machine_on_nsrt)
        return nsrts
