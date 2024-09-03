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
        table_type = types["table"]

        # Predicates
        On = predicates["On"]
        OnTable = predicates["OnTable"]
        GripperOpen = predicates["GripperOpen"]
        Holding = predicates["Holding"]
        Clear = predicates["Clear"]
        Balanced = predicates["Balanced"]
        MachineOn = predicates["MachineOn"]

        # Options
        Pick = options["Pick"]
        Stack = options["Stack"]
        PutOnTable = options["PutOnTable"]
        TurnMachineOn = options["TurnMachineOn"]

        nsrts = set()

        # PickFromTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)
        parameters = [block, robot, table]
        option_vars = [robot, block]
        option = Pick
        preconditions = {
            LiftedAtom(OnTable, [block, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(Holding, [block])}
        delete_effects = {
            LiftedAtom(OnTable, [block, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
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

        # PutOnTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)
        parameters = [block, robot, table]
        # option_vars = [block, robot]
        option_vars = [robot]
        option = PutOnTable
        preconditions = {LiftedAtom(Holding, [block])}
        add_effects = {
            LiftedAtom(OnTable, [block, table]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {LiftedAtom(Holding, [block])}

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal # unused
            block, robot, table = objs
            # Note: normalized coordinates w.r.t. workspace.
            x = rng.uniform()
            if table.name == "table1":
                y = rng.uniform(0, 
                    (env_cls.y_table1_ub - env_cls.y_lb) /
                    (env_cls.y_ub - env_cls.y_lb))
            elif table.name == "table3":
                y = rng.uniform(
                    (env_cls.y_table3_lb - env_cls.y_lb) /
                    (env_cls.y_ub - env_cls.y_lb)
                    , 1)
            else:
                raise ValueError(f"Unknown table: {table.name}")
            return np.array([x, y], dtype=np.float32)

        putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, putontable_sampler)
        nsrts.add(putontable_nsrt)

        # TurnMachineOn
        machine = Variable("?machine", machine_type)
        robot = Variable("?robot", robot_type)
        table1 = Variable("?table1", table_type)
        table2 = Variable("?table2", table_type)
        parameters = [robot, machine, table1, table2]
        option_vars = [robot, machine, table1, table2]
        option = TurnMachineOn
        preconditions = {LiftedAtom(Balanced, [table1, table2]),
                         LiftedAtom(GripperOpen, [robot])}
        add_effects = {LiftedAtom(MachineOn, [machine])}
        delete_effects = set()

        turn_machine_on_nsrt = NSRT("TurnMachineOn", parameters, preconditions,
                                  add_effects, delete_effects, set(), option,
                                  option_vars, null_sampler)
        nsrts.add(turn_machine_on_nsrt)
        return nsrts
