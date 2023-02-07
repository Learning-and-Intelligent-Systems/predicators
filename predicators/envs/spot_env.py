from typing import Set

import numpy as np
from gym.spaces import Box

from predicators.envs import BaseEnv
from predicators.envs.pddl_env import _strips_operator_to_parameterized_option
from predicators.structs import LiftedAtom, Predicate, STRIPSOperator, Type, \
    Variable, ParameterizedOption


class SpotEnv(BaseEnv):
    """An environment containing tasks for a real Spot robot to execute.

    (TODO: description of current simple setup and
    mechanics of env).
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._robot_type = Type("robot", [])
        self._can_type = Type("soda_can", [])
        self._surface_type = Type("flat_surface", [])

        # Predicates
        # Note that all classifiers assigned here are dummies.
        self._On = Predicate("On", [self._can_type, self._surface_type],
                             lambda s, o: False)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    lambda s, o: False)
        self._HoldingCan = Predicate("HoldingCan",
                                     [self._robot_type, self._can_type],
                                     lambda s, o: False)
        self._ReachableCan = Predicate("ReachableCan",
                                       [self._robot_type, self._can_type],
                                       lambda s, o: False)
        self._ReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            lambda s, o: False)

        # STRIPS Operators (needed for option creation)
        # MoveToCan
        spot = Variable("?robot", self._robot_type)
        can = Variable("?can", self._can_type)
        add_effs = {LiftedAtom(self._ReachableCan, [spot, can])}
        ignore_effs = {self._ReachableCan}
        self._MoveToCanOp = STRIPSOperator("MoveToCan", [spot, can], set(),
                                           add_effs, set(), ignore_effs)
        # MoveToSurface
        spot = Variable("?robot", self._robot_type)
        surface = Variable("?surface", self._surface_type)
        add_effs = {LiftedAtom(self._ReachableSurface, [spot, surface])}
        ignore_effs = {self._ReachableSurface}
        self._MoveToSurfaceOp = STRIPSOperator("MoveToSurface",
                                               [spot, surface], set(),
                                               add_effs, set(), ignore_effs)
        # GraspCan
        spot = Variable("?robot", self._robot_type)
        can = Variable("?can", self._can_type)
        preconds = {
            LiftedAtom(self._ReachableCan, [spot, can]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        add_effs = {LiftedAtom(self._HoldingCan, [spot, can])}
        del_effs = {LiftedAtom(self._HandEmpty, [spot])}
        self._GraspCanOp = STRIPSOperator("GraspCan", [spot, can], preconds,
                                          add_effs, del_effs, set())
        # Place Can
        spot = Variable("?robot", self._robot_type)
        can = Variable("?can", self._can_type)
        surface = Variable("?surface", self._surface_type)
        preconds = {
            LiftedAtom(self._ReachableSurface, [spot, surface]),
            LiftedAtom(self._HoldingCan, [spot, can])
        }
        add_effs = {
            LiftedAtom(self._On, [can, surface]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        del_effs = {LiftedAtom(self._HoldingCan, [spot, can])}
        self._PlaceCanOp = STRIPSOperator("PlaceCanOnTop",
                                          [spot, can, surface], preconds,
                                          add_effs, del_effs, set())
        self._strips_operators = {
            self._MoveToCanOp, self._MoveToSurfaceOp, self._GraspCanOp,
            self._PlaceCanOp
        }
        self._ordered_strips_operators = sorted(self._strips_operators)

        # Options (aka Controllers)
        # Note that these are 1:1 with the operators; in the future, we will actually
        # implement these with robot-specific API calls.
        self._options = {
            _strips_operator_to_parameterized_option(
                op, self._ordered_strips_operators, self.action_space.shape[0])
            for op in self._strips_operators
        }

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._can_type, self._surface_type}

    @property
    def predicates(self) -> Set[Type]:
        return {
            self._On, self._HandEmpty, self._HoldingCan, self._ReachableCan,
            self._ReachableSurface
        }

    @property
    def options(self) -> Set[ParameterizedOption]:
        return self._options

    @property
    def action_space(self) -> Box:
        # See class docstring for explanation.
        num_ops = len(self._strips_operators)
        max_arity = max(len(op.parameters) for op in self._strips_operators)
        lb = np.array([0.0 for _ in range(max_arity + 1)], dtype=np.float32)
        ub = np.array([num_ops - 1.0] + [np.inf for _ in range(max_arity)],
                      dtype=np.float32)
        return Box(lb, ub, dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "realworld_spot"
