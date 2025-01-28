"""Ground-truth NSRTs for the coffee environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, DummyParameterizedOption, LiftedAtom, \
    ParameterizedOption, Predicate, Type, Variable
from predicators.utils import null_sampler
from predicators.settings import CFG


class PyBulletFanGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the fan environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_fan"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        wall_type = types["wall"]
        position_type = types["position"]
        switch_type = types["switch"]
        ball_type = types["ball"]


        # Predicates
        BallAtPos = predicates["BallAtPos"]
        ClearPos = predicates["ClearPos"]

        LeftOf = predicates["LeftOf"]
        RightOf = predicates["RightOf"]
        UpOf = predicates["UpOf"]
        DownOf = predicates["DownOf"]

        LeftFanSwitch = predicates["LeftFanSwitch"]
        RightFanSwitch = predicates["RightFanSwitch"]
        FrontFanSwitch = predicates["FrontFanSwitch"]
        BackFanSwitch = predicates["BackFanSwitch"]

        # Options
        TurnSwitchOnOff = options["SwitchOnOff"]

        nsrts = set()

        # MoveRight
        robot = Variable("?robot", robot_type)
        pos1 = Variable("?pos1", position_type)
        pos2 = Variable("?pos2", position_type)
        ball = Variable("?ball", ball_type)
        switch = Variable("?switch", switch_type)
        parameters = [robot, ball, pos1, pos2, switch]
        option_vars = [robot, switch]
        option = TurnSwitchOnOff
        preconditions = {
            LiftedAtom(BallAtPos, [ball, pos1]),
            LiftedAtom(ClearPos, [pos2]),
            LiftedAtom(LeftOf, [pos1, pos2]),
            LiftedAtom(LeftFanSwitch, [switch]) if \
                            not CFG.fan_fans_blow_opposite_direction else \
                                LiftedAtom(RightFanSwitch, [switch]),
        }
        add_effects = {
            LiftedAtom(BallAtPos, [ball, pos2]),
        }
        delete_effects = {}
        move_right_nsrt = NSRT("MoveRight", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, null_sampler)
        nsrts.add(move_right_nsrt)

        # MoveLeft
        robot = Variable("?robot", robot_type)
        pos1 = Variable("?pos1", position_type)
        pos2 = Variable("?pos2", position_type)
        ball = Variable("?ball", ball_type)
        switch = Variable("?switch", switch_type)
        parameters = [robot, ball, pos1, pos2, switch]
        option_vars = [robot, switch]
        option = TurnSwitchOnOff
        preconditions = {
            LiftedAtom(BallAtPos, [ball, pos1]),
            LiftedAtom(ClearPos, [pos2]),
            LiftedAtom(RightOf, [pos1, pos2]),
            LiftedAtom(RightFanSwitch, [switch]) if \
                            not CFG.fan_fans_blow_opposite_direction else \
                                LiftedAtom(LeftFanSwitch, [switch]),
        }
        add_effects = {
            LiftedAtom(BallAtPos, [ball, pos2]),
        }
        delete_effects = {}
        move_left_nsrt = NSRT("MoveLeft", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, null_sampler)
        nsrts.add(move_left_nsrt)

        # MoveDown
        robot = Variable("?robot", robot_type)
        pos1 = Variable("?pos1", position_type)
        pos2 = Variable("?pos2", position_type)
        ball = Variable("?ball", ball_type)
        switch = Variable("?switch", switch_type)
        parameters = [robot, ball, pos1, pos2, switch]
        option_vars = [robot, switch]
        option = TurnSwitchOnOff
        preconditions = {
            LiftedAtom(BallAtPos, [ball, pos1]),
            LiftedAtom(ClearPos, [pos2]),
            LiftedAtom(UpOf, [pos1, pos2]),
            LiftedAtom(FrontFanSwitch, [switch]) if \
                            not CFG.fan_fans_blow_opposite_direction else \
                                LiftedAtom(BackFanSwitch, [switch]),
        }
        add_effects = {
            LiftedAtom(BallAtPos, [ball, pos2]),
        }
        delete_effects = {}
        move_down_nsrt = NSRT("MoveDown", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, null_sampler)
        nsrts.add(move_down_nsrt)

        # MoveUp
        robot = Variable("?robot", robot_type)
        pos1 = Variable("?pos1", position_type)
        pos2 = Variable("?pos2", position_type)
        ball = Variable("?ball", ball_type)
        switch = Variable("?switch", switch_type)
        parameters = [robot, ball, pos1, pos2, switch]
        option_vars = [robot, switch]
        option = TurnSwitchOnOff
        preconditions = {
            LiftedAtom(BallAtPos, [ball, pos1]),
            LiftedAtom(ClearPos, [pos2]),
            LiftedAtom(DownOf, [pos1, pos2]),
            LiftedAtom(BackFanSwitch, [switch]) if \
                            not CFG.fan_fans_blow_opposite_direction else \
                                LiftedAtom(FrontFanSwitch, [switch]),
        }
        add_effects = {
            LiftedAtom(BallAtPos, [ball, pos2]),
        }
        delete_effects = {}
        move_up_nsrt = NSRT("MoveUp", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, null_sampler)
        nsrts.add(move_up_nsrt)


        return nsrts
