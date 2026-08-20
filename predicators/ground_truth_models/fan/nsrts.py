"""Ground-truth NSRTs for the fan environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


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
        position_type = types["loc"]
        switch_type = types["switch"]
        ball_type = types["ball"]

        # Predicates
        BallAtLoc = predicates["BallAtLoc"]
        ClearPos = predicates["ClearLoc"]

        # LeftOf = predicates["LeftOf"]
        # RightOf = predicates["RightOf"]
        # UpOf = predicates["UpOf"]
        # DownOf = predicates["DownOf"]

        # LeftFanSwitch = predicates["LeftFanSwitch"]
        # RightFanSwitch = predicates["RightFanSwitch"]
        # FrontFanSwitch = predicates["FrontFanSwitch"]
        # BackFanSwitch = predicates["BackFanSwitch"]

        # Predicates for switch/fan control
        if CFG.fan_known_controls_relation:
            FanOn = predicates["FanOn"]
            FanOff = predicates["FanOff"]
            fan_type = types["fan"]
        else:
            SwitchOn = predicates["SwitchOn"]
            SwitchOff = predicates["SwitchOff"]

        # Options
        switch_option = None
        switch_on_option = None
        switch_off_option = None
        Wait = options["Wait"]

        if CFG.fan_combine_switch_on_off:
            switch_option = options["SwitchOnOff"]
        else:
            switch_on_option = options["SwitchOn"]
            switch_off_option = options["SwitchOff"]

        nsrts = set()

        # Helper function to create fan/switch toggle NSRTs
        def _make_fan_toggle_nsrt(
                name: str, start_predicate: Predicate,
                add_predicate: Predicate, delete_predicate: Predicate,
                selected_option: ParameterizedOption) -> NSRT:
            robot = Variable("?robot", robot_type)
            if CFG.fan_known_controls_relation:
                controlled_obj = Variable("?fan", fan_type)
            else:
                controlled_obj = Variable("?switch", switch_type)
            parameters = [robot, controlled_obj]
            option_vars = [robot, controlled_obj]
            preconditions: Set[LiftedAtom] = {
                LiftedAtom(start_predicate, [controlled_obj]),
            }
            add_effects: Set[LiftedAtom] = {
                LiftedAtom(add_predicate, [controlled_obj]),
            }
            delete_effects: Set[LiftedAtom] = {
                LiftedAtom(delete_predicate, [controlled_obj]),
            }
            return NSRT(name,
                        parameters, preconditions, add_effects, delete_effects,
                        set(), selected_option, option_vars, null_sampler)

        # Switch/Fan control NSRTs
        if CFG.fan_known_controls_relation:
            if CFG.fan_combine_switch_on_off:
                # Combined switch option for both on/off
                assert switch_option is not None
                nsrts.add(
                    _make_fan_toggle_nsrt("TurnFanOn", FanOff, FanOn, FanOff,
                                          switch_option))
                nsrts.add(
                    _make_fan_toggle_nsrt("TurnFanOff", FanOn, FanOff, FanOn,
                                          switch_option))
            else:
                # Separate switch options
                assert switch_on_option is not None
                assert switch_off_option is not None
                nsrts.add(
                    _make_fan_toggle_nsrt("TurnFanOn", FanOff, FanOn, FanOff,
                                          switch_on_option))
                nsrts.add(
                    _make_fan_toggle_nsrt("TurnFanOff", FanOn, FanOff, FanOn,
                                          switch_off_option))
        else:
            if CFG.fan_combine_switch_on_off:
                # Combined switch option for both on/off
                assert switch_option is not None
                nsrts.add(
                    _make_fan_toggle_nsrt("TurnSwitchOn", SwitchOff, SwitchOn,
                                          SwitchOff, switch_option))
                nsrts.add(
                    _make_fan_toggle_nsrt("TurnSwitchOff", SwitchOn, SwitchOff,
                                          SwitchOn, switch_option))
            else:
                # Separate switch options
                assert switch_on_option is not None
                assert switch_off_option is not None
                nsrts.add(
                    _make_fan_toggle_nsrt("TurnSwitchOn", SwitchOff, SwitchOn,
                                          SwitchOff, switch_on_option))
                nsrts.add(
                    _make_fan_toggle_nsrt("TurnSwitchOff", SwitchOn, SwitchOff,
                                          SwitchOn, switch_off_option))

        # Wait
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        preconditions: Set[LiftedAtom] = set()
        add_effects: Set[LiftedAtom] = set()
        delete_effects: Set[LiftedAtom] = set()
        wait_nsrt = NSRT("Wait", parameters, preconditions, add_effects,
                         delete_effects, set(), Wait, option_vars,
                         null_sampler)
        nsrts.add(wait_nsrt)

        # Helper function to create movement NSRTs that use appropriate switch
        # options
        def _make_movement_nsrt(name: str,
                                option_to_use: ParameterizedOption) -> NSRT:
            robot = Variable("?robot", robot_type)
            pos1 = Variable("?pos1", position_type)
            pos2 = Variable("?pos2", position_type)
            ball = Variable("?ball", ball_type)
            if CFG.fan_known_controls_relation:
                control_obj = Variable("?fan", fan_type)
            else:
                control_obj = Variable("?switch", switch_type)
            parameters = [robot, ball, pos1, pos2, control_obj]
            option_vars = [robot, control_obj]
            preconditions = {
                LiftedAtom(BallAtLoc, [ball, pos1]),
                LiftedAtom(ClearPos, [pos2]),
            }
            add_effects = {
                LiftedAtom(BallAtLoc, [ball, pos2]),
            }
            delete_effects = {
                LiftedAtom(BallAtLoc, [ball, pos1]),
            }
            return NSRT(name,
                        parameters, preconditions, add_effects, delete_effects,
                        set(), option_to_use, option_vars, null_sampler)

        # Movement NSRTs using appropriate switch options
        if CFG.fan_combine_switch_on_off:
            # Use combined switch option
            assert switch_option is not None
            nsrts.add(_make_movement_nsrt("MoveRight", switch_option))
            nsrts.add(_make_movement_nsrt("MoveLeft", switch_option))
            nsrts.add(_make_movement_nsrt("MoveDown", switch_option))
            nsrts.add(_make_movement_nsrt("MoveUp", switch_option))
        else:
            # Use separate switch options - movement typically requires turning
            # fan on
            assert switch_on_option is not None
            nsrts.add(_make_movement_nsrt("MoveRight", switch_on_option))
            nsrts.add(_make_movement_nsrt("MoveLeft", switch_on_option))
            nsrts.add(_make_movement_nsrt("MoveDown", switch_on_option))
            nsrts.add(_make_movement_nsrt("MoveUp", switch_on_option))

        return nsrts
