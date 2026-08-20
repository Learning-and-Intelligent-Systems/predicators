"""Ground-truth processes for the fan environment."""

from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, DelayDistribution, \
    EndogenousProcess, ExogenousProcess, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler


def _push_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return fixed push params for fan switch push.

    Approach 0.075 sits in a measured window: with the side-oriented
    gripper (the fetch's push_ee_yaw_offset, 0) the hand extends along
    the approach axis, so below ~0.073 the descend waypoint collides
    with an end-of-row switch (BiRRT goal-in-collision), while at 0.08
    the far-side (Off-push) waypoint already stalls at the fetch arm's
    reach limit. 0.075 toggles all four switches both ways.
    """
    if not CFG.fan_use_skill_factories:
        return np.array([], dtype=np.float32)
    del state, goal, rng, objs
    return np.array([0.075, 0.1], dtype=np.float32)


class PyBulletFanGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the fan environment.

    Endogenous: toggling switches (modeled as turning fans on/off).
    Exogenous: ball moves between grid positions when blown by active fans.
    """

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_fan"}

    @classmethod
    def get_processes(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

        # Types
        robot_type = types["robot"]
        switch_type = types["switch"]
        fan_type = types["fan"]
        ball_type = types["ball"]
        location_type = types["loc"]
        side_type = types["side"]

        # Predicates
        FanOn = predicates["FanOn"]
        FanOff = predicates["FanOff"]
        BallAtLoc = predicates["BallAtLoc"]
        ClearLoc = predicates["ClearLoc"]
        FanFacingSide = predicates["FanFacingSide"]
        SideOf = predicates["SideOf"]
        OppositeFan = predicates["OppositeFan"]
        if not CFG.fan_known_controls_relation:
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

        processes: Set[CausalProcess] = set()

        def _make_fan_toggle_process(name: str, start_predicate: Predicate,
                                     add_predicate: Predicate,
                                     delete_predicate: Predicate,
                                     delay_mu: float) -> EndogenousProcess:
            """Helper function to create fan on/off toggle processes."""
            robot = Variable("?robot", robot_type)
            if CFG.fan_known_controls_relation:
                controled_obj = Variable("?fan", fan_type)
            else:
                controled_obj = Variable("?switch", switch_type)
            parameters = [robot, controled_obj]
            option_vars = [robot, controled_obj]
            condition_at_start = {
                LiftedAtom(start_predicate, [controled_obj]),
            }
            add_effects: Set[LiftedAtom] = {
                LiftedAtom(add_predicate, [controled_obj]),
            }
            delete_effects = {
                LiftedAtom(delete_predicate, [controled_obj]),
            }
            delay_distribution = DiscreteGaussianDelay(
                mu=torch.tensor(delay_mu), sigma=torch.tensor(0.1))

            # Select the appropriate option based on configuration
            if CFG.fan_combine_switch_on_off:
                selected_option = switch_option
            else:
                if name in ["TurnFanOn", "TurnSwitchOn"]:
                    selected_option = switch_on_option
                else:
                    selected_option = switch_off_option

            assert selected_option is not None
            return EndogenousProcess(name, parameters, condition_at_start,
                                     set(), set(), add_effects,
                                     delete_effects, delay_distribution,
                                     torch.tensor(1.0), selected_option,
                                     option_vars, _push_sampler)

        # --- Endogenous processes: Switch toggling ---
        # For the harder setting of having to figure out which switch controls
        # which fan, we can have effects to be turn swtch on/off, and have it
        # to invent Control(fan, switch) predicate.
        # Delays are calibrated to execution. A switch press runs ~28 env
        # steps while a MoveToSide cell crossing runs ~35 (mu=4 ticks),
        # but the wind state flips when the FINGER FLIPS THE SWITCH,
        # about a quarter into the press, not at press end. The
        # asymmetry matters for planning:
        # - TurnFanOn mu=4: the wind starting late is harmless (each
        #   executed Wait ends on the actual cell crossing, so an early
        #   real start only makes Waits terminate sooner), and the long
        #   delay keeps the believed first crossing after the press, in
        #   line with execution, so the plan's Wait count matches the
        #   crossings the executor must observe.
        # - TurnFanOff mu=1: the honest flip time. Modeling it as press
        #   END (mu=4) lets the planner bank on a full extra cell of
        #   drift during the off-press; the real press delivers less,
        #   the ball stops short of the target, and the plan needs an
        #   execution replan. With mu=1 the planner keeps the fan on
        #   until the ball has actually entered the target cell and the
        #   small real off-press drift only recenters it in the cell.
        if CFG.fan_known_controls_relation:
            processes.add(
                _make_fan_toggle_process("TurnFanOn",
                                         FanOff,
                                         FanOn,
                                         FanOff,
                                         delay_mu=4.0))
            processes.add(
                _make_fan_toggle_process("TurnFanOff",
                                         FanOn,
                                         FanOff,
                                         FanOn,
                                         delay_mu=1.0))
        else:
            processes.add(
                _make_fan_toggle_process("TurnSwitchOn",
                                         SwitchOff,
                                         SwitchOn,
                                         SwitchOff,
                                         delay_mu=4.0))
            processes.add(
                _make_fan_toggle_process("TurnSwitchOff",
                                         SwitchOn,
                                         SwitchOff,
                                         SwitchOn,
                                         delay_mu=1.0))

        # Wait
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = Wait
        delay_distribution: DelayDistribution = ConstantDelay(1)
        wait_process = EndogenousProcess("Wait", parameters, set(), set(),
                                         set(), set(),
                                         set(), delay_distribution,
                                         torch.tensor(1.0), option,
                                         option_vars, null_sampler)
        processes.add(wait_process)

        # --- Exogenous processes: Ball movement due to active fans ---
        fan = Variable("?fan", fan_type)
        op_fan = Variable("?op_fan", fan_type)
        ball = Variable("?ball", ball_type)
        pos1 = Variable("?pos1", location_type)
        pos2 = Variable("?pos2", location_type)
        direction = Variable("?dir", side_type)
        parameters = [ball, pos1, pos2, direction]
        condition_at_start = {
            LiftedAtom(BallAtLoc, [ball, pos2]),
            LiftedAtom(ClearLoc, [pos1]),
            LiftedAtom(SideOf, [pos1, pos2, direction]),  # could be invented
            LiftedAtom(FanFacingSide, [fan, direction]),  # could be invented
        }
        if CFG.fan_known_controls_relation:
            parameters.extend([fan, op_fan])
            condition_at_start.add(LiftedAtom(
                OppositeFan, [fan, op_fan]))  # could be invented
            condition_at_start.add(LiftedAtom(FanOn, [fan]))
            condition_at_start.add(LiftedAtom(FanOff, [op_fan]))
        else:
            raise NotImplementedError

        condition_overall = set(condition_at_start)
        add_effects = {
            LiftedAtom(BallAtLoc, [ball, pos1]),
        }
        delete_effects = {
            LiftedAtom(BallAtLoc, [ball, pos2]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                                   sigma=torch.tensor(0.1))
        move_to_side = ExogenousProcess("MoveToSide", parameters,
                                        condition_at_start, condition_overall,
                                        set(), add_effects, delete_effects,
                                        delay_distribution, torch.tensor(1.0))
        processes.add(move_to_side)
        return processes
