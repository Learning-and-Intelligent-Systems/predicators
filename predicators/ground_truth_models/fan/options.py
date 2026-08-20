"""Ground-truth options for the coffee environment."""

from typing import Callable, ClassVar, Dict, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    create_push_skill, create_wait_option, shared_skill_robot, \
    shared_skill_simulator
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type

from .options_legacy import _FanLegacyOptionsMixin


class PyBulletFanGroundTruthOptionFactory(_FanLegacyOptionsMixin,
                                          GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletFanEnv]] = PyBulletFanEnv
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _hand_empty_move_z: ClassVar[float] = env_cls.z_ub - 0.3
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.5
    _z_offset: ClassVar[float] = 0.1
    _y_offset: ClassVar[float] = 0.03

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_fan"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the fan environment."""
        if CFG.fan_use_skill_factories:
            return cls._get_options_skill_factories(env_name, types,
                                                    predicates, action_space)
        return cls._get_options_legacy(env_name, types, predicates,
                                       action_space)

    # ------------------------------------------------------------------
    # Skill-factory path
    # ------------------------------------------------------------------

    @classmethod
    def _get_options_skill_factories(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:
        """Skill-factory-based option implementations for the fan env."""
        del env_name, predicates, action_space  # unused

        pybullet_robot = shared_skill_robot(PyBulletFanEnv)

        robot_type = types["robot"]
        switch_type = types["switch"]
        fan_type = types["fan"]

        env_cls = cls.env_cls

        _push_transport_z = cls._hand_empty_move_z

        simulator = shared_skill_simulator(env_cls) \
            if CFG.skill_phase_use_motion_planning else None
        config = SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=PyBulletFanEnv._fingers_state_to_joint,  # pylint: disable=protected-access
            robot_init_tilt=PyBulletFanEnv.robot_init_tilt,
            robot_init_wrist=PyBulletFanEnv.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=_push_transport_z,
            simulator=simulator,
        )

        if CFG.fan_known_controls_relation:
            control_obj_type = fan_type

            # With fan_known_controls_relation, the second object is the fan.
            # We look up the switch via the fan's controls_fan/facing_side.
            def _get_switch_pose_via_fan(
                state: State,
                objects: Sequence[Object],
                params: Array,
                config: SkillConfig,
            ) -> Tuple[float, float, float, float]:
                del params, config
                _, fan = objects
                switch = next(
                    (s for s in state.get_objects(switch_type) if state.get(
                        s, "controls_fan") == state.get(fan, "facing_side")),
                    None)
                if switch is None:
                    raise utils.OptionExecutionFailure(
                        "No switch found for fan (controls_fan mismatch)")
                return (state.get(switch, "x"), state.get(switch, "y"),
                        state.get(switch, "z"), state.get(switch, "rot"))

            _get_switch_pose: Callable = _get_switch_pose_via_fan
        else:
            control_obj_type = switch_type

            def _get_switch_pose_direct(
                state: State,
                objects: Sequence[Object],
                params: Array,
                config: SkillConfig,
            ) -> Tuple[float, float, float, float]:
                del params, config
                _, switch = objects
                return (state.get(switch, "x"), state.get(switch, "y"),
                        state.get(switch, "z"), state.get(switch, "rot"))

            _get_switch_pose = _get_switch_pose_direct

        # Adjust yaw to match standard facing convention:
        # standard facing = (sin(yaw), cos(yaw)),
        # switch push_dir = (cos(rot), sin(rot)) → yaw = π/2 − rot
        def _get_switch_on_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            x, y, z, rot = _get_switch_pose(state, objects, params, cfg)
            return x, y, z, rot - np.pi / 2

        def _get_switch_off_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            x, y, z, rot = _get_switch_pose(state, objects, params, cfg)
            return x, y, z, rot + np.pi / 2

        option_type = [robot_type, control_obj_type]

        Wait = create_wait_option("Wait", config, robot_type)

        if CFG.fan_combine_switch_on_off:
            # Combined SwitchOnOff: chain two standard push skills.
            _SwitchOn = create_push_skill(
                name="_SwitchOn",
                types=option_type,
                config=config,
                get_target_pose_fn=_get_switch_on_pose,
            )
            _SwitchOff = create_push_skill(
                name="_SwitchOff",
                types=option_type,
                config=config,
                get_target_pose_fn=_get_switch_off_pose,
            )
            SwitchOnOff = utils.LinearChainParameterizedOption(
                "SwitchOnOff", [_SwitchOn, _SwitchOff])
            return {SwitchOnOff, Wait}

        SwitchOn = create_push_skill(
            name="SwitchOn",
            types=option_type,
            config=config,
            get_target_pose_fn=_get_switch_on_pose,
        )
        SwitchOff = create_push_skill(
            name="SwitchOff",
            types=option_type,
            config=config,
            get_target_pose_fn=_get_switch_off_pose,
        )
        return {SwitchOn, SwitchOff, Wait}
