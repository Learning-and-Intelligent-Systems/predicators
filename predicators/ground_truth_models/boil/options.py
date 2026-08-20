"""Ground-truth options for the boil environment."""

from dataclasses import replace
from typing import ClassVar, Dict, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_boil import PyBulletBoilEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    create_pick_skill, create_place_skill, create_push_skill, \
    create_wait_option, shared_skill_robot, shared_skill_simulator
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type

from .options_legacy import _BoilLegacyOptionsMixin


class PyBulletBoilGroundTruthOptionFactory(_BoilLegacyOptionsMixin,
                                           GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletBoilEnv]] = PyBulletBoilEnv
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _hand_empty_move_z: ClassVar[float] = env_cls.z_ub - 0.3
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.35
    _z_offset: ClassVar[float] = 0.1
    _y_offset: ClassVar[float] = 0.03

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_boil"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the boil environment."""
        if CFG.boil_use_skill_factories:
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
        """Skill-factory-based option implementations for the boil env."""
        del env_name, action_space, predicates  # unused

        pybullet_robot = shared_skill_robot(PyBulletBoilEnv)

        robot_type = types["robot"]
        switch_type = types["switch"]
        jug_type = types["jug"]
        burner_type = types["burner"]
        faucet_type = types["faucet"]

        env_cls = cls.env_cls

        simulator = shared_skill_simulator(env_cls) \
            if CFG.skill_phase_use_motion_planning else None
        config = SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=PyBulletBoilEnv._fingers_state_to_joint,  # pylint: disable=protected-access
            robot_init_tilt=PyBulletBoilEnv.robot_init_tilt,
            robot_init_wrist=PyBulletBoilEnv.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=cls._transport_z,
            # Mobile-base (mobile_fetch) positioning: park the base 0.6 m in
            # front of each reach target with its x aligned to the target x, so
            # the arm reaches straight forward at a comfortable distance instead
            # of sideways over the burner or fully extended. base_y is clamped
            # to keep the base clear of the table front (y_lb).
            base_standoff=(CFG.boil_mobile_base_standoff
                           if CFG.boil_mobile_base_park else None),
            base_y_max=env_cls.y_lb - 0.28,
            base_align_x=CFG.boil_mobile_base_align_x,
            base_home_xy=(env_cls.robot_base_pos[0],
                          env_cls.robot_base_pos[1]),
            simulator=simulator,
        )

        # ---------------------------------------------------------------
        # Helper: find the switch object associated with a faucet/burner.
        # The env sets obj.switch_id in _set_state.
        # ---------------------------------------------------------------
        def _get_switch_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            config: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params, config
            _, obj = objects
            switch = next((s for s in state.get_objects(switch_type)
                           if s.id == obj.switch_id), None)
            if switch is None:
                raise utils.OptionExecutionFailure(
                    f"No switch found for {obj} (switch_id={obj.switch_id})")
            return (state.get(switch, "x"), state.get(switch, "y"),
                    state.get(switch, "z"), state.get(switch, "rot"))

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
            return x, y, z, rot + np.pi / 2

        def _get_switch_off_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            x, y, z, rot = _get_switch_pose(state, objects, params, cfg)
            return x, y, z, rot - np.pi / 2

        _push_transport_z = cls._hand_empty_move_z
        push_config = replace(config, transport_z=_push_transport_z)

        # ---------------------------------------------------------------
        # PickJug: grasp at handle position with is_held terminal.
        # ---------------------------------------------------------------
        def _get_jug_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            config: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params, config
            _, jug = objects
            rot = state.get(jug, "rot")
            gx = (state.get(jug, "x") +
                  np.cos(rot) * env_cls.jug_handle_offset)
            gy = (state.get(jug, "y") +
                  np.sin(rot) * env_cls.jug_handle_offset)
            gz = env_cls.table_height + env_cls.jug_handle_height
            return gx, gy, gz, rot

        PickJug = create_pick_skill(
            name="PickJug",
            types=[robot_type, jug_type],
            config=config,
            get_target_pose_fn=_get_jug_pose,
        )

        # ---------------------------------------------------------------
        # SwitchFaucetOn / SwitchFaucetOff (take [robot, faucet] objects)
        # ---------------------------------------------------------------
        SwitchFaucetOn = create_push_skill(
            name="SwitchFaucetOn",
            types=[robot_type, faucet_type],
            config=push_config,
            get_target_pose_fn=_get_switch_on_pose,
        )
        SwitchFaucetOff = create_push_skill(
            name="SwitchFaucetOff",
            types=[robot_type, faucet_type],
            config=push_config,
            get_target_pose_fn=_get_switch_off_pose,
        )

        # ---------------------------------------------------------------
        # SwitchBurnerOn / SwitchBurnerOff (take [robot, burner] objects)
        # ---------------------------------------------------------------
        SwitchBurnerOn = create_push_skill(
            name="SwitchBurnerOn",
            types=[robot_type, burner_type],
            config=push_config,
            get_target_pose_fn=_get_switch_on_pose,
        )
        SwitchBurnerOff = create_push_skill(
            name="SwitchBurnerOff",
            types=[robot_type, burner_type],
            config=push_config,
            get_target_pose_fn=_get_switch_off_pose,
        )

        # ---------------------------------------------------------------
        # Place option (unified – only needs the robot)
        # ---------------------------------------------------------------
        Place = create_place_skill(
            name="Place",
            types=[robot_type],
            config=config,
        )

        # ---------------------------------------------------------------
        # Wait
        # ---------------------------------------------------------------
        Wait = create_wait_option("Wait", config, robot_type)

        return {
            PickJug, SwitchFaucetOn, SwitchFaucetOff, SwitchBurnerOn,
            SwitchBurnerOff, Place, Wait
        }
