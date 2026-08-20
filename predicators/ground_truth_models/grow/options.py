"""Ground-truth options for the coffee environment."""

from typing import ClassVar, Dict, Sequence, Set, Tuple
from typing import Type as TypingType

from gym.spaces import Box

from predicators.envs.pybullet_grow import PyBulletGrowEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    create_pick_skill, create_place_skill, create_pour_skill, \
    create_wait_option, shared_skill_robot, shared_skill_simulator
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type

from .options_legacy import _GrowLegacyOptionsMixin


class PyBulletGrowGroundTruthOptionFactory(_GrowLegacyOptionsMixin,
                                           GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletGrowEnv]] = PyBulletGrowEnv
    pick_policy_tol: ClassVar[float] = 1e-3
    pour_policy_tol: ClassVar[float] = 1e-3 / 2
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_grow"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the grow environment."""
        if CFG.grow_use_skill_factories:
            return cls._get_options_skill_factories(env_name, types,
                                                    predicates, action_space)
        return cls._get_options_legacy(env_name, types, predicates,
                                       action_space)

    # ------------------------------------------------------------------
    # Skill-factory path
    # ------------------------------------------------------------------

    @classmethod
    def _get_options_skill_factories(
            cls, _env_name: str, types: Dict[str, Type],
            predicates: Dict[str, Predicate],
            _action_space: Box) -> Set[ParameterizedOption]:
        """Skill-factory-based option implementations for the grow env.

        PickJug and Place use the PhaseSkill framework.  Pour falls back
        to the legacy implementation because it involves continuous
        tilting that doesn't map directly to MOVE_TO_POSE /
        CHANGE_FINGERS phases.
        """
        del predicates  # unused in skill factory implementation

        pybullet_robot = shared_skill_robot(PyBulletGrowEnv)

        robot_type = types["robot"]
        jug_type = types["jug"]
        cup_type = types["cup"]

        env_cls = cls.env_cls

        simulator = shared_skill_simulator(env_cls) \
            if CFG.skill_phase_use_motion_planning else None
        config = SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=PyBulletGrowEnv._fingers_state_to_joint,  # pylint: disable=protected-access
            robot_init_tilt=PyBulletGrowEnv.robot_init_tilt,
            robot_init_wrist=PyBulletGrowEnv.robot_init_wrist,
            transport_z=env_cls.z_ub - 0.35,
            simulator=simulator,
            ik_validate=CFG.pybullet_ik_validate if hasattr(
                CFG, 'pybullet_ik_validate') else False,
            extra={"jug_handle_height": env_cls.jug_handle_height},
        )

        # ---------------------------------------------------------------
        # PickJug: grow uses a very permissive grasp tolerance (5e-2), so
        # the default joint-value terminal is sufficient (no is_held check).
        # ---------------------------------------------------------------
        def _get_jug_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            config: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params, config
            _, jug = objects
            hx, hy, hz = env_cls._get_jug_handle_grasp(  # type: ignore[attr-defined]  # pylint: disable=protected-access
                state, jug)
            return (hx, hy, hz, state.get(jug, "rot"))

        PickJug = create_pick_skill(
            name="PickJug",
            types=[robot_type, jug_type],
            config=config,
            get_target_pose_fn=_get_jug_pose,
        )

        # ---------------------------------------------------------------
        # Place
        # ---------------------------------------------------------------
        Place = create_place_skill(
            name="Place",
            types=[robot_type, jug_type],
            config=config,
        )

        # ---------------------------------------------------------------
        # Pour
        # ---------------------------------------------------------------

        def _get_cup_position(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params
            _, _, cup = objects
            return (state.get(cup, "x"), state.get(cup, "y"),
                    state.get(cup, "z"), cfg.robot_init_wrist)

        Pour = create_pour_skill(
            name="Pour",
            types=[robot_type, jug_type, cup_type],
            config=config,
            get_target_pose_fn=_get_cup_position,
        )

        # ---------------------------------------------------------------
        # Wait
        # ---------------------------------------------------------------
        Wait = create_wait_option("Wait", config, robot_type)

        return {PickJug, Place, Pour, Wait}
