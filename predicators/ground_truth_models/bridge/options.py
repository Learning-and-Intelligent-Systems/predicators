"""Ground-truth options (skills) for the bridge environment.

Built on the shared skill factories: PickBlock / PickBottle (pick),
Place (generic place; all geometry via continuous params from the
samplers in ``processes.py``), three ApplyGlue skills (a custom phase
composition that dabs the held bottle's tip onto a block face), and
Wait.
"""

from typing import ClassVar, Dict, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import PhaseSkill, \
    SkillConfig, build_params_space, create_pick_skill, create_place_skill, \
    create_wait_option, shared_skill_robot, shared_skill_simulator
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type

# Place params: canonical (x, y, release_z, yaw) order with a wider
# release_z band than the factory default [0.5, 0.6] -- placing a span
# block flat on the table releases at ~0.46, and seating the span on
# full-variant leg stacks releases at ~0.66.
_BRIDGE_PLACE_PARAMS = [
    ("target_x (world x position for placement)", 0.4, 1.1),
    ("target_y (world y position for placement)", 1.1, 1.6),
    ("release_z (world z height to open gripper)", 0.44, 0.72),
    ("target_yaw (placement orientation in radians)", -np.pi, np.pi),
]

_APPLY_GLUE_PARAMS = [
    ("dab_z_offset (extra height above the face dab point)", 0.0, 0.02),
]


class PyBulletBridgeGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the bridge environment."""

    env_cls: ClassVar[TypingType[PyBulletBridgeEnv]] = PyBulletBridgeEnv

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_bridge"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        del env_name, predicates, action_space  # unused

        pybullet_robot = shared_skill_robot(cls.env_cls)

        robot_type = types["robot"]
        block_type = types["block"]
        bottle_type = types["bottle"]

        config = cls._build_skill_config(pybullet_robot)

        # -- PickBlock -------------------------------------------------------
        def _get_block_grasp_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params, cfg
            _, blk = objects
            half = cls.env_cls._block_half_extents(blk)  # pylint: disable=protected-access
            # Grasp near the top of the block. At wrist yaw = block rot
            # the fingers straddle the block's local-y axis -- the 5 cm
            # width for both shapes (a quarter-turn would straddle the
            # span's 10 cm length, wider than the ~8 cm finger opening,
            # so the fingers press the top face and shove the block
            # into the table instead of wrapping it).
            return (state.get(blk, "x"), state.get(blk, "y"),
                    state.get(blk, "z") + half[2], state.get(blk, "rot"))

        PickBlock = create_pick_skill(
            name="PickBlock",
            types=[robot_type, block_type],
            config=config,
            get_target_pose_fn=_get_block_grasp_pose,
            # Open-fingered approach avoids dragging the block before
            # the grasp; anchored lift avoids the unreachable chase of
            # the held block's xy near the reach limit (see bond).
            approach_open=True,
            anchor_lift=True,
            # Staging is dense (grid slots ~11 cm apart), so end the
            # pick 3 cm up: a 1 cm lift can leave the gripper grazing
            # a neighboring block, poisoning the next option's BiRRT
            # start config.
            lift_dz=0.03,
        )

        # -- PickBottle ------------------------------------------------------
        def _get_bottle_grasp_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params, cfg
            _, bottle = objects
            return (state.get(bottle,
                              "x"), state.get(bottle,
                                              "y"), state.get(bottle, "z") +
                    cls.env_cls.bottle_half_extents[2],
                    state.get(bottle, "rot"))

        PickBottle = create_pick_skill(
            name="PickBottle",
            types=[robot_type, bottle_type],
            config=config,
            get_target_pose_fn=_get_bottle_grasp_pose,
            approach_open=True,
            anchor_lift=True,
        )

        # -- Place (generic; geometry via params) ---------------------------
        # use_move_above keeps the carry at transport_z so a held block
        # (or a whole welded span assembly) clears staged objects and
        # the standing legs.
        Place = create_place_skill(
            name="Place",
            types=[robot_type],
            config=config,
            use_move_above=True,
            param_defs=_BRIDGE_PLACE_PARAMS,
            # Land the HELD BLOCK (not the gripper) on the sampled
            # target: blocks staged near the reach limit grasp with up
            # to ~2 cm of EE-to-block IK residual, and an uncompensated
            # place transfers that error into the butt joints and seat
            # alignment, past the cure gates' tolerance.
            compensate_held_offset=True,
        )

        # -- ApplyGlue (one per face) ----------------------------------------
        options = {
            PickBlock, PickBottle, Place,
            create_wait_option("Wait", config, robot_type)
        }
        for face, name in (("top", "ApplyGlueTop"), ("end_a", "ApplyGlueEndA"),
                           ("end_b", "ApplyGlueEndB")):
            options.add(
                cls._create_apply_glue_skill(name, face, robot_type,
                                             bottle_type, block_type, config))
        return options

    @classmethod
    def _build_skill_config(
            cls, pybullet_robot: SingleArmPyBulletRobot) -> SkillConfig:
        simulator = shared_skill_simulator(cls.env_cls) \
            if CFG.skill_phase_use_motion_planning else None
        env_cls = cls.env_cls
        return SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=env_cls._fingers_state_to_joint,  # pylint: disable=protected-access
            ik_validate=CFG.pybullet_ik_validate,
            robot_init_tilt=env_cls.robot_init_tilt,
            robot_init_wrist=env_cls.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=env_cls.transport_z,
            simulator=simulator,
        )

    @classmethod
    def _create_apply_glue_skill(cls, name: str, face: str, robot_type: Type,
                                 bottle_type: Type, block_type: Type,
                                 config: SkillConfig) -> ParameterizedOption:
        """Dab the held bottle's tip onto ``face`` of the target block.

        Phases: MoveAboveDab (transport height) -> DescendToDab
        (validated IK) -> Dwell (holds the dab pose until the env's
        proximity detector flips the face's wet-glue feature -- a
        closed-loop terminal) -> Retreat (back to transport height).

        Objects: (robot, bottle, block). The bottle must already be
        held. Continuous param: extra dab height offset.
        """
        params_space, params_description = build_params_space(
            _APPLY_GLUE_PARAMS)
        env_cls = cls.env_cls

        def _dab_xy_yaw(
                state: State,
                objects: Sequence[Object]) -> Tuple[float, float, float]:
            blk = objects[2]
            dab = env_cls._face_dab_point(state, blk, face)  # pylint: disable=protected-access
            return dab[0], dab[1], state.get(blk, "rot")

        def _above_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params
            x, y, yaw = _dab_xy_yaw(state, objects)
            return x, y, cfg.transport_z, yaw

        def _dab_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del cfg
            robot, bottle, blk = objects
            x, y, yaw = _dab_xy_yaw(state, objects)
            dab_z = env_cls._face_dab_point(state, blk, face)[2]  # pylint: disable=protected-access
            # Descend so the held bottle's TIP lands at the dab point.
            # The EE-to-bottle hang offset is read live from the state
            # (exact for whatever grasp depth the pick ended with; a
            # constant estimate here put the tip inside the block and
            # BiRRT rejected the goal as in-collision).
            hang = state.get(robot, "z") - state.get(bottle, "z")
            z = dab_z + env_cls.bottle_half_extents[2] + hang + \
                float(params[0])
            return x, y, z, yaw

        def _glue_applied(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> bool:
            del params, cfg
            blk = objects[2]
            return state.get(blk, f"glue_{face}") > 0.5

        dwell = make_move_to_phase("Dwell",
                                   _dab_pose,
                                   "closed",
                                   allow_shallow_held_object_contacts=True)
        # Closed-loop terminal: the phase ends when the env's proximity
        # detector wets the face, not when a pose tolerance is met.
        dwell.terminal_fn = _glue_applied

        phases = [
            make_move_to_phase("MoveAboveDab", _above_pose, "closed"),
            make_move_to_phase("DescendToDab",
                               _dab_pose,
                               "closed",
                               validate_ik=True),
            dwell,
            # The dab can leave the bottle tip grazing the face edge by
            # ~1 mm; without the shallow-contact allowance that graze
            # invalidates the Retreat's BiRRT start config.
            make_move_to_phase("Retreat",
                               _above_pose,
                               "closed",
                               allow_shallow_held_object_contacts=True),
        ]

        return PhaseSkill(name, [robot_type, bottle_type, block_type],
                          params_space,
                          config,
                          phases,
                          params_description=params_description,
                          base_mode="home").build()
