"""Ground-truth options for the Kitchen environment."""

from typing import ClassVar, Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.envs.kitchen import KitchenEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.geometry import Pose3D
from predicators.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Type

try:
    from gymnasium_robotics.utils.rotations import euler2quat, quat2euler, \
        subtract_euler
    _MJKITCHEN_IMPORTED = True
except (ImportError, RuntimeError):
    _MJKITCHEN_IMPORTED = False


class KitchenGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the Kitchen environment."""

    moveto_tol: ClassVar[float] = 0.01  # for terminating moving
    max_delta_mag: ClassVar[float] = 1.0  # don't move more than this per step
    # A reasonable home position for the end effector.
    home_pos: ClassVar[Pose3D] = (0.0, 0.3, 2.0)
    # Move forward by this amount while pushing left/right.
    push_lr_forward_mag: ClassVar[float] = 0.1
    # Keep pushing a bit even if the On classifier holds.
    push_lr_thresh_pad: ClassVar[float] = 0.01

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"kitchen"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        assert _MJKITCHEN_IMPORTED, "See kitchen.py"

        # Need to define these here because uses may not have euler2quat.
        down_quat = euler2quat((-np.pi, 0.0, -np.pi / 2))
        # End effector facing forward (e.g., toward the knobs.)
        fwd_quat = euler2quat((-np.pi / 2, 0.0, -np.pi / 2))

        # Types
        gripper_type = types["gripper"]
        object_type = types["obj"]

        # Predicates
        OnTop = predicates["OnTop"]

        options: Set[ParameterizedOption] = set()

        # MoveTo
        def _MoveTo_initiable(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            # Store the target pose.
            _, obj = objects
            ox = state.get(obj, "x")
            oy = state.get(obj, "y")
            oz = state.get(obj, "z")
            target_pose = params + (ox, oy, oz)
            memory["target_pose"] = target_pose
            # Turn the knobs by pushing from a "forward" position.
            if "knob" in obj.name:
                memory["target_quat"] = fwd_quat
            else:
                memory["target_quat"] = down_quat
            memory["reset_pose"] = np.array(cls.home_pos, dtype=np.float32)
            memory["reset_quat"] = down_quat
            memory["has_reset"] = False
            return True

        def _MoveTo_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
            del params  # unused
            gripper = objects[0]
            gx = state.get(gripper, "x")
            gy = state.get(gripper, "y")
            gz = state.get(gripper, "z")
            gqw = state.get(gripper, "qw")
            gqx = state.get(gripper, "qx")
            gqy = state.get(gripper, "qy")
            gqz = state.get(gripper, "qz")
            current_euler = quat2euler([gqw, gqx, gqy, gqz])
            if not memory["has_reset"]:
                if np.allclose((gx, gy, gz),
                               memory["reset_pose"],
                               atol=cls.moveto_tol):
                    target_pose = memory["target_pose"]
                    target_quat = memory["target_quat"]
                    memory["has_reset"] = True
                else:
                    target_pose = memory["reset_pose"]
                    target_quat = memory["reset_quat"]
            else:
                target_pose = memory["target_pose"]
                target_quat = memory["target_quat"]
            dx, dy, dz = np.subtract(target_pose, (gx, gy, gz))
            target_euler = quat2euler(target_quat)
            droll, dpitch, dyaw = subtract_euler(target_euler, current_euler)
            arr = np.array([dx, dy, dz, droll, dpitch, dyaw, 0.0],
                           dtype=np.float32)
            action_mag = np.linalg.norm(arr)
            if action_mag > cls.max_delta_mag:
                scale = cls.max_delta_mag / action_mag
                arr = arr * scale
            return Action(arr)

        def _MoveTo_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del params  # unused
            gripper = objects[0]
            gx = state.get(gripper, "x")
            gy = state.get(gripper, "y")
            gz = state.get(gripper, "z")
            return np.allclose((gx, gy, gz),
                               memory["target_pose"],
                               atol=cls.moveto_tol)

        MoveTo = ParameterizedOption(
            "MoveTo",
            types=[gripper_type, object_type],
            # Parameter is a position to move to relative to the object.
            params_space=Box(-5, 5, (3, )),
            policy=_MoveTo_policy,
            initiable=_MoveTo_initiable,
            terminal=_MoveTo_terminal)

        options.add(MoveTo)

        # PushObjOnObjForward
        def _PushObjOnObjForward_policy(state: State, memory: Dict,
                                        objects: Sequence[Object],
                                        params: Array) -> Action:
            del state, memory, objects  # unused
            arr = np.array([0.0, params[0], 0.0, 0.0, 0.0, 0.0, 0.0],
                           dtype=np.float32)
            return Action(arr)

        def _PushObjOnObjForward_terminal(state: State, memory: Dict,
                                          objects: Sequence[Object],
                                          params: Array) -> bool:
            del memory, params  # unused
            gripper, obj, obj2 = objects
            gripper_y = state.get(gripper, "y")
            obj_y = state.get(obj, "y")
            obj2_y = state.get(obj2, "y")
            # Terminate early if the gripper is far past either of the objects.
            if gripper_y - obj_y > 2 * cls.moveto_tol or \
               gripper_y - obj2_y > 2 * cls.moveto_tol:
                return True
            if not GroundAtom(OnTop, [obj, obj2]).holds(state):
                return False
            # Stronger check to deal with case where push release leads object
            # to be no longer OnTop.
            return obj_y > obj2_y - cls.moveto_tol / 2

        PushObjOnObjForward = ParameterizedOption(
            "PushObjOnObjForward",
            types=[gripper_type, object_type, object_type],
            # Parameter is a magnitude for pushing forward.
            params_space=Box(0.0, cls.max_delta_mag, (1, )),
            policy=_PushObjOnObjForward_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_PushObjOnObjForward_terminal)

        options.add(PushObjOnObjForward)

        # PushObjTurnOnLeftRight
        def _PushObjTurnOnLeftRight_initiable(state: State, memory: Dict,
                                              objects: Sequence[Object],
                                              params: Array) -> bool:
            del params  # unused
            # Memorize whether to push left or right based on the relative
            # position of the gripper and object when pushing starts.
            gripper, obj = objects
            ox = state.get(obj, "x")
            gx = state.get(gripper, "x")
            if gx > ox:
                sign = -1
            else:
                sign = 1
            memory["sign"] = sign
            return True

        def _PushObjTurnOnLeftRight_policy(state: State, memory: Dict,
                                           objects: Sequence[Object],
                                           params: Array) -> Action:
            del state, objects  # unused
            sign = memory["sign"]
            # Also push a little bit forward.
            arr = np.array([
                sign * params[0], cls.push_lr_forward_mag, 0.0, 0.0, 0.0, 0.0,
                0.0
            ],
                           dtype=np.float32)
            return Action(arr)

        def _PushObjTurnOnLeftRight_terminal(state: State, memory: Dict,
                                             objects: Sequence[Object],
                                             params: Array) -> bool:
            del memory, params  # unused
            _, obj = objects
            # Use a more stringent threshold to avoid numerical issues.
            return KitchenEnv.On_holds(state, [obj],
                                       thresh_pad=cls.push_lr_thresh_pad)

        PushObjTurnOnLeftRight = ParameterizedOption(
            "PushObjTurnOnLeftRight",
            types=[gripper_type, object_type],
            # Parameter is a magnitude for pushing left/right.
            params_space=Box(0.0, cls.max_delta_mag, (1, )),
            policy=_PushObjTurnOnLeftRight_policy,
            initiable=_PushObjTurnOnLeftRight_initiable,
            terminal=_PushObjTurnOnLeftRight_terminal)

        options.add(PushObjTurnOnLeftRight)

        return options
