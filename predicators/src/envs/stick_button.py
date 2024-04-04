"""An environment where a robot must press buttons with its hand or a stick.

Example to make videos:

python src/main.py --env stick_button --approach oracle --seed 0 --make_test_videos \
    --pybullet_use_gui True --num_test_tasks 1 --stick_button_render_mode pybullet \
    --pybullet_control_mode reset

python predicators/src/main.py --env stick_button --approach oracle --seed 0 --make_test_videos \
    --num_test_tasks 1 --stick_button_render_mode pybullet \
    --pybullet_control_mode reset
"""

import logging
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Set, \
    Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
import pybullet as p

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.envs.pybullet_robots import \
    create_single_arm_pybullet_robot
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Pose3D, Predicate, State, Task, Type, Video, Image
from predicators.src.utils import _Geom2D


class StickButtonEnv(BaseEnv):
    """An environment where a robot must press buttons with its hand or a
    stick."""
    x_lb: ClassVar[float] = 0.4
    y_lb: ClassVar[float] = 1.1
    theta_lb: ClassVar[float] = -np.pi  # radians
    x_ub: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 2.0
    theta_ub: ClassVar[float] = np.pi  # radians
    # Reachable zone boundaries.
    rz_x_lb: ClassVar[float] = x_lb
    rz_x_ub: ClassVar[float] = x_ub
    rz_y_lb: ClassVar[float] = y_lb
    rz_y_ub: ClassVar[float] = 1.5
    max_speed: ClassVar[float] = 0.05  # shared by dx, dy
    max_angular_speed: ClassVar[float] = np.pi / 4
    robot_radius: ClassVar[float] = 0.02
    button_radius: ClassVar[float] = 0.02
    # Note that the stick_width is the longer dimension.
    stick_width: ClassVar[float] = 0.55
    stick_height: ClassVar[float] = stick_width / 30
    # Note that the holder width is set in the class because it uses CFG.
    holder_height: ClassVar[float] = 2.5 * stick_height
    stick_tip_width: ClassVar[float] = stick_height
    init_padding: ClassVar[float] = 0.01  # used to space objects in init states
    stick_init_lb: ClassVar[float] = rz_y_lb + robot_radius + init_padding
    stick_init_ub: ClassVar[float] = stick_init_lb + 0.2 * (rz_y_ub - rz_y_lb)
    pick_grasp_tol: ClassVar[float] = 1e-3
    # PyBullet settings.
    robot_init_x: ClassVar[float] = (rz_y_ub + rz_y_lb) / 2.0
    robot_init_y: ClassVar[float] = (rz_x_ub + rz_x_lb) / 2.0
    robot_init_z: ClassVar[float] = 0.65
    _camera_distance: ClassVar[float] = 1.0
    _camera_yaw: ClassVar[float] = 140
    _camera_pitch: ClassVar[float] = -72
    _camera_target: ClassVar[Pose3D] = (1.5, 0.75, 0.42)
    _table_pose: ClassVar[Pose3D] = (1.75, 0.75, 0.0)
    _table_orientation: ClassVar[Sequence[float]] = [0., 0., 0., 1.]
    _default_obj_orn: ClassVar[Sequence[float]] = [0.0, 0.0, 0.0, 1.0]
    _pybullet_move_to_pose_tol: ClassVar[float] = 1e-4
    _pybullet_max_vel_norm: ClassVar[float] = 0.02
    _holder_base_z_len: ClassVar[float] = 0.05
    _holder_side_z_len: ClassVar[float] = 2 * _holder_base_z_len
    _holder_side_height: ClassVar[float] = 0.2 * holder_height
    _z_lb: ClassVar[float] = 0.2
    _stick_z_len: ClassVar[float] = stick_height
    _button_z_len: ClassVar[float] = button_radius * 0.25
    _button_press_z_offset: ClassVar[float] = _button_z_len + 0.035
    _out_of_view_xy: ClassVar[Sequence[float]] = [10.0, 10.0]

    def __init__(self) -> None:
        super().__init__()
        # Types
        # The (x, y) is the center of the robot. Theta is only relevant when
        # the robot is holding the stick.
        self._robot_type = Type("robot", ["x", "y", "theta"])
        # The (x, y) is the center of the button.
        self._button_type = Type("button", ["x", "y", "pressed"])
        # The (x, y) is the bottom left-hand corner of the stick, and theta
        # is CCW angle in radians, consistent with utils.Rectangle.
        self._stick_type = Type("stick", ["x", "y", "theta", "held"])
        # Holds the stick up so that it can be grasped by the robot.
        self._holder_type = Type("holder", ["x", "y", "theta"])
        # Predicates
        self._Pressed = Predicate("Pressed", [self._button_type],
                                  self._Pressed_holds)
        self._StickAboveButton = Predicate(
            "StickAboveButton", [self._stick_type, self._button_type],
            self._Above_holds)
        self._RobotAboveButton = Predicate(
            "RobotAboveButton", [self._robot_type, self._button_type],
            self._Above_holds)
        self._Grasped = Predicate("Grasped",
                                  [self._robot_type, self._stick_type],
                                  self._Grasped_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._AboveNoButton = Predicate("AboveNoButton", [],
                                        self._AboveNoButton_holds)
        # Options
        self._RobotPressButton = ParameterizedOption(
            "RobotPressButton",
            types=[self._robot_type, self._button_type],
            params_space=Box(0, 1, (0, )),
            policy=self._RobotPressButton_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._RobotPressButton_terminal,
        )

        self._PickStick = ParameterizedOption(
            "PickStick",
            types=[self._robot_type, self._stick_type],
            params_space=Box(0, 1, (1, )),  # normalized w.r.t. stick width
            policy=self._PickStick_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._PickStick_terminal,
        )

        self._StickPressButton = ParameterizedOption(
            "StickPressButton",
            types=[self._robot_type, self._stick_type, self._button_type],
            params_space=Box(0, 1, (0, )),
            policy=self._StickPressButton_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._StickPressButton_terminal,
        )

        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._stick = Object("stick", self._stick_type)
        self._holder = Object("holder", self._holder_type)

        assert 0 < CFG.stick_button_holder_scale < 1
        self._holder_width = self.stick_width * CFG.stick_button_holder_scale

        # For PyBullet rendering.
        self._physics_client_id: Optional[int] = None

    @classmethod
    def get_name(cls) -> str:
        return "stick_button"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        norm_dx, norm_dy, norm_dtheta, press = action.arr
        # Actions are normalized to [-1, 1]. Denormalize them here.
        dx = norm_dx * self.max_speed
        dy = norm_dy * self.max_speed
        if CFG.stick_button_disable_angles:
            dtheta = 0.0
        else:
            dtheta = norm_dtheta * self.max_angular_speed
        # Update the robot state.
        rx = state.get(self._robot, "x")
        ry = state.get(self._robot, "y")
        rtheta = state.get(self._robot, "theta")
        new_rx = rx + dx
        new_ry = ry + dy
        new_rtheta = rtheta + dtheta
        # The robot cannot leave the reachable zone. If it tries to, raise
        # an EnvironmentFailure, which represents a terminal state.
        rad = self.robot_radius
        if not self.rz_x_lb + rad <= new_rx <= self.rz_x_ub - rad or \
           not self.rz_y_lb + rad <= new_ry <= self.rz_y_ub - rad:
            raise utils.EnvironmentFailure("Left reachable zone.")
        next_state = state.copy()
        next_state.set(self._robot, "x", new_rx)
        next_state.set(self._robot, "y", new_ry)
        next_state.set(self._robot, "theta", new_rtheta)
        robot_circ = self._object_to_geom(self._robot, next_state)

        # Check if the stick is held. If so, we need to move and rotate it.
        stick_held = state.get(self._stick, "held") > 0.5
        stick_rect = self._object_to_geom(self._stick, state)
        assert isinstance(stick_rect, utils.Rectangle)
        if stick_held:
            if not CFG.stick_button_disable_angles:
                stick_rect = stick_rect.rotate_about_point(rx, ry, dtheta)
            stick_rect = utils.Rectangle(x=(stick_rect.x + dx),
                                         y=(stick_rect.y + dy),
                                         width=stick_rect.width,
                                         height=stick_rect.height,
                                         theta=stick_rect.theta)
            next_state.set(self._stick, "x", stick_rect.x)
            next_state.set(self._stick, "y", stick_rect.y)
            next_state.set(self._stick, "theta", stick_rect.theta)

        if press > 0:
            # Check if the stick is now held for the first time.
            if not stick_held and stick_rect.intersects(robot_circ):
                # Check for a collision with the stick holder. The reason that
                # we only check for a collision here, as opposed to every
                # timestep, is that we imagine the robot moving down in the z
                # direction to pick up the stick, at which button it may
                # collide with the stick holder. On other timesteps, the robot
                # would be high enough above the holder to avoid collisions.
                holder_rect = self._object_to_geom(self._holder, state)
                if robot_circ.intersects(holder_rect):
                    # Immediately fail in case of collision.
                    raise utils.EnvironmentFailure(
                        "Collided with holder.",
                        {"offending_objects": {self._holder}})

                next_state.set(self._stick, "held", 1.0)

            # Check if any button is now pressed.
            tip_rect = self._stick_rect_to_tip_rect(stick_rect)
            for button in state.get_objects(self._button_type):
                circ = self._object_to_geom(button, state)
                if (circ.intersects(tip_rect) and stick_held) or \
                   (circ.intersects(robot_circ) and not stick_held):
                    next_state.set(button, "pressed", 1.0)

        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(
            num=CFG.num_train_tasks,
            num_button_lst=CFG.stick_button_num_buttons_train,
            rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(
            num=CFG.num_test_tasks,
            num_button_lst=CFG.stick_button_num_buttons_test,
            rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Pressed, self._RobotAboveButton, self._StickAboveButton,
            self._Grasped, self._HandEmpty, self._AboveNoButton
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Pressed}

    @property
    def types(self) -> Set[Type]:
        return {
            self._holder_type, self._robot_type, self._stick_type,
            self._button_type
        }

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {
            self._RobotPressButton, self._PickStick, self._StickPressButton
        }

    @property
    def action_space(self) -> Box:
        # Normalized dx, dy, dtheta, press.
        return Box(low=-1., high=1., shape=(4, ), dtype=np.float32)

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        if CFG.stick_button_render_mode == "matplotlib":
            return super().render_state(state, task, action, caption)
        assert CFG.stick_button_render_mode == "pybullet"
        assert CFG.stick_button_disable_angles
        return self._render_state_pybullet(state, task, action, caption)

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        figsize = (10 * (self.x_ub - self.x_lb), 10 * (self.y_ub - self.y_lb))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.suptitle(caption, wrap=True)
        # Draw a light green rectangle for the reachable zone.
        reachable_zone = utils.Rectangle(x=self.rz_x_lb,
                                         y=self.rz_y_lb,
                                         width=(self.rz_x_ub - self.rz_x_lb),
                                         height=(self.rz_y_ub - self.rz_y_lb),
                                         theta=0)
        reachable_zone.plot(ax, color="lightgreen", alpha=0.25)
        # Draw the buttons.
        for button in state.get_objects(self._button_type):
            color = "blue" if state.get(button, "pressed") > 0.5 else "yellow"
            circ = self._object_to_geom(button, state)
            circ.plot(ax, facecolor=color, edgecolor="black", alpha=0.75)
        # Draw the holder.
        holder, = state.get_objects(self._holder_type)
        rect = self._object_to_geom(holder, state)
        assert isinstance(rect, utils.Rectangle)
        rect.plot(ax, color="gray")
        # Draw the stick.
        stick, = state.get_objects(self._stick_type)
        rect = self._object_to_geom(stick, state)
        assert isinstance(rect, utils.Rectangle)
        color = "black" if state.get(stick, "held") > 0.5 else "white"
        rect.plot(ax, facecolor="firebrick", edgecolor=color)
        rect = self._stick_rect_to_tip_rect(rect)
        rect.plot(ax, facecolor="saddlebrown", edgecolor=color)
        # Uncomment for debugging.
        # tx, ty = self._get_stick_grasp_loc(state, stick, np.array([0.1]))
        # circ = utils.Circle(tx, ty, radius=0.025)
        # circ.plot(ax, color="black")
        # Draw the robot.
        robot, = state.get_objects(self._robot_type)
        circ = self._object_to_geom(robot, state)
        assert isinstance(circ, utils.Circle)
        circ.plot(ax, facecolor="red", edgecolor="black")
        # Show the direction that the robot is facing.
        if not CFG.stick_button_disable_angles:
            theta = state.get(robot, "theta")
            l = 1.5 * self.robot_radius  # arrow length
            w = 0.1 * self.robot_radius  # arrow width
            ax.arrow(circ.x,
                     circ.y,
                     l * np.cos(theta),
                     l * np.sin(theta),
                     width=w)
        ax.set_xlim(self.x_lb, self.x_ub)
        ax.set_ylim(self.y_lb, self.y_ub)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int, num_button_lst: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        for _ in range(num):
            state_dict = {}
            num_buttons = num_button_lst[rng.choice(len(num_button_lst))]
            buttons = [
                Object(f"button{i}", self._button_type)
                for i in range(num_buttons)
            ]
            goal = {GroundAtom(self._Pressed, [p]) for p in buttons}
            # Sample initial positions for buttons, making sure to keep them
            # far enough apart from one another.
            collision_geoms: Set[utils.Circle] = set()
            radius = self.button_radius + self.init_padding
            for button in buttons:
                # Assuming that the dimensions are forgiving enough that
                # infinite loops are impossible.
                while True:
                    x = rng.uniform(self.x_lb + radius, self.x_ub - radius)
                    y = rng.uniform(self.y_lb + radius, self.y_ub - radius)
                    geom = utils.Circle(x, y, radius)
                    # Keep only if no intersections with existing objects.
                    if not any(geom.intersects(g) for g in collision_geoms):
                        break
                collision_geoms.add(geom)
                state_dict[button] = {"x": x, "y": y, "pressed": 0.0}
            # Sample an initial position for the robot, making sure that it
            # doesn't collide with buttons and that it's in the reachable zone.
            radius = self.robot_radius + self.init_padding
            while True:
                x = rng.uniform(self.rz_x_lb + radius, self.rz_x_ub - radius)
                y = rng.uniform(self.rz_y_lb + radius, self.rz_y_ub - radius)
                geom = utils.Circle(x, y, radius)
                # Keep only if no intersections with existing objects.
                if not any(geom.intersects(g) for g in collision_geoms):
                    break
            collision_geoms.add(geom)
            if CFG.stick_button_disable_angles:
                theta = np.pi / 2
            else:
                theta = rng.uniform(self.theta_lb, self.theta_ub)
            state_dict[self._robot] = {"x": x, "y": y, "theta": theta}
            # Sample the stick, making sure that the origin is in the
            # reachable zone, and that the stick doesn't collide with anything.
            radius = self.robot_radius + self.init_padding
            while True:
                # The radius here is to prevent the stick from being very
                # slightly in the reachable zone, but not grabbable.
                x = rng.uniform(self.rz_x_lb + radius, self.rz_x_ub - radius)
                y = rng.uniform(self.stick_init_lb, self.stick_init_ub)
                assert self.rz_y_lb + radius <= y <= self.rz_y_ub - radius
                if CFG.stick_button_disable_angles:
                    theta = np.pi / 2
                else:
                    theta = rng.uniform(self.theta_lb, self.theta_ub)
                rect = utils.Rectangle(x, y, self.stick_width,
                                       self.stick_height, theta)
                # Keep only if no intersections with existing objects.
                if not any(rect.intersects(g) for g in collision_geoms):
                    break
            state_dict[self._stick] = {
                "x": x,
                "y": y,
                "theta": theta,
                "held": 0.0
            }
            # Create the holder for the stick, sampling the position so that it
            # is somewhere along the long dimension of the stick. To make sure
            # that the problem is solvable, check that if the stick were
            # grasped at the lowest reachable position, it would still be
            # able to press the highest button.
            max_button_y = max(state_dict[p]["y"] for p in buttons)
            necessary_reach = max_button_y - self.rz_y_ub
            while True:
                # Allow the stick to start in the middle of the holder.
                x_offset = rng.uniform(-self._holder_width,
                                       self.stick_width / 2)
                # Check solvability.
                # Case 0: If all buttons are within reach, we're all set.
                if necessary_reach < 0:
                    break
                # Case 1: we can grasp the stick from the bottom.
                if x_offset > 2 * self.robot_radius:
                    break
                # Case 2: we can grasp the stick above the holder, but we can
                # still reach the highest button.
                min_rel_grasp = x_offset + self._holder_width
                grasp_to_top = self.stick_width - min_rel_grasp
                if grasp_to_top > necessary_reach:
                    break
            # First orient the rectangle at 0 and then rotate it.
            # Along the shorter dimension, we want the stick to be in the
            # center of the holder, so we need to translate the holder's y
            # position relative to the stick's y position.
            assert self.holder_height > self.stick_height
            height_diff = self.holder_height - self.stick_height
            holder_rect = utils.Rectangle(
                x=x + x_offset,
                y=(y - height_diff / 2),
                width=self._holder_width,
                height=self.holder_height,
                theta=0,
            )
            holder_rect = holder_rect.rotate_about_point(x, y, theta)
            state_dict[self._holder] = {
                "x": holder_rect.x,
                "y": holder_rect.y,
                "theta": holder_rect.theta,
            }
            init_state = utils.create_state_from_dict(state_dict)
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    def _object_to_geom(self, obj: Object, state: State) -> _Geom2D:
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        if obj.is_instance(self._robot_type):
            return utils.Circle(x, y, self.robot_radius)
        if obj.is_instance(self._button_type):
            return utils.Circle(x, y, self.button_radius)
        if obj.is_instance(self._holder_type):
            theta = state.get(obj, "theta")
            return utils.Rectangle(x=x,
                                   y=y,
                                   width=self._holder_width,
                                   height=self.holder_height,
                                   theta=theta)
        assert obj.is_instance(self._stick_type)
        theta = state.get(obj, "theta")
        return utils.Rectangle(x=x,
                               y=y,
                               width=self.stick_width,
                               height=self.stick_height,
                               theta=theta)

    def _stick_rect_to_tip_rect(
            self, stick_rect: utils.Rectangle) -> utils.Rectangle:
        theta = stick_rect.theta
        width = self.stick_tip_width
        scale = stick_rect.width - width
        return utils.Rectangle(x=(stick_rect.x + scale * np.cos(theta)),
                               y=(stick_rect.y + scale * np.sin(theta)),
                               width=self.stick_tip_width,
                               height=stick_rect.height,
                               theta=theta)

    def _get_stick_grasp_loc(self, state: State, stick: Object,
                             params: Array) -> Tuple[float, float]:
        stheta = state.get(stick, "theta")
        # We always aim for the center of the shorter dimension. The params
        # selects a position along the longer dimension.
        h = self.stick_height
        sx = state.get(stick, "x") + (h / 2) * np.cos(stheta + np.pi / 2)
        sy = state.get(stick, "y") + (h / 2) * np.sin(stheta + np.pi / 2)
        # Calculate the target button to reach based on the parameter.
        pick_param, = params
        scale = self.stick_width * pick_param
        tx = sx + scale * np.cos(stheta)
        ty = sy + scale * np.sin(stheta)
        return (tx, ty)

    def _RobotPressButton_policy(self, state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
        del memory, params  # unused
        # If the robot and button are already pressing, press.
        if self._Above_holds(state, objects):
            return Action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        # Otherwise, move toward the button.
        robot, button = objects
        rx = state.get(robot, "x")
        ry = state.get(robot, "y")
        px = state.get(button, "x")
        py = state.get(button, "y")
        dx = np.clip(px - rx, -self.max_speed, self.max_speed)
        dy = np.clip(py - ry, -self.max_speed, self.max_speed)
        # Normalize.
        dx = dx / self.max_speed
        dy = dy / self.max_speed
        # No need to rotate, and we don't want to press until we're there.
        return Action(np.array([dx, dy, 0.0, -1.0], dtype=np.float32))

    def _RobotPressButton_terminal(self, state: State, memory: Dict,
                                   objects: Sequence[Object],
                                   params: Array) -> bool:
        del memory, params  # unused
        _, button = objects
        return self._Pressed_holds(state, [button])

    def _PickStick_policy(self, state: State, memory: Dict,
                          objects: Sequence[Object], params: Array) -> Action:
        del memory  # unused
        robot, stick = objects
        rx = state.get(robot, "x")
        ry = state.get(robot, "y")
        tx, ty = self._get_stick_grasp_loc(state, stick, params)
        # If we're close enough to the grasp button, press.
        if (tx - rx)**2 + (ty - ry)**2 < self.pick_grasp_tol:
            return Action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        # Move toward the target.
        dx = np.clip(tx - rx, -self.max_speed, self.max_speed)
        dy = np.clip(ty - ry, -self.max_speed, self.max_speed)
        # Normalize.
        dx = dx / self.max_speed
        dy = dy / self.max_speed
        # No need to rotate or press.
        return Action(np.array([dx, dy, 0.0, -1.0], dtype=np.float32))

    def _PickStick_terminal(self, state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        return self._Grasped_holds(state, objects)

    def _StickPressButton_policy(self, state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
        del memory, params  # unused
        _, stick, button = objects
        button_circ = self._object_to_geom(button, state)
        stick_rect = self._object_to_geom(self._stick, state)
        assert isinstance(stick_rect, utils.Rectangle)
        tip_rect = self._stick_rect_to_tip_rect(stick_rect)
        # If the stick tip is pressing the button, press.
        if tip_rect.intersects(button_circ):
            return Action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        # If the stick is vertical, move the tip toward the button.
        stheta = state.get(stick, "theta")
        desired_theta = np.pi / 2
        if abs(stheta - desired_theta) < 1e-3:
            tx = tip_rect.x
            ty = tip_rect.y
            px = state.get(button, "x")
            py = state.get(button, "y")
            dx = np.clip(px - tx, -self.max_speed, self.max_speed)
            dy = np.clip(py - ty, -self.max_speed, self.max_speed)
            # Normalize.
            dx = dx / self.max_speed
            dy = dy / self.max_speed
            # No need to rotate or press.
            return Action(np.array([dx, dy, 0.0, -1.0], dtype=np.float32))
        assert not CFG.stick_button_disable_angles
        # Otherwise, rotate the stick.
        dtheta = np.clip(desired_theta - stheta, -self.max_angular_speed,
                         self.max_angular_speed)
        # Normalize.
        dtheta = dtheta / self.max_angular_speed
        return Action(np.array([0.0, 0.0, dtheta, -1.0], dtype=np.float32))

    def _StickPressButton_terminal(self, state: State, memory: Dict,
                                   objects: Sequence[Object],
                                   params: Array) -> bool:
        del memory, params  # unused
        _, _, button = objects
        return self._Pressed_holds(state, [button])

    @staticmethod
    def _Pressed_holds(state: State, objects: Sequence[Object]) -> bool:
        button, = objects
        return state.get(button, "pressed") > 0.5

    def _Above_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        geom1 = self._object_to_geom(obj1, state)
        geom2 = self._object_to_geom(obj2, state)
        return geom1.intersects(geom2)

    @staticmethod
    def _Grasped_holds(state: State, objects: Sequence[Object]) -> bool:
        _, stick = objects
        return state.get(stick, "held") > 0.5

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        robot, = objects
        stick, = state.get_objects(self._stick_type)
        return not self._Grasped_holds(state, [robot, stick])

    def _AboveNoButton_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        assert not objects
        robot, = state.get_objects(self._robot_type)
        stick, = state.get_objects(self._stick_type)
        for button in state.get_objects(self._button_type):
            if self._Above_holds(state, [robot, button]):
                return False
            if self._Above_holds(state, [stick, button]):
                return False
        return True

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:
        assert CFG.stick_button_disable_angles
        logging.info("Controls: mouse click to move, any key to press")

        def _event_to_action(state: State,
                             event: matplotlib.backend_bases.Event) -> Action:
            if event.key is not None:
                # Press action.
                return Action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            # Move action.
            rx = state.get(self._robot, "x")
            ry = state.get(self._robot, "y")
            tx = event.xdata
            ty = event.ydata
            assert tx is not None and ty is not None, "Out-of-bounds click"
            # Move toward the target.
            dx = np.clip(tx - rx, -self.max_speed, self.max_speed)
            dy = np.clip(ty - ry, -self.max_speed, self.max_speed)
            # Normalize.
            dx = dx / self.max_speed
            dy = dy / self.max_speed
            # No need to rotate or press.
            return Action(np.array([dx, dy, 0.0, -1.0], dtype=np.float32))

        return _event_to_action

    def _render_state_pybullet(self,
                               state: State,
                               task: Task,
                               action: Optional[Action] = None,
                               caption: Optional[str] = None) -> Video:
        # NOTE: the axes are unfortunately not the same here. In the original
        # environment, the x axis is the negative y axis, and the y axis is the
        # x axis of the pybullet environment.
        assert CFG.pybullet_control_mode == "reset"

        if self._physics_client_id is None:
            self._initialize_pybullet()

        # Update based on the input state.
        self._update_pybullet_from_state(state)

        # Take the first image.
        imgs = [self._capture_pybullet_image()]

        if action is None:
            return imgs

        next_state = self.simulate(state, action)
        hit_special_case = False

        # Case 1: robot is pressing the button directly.
        for button in state.get_objects(self._button_type):
            if self._Above_holds(next_state, [self._robot, button]) and \
               not self._Pressed_holds(state, [button]) and \
               self._Pressed_holds(next_state, [button]):
                hit_special_case = True

                # Move down to press.
                target = (
                    state.get(button, "y"),
                    state.get(button, "x"),
                    self._z_lb + self._button_press_z_offset,
                )
                self._pybullet_move_robot_to_target(target, imgs,
                    self._pybullet_robot.closed_fingers)

                # Change the button color.
                button_to_button_id = {b: bid for bid, b in self._button_id_to_button.items()}
                button_id = button_to_button_id[button]
                color = (0.2, 0.9, 0.2, 1.0)
                p.changeVisualShape(button_id, -1, rgbaColor=color,
                    physicsClientId=self._physics_client_id)

                # Move up.
                target = (
                    next_state.get(self._robot, "y"),
                    next_state.get(self._robot, "x"),
                    self.robot_init_z
                )
                self._pybullet_move_robot_to_target(target, imgs,
                    self._pybullet_robot.closed_fingers)

        # Case 2: picking up the stick.
        if not self._Grasped_holds(state, [self._robot, self._stick]) and \
            self._Grasped_holds(next_state, [self._robot, self._stick]):
            hit_special_case = True

            # Move down to pick.
            target = (
                state.get(self._robot, "y"),
                state.get(self._stick, "x") + self.stick_height / 2,
                self._z_lb + self._holder_base_z_len,
            )
            self._pybullet_move_robot_to_target(target, imgs,
                self._pybullet_robot.open_fingers)

            # Create a grasp constraint.
            base_link_to_world = np.r_[p.invertTransform(*p.getLinkState(
                    self._pybullet_robot.robot_id,
                self._pybullet_robot.end_effector_id,
                physicsClientId=self._physics_client_id)[:2])]
            world_to_obj = np.r_[p.getBasePositionAndOrientation(
                self._stick_id, physicsClientId=self._physics_client_id)]
            self._held_obj_to_base_link = p.invertTransform(
                *p.multiplyTransforms(base_link_to_world[:3],
                                      base_link_to_world[3:],
                                      world_to_obj[:3], world_to_obj[3:]))

            # Move back up.
            target = (
                next_state.get(self._robot, "y"),
                next_state.get(self._robot, "x"),
                self.robot_init_z
            )
            self._pybullet_move_robot_to_target(target, imgs,
                self._pybullet_robot.closed_fingers)

            # while True:
            #     p.stepSimulation(physicsClientId=self._physics_client_id)

        # Case 3: stick pressing button
        for button in state.get_objects(self._button_type):
            if self._Above_holds(next_state, [self._stick, button]) and \
               not self._Pressed_holds(state, [button]) and \
               self._Pressed_holds(next_state, [button]):
                hit_special_case = True

                # Move down to press.
                target = (
                    state.get(self._robot, "y"),
                    state.get(self._robot, "x") + self.button_radius / 2,
                    self._z_lb + self._button_press_z_offset - 0.01,
                )
                self._pybullet_move_robot_to_target(target, imgs,
                    self._pybullet_robot.closed_fingers)

                # Change the button color.
                button_to_button_id = {b: bid for bid, b in self._button_id_to_button.items()}
                button_id = button_to_button_id[button]
                color = (0.2, 0.9, 0.2, 1.0)
                p.changeVisualShape(button_id, -1, rgbaColor=color,
                    physicsClientId=self._physics_client_id)

                # Move up.
                target = (
                    next_state.get(self._robot, "y"),
                    next_state.get(self._robot, "x"),
                    self.robot_init_z
                )
                self._pybullet_move_robot_to_target(target, imgs,
                    self._pybullet_robot.closed_fingers)

        # Case 4: just moving normally
        if not hit_special_case:
            target = (
                next_state.get(self._robot, "y"),
                next_state.get(self._robot, "x"),
                self.robot_init_z,
            )
            self._pybullet_move_robot_to_target(target, imgs,
                self._state_to_fingers(next_state))


        # TODO
        return imgs


    def _initialize_pybullet(self) -> None:
        self._physics_client_id = p.connect(p.DIRECT)
        # Disable the preview windows for faster rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,
                                   False,
                                   physicsClientId=self._physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                                   False,
                                   physicsClientId=self._physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                                   False,
                                   physicsClientId=self._physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                                   False,
                                   physicsClientId=self._physics_client_id)
        p.resetDebugVisualizerCamera(self._camera_distance,
                                     self._camera_yaw,
                                     self._camera_pitch,
                                     self._camera_target,
                                     physicsClientId=self._physics_client_id)
        p.resetSimulation(physicsClientId=self._physics_client_id)

        # Load plane.
        p.loadURDF(utils.get_env_asset_path("urdf/plane.urdf"), [0, 0, -1],
                   useFixedBase=True,
                   physicsClientId=self._physics_client_id)

        # Load robot.
        ee_home = (self.robot_init_x, self.robot_init_y, self.robot_init_z)
        ee_orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
        self._pybullet_robot = create_single_arm_pybullet_robot(
            CFG.pybullet_robot, ee_home, ee_orn, self._physics_client_id)

        # Load table.
        self._table_id = p.loadURDF(
            utils.get_env_asset_path("urdf/extended_table.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._table_id,
            self._table_pose,
            self._table_orientation,
            physicsClientId=self._physics_client_id)
        p.changeVisualShape(self._table_id, -1, rgbaColor=(0.1, 0.1, 0.1, 1.0),
            physicsClientId=self._physics_client_id)

        # Load stick holder.
        self._holder_id = self._create_pybullet_stick_holder()

        # Load stick.
        self._stick_id = self._create_pybullet_stick()

        # Load buttons.
        self._button_ids = self._create_pybullet_buttons()

        # while True:
        #     p.stepSimulation(physicsClientId=self._physics_client_id)

    def _create_pybullet_stick_holder(self) -> int:
        color = (0.7, 0.7, 0.7, 1.0)

        # Create the main holder body.
        main_half_extents = (
            self._holder_width / 2,
            self.holder_height / 2,
            self._holder_base_z_len / 2,
        )
        main_collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=main_half_extents,
            physicsClientId=self._physics_client_id)

        main_visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=main_half_extents,
            rgbaColor=color,
            physicsClientId=self._physics_client_id)

        x = (self.stick_init_lb + self.stick_init_ub) / 2 + self._holder_width / 2
        main_y = (self.rz_x_lb + self.rz_x_ub) / 2 + self.holder_height / 2
        z = self._z_lb + self._holder_base_z_len / 2
        main_pose = (x, main_y, z)
        main_orientation = self._default_obj_orn

        # Create the sides.
        side_height = self._holder_side_height
        side_half_extents = (
            self._holder_width / 2,
            side_height / 2,
            self._holder_side_z_len / 2,
        )
        side_collision_ids = []
        side_visual_ids = []
        side_positions = []
        side_orientations = []
        for y_offset in [self.holder_height / 2 + side_height / 2,
                         -(self.holder_height / 2 + side_height / 2)]:
            side_collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=side_half_extents,
                physicsClientId=self._physics_client_id)
            side_collision_ids.append(side_collision_id)

            side_visual_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=side_half_extents,
                rgbaColor=color,
                physicsClientId=self._physics_client_id)
            side_visual_ids.append(side_visual_id)

            x = 0
            y = y_offset
            z = self._holder_side_z_len / 2 - self._holder_base_z_len / 2
            side_pose = (x, y, z)
            side_positions.append(side_pose)
            side_orientation = self._default_obj_orn
            side_orientations.append(side_orientation)

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=main_collision_id,
            baseVisualShapeIndex=main_visual_id,
            basePosition=main_pose,
            baseOrientation=main_orientation,
            linkMasses=[0, 0],
            linkCollisionShapeIndices=side_collision_ids,
            linkVisualShapeIndices=side_visual_ids,
            linkPositions=side_positions,
            linkOrientations=side_orientations,
            linkParentIndices=[0, 0],
            linkInertialFramePositions=[(0, 0, 0), (0, 0, 0)],
            linkInertialFrameOrientations=[(0, 0, 0, 1), (0, 0, 0, 1)],
            linkJointAxis=[(0, 0, 0), (0, 0, 0)],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
            physicsClientId=self._physics_client_id)

    def _create_pybullet_stick(self) -> int:
        main_half_extents = (
            self.stick_width / 2,
            self.stick_height / 2,
            self._stick_z_len / 2,
        )
        main_collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=main_half_extents,
            physicsClientId=self._physics_client_id)

        main_visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=main_half_extents,
            physicsClientId=self._physics_client_id)

        height_diff = self.holder_height - self.stick_height
        x = (self.stick_init_lb + self.stick_init_ub) / 2 + self.stick_width / 2
        y = (self.rz_x_lb + self.rz_x_ub) / 2 + self.stick_height / 2 + height_diff / 2
        z = self._z_lb + self._stick_z_len / 2 + self._holder_base_z_len
        main_pose = (x, y, z)
        main_orientation = self._default_obj_orn

        stick_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=main_collision_id,
            baseVisualShapeIndex=main_visual_id,
            basePosition=main_pose,
            baseOrientation=main_orientation,
            physicsClientId=self._physics_client_id)

        texture_id = p.loadTexture(utils.get_env_asset_path("urdf/wood.png"))
        p.changeVisualShape(stick_id, -1, textureUniqueId=texture_id)

        return stick_id

    def _create_pybullet_buttons(self) -> List[int]:
        num_buttons = max(max(CFG.stick_button_num_buttons_train),
                          max(CFG.stick_button_num_buttons_test))

        button_ids = []

        for i in range(num_buttons):
            collision_id = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=self.button_radius,
                physicsClientId=self._physics_client_id)

            # Create the visual_shape.
            visual_id = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=self.button_radius,
                rgbaColor=(0.9, 0.2, 0.2, 1.0),
                physicsClientId=self._physics_client_id)

            # Create the body.
            delta_x = (self.x_ub - self.x_lb) / (num_buttons - 1)
            pose = (
                (self.y_lb + self.y_ub) / 2,
                self.x_lb + i * delta_x,
                self._z_lb + 2*self.button_radius,
            )
            
            # Facing outward.
            orientation = self._default_obj_orn
            button_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=pose,
                baseOrientation=orientation,
                physicsClientId=self._physics_client_id)

            button_ids.append(button_id)

        return button_ids

    def _capture_pybullet_image(self) -> Image:

        camera_distance = self._camera_distance
        camera_yaw = self._camera_yaw
        camera_pitch = self._camera_pitch            

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._camera_target,
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._physics_client_id)

        width = CFG.pybullet_camera_width
        height = CFG.pybullet_camera_height

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width / height),
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self._physics_client_id)

        (_, _, px, _,
         _) = p.getCameraImage(width=width,
                               height=height,
                               viewMatrix=view_matrix,
                               projectionMatrix=proj_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL,
                               physicsClientId=self._physics_client_id)

        rgb_array = np.array(px).reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _update_pybullet_from_state(self, state: State) -> None:
        # Reset buttons based on the state.
        buttons = state.get_objects(self._button_type)
        self._button_id_to_button = {}
        for i, button in enumerate(buttons):
            button_id = self._button_ids[i]
            self._button_id_to_button[button_id] = button
            bx = state.get(button, "y")
            by = state.get(button, "x")
            bz = self._z_lb + self._button_z_len / 2
            p.resetBasePositionAndOrientation(
                button_id, [bx, by, bz],
                self._default_obj_orn,
                physicsClientId=self._physics_client_id)
            # Change the color if pressed.
            if self._Pressed_holds(state, [button]):
                color = (0.2, 0.9, 0.2, 1.0)
            else:
                color = (0.9, 0.2, 0.2, 1.0)
            p.changeVisualShape(button_id, -1, rgbaColor=color,
                physicsClientId=self._physics_client_id)

        # For any buttons not involved, put them out of view.
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(buttons), len(self._button_ids)):
            button_id = self._button_ids[i]
            assert button_id not in self._button_id_to_button
            p.resetBasePositionAndOrientation(
                button_id, [oov_x, oov_y, self._z_lb],
                self._default_obj_orn,
                physicsClientId=self._physics_client_id)

        # Update the stick holder.
        x = state.get(self._holder, "y") + self._holder_width / 2
        y = state.get(self._holder, "x") - self._holder_side_height / 2
        z = self._z_lb + self._holder_base_z_len / 2
        p.resetBasePositionAndOrientation(
            self._holder_id, [x, y, z],
            self._default_obj_orn,
            physicsClientId=self._physics_client_id)

        # Update the robot.
        self._pybullet_robot.reset_state(self._extract_robot_state(state))

        # If we are currently holding the stick, create a constraint.
        if self._Grasped_holds(state, [self._robot, self._stick]):
            if self._held_obj_to_base_link is None:
                base_link_to_world = np.r_[p.invertTransform(*p.getLinkState(
                    self._pybullet_robot.robot_id,
                    self._pybullet_robot.end_effector_id,
                    physicsClientId=self._physics_client_id)[:2])]
                world_to_obj = np.r_[p.getBasePositionAndOrientation(
                    self._stick_id, physicsClientId=self._physics_client_id)]
                self._held_obj_to_base_link = p.invertTransform(
                    *p.multiplyTransforms(base_link_to_world[:3],
                                          base_link_to_world[3:],
                                          world_to_obj[:3], world_to_obj[3:]))
        # Otherwise, update the state of the stick.
        else:
            self._held_obj_to_base_link = None
            x = state.get(self._stick, "y") + self.stick_width / 2
            y = state.get(self._stick, "x") + self.stick_height / 2
            z = self._z_lb + self._stick_z_len / 2 + self._holder_base_z_len
            p.resetBasePositionAndOrientation(
                self._stick_id, [x, y, z],
                self._default_obj_orn,
                physicsClientId=self._physics_client_id)

        # Update the held object.
        if self._held_obj_to_base_link:
            world_to_base_link = p.getLinkState(
                self._pybullet_robot.robot_id,
                self._pybullet_robot.end_effector_id,
                physicsClientId=self._physics_client_id)[:2]
            base_link_to_held_obj = p.invertTransform(
                *self._held_obj_to_base_link)
            world_to_held_obj = p.multiplyTransforms(
                world_to_base_link[0], world_to_base_link[1],
                base_link_to_held_obj[0], base_link_to_held_obj[1])
            p.resetBasePositionAndOrientation(
                self._stick_id,
                world_to_held_obj[0],
                world_to_held_obj[1],
                physicsClientId=self._physics_client_id)

        # while True:
        #     p.stepSimulation(physicsClientId=self._physics_client_id)


    def _extract_robot_state(self, state: State) -> Array:
        return np.array([
            state.get(self._robot, "y"),
            state.get(self._robot, "x"),
            self.robot_init_z,
            self._state_to_fingers(state),
        ],
                        dtype=np.float32)

    def _state_to_fingers(self, state: State) -> float:
        if self._Grasped_holds(state, [self._robot, self._stick]):
            return self._pybullet_robot.closed_fingers
        return self._pybullet_robot.open_fingers

    def _pybullet_move_robot_to_target(self, target: Pose3D, imgs: Video, finger_joint: float) -> None:

        while True:
            rx, ry, rz, _ = self._pybullet_robot.get_state()
            current = (rx, ry, rz)
            sq_dist = np.sum(np.square(np.subtract(current, target)))
            if sq_dist < self._pybullet_move_to_pose_tol:
                break
            # Run IK to determine the target joint positions.
            ee_delta = np.subtract(target, current)
            # Reduce the target to conform to the max velocity constraint.
            ee_norm = np.linalg.norm(ee_delta)
            if ee_norm > self._pybullet_max_vel_norm:
                ee_delta = ee_delta * self._pybullet_max_vel_norm / ee_norm
            ee_action = np.add(current, ee_delta)
            joints_state = self._pybullet_robot.inverse_kinematics(
                (ee_action[0], ee_action[1], ee_action[2]),
                validate=True)
            # Override the meaningless finger values in joint_action.
            joints_state[
                self._pybullet_robot.left_finger_joint_idx] = finger_joint
            joints_state[
                self._pybullet_robot.right_finger_joint_idx] = finger_joint
            action_arr = np.array(joints_state, dtype=np.float32)
            # This clipping is needed sometimes for the joint limits.
            action_arr = np.clip(action_arr,
                                 self._pybullet_robot.action_space.low,
                                 self._pybullet_robot.action_space.high)
            assert self._pybullet_robot.action_space.contains(action_arr)

            # Take action in PyBullet.
            self._pybullet_robot.set_motors(action_arr.tolist())

            # Update the held object.
            if self._held_obj_to_base_link:
                world_to_base_link = p.getLinkState(
                    self._pybullet_robot.robot_id,
                    self._pybullet_robot.end_effector_id,
                    physicsClientId=self._physics_client_id)[:2]
                base_link_to_held_obj = p.invertTransform(
                    *self._held_obj_to_base_link)
                world_to_held_obj = p.multiplyTransforms(
                    world_to_base_link[0], world_to_base_link[1],
                    base_link_to_held_obj[0], base_link_to_held_obj[1])
                p.resetBasePositionAndOrientation(
                    self._stick_id,
                    world_to_held_obj[0],
                    world_to_held_obj[1],
                    physicsClientId=self._physics_client_id)
            
            # Take an image.
            imgs.append(self._capture_pybullet_image())
