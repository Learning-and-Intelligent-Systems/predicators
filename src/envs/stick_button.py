"""An environment where a robot must press buttons with its hand or a stick."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type
from predicators.src.utils import _Geom2D


class StickButtonEnv(BaseEnv):
    """An environment where a robot must press buttons with its hand or a
    stick."""
    x_lb: ClassVar[float] = 0.0
    y_lb: ClassVar[float] = 0.0
    theta_lb: ClassVar[float] = -np.pi  # radians
    x_ub: ClassVar[float] = 10.0
    y_ub: ClassVar[float] = 6.0
    theta_ub: ClassVar[float] = np.pi  # radians
    # Reachable zone boundaries.
    rz_x_lb: ClassVar[float] = x_lb
    rz_x_ub: ClassVar[float] = x_ub
    rz_y_lb: ClassVar[float] = y_lb
    rz_y_ub: ClassVar[float] = y_lb + 3.0
    max_speed: ClassVar[float] = 0.5  # shared by dx, dy
    max_angular_speed: ClassVar[float] = np.pi / 4
    robot_radius: ClassVar[float] = 0.1
    button_radius: ClassVar[float] = 0.1
    # Note that the stick_width is the longer dimension.
    stick_width: ClassVar[float] = 3.0
    stick_height: ClassVar[float] = 0.05
    # Note that the holder width is set in the class because it uses CFG.
    holder_height: ClassVar[float] = 2.5 * stick_height
    stick_tip_width: ClassVar[float] = 0.05
    init_padding: ClassVar[float] = 0.5  # used to space objects in init states
    stick_init_lb: ClassVar[float] = 0.6
    stick_init_ub: ClassVar[float] = 1.6  # start low in the reachable zone
    pick_grasp_tol: ClassVar[float] = 1e-3

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

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        figsize = (self.x_ub - self.x_lb, self.y_ub - self.y_lb)
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

    def event_to_action(self, state: State,
                        event: matplotlib.backend_bases.Event) -> Action:
        """Controls: mouse click to move, any key to press.
        """
        if event.key is not None:
            # Press action.
            return Action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        # Move action.
        rx = state.get(self._robot, "x")
        ry = state.get(self._robot, "y")
        tx = event.xdata
        ty = event.ydata
        # Move toward the target.
        dx = np.clip(tx - rx, -self.max_speed, self.max_speed)
        dy = np.clip(ty - ry, -self.max_speed, self.max_speed)
        # Normalize.
        dx = dx / self.max_speed
        dy = dy / self.max_speed
        # No need to rotate or press.
        return Action(np.array([dx, dy, 0.0, -1.0], dtype=np.float32))
