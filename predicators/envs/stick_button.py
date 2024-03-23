"""An environment where a robot must press buttons with its hand or a stick."""

import logging
from typing import Callable, ClassVar, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type
from predicators.utils import Rectangle, _Geom2D


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

    # Types
    # The (x, y) is the center of the robot. Theta is only relevant when
    # the robot is holding the stick.
    _robot_type = Type("robot", ["x", "y", "theta"])
    # The (x, y) is the center of the button.
    _button_type = Type("button", ["x", "y", "pressed"])
    # The (x, y) is the bottom left-hand corner of the stick, and theta
    # is CCW angle in radians, consistent with utils.Rectangle.
    _stick_type = Type("stick", ["x", "y", "theta", "held"])
    # Holds the stick up so that it can be grasped by the robot.
    _holder_type = Type("holder", ["x", "y", "theta"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates
        self._Pressed = Predicate("Pressed", [self._button_type],
                                  self._Pressed_holds)
        self._StickAboveButton = Predicate(
            "StickAboveButton", [self._stick_type, self._button_type],
            self.Above_holds)
        self._RobotAboveButton = Predicate(
            "RobotAboveButton", [self._robot_type, self._button_type],
            self.Above_holds)
        self._Grasped = Predicate("Grasped",
                                  [self._robot_type, self._stick_type],
                                  self._Grasped_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._AboveNoButton = Predicate("AboveNoButton", [],
                                        self._AboveNoButton_holds)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._stick = Object("stick", self._stick_type)
        self._holder = Object("holder", self._holder_type)

        assert 0 < CFG.stick_button_holder_scale < 1

    @classmethod
    def get_name(cls) -> str:
        return "stick_button"

    @classmethod
    def _get_holder_width(cls) -> float:
        return cls.stick_width * CFG.stick_button_holder_scale

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
        # The robot cannot leave the reachable zone. If it tries to, noop.
        rad = self.robot_radius
        if not self.rz_x_lb + rad <= new_rx <= self.rz_x_ub - rad or \
           not self.rz_y_lb + rad <= new_ry <= self.rz_y_ub - rad:
            return state.copy()
        next_state = state.copy()
        next_state.set(self._robot, "x", new_rx)
        next_state.set(self._robot, "y", new_ry)
        next_state.set(self._robot, "theta", new_rtheta)
        robot_circ = self.object_to_geom(self._robot, next_state)

        # Check if the stick is held. If so, we need to move and rotate it.
        stick_held = state.get(self._stick, "held") > 0.5
        stick_rect = self.object_to_geom(self._stick, state)
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
            # Check for placing the stick.
            holder_rect = self.object_to_geom(self._holder, state)
            if stick_held and stick_rect.intersects(holder_rect):
                # Place the stick back on the holder.
                next_state.set(self._stick, "held", 0.0)

            # Check if the stick is now held for the first time.
            if not stick_held and stick_rect.intersects(robot_circ):
                # Check for a collision with the stick holder. The reason that
                # we only check for a collision here, as opposed to every
                # timestep, is that we imagine the robot moving down in the z
                # direction to pick up the stick, at which button it may
                # collide with the stick holder. On other timesteps, the robot
                # would be high enough above the holder to avoid collisions.
                if robot_circ.intersects(holder_rect):
                    # No-op in case of collision.
                    return state.copy()

                next_state.set(self._stick, "held", 1.0)

            # Check if any button is now pressed.
            tip_rect = self.stick_rect_to_tip_rect(stick_rect)
            for button in state.get_objects(self._button_type):
                circ = self.object_to_geom(button, state)
                if (circ.intersects(tip_rect) and stick_held) or \
                   (circ.intersects(robot_circ) and not stick_held):
                    next_state.set(button, "pressed", 1.0)

        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num=CFG.num_train_tasks,
            num_button_lst=CFG.stick_button_num_buttons_train,
            rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
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
    def action_space(self) -> Box:
        # Normalized dx, dy, dtheta, press.
        return Box(low=-1., high=1., shape=(4, ), dtype=np.float32)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
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
            circ = self.object_to_geom(button, state)
            circ.plot(ax, facecolor=color, edgecolor="black", alpha=0.75)
        # Draw the holder.
        holder, = state.get_objects(self._holder_type)
        rect = self.object_to_geom(holder, state)
        assert isinstance(rect, utils.Rectangle)
        rect.plot(ax, color="gray")
        # Draw the stick.
        stick, = state.get_objects(self._stick_type)
        rect = self.object_to_geom(stick, state)
        assert isinstance(rect, utils.Rectangle)
        color = "black" if state.get(stick, "held") > 0.5 else "white"
        rect.plot(ax, facecolor="firebrick", edgecolor=color)
        rect = self.stick_rect_to_tip_rect(rect)
        rect.plot(ax, facecolor="saddlebrown", edgecolor=color)
        # Draw the robot.
        robot, = state.get_objects(self._robot_type)
        circ = self.object_to_geom(robot, state)
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
                   rng: np.random.Generator) -> List[EnvironmentTask]:
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
                x_offset = rng.uniform(-self._get_holder_width(),
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
                min_rel_grasp = x_offset + self._get_holder_width()
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
                width=self._get_holder_width(),
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
            task = EnvironmentTask(init_state, goal)
            tasks.append(task)
        return tasks

    @classmethod
    def object_to_geom(cls, obj: Object, state: State) -> _Geom2D:
        """Public for use by oracle options."""
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        if obj.is_instance(cls._robot_type):
            return utils.Circle(x, y, cls.robot_radius)
        if obj.is_instance(cls._button_type):
            return utils.Circle(x, y, cls.button_radius)
        if obj.is_instance(cls._holder_type):
            theta = state.get(obj, "theta")
            return utils.Rectangle(x=x,
                                   y=y,
                                   width=cls._get_holder_width(),
                                   height=cls.holder_height,
                                   theta=theta)
        theta = state.get(obj, "theta")
        return utils.Rectangle(x=x,
                               y=y,
                               width=cls.stick_width,
                               height=cls.stick_height,
                               theta=theta)

    @classmethod
    def stick_rect_to_tip_rect(cls,
                               stick_rect: utils.Rectangle) -> utils.Rectangle:
        """Public for use by oracle options."""
        theta = stick_rect.theta
        width = cls.stick_tip_width
        scale = stick_rect.width - width
        return utils.Rectangle(x=(stick_rect.x + scale * np.cos(theta)),
                               y=(stick_rect.y + scale * np.sin(theta)),
                               width=cls.stick_tip_width,
                               height=stick_rect.height,
                               theta=theta)

    @staticmethod
    def _Pressed_holds(state: State, objects: Sequence[Object]) -> bool:
        button, = objects
        return state.get(button, "pressed") > 0.5

    @classmethod
    def Above_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Public for use by oracle options."""
        assert len(objects) == 2
        obj1, obj2 = objects
        geom1 = cls.object_to_geom(obj1, state)
        geom2 = cls.object_to_geom(obj2, state)
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
            if self.Above_holds(state, [robot, button]):
                return False
            if self.Above_holds(state, [stick, button]):
                return False
        return True

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:
        assert CFG.stick_button_disable_angles
        logging.info("Controls: mouse click to move, (q) to quit, any other "
                     "key to press")

        def _event_to_action(state: State,
                             event: matplotlib.backend_bases.Event) -> Action:
            if event.key == "q":
                raise utils.HumanDemonstrationFailure("Human quit.")
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


class StickButtonMovementEnv(StickButtonEnv):
    """An extension to the stick button env that also has movement options (the
    pick and place options don't implicitly contain movement."""

    # Make x_ub smaller to make predicate invention constant finding easier.
    x_ub: ClassVar[float] = 6.0
    rz_x_ub: ClassVar[float] = x_ub
    # The (x, y) is the bottom left-hand corner of the stick, and theta
    # is CCW angle in radians, consistent with utils.Rectangle. The tip
    # x and y correspond to the end of the stick.
    _stick_type = Type("stick", ["x", "y", "tip_x", "tip_y", "theta", "held"])

    def _get_tasks(self, num: int, num_button_lst: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
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
                    # Also enforce that the button is clearly on one side
                    # of the boundary between robot's reachable vs
                    # unreachable regions to make predicate invention
                    # easier.
                    if not any(geom.intersects(g)
                               for g in collision_geoms) and abs(
                                   y - self.rz_y_ub) > radius:
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
            tip_rect = self.stick_rect_to_tip_rect(rect)
            state_dict[self._stick] = {
                "x": x,
                "y": y,
                "tip_x": tip_rect.x,
                "tip_y": tip_rect.y,
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
                x_offset = rng.uniform(-self._get_holder_width(),
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
                min_rel_grasp = x_offset + self._get_holder_width()
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
                width=self._get_holder_width(),
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
            task = EnvironmentTask(init_state, goal)
            tasks.append(task)
        return tasks

    def simulate(self, state: State, action: Action) -> State:
        """Run simulation and update tip_x and tip_y."""
        next_state = super().simulate(state, action)
        stick_rect = self.object_to_geom(self._stick, next_state)
        assert isinstance(stick_rect, Rectangle)
        tip_rect = self.stick_rect_to_tip_rect(stick_rect)
        next_state.set(self._stick, "tip_x", tip_rect.x)
        next_state.set(self._stick, "tip_y", tip_rect.y)
        return next_state

    @classmethod
    def get_name(cls) -> str:
        return "stick_button_move"
