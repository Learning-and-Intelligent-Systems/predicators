"""An environment where a robot must brew and pour coffee."""

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


class CoffeeEnv(BaseEnv):
    """An environment where a robot must brew and pour coffee."""

    # Tolerances.
    grasp_finger_tol: ClassVar[float] = 1e-2
    grasp_position_tol: ClassVar[float] = 0.5
    dispense_tol: ClassVar[float] = 1.0
    pour_angle_tol: ClassVar[float] = 1e-1
    pour_pos_tol: ClassVar[float] = 1.0
    init_padding: ClassVar[float] = 0.5  # used to space objects in init states
    pick_jug_y_padding: ClassVar[float] = 1.5
    pick_jug_rot_tol: ClassVar[float] = np.pi / 3
    safe_z_tol: ClassVar[float] = 1e-1
    twist_policy_tol: ClassVar[float] = 1e-1
    pick_policy_tol: ClassVar[float] = 1e-1
    place_jug_in_machine_tol: ClassVar[float] = 1e-1
    pour_policy_tol: ClassVar[float] = 1e-1
    # Robot settings.
    x_lb: ClassVar[float] = 0.0
    x_ub: ClassVar[float] = 10.0
    y_lb: ClassVar[float] = 0.0
    y_ub: ClassVar[float] = 10.0
    z_lb: ClassVar[float] = 0.0
    z_ub: ClassVar[float] = 10.0
    tilt_lb: ClassVar[float] = 0.0
    tilt_ub: ClassVar[float] = np.pi / 4
    wrist_lb: ClassVar[float] = -np.pi
    wrist_ub: ClassVar[float] = np.pi
    robot_init_x: ClassVar[float] = (x_ub + x_lb) / 2.0
    robot_init_y: ClassVar[float] = (y_ub + y_lb) / 2.0
    robot_init_z: ClassVar[float] = z_ub
    robot_init_tilt: ClassVar[float] = 0.0
    robot_init_wrist: ClassVar[float] = 0.0
    open_fingers: ClassVar[float] = 0.4
    closed_fingers: ClassVar[float] = 0.1
    # Machine settings.
    machine_x_len: ClassVar[float] = 0.1 * (x_ub - x_lb)
    machine_y_len: ClassVar[float] = 0.2 * (y_ub - y_lb)
    machine_z_len: ClassVar[float] = 0.4 * (z_ub - z_lb)
    machine_x: ClassVar[float] = x_ub - machine_x_len - init_padding
    machine_y: ClassVar[float] = y_ub - machine_y_len - init_padding
    button_x: ClassVar[float] = machine_x + machine_x_len / 2
    button_y: ClassVar[float] = machine_y
    button_z: ClassVar[float] = 3 * machine_z_len / 4
    button_radius: ClassVar[float] = 0.2 * machine_x_len
    # Jug settings.
    jug_radius: ClassVar[float] = (0.8 * machine_x_len) / 2.0
    jug_height: ClassVar[float] = 0.15 * (z_ub - z_lb)
    jug_init_x_lb: ClassVar[float] = machine_x - machine_x_len + init_padding
    jug_init_x_ub: ClassVar[float] = machine_x + machine_x_len - init_padding
    jug_init_y_lb: ClassVar[float] = y_lb + jug_radius + pick_jug_y_padding + \
                                     init_padding
    jug_init_y_ub: ClassVar[
        float] = machine_y - machine_y_len - jug_radius - init_padding
    jug_handle_offset: ClassVar[float] = 1.05 * jug_radius
    jug_handle_height: ClassVar[float] = 3 * jug_height / 4
    jug_handle_radius: ClassVar[float] = 1e-1  # just for rendering
    # Dispense area settings.
    dispense_area_x: ClassVar[float] = machine_x + machine_x_len / 2
    dispense_area_y: ClassVar[float] = machine_y - 1.1 * jug_radius
    # Cup settings.
    cup_radius: ClassVar[float] = 0.6 * jug_radius
    cup_init_x_lb: ClassVar[float] = x_lb + cup_radius + init_padding
    cup_init_x_ub: ClassVar[
        float] = machine_x - machine_x_len - cup_radius - init_padding
    cup_init_y_lb: ClassVar[float] = jug_init_y_lb
    cup_init_y_ub: ClassVar[float] = jug_init_y_ub
    cup_capacity_lb: ClassVar[float] = 0.075 * (z_ub - z_lb)
    cup_capacity_ub: ClassVar[float] = 0.15 * (z_ub - z_lb)
    cup_target_frac: ClassVar[float] = 0.75  # fraction of the capacity
    # Simulation settings.
    pour_x_offset: ClassVar[float] = 1.5 * (cup_radius + jug_radius)
    pour_y_offset: ClassVar[float] = cup_radius
    pour_z_offset: ClassVar[float] = 1.1 * (cup_capacity_ub + jug_height - \
                                            jug_handle_height)
    pour_velocity: ClassVar[float] = cup_capacity_ub / 10.0
    max_position_vel: ClassVar[float] = 2.5
    max_angular_vel: ClassVar[float] = tilt_ub
    max_finger_vel: ClassVar[float] = 1.0

    def __init__(self) -> None:
        super().__init__()

        # Types
        self._robot_type = Type("robot",
                                ["x", "y", "z", "tilt", "wrist", "fingers"])
        self._jug_type = Type("jug", ["x", "y", "rot", "is_held", "is_filled"])
        self._machine_type = Type("machine", ["is_on"])
        self._cup_type = Type(
            "cup",
            ["x", "y", "capacity_liquid", "target_liquid", "current_liquid"])

        # Predicates
        self._CupFilled = Predicate("CupFilled", [self._cup_type],
                                    self._CupFilled_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._jug_type],
                                  self._Holding_holds)
        self._JugInMachine = Predicate("JugInMachine",
                                       [self._jug_type, self._machine_type],
                                       self._JugInMachine_holds)
        self._MachineOn = Predicate("MachineOn", [self._machine_type],
                                    self._MachineOn_holds)
        self._OnTable = Predicate("OnTable", [self._jug_type],
                                  self._OnTable_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._JugFilled = Predicate("JugFilled", [self._jug_type],
                                    self._JugFilled_holds)
        self._RobotAboveCup = Predicate("RobotAboveCup",
                                        [self._robot_type, self._cup_type],
                                        self._RobotAboveCup_holds)
        self._JugAboveCup = Predicate("JugAboveCup",
                                      [self._jug_type, self._cup_type],
                                      self._JugAboveCup_holds)
        self._NotAboveCup = Predicate("NotAboveCup",
                                      [self._robot_type, self._jug_type],
                                      self._NotAboveCup_holds)
        self._Twisting = Predicate("Twisting",
                                   [self._robot_type, self._jug_type],
                                   self._Twisting_holds)
        self._PressingButton = Predicate(
            "PressingButton", [self._robot_type, self._machine_type],
            self._PressingButton_holds)

        # Options
        self._MoveToTwistJug = ParameterizedOption(
            "MoveToTwistJug",
            types=[self._robot_type, self._jug_type],
            params_space=Box(0, 1, (0, )),
            policy=self._MoveToTwistJug_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._MoveToTwistJug_terminal,
        )
        self._TwistJug = ParameterizedOption(
            "TwistJug",
            types=[self._robot_type, self._jug_type],
            # The parameter is a normalized amount to twist by.
            params_space=Box(-1, 1, (1, )),
            policy=self._TwistJug_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._TwistJug_terminal,
        )
        self._PickJug = ParameterizedOption(
            "PickJug",
            types=[self._robot_type, self._jug_type],
            params_space=Box(0, 1, (0, )),
            policy=self._PickJug_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._PickJug_terminal,
        )
        self._PlaceJugInMachine = ParameterizedOption(
            "PlaceJugInMachine",
            types=[self._robot_type, self._jug_type, self._machine_type],
            params_space=Box(0, 1, (0, )),
            policy=self._PlaceJugInMachine_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._PlaceJugInMachine_terminal,
        )
        self._TurnMachineOn = ParameterizedOption(
            "TurnMachineOn",
            types=[self._robot_type, self._machine_type],
            params_space=Box(0, 1, (0, )),
            policy=self._TurnMachineOn_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._TurnMachineOn_terminal,
        )
        self._Pour = ParameterizedOption(
            "Pour",
            types=[self._robot_type, self._jug_type, self._cup_type],
            params_space=Box(0, 1, (0, )),
            policy=self._Pour_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._Pour_terminal,
        )

        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._jug = Object("juggy", self._jug_type)
        self._machine = Object("coffee_machine", self._machine_type)
        # Settings from CFG.
        self.jug_init_rot_lb = -CFG.coffee_jug_init_rot_amt
        self.jug_init_rot_ub = CFG.coffee_jug_init_rot_amt

    @classmethod
    def get_name(cls) -> str:
        return "coffee"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        norm_dx, norm_dy, norm_dz, norm_dtilt, norm_dwrist, norm_dfingers = \
            action.arr
        # Denormalize the action.
        dx = norm_dx * self.max_position_vel
        dy = norm_dy * self.max_position_vel
        dz = norm_dz * self.max_position_vel
        dtilt = norm_dtilt * self.max_angular_vel
        dwrist = norm_dwrist * self.max_angular_vel
        dfingers = norm_dfingers * self.max_finger_vel
        # Apply changes to the robot, taking bounds into account.
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_z = state.get(self._robot, "z")
        x = np.clip(robot_x + dx, self.x_lb, self.x_ub)
        y = np.clip(robot_y + dy, self.y_lb, self.y_ub)
        z = np.clip(robot_z + dz, self.z_lb, self.z_ub)
        current_tilt = state.get(self._robot, "tilt")
        tilt = np.clip(current_tilt + dtilt, self.tilt_lb, self.tilt_ub)
        current_wrist = state.get(self._robot, "wrist")
        wrist = np.clip(current_wrist + dwrist, self.wrist_lb, self.wrist_ub)
        current_fingers = state.get(self._robot, "fingers")
        fingers = np.clip(current_fingers + dfingers, self.closed_fingers,
                          self.open_fingers)
        # The deltas may be outdated because of the clipping, so recompute
        # or delete them.
        dx = x - state.get(self._robot, "x")
        dy = y - state.get(self._robot, "y")
        dwrist = wrist - state.get(self._robot, "wrist")
        del dz, dtilt, dfingers
        # Update the robot in the next state.
        next_state.set(self._robot, "x", x)
        next_state.set(self._robot, "y", y)
        next_state.set(self._robot, "z", z)
        next_state.set(self._robot, "tilt", tilt)
        next_state.set(self._robot, "wrist", wrist)
        next_state.set(self._robot, "fingers", fingers)
        # Get jug state info for later checks.
        handle_pos = self._get_jug_handle_grasp(state, self._jug)
        sq_dist_to_handle = np.sum(np.subtract(handle_pos, (x, y, z))**2)
        jug_rot = state.get(self._jug, "rot")
        # Check if the button should be pressed for the first time.
        machine_was_on = self._MachineOn_holds(state, [self._machine])
        pressing_button = self._PressingButton_holds(
            next_state, [self._robot, self._machine])
        jug_held = self._Holding_holds(state, [self._robot, self._jug])
        if pressing_button and not machine_was_on:
            next_state.set(self._machine, "is_on", 1.0)
            # Snap the robot to the center of the button.
            next_state.set(self._robot, "x", self.button_x)
            next_state.set(self._robot, "y", self.button_y)
            next_state.set(self._robot, "z", self.button_z)
            next_state.set(self._robot, "tilt", self.tilt_lb)
            next_state.set(self._robot, "wrist", self.robot_init_wrist)
        # If the jug is already held, move its position, and process drops.
        elif jug_held:
            # If the jug should be dropped, drop it first.
            if abs(fingers - self.open_fingers) < self.grasp_finger_tol:
                next_state.set(self._jug, "is_held", 0.0)
            # Otherwise, move it, and process pouring.
            else:
                # Check for pouring.
                if abs(tilt - self.tilt_ub) < self.pour_angle_tol:
                    # Find the cup to pour into, if any.
                    cup = self._get_cup_to_pour(next_state)
                    # If pouring into nothing, no-op.
                    if cup is None:
                        return state.copy()
                    # Increase the liquid in the cup.
                    current_liquid = state.get(cup, "current_liquid")
                    new_liquid = current_liquid + self.pour_velocity
                    # If we have exceeded the capacity of the cup, no-op.
                    if new_liquid > state.get(cup, "capacity_liquid"):
                        return state.copy()
                    next_state.set(cup, "current_liquid", new_liquid)
                    # If successfully poured, prevent movement and dropping.
                    next_state.set(self._robot, "x", robot_x)
                    next_state.set(self._robot, "y", robot_y)
                    next_state.set(self._robot, "z", robot_z)
                    next_state.set(self._robot, "fingers", self.closed_fingers)
                # Move the jug.
                else:
                    new_jug_x = state.get(self._jug, "x") + dx
                    new_jug_y = state.get(self._jug, "y") + dy
                    next_state.set(self._jug, "x", new_jug_x)
                    next_state.set(self._jug, "y", new_jug_y)
                    next_state.set(self._robot, "tilt", self.tilt_lb)
                    next_state.set(self._robot, "wrist", self.robot_init_wrist)
                    next_state.set(self._robot, "fingers", self.closed_fingers)
        # Check if the jug should be grasped for the first time.
        elif abs(fingers - self.closed_fingers) < self.grasp_finger_tol and \
            sq_dist_to_handle < self.grasp_position_tol and \
            abs(jug_rot) < self.pick_jug_rot_tol:
            # Snap to the handle.
            handle_x, handle_y, handle_z = handle_pos
            next_state.set(self._robot, "x", handle_x)
            next_state.set(self._robot, "y", handle_y)
            next_state.set(self._robot, "z", handle_z)
            next_state.set(self._robot, "tilt", self.tilt_lb)
            next_state.set(self._robot, "wrist", self.robot_init_wrist)
            # Grasp the jug.
            next_state.set(self._jug, "is_held", 1.0)
        # Check if the jug should be rotated.
        elif self._Twisting_holds(state, [self._robot, self._jug]):
            # Rotate the jug.
            rot = state.get(self._jug, "rot")
            next_state.set(self._jug, "rot", rot + dwrist)
        # If the jug is close enough to the dispense area and the machine is
        # on, the jug should get filled.
        jug_in_machine = self._JugInMachine_holds(next_state,
                                                  [self._jug, self._machine])
        machine_on = self._MachineOn_holds(next_state, [self._machine])
        if jug_in_machine and machine_on:
            next_state.set(self._jug, "is_filled", 1.0)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_cups_lst=CFG.coffee_num_cups_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               num_cups_lst=CFG.coffee_num_cups_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._CupFilled, self._JugInMachine, self._Holding,
            self._MachineOn, self._OnTable, self._HandEmpty, self._JugFilled,
            self._RobotAboveCup, self._JugAboveCup, self._NotAboveCup,
            self._PressingButton, self._Twisting
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._CupFilled}

    @property
    def types(self) -> Set[Type]:
        return {
            self._cup_type, self._jug_type, self._machine_type,
            self._robot_type
        }

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {
            self._TwistJug,
            self._PickJug,
            self._PlaceJugInMachine,
            self._TurnMachineOn,
            self._Pour,
            self._MoveToTwistJug,
        }

    @property
    def action_space(self) -> Box:
        # Normalized dx, dy, dz, dtilt, dwrist, dfingers.
        return Box(low=-1., high=1., shape=(6, ), dtype=np.float32)

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        del caption  # unused
        fig_width = (2 * (self.x_ub - self.x_lb))
        fig_height = max((self.y_ub - self.y_lb), (self.z_ub - self.z_lb))
        fig_size = (fig_width, fig_height)
        fig, axes = plt.subplots(1, 2, figsize=fig_size)
        xy_ax, xz_ax = axes
        # Draw the cups.
        color = "none"  # transparent cups
        for cup in state.get_objects(self._cup_type):
            x = state.get(cup, "x")
            y = state.get(cup, "y")
            capacity = state.get(cup, "capacity_liquid")
            current = state.get(cup, "current_liquid")
            z = self.z_lb + self.cup_radius
            circ = utils.Circle(x, y, self.cup_radius)
            circ.plot(xy_ax, facecolor=color, edgecolor="black")
            # Cups are cylinders, so in the xz plane, they look like rects.
            rect = utils.Rectangle(x=x,
                                   y=z,
                                   width=(self.cup_radius * 2),
                                   height=capacity,
                                   theta=0)
            rect.plot(xz_ax, facecolor=color, edgecolor="black")
            # Draw an inner rect to represent the filled level.
            if current > 0:
                rect = utils.Rectangle(x=x,
                                       y=z,
                                       width=(self.cup_radius * 2),
                                       height=current,
                                       theta=0)
                rect.plot(xz_ax, facecolor="lightblue", edgecolor="black")
        # Draw the machine.
        color = "gray"
        rect = utils.Rectangle(x=self.machine_x,
                               y=self.machine_y,
                               width=self.machine_x_len,
                               height=self.machine_y_len,
                               theta=0.0)
        rect.plot(xy_ax, facecolor=color, edgecolor="black")
        rect = utils.Rectangle(x=self.machine_x,
                               y=self.z_lb,
                               width=self.machine_x_len,
                               height=self.machine_z_len,
                               theta=0.0)
        rect.plot(xz_ax, facecolor=color, edgecolor="black")
        # Draw a button on the machine (xz plane only).
        machine_on = self._MachineOn_holds(state, [self._machine])
        color = "red" if machine_on else "brown"
        circ = utils.Circle(x=self.button_x,
                            y=self.button_z,
                            radius=self.button_radius)
        circ.plot(xz_ax, facecolor=color, edgecolor="black")
        # Draw the jug.
        jug_full = self._JugFilled_holds(state, [self._jug])
        jug_held = self._Holding_holds(state, [self._robot, self._jug])
        color = {
            # (jug_full, jug_held)
            (True, False): "lightblue",
            (True, True): "darkblue",
            (False, False): "lightgreen",
            (False, True): "darkgreen",
        }[(jug_full, jug_held)]
        x = state.get(self._jug, "x")
        y = state.get(self._jug, "y")
        z = self._get_jug_z(state, self._jug)
        circ = utils.Circle(x=x, y=y, radius=self.jug_radius)
        circ.plot(xy_ax, facecolor=color, edgecolor="black")
        # The jug is a cylinder, so in the xz plane it looks like a rect.
        rect = utils.Rectangle(x=(x - self.jug_radius),
                               y=z,
                               width=(2 * self.jug_radius),
                               height=self.jug_height,
                               theta=0.0)
        # Rotate if held.
        if jug_held:
            tilt = state.get(self._robot, "tilt")
            robot_x = state.get(self._robot, "x")
            robot_z = state.get(self._robot, "z")
            rect = rect.rotate_about_point(robot_x, robot_z, tilt)
        rect.plot(xz_ax, facecolor=color, edgecolor="black")
        # Draw the jug handle.
        if jug_held:
            # Offset to account for handle.
            handle_x = state.get(self._robot, "x")
            handle_y = state.get(self._robot, "y")
            handle_z = state.get(self._robot, "z")
        else:
            handle_x, handle_y, handle_z = self._get_jug_handle_grasp(
                state, self._jug)
        color = "darkgray"
        circ = utils.Circle(x=handle_x,
                            y=handle_y,
                            radius=self.jug_handle_radius)
        circ.plot(xy_ax, facecolor=color, edgecolor="black")
        circ = utils.Circle(x=handle_x,
                            y=handle_z,
                            radius=self.jug_handle_radius)
        circ.plot(xz_ax, facecolor=color, edgecolor="black")
        # Draw the robot.
        color = "gold"
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        z = state.get(self._robot, "z")
        circ = utils.Circle(
            x=x,
            y=y,
            radius=self.cup_radius  # robot in reality has no 'radius'
        )
        circ.plot(xy_ax, facecolor=color, edgecolor="black")
        circ = utils.Circle(
            x=x,
            y=z,
            radius=self.cup_radius  # robot in reality has no 'radius'
        )
        circ.plot(xz_ax, facecolor=color, edgecolor="black")
        ax_pad = 0.5
        xy_ax.set_xlim((self.x_lb - ax_pad), (self.x_ub + ax_pad))
        xy_ax.set_ylim((self.y_lb - ax_pad), (self.y_ub + ax_pad))
        xy_ax.set_xlabel("x")
        xy_ax.set_ylabel("y")
        xz_ax.set_xlim((self.x_lb - ax_pad), (self.x_ub + ax_pad))
        xz_ax.set_ylim((self.z_lb - ax_pad), (self.z_ub + ax_pad))
        xz_ax.set_xlabel("x")
        xz_ax.set_ylabel("z")
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int, num_cups_lst: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        # Create the parts of the initial state that do not change between
        # tasks, which includes the robot and the machine.
        common_state_dict = {}
        # Create the robot.
        common_state_dict[self._robot] = {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
            "fingers": self.open_fingers,  # robot fingers start open
        }
        # Create the machine.
        common_state_dict[self._machine] = {
            "is_on": 0.0,  # machine starts off
        }
        for _ in range(num):
            state_dict = {k: v.copy() for k, v in common_state_dict.items()}
            num_cups = num_cups_lst[rng.choice(len(num_cups_lst))]
            cups = [Object(f"cup{i}", self._cup_type) for i in range(num_cups)]
            goal = {GroundAtom(self._CupFilled, [c]) for c in cups}
            # Sample initial positions for cups, making sure to keep them
            # far enough apart from one another.
            radius = self.cup_radius + self.init_padding
            # Assuming that the dimensions are forgiving enough that
            # infinite loops are impossible.
            while True:
                collision_geoms: Set[utils.Circle] = set()
                cup_state_dict: Dict[Object, Dict[str, float]] = {}
                for cup in cups:
                    # Try to sample a position for the cup. If sampling does
                    # not quickly succeed, throw out the whole set of cup
                    # positions and start over.
                    for _ in range(10):
                        x = rng.uniform(self.cup_init_x_lb, self.cup_init_x_ub)
                        y = rng.uniform(self.cup_init_y_lb, self.cup_init_y_ub)
                        gm = utils.Circle(x, y, radius)
                        # Keep only if no intersections with existing objects.
                        if not any(gm.intersects(g) for g in collision_geoms):
                            break
                    else:
                        # Failed to sample a position for the cup.
                        break
                    collision_geoms.add(gm)
                    # Sample a cup capacity, which also defines its height.
                    cap = rng.uniform(self.cup_capacity_lb,
                                      self.cup_capacity_ub)
                    # Target liquid amount for filling the cup.
                    target = cap * self.cup_target_frac
                    # The initial liquid amount is always 0.
                    current = 0.0
                    cup_state_dict[cup] = {
                        "x": x,
                        "y": y,
                        "capacity_liquid": cap,
                        "target_liquid": target,
                        "current_liquid": current,
                    }
                else:
                    # We made it through without breaking, so we're done.
                    assert len(cup_state_dict) == len(cups)
                    # It is very rare that this while True loop fails on the
                    # first try, but it can happen. It doesn't happen during
                    # normal testing, so coverage complains (because the case
                    # where this else block is not hit is not covered).
                    break  # pragma: no cover
            state_dict.update(cup_state_dict)
            # Create the jug.
            x = rng.uniform(self.jug_init_x_lb, self.jug_init_x_ub)
            y = rng.uniform(self.jug_init_y_lb, self.jug_init_y_ub)
            rot = rng.uniform(self.jug_init_rot_lb, self.jug_init_rot_ub)
            state_dict[self._jug] = {
                "x": x,
                "y": y,
                "rot": rot,
                "is_held": 0.0,  # jug starts off not held
                "is_filled": 0.0  # jug starts off empty
            }
            init_state = utils.create_state_from_dict(state_dict)
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    @staticmethod
    def _CupFilled_holds(state: State, objects: Sequence[Object]) -> bool:
        cup, = objects
        current = state.get(cup, "current_liquid")
        target = state.get(cup, "target_liquid")
        return current > target

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, jug = objects
        return state.get(jug, "is_held") > 0.5

    def _JugInMachine_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        jug, _ = objects
        if self._Holding_holds(state, [self._robot, jug]):
            return False
        dispense_pos = (self.dispense_area_x, self.dispense_area_y, self.z_lb)
        x = state.get(jug, "x")
        y = state.get(jug, "y")
        z = self._get_jug_z(state, jug)
        jug_pos = (x, y, z)
        sq_dist_to_dispense = np.sum(np.subtract(dispense_pos, jug_pos)**2)
        return sq_dist_to_dispense < self.dispense_tol

    @staticmethod
    def _MachineOn_holds(state: State, objects: Sequence[Object]) -> bool:
        machine, = objects
        return state.get(machine, "is_on") > 0.5

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        jug, = objects
        if self._Holding_holds(state, [self._robot, jug]):
            return False
        return not self._JugInMachine_holds(state, [jug, self._machine])

    def _Twisting_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, jug = objects
        x = state.get(robot, "x")
        y = state.get(robot, "y")
        z = state.get(robot, "z")
        jug_x = state.get(jug, "x")
        jug_y = state.get(jug, "y")
        jug_top = (jug_x, jug_y, self.jug_height)
        # To prevent false positives, if the distance to the handle is less
        # than the distance to the jug top, we are not twisting.
        handle_pos = self._get_jug_handle_grasp(state, jug)
        sq_dist_to_handle = np.sum(np.subtract(handle_pos, (x, y, z))**2)
        sq_dist_to_jug_top = np.sum(np.subtract(jug_top, (x, y, z))**2)
        if sq_dist_to_handle < sq_dist_to_jug_top:
            return False
        return sq_dist_to_jug_top < self.grasp_position_tol

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        robot, = objects
        if self._Twisting_holds(state, [robot, self._jug]):
            return False
        return not self._Holding_holds(state, [robot, self._jug])

    @staticmethod
    def _JugFilled_holds(state: State, objects: Sequence[Object]) -> bool:
        jug, = objects
        return state.get(jug, "is_filled") > 0.5

    def _RobotAboveCup_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        robot, cup = objects
        assert robot == self._robot
        return self._robot_jug_above_cup(state, cup)

    def _JugAboveCup_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        jug, cup = objects
        assert jug == self._jug
        return self._robot_jug_above_cup(state, cup)

    def _NotAboveCup_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        robot, jug = objects
        assert robot == self._robot
        assert jug == self._jug
        for cup in state.get_objects(self._cup_type):
            if self._robot_jug_above_cup(state, cup):
                return False
        return True

    def _PressingButton_holds(self, state: State,
                              objects: Sequence[Object]) -> bool:
        robot, _ = objects
        button_pos = (self.button_x, self.button_y, self.button_z)
        x = state.get(robot, "x")
        y = state.get(robot, "y")
        z = state.get(robot, "z")
        sq_dist_to_button = np.sum(np.subtract(button_pos, (x, y, z))**2)
        return sq_dist_to_button < self.button_radius

    def _MoveToTwistJug_policy(self, state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> Action:
        # This policy moves the robot to above the jug, then moves down.
        del memory, params  # unused
        robot, jug = objects
        x = state.get(robot, "x")
        y = state.get(robot, "y")
        z = state.get(robot, "z")
        robot_pos = (x, y, z)
        jug_x = state.get(jug, "x")
        jug_y = state.get(jug, "y")
        jug_z = self.jug_height
        jug_top = (jug_x, jug_y, jug_z)
        xy_sq_dist = (jug_x - x)**2 + (jug_y - y)**2
        # If at the correct x and y position, move directly toward the target.
        if xy_sq_dist < self.twist_policy_tol:
            return self._get_move_action(jug_top, robot_pos)
        # Move to the position above the jug.
        return self._get_move_action((jug_x, jug_y, self.robot_init_z),
                                     robot_pos)

    def _MoveToTwistJug_terminal(self, state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> bool:
        del memory, params  # unused
        robot, jug = objects
        return self._Twisting_holds(state, [robot, jug])

    def _TwistJug_policy(self, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> Action:
        # This policy twists until the jug is in the desired rotation, and then
        # moves up to break contact with the jug.
        del memory  # unused
        robot, jug = objects
        current_rot = state.get(jug, "rot")
        norm_desired_rot, = params
        desired_rot = norm_desired_rot * CFG.coffee_jug_init_rot_amt
        delta_rot = np.clip(desired_rot - current_rot, -self.max_angular_vel,
                            self.max_angular_vel)
        if abs(delta_rot) < self.twist_policy_tol:
            # Move up to stop twisting.
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            robot_pos = (x, y, z)
            return self._get_move_action((x, y, self.robot_init_z), robot_pos)
        dtwist = delta_rot / self.max_angular_vel
        return Action(
            np.array([0.0, 0.0, 0.0, 0.0, dtwist, 0.0], dtype=np.float32))

    def _TwistJug_terminal(self, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        robot, _ = objects
        return self._HandEmpty_holds(state, [robot])

    def _PickJug_policy(self, state: State, memory: Dict,
                        objects: Sequence[Object], params: Array) -> Action:
        # This policy moves the robot to a safe height, then moves to behind
        # the handle in the y direction, then moves down in the z direction,
        # then moves forward in the y direction before finally grasping.
        del memory, params  # unused
        robot, jug = objects
        x = state.get(robot, "x")
        y = state.get(robot, "y")
        z = state.get(robot, "z")
        robot_pos = (x, y, z)
        handle_pos = self._get_jug_handle_grasp(state, jug)
        # If close enough, pick.
        sq_dist_to_handle = np.sum(np.subtract(handle_pos, robot_pos)**2)
        if sq_dist_to_handle < self.pick_policy_tol:
            return Action(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32))
        target_x, target_y, target_z = handle_pos
        # Distance to the handle in the x/z plane.
        xz_handle_sq_dist = (target_x - x)**2 + (target_z - z)**2
        # Distance to the penultimate waypoint in the x/y plane.
        waypoint_y = target_y - self.pick_jug_y_padding
        # Distance in the z direction to a safe move distance.
        safe_z_sq_dist = (self.robot_init_z - z)**2
        xy_waypoint_sq_dist = (target_x - x)**2 + (waypoint_y - y)**2
        # If at the correct x and z position and behind in the y direction,
        # move directly toward the target.
        if target_y > y and xz_handle_sq_dist < self.pick_policy_tol:
            return self._get_move_action(handle_pos, robot_pos)
        # If close enough to the penultimate waypoint in the x/y plane,
        # move to the waypoint (in the z direction).
        if xy_waypoint_sq_dist < self.pick_policy_tol:
            return self._get_move_action((target_x, waypoint_y, target_z),
                                         robot_pos)
        # If at a safe height, move to the position above the penultimate
        # waypoint, still at a safe height.
        if safe_z_sq_dist < self.safe_z_tol:
            return self._get_move_action(
                (target_x, waypoint_y, self.robot_init_z), robot_pos)
        # Move up to a safe height.
        return self._get_move_action((x, y, self.robot_init_z), robot_pos)

    def _PickJug_terminal(self, state: State, memory: Dict,
                          objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        robot, jug = objects
        return self._Holding_holds(state, [robot, jug])

    def _PlaceJugInMachine_policy(self, state: State, memory: Dict,
                                  objects: Sequence[Object],
                                  params: Array) -> Action:
        # This policy picks the jug up slightly above the table to avoid
        # worrying about friction, then moves directly to the place position,
        # then places the jug.
        del memory, params  # unused
        robot, jug, _ = objects
        # Use the jug position as the origin.
        x = state.get(jug, "x")
        y = state.get(jug, "y")
        z = state.get(robot, "z") - self.jug_handle_height
        jug_pos = (x, y, z)
        place_pos = (self.dispense_area_x, self.dispense_area_y, self.z_lb)
        # If close enough, place.
        sq_dist_to_place = np.sum(np.subtract(jug_pos, place_pos)**2)
        if sq_dist_to_place < self.place_jug_in_machine_tol:
            return Action(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        # If already above the table, move directly toward the place pos.
        if z > self.z_lb:
            return self._get_move_action(place_pos, jug_pos)
        # Move up.
        return self._get_move_action((x, y, z + self.max_position_vel),
                                     jug_pos)

    def _PlaceJugInMachine_terminal(self, state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> bool:
        del memory, params  # unused
        robot, jug, machine = objects
        return not self._Holding_holds(state, [robot, jug]) and \
            self._JugInMachine_holds(state, [jug, machine])

    def _TurnMachineOn_policy(self, state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> Action:
        # This policy moves the robot up to be level with the button in the
        # z direction and then moves forward in the y direction to press it.
        del memory, params  # unused
        robot, _ = objects
        x = state.get(robot, "x")
        y = state.get(robot, "y")
        z = state.get(robot, "z")
        robot_pos = (x, y, z)
        button_pos = (self.button_x, self.button_y, self.button_z)
        if (self.button_z - z)**2 < self.button_radius**2:
            # Move directly toward the button.
            return self._get_move_action(button_pos, robot_pos)
        # Move only in the z direction.
        return self._get_move_action((x, y, self.button_z), robot_pos)

    def _TurnMachineOn_terminal(self, state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> bool:
        del memory, params  # unused
        _, machine = objects
        return self._MachineOn_holds(state, [machine])

    def _Pour_policy(self, state: State, memory: Dict,
                     objects: Sequence[Object], params: Array) -> Action:
        # This policy moves the robot next to the cup and then pours until
        # the cup is filled. Note that if starting out at the end of another
        # pour, we need to start by rotating the cup to prevent any further
        # pouring until we've moved over the next cup.
        del memory, params  # unused
        move_tilt = self.tilt_lb
        pour_tilt = self.tilt_ub
        robot, jug, cup = objects
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        robot_z = state.get(robot, "z")
        robot_pos = (robot_x, robot_y, robot_z)
        tilt = state.get(robot, "tilt")
        jug_x = state.get(jug, "x")
        jug_y = state.get(jug, "y")
        jug_z = self._get_jug_z(state, jug)
        jug_pos = (jug_x, jug_y, jug_z)
        pour_x, pour_y, _ = pour_pos = self._get_pour_position(state, cup)
        # If we're close enough to the pour position, pour.
        sq_dist_to_pour = np.sum(np.subtract(jug_pos, pour_pos)**2)
        if sq_dist_to_pour < self.pour_policy_tol:
            dtilt = pour_tilt - tilt
            return self._get_move_action(jug_pos, jug_pos, dtilt=dtilt)
        dtilt = move_tilt - tilt
        # If we're above the pour position, move down to pour.
        xy_pour_sq_dist = (jug_x - pour_x)**2 + (jug_y - pour_y)**2
        if xy_pour_sq_dist < self.safe_z_tol:
            return self._get_move_action(pour_pos, jug_pos, dtilt=dtilt)
        # If we're at a safe height, move toward above the pour position.
        if (robot_z - self.robot_init_z)**2 < self.safe_z_tol:
            return self._get_move_action((pour_x, pour_y, jug_z),
                                         jug_pos,
                                         dtilt=dtilt)
        # Move to a safe moving height.
        return self._get_move_action((robot_x, robot_y, self.robot_init_z),
                                     robot_pos,
                                     dtilt=dtilt)

    def _Pour_terminal(self, state: State, memory: Dict,
                       objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        _, _, cup = objects
        return self._CupFilled_holds(state, [cup])

    def _robot_jug_above_cup(self, state: State, cup: Object) -> bool:
        if not self._Holding_holds(state, [self._robot, self._jug]):
            return False
        jug_x = state.get(self._jug, "x")
        jug_y = state.get(self._jug, "y")
        jug_z = state.get(self._robot, "z") - self.jug_handle_height
        jug_pos = (jug_x, jug_y, jug_z)
        pour_pos = self._get_pour_position(state, cup)
        sq_dist_to_pour = np.sum(np.subtract(jug_pos, pour_pos)**2)
        return sq_dist_to_pour < self.pour_pos_tol

    def _get_jug_handle_grasp(self, state: State,
                              jug: Object) -> Tuple[float, float, float]:
        # Orient pointing down.
        rot = state.get(jug, "rot") - np.pi / 2
        target_x = state.get(jug, "x") + np.cos(rot) * self.jug_handle_offset
        target_y = state.get(jug, "y") + np.sin(rot) * self.jug_handle_offset
        target_z = self.jug_handle_height
        return (target_x, target_y, target_z)

    def _get_jug_z(self, state: State, jug: Object) -> float:
        if state.get(jug, "is_held") > 0.5:
            # Offset to account for handle.
            return state.get(self._robot, "z") - self.jug_handle_height
        # On the table.
        return self.z_lb

    def _get_pour_position(self, state: State,
                           cup: Object) -> Tuple[float, float, float]:
        target_x = state.get(cup, "x") + self.pour_x_offset
        target_y = state.get(cup, "y") + self.pour_y_offset
        target_z = self.pour_z_offset
        return (target_x, target_y, target_z)

    def _get_cup_to_pour(self, state: State) -> Optional[Object]:
        jug_x = state.get(self._jug, "x")
        jug_y = state.get(self._jug, "y")
        jug_z = self._get_jug_z(state, self._jug)
        jug_pos = (jug_x, jug_y, jug_z)
        closest_cup = None
        closest_cup_dist = float("inf")
        for cup in state.get_objects(self._cup_type):
            target = self._get_pour_position(state, cup)
            sq_dist = np.sum(np.subtract(jug_pos, target)**2)
            if sq_dist < self.pour_pos_tol and sq_dist < closest_cup_dist:
                closest_cup = cup
                closest_cup_dist = sq_dist
        return closest_cup

    def _get_move_action(self,
                         target_pos: Tuple[float, float, float],
                         robot_pos: Tuple[float, float, float],
                         dtilt: float = 0.0,
                         dwrist: float = 0.0) -> Action:
        # We want to move in this direction.
        delta = np.subtract(target_pos, robot_pos)
        # But we can only move at most max_position_vel in one step.
        # Get the norm full move delta.
        pos_norm = float(np.linalg.norm(delta))
        # If the norm is more than max_position_vel, rescale the delta so that
        # its norm is max_position_vel.
        if pos_norm > self.max_position_vel:
            delta = self.max_position_vel * (delta / pos_norm)
            pos_norm = self.max_position_vel
        # Now normalize so that the action values are between -1 and 1, as
        # expected by simulate and the action space.
        if pos_norm > 0:
            delta = delta / self.max_position_vel
        dx, dy, dz = delta
        dtilt = np.clip(dtilt, -self.max_angular_vel, self.max_angular_vel)
        dtilt = dtilt / self.max_angular_vel
        return Action(
            np.array([dx, dy, dz, dtilt, dwrist, 0.0], dtype=np.float32))
