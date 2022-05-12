"""Toy screw picking environment."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Image, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class ScrewsEnv(BaseEnv):
    """Toy screw picking env.

    In this environment, the agent controls a magnetic gripper that can
    be used to pick up screws of different colors/sizes. The challennge
    is for the agent to learn actions that don't precisely model all the
    irrelevant screws that might be picked up when picking a desired
    screw.
    """
    _gripper_width: ClassVar[float] = 0.5
    _receptacle_width: ClassVar[float] = 0.6
    _receptacle_height: ClassVar[float] = 0.2
    _screw_width: ClassVar[float] = 0.05
    _screw_height: ClassVar[float] = 0.05
    _magnetic_field_dist: ClassVar[float] = 0.1
    # Reachable zone boundaries.
    rz_x_lb = -5.0
    rz_x_ub = 5.0
    rz_y_lb = 0.0
    rz_y_ub = 5.0

    def __init__(self) -> None:
        super.__init__()
        # Types
        self._screw_type = Type(
            "screw", ["pose_x", "pose_y", "color", "width", "height", "held"])
        self._gripper_type = Type("gripper", ["pose_x", "pose_y", "width"])
        self._receptacle_type = Type("receptacle",
                                     ["pose_x", "pose_y", "width"])
        # Predicates
        self._ScrewPickupable = Predicate(
            "AboveScrew", [self._gripper_type, self._screw_type],
            self._ScrewPickupable_holds)
        self._AboveReceptacle = Predicate(
            "AboveReceptacle", [self._gripper_type, self._receptacle_type],
            self._AboveReceptacle_holds)
        self._HoldingScrew = Predicate("HoldingScrew", [self._screw_type],
                                       self._HoldingScrew_holds)
        self._ScrewInReceptacle = Predicate(
            "ScrewInReceptacle", [self._screw_type, self._receptacle_type],
            self._ScrewInReceptacle_holds)

        # Options
        self._MoveToScrew: ParameterizedOption = utils.SingletonParameterizedOption(
            # variables: [robot, screw to pick up]
            # params: []
            "MoveToScrew",
            self._MoveToScrew_policy,
            types=[self._gripper_type, self._screw_type])
        self._MoveToReceptacle: ParameterizedOption = utils.SingletonParameterizedOption(
            # variables: [robot, receptacle]
            # params: []
            "MoveToReceptacle",
            self._MoveToReceptacle_policy,
            types=[self._gripper_type, self._receptacle_type])
        self._MagnetizeGripper: ParameterizedOption = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: []
            "MagnetizeGripper",
            self._MagnetizeGripper_policy,
            types=[self._gripper_type])
        self._DemagnetizeGripper: ParameterizedOption = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: []
            "DemagnetizeGripper",
            self._DemagnetizeGripper_policy,
            types=[self._gripper_type])

        # TODO (somewhere else):
        # - sampling initial positions for things/training problems
        # - rendering env

        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._gripper_type)
        self._receptacle = Object("receptacle", self._receptacle_type)

    @classmethod
    def get_name(cls) -> str:
        return "screws"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        # NOTE: currently, the agent can only pick things up or
        # drop them iff its staying still. This isn't particularly
        # realistic...
        dx, dy, magnetization = action
        if dx != 0.0 or dy != 0.0:
            return self._transition_move(state, dx, dy)

        if magnetization > 0.5:
            return self._transition_magnetize(state)
        else:
            return self._transition_demagnetize(state)

    def _transition_move(self, state: State, x: float, y: float) -> State:
        # TODO: Detect collisions and either don't allow move when there's
        # collisions, or stop the move at that point.
        next_state = state.copy()
        rx = state.get(self._robot, "pose_x")
        ry = state.get(self._robot, "pose_y")
        rwidth = state.get(self._robot, "width")
        new_rx = rx + x
        new_rx_min = new_rx - rwidth / 2.0
        new_rx_max = new_rx + rwidth / 2.0
        new_ry = ry + y

        # If the new position is outside the boundaries, move till
        # the robot collides with the boundary.
        if new_rx_min < self.rz_x_lb:
            new_rx = self.rz_x_lb + (rwidth / 2.0)
        if new_rx_max > self.rz_x_ub:
            new_rx = self.rz_x_ub - (rwidth / 2.0)
        if new_ry < self.rz_y_lb:
            new_ry = self.rz_y_lb
        if new_ry > self.rz_y_ub:
            new_ry = self.rz_y_ub
        next_state.set(self._robot, "pose_x", new_rx)
        next_state.set(self._robot, "pose_y", new_ry)

        # Recompute dx and dy in case it's changed.
        dx = new_rx - rx
        dy = new_ry - ry

        # Find out which (if any) screws are currently being held by
        # the gripper and update their positions to follow the gripper.
        all_screws = state.get_objects(self._screw_type)
        for screw in all_screws:
            if self._HoldingScrew_holds(state, [screw]):
                screw_x = state.get(screw, "pose_x")
                screw_y = state.get(screw, "pose_y")
                next_state.set(screw, "pose_x", screw_x + dx)
                next_state.set(screw, "pose_y", screw_y + dy)

        return next_state

    def _transition_magnetize(self, state: State) -> State:
        next_state = state.copy()
        ry = state.get(self._robot, "pose_y")
        all_screws = state.get_objects(self._screw_type)
        for screw in all_screws:
            if not self._HoldingScrew_holds(state, [screw]):
                # Check if the gripper is in the right position to
                # pick up this screw, and if so, move the screw to
                # the gripper.
                if self._ScrewPickupable_holds(state, [self._robot, screw]):
                    screw_height = state.get(screw, "height")
                    next_state.update(screw,
                                      {"pose_y": ry - (screw_height / 2.0)})
                    next_state.update(screw, {"held": True})

        return next_state

    def _transition_demagnetize(self, state: State) -> State:
        next_state = state.copy()
        all_screws = state.get_objects(self._screw_type)
        for screw in all_screws:
            if self._HoldingScrew_holds(state, [screw]):
                # NOTE: This currently will drop all screws into the receptacle
                # iff AboveReceptacle is true. Otherwise, it will drop all screws
                # onto the floor. This neglects the case where the gripper is
                # half over the receptacle and half not.
                screw_height = state.get(screw, "height")
                if self._AboveReceptacle_holds(
                        state, [self._robot, self._receptacle]):
                    receptacle_y = state.get(self._receptacle, "pose_y")
                    next_state.update(
                        screw, {"pose_y": receptacle_y + (screw_height / 2.0)})
                else:
                    next_state.update(
                        screw, {"pose_y": self.rz_y_lb + (screw_height / 2.0)})

        return next_state

    @property
    def action_space(self) -> Box:
        # dimensions: [dx, dy, magnetized vs. unmagnetized].
        # Note that dx and dy are normalized to be between -1.0
        # and 1.0.
        lowers = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
        uppers = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return Box(lowers, uppers)

    @staticmethod
    def _ScrewPickupable_holds(self, state: State,
                               objects: Sequence[Object]) -> bool:
        gripper, screw = objects
        gripper_y = state.get(gripper, "pose_y")
        screw_y = state.get(screw, "pose_y")
        screw_height = state.get(screw, "height")
        screw_maxy = screw_y + screw_height / 2.0

        gripper_center_pose = state.get(gripper, "pose_x")
        gripper_width = state.get(gripper, "width")
        gripper_minx = gripper_center_pose - gripper_width / 2.0
        gripper_maxx = gripper_center_pose + gripper_width / 2.0
        screw_x = state.get(screw, "pose_x")
        screw_width = state.get(screw, "width")
        screw_minx = screw_x - screw_width / 2.0
        screw_maxx = screw_x + screw_width / 2.0

        return screw_maxy < gripper_y and \
            screw_maxx > gripper_minx and \
            screw_minx < gripper_maxx

    @staticmethod
    def _AboveReceptacle_holds(self, state: State,
                               objects: Sequence[Object]) -> bool:
        gripper, receptacle = objects
        gripper_y = state.get(gripper, "pose_y")
        receptacle_y = state.get(receptacle, "pose_y")
        receptacle_height = state.get(receptacle, "height")
        receptacle_maxy = receptacle_y + receptacle_height / 2.0

        gripper_center_pose = state.get(gripper, "pose_x")
        gripper_width = state.get(gripper, "width")
        gripper_minx = gripper_center_pose - gripper_width / 2.0
        gripper_maxx = gripper_center_pose + gripper_width / 2.0
        receptacle_x = state.get(receptacle, "pose_x")
        receptacle_width = state.get(receptacle, "width")
        receptacle_minx = receptacle_x - receptacle_width / 2.0
        receptacle_maxx = receptacle_x + receptacle_width / 2.0

        return receptacle_maxy < gripper_y and \
            receptacle_maxy > gripper_y - self._magnetic_field_dist and \
            receptacle_minx < gripper_minx and \
            receptacle_maxx > gripper_maxx

    @staticmethod
    def _HoldingScrew_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        screw, = objects
        return state.get(screw, "held") != -1

    @staticmethod
    def _ScrewInReceptacle_holds(self, state: State,
                                 objects: Sequence[Object]) -> bool:
        screw, receptacle = objects
        screw_y = state.get(screw, "pose_y")
        receptacle_y = state.get(receptacle, "pose_y")
        receptacle_height = state.get(receptacle, "height")
        screw_height = state.get(screw, "height")
        screw_maxy = screw_y + screw_height / 2.0
        screw_miny = screw_y - screw_height / 2.0
        receptacle_maxy = receptacle_y + receptacle_height / 2.0
        receptacle_miny = receptacle_y - receptacle_height / 2.0

        screw_center_pose = state.get(screw, "pose_x")
        screw_width = state.get(screw, "width")
        screw_minx = screw_center_pose - screw_width / 2.0
        screw_maxx = screw_center_pose + screw_width / 2.0
        receptacle_x = state.get(receptacle, "pose_x")
        receptacle_width = state.get(receptacle, "width")
        receptacle_maxx = receptacle_x + receptacle_width / 2.0
        receptacle_minx = receptacle_x - receptacle_width / 2.0

        return screw_maxy < receptacle_maxy and \
            screw_miny > receptacle_miny and \
            screw_minx > receptacle_minx and \
            screw_maxx < receptacle_maxx

    def _MoveToScrew_policy(self, state: State, memory: Dict,
                            objects: Sequence[Object],
                            params: Array) -> Action:
        del memory, params  # unused
        _, screw = objects
        screw_x = state.get(screw, "pose_x")
        screw_y = state.get(screw, "pose_y")
        screw_height = state.get(screw, "height")

        target_x = screw_x
        target_y = screw_y + (screw_height /
                              2.0) + (self._magnetic_field_dist / 2.0)

        current_x = state.get(self._gripper, "pose_x")
        current_y = state.get(self._gripper, "pose_y")

        return np.array([target_x - current_x, target_y - current_y, 0.0],
                        dtype=np.float32)

    def _MoveToReceptacle_policy(self, state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
        del memory, params  # unused
        _, receptacle = objects
        receptacle_x = state.get(receptacle, "pose_x")
        receptacle_y = state.get(receptacle, "pose_y")

        target_x = receptacle_x
        target_y = receptacle_y + (self._magnetic_field_dist)

        current_x = state.get(self._gripper, "pose_x")
        current_y = state.get(self._gripper, "pose_y")

        return np.array([target_x - current_x, target_y - current_y, 1.0],
                        dtype=np.float32)

    def _MagnetizeGripper_policy(self, state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
        del state, memory, objects, params  # unused
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _DemagnetizeGripper_policy(self, state: State, memory: Dict,
                                   objects: Sequence[Object],
                                   params: Array) -> Action:
        del state, memory, objects, params  # unused
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
