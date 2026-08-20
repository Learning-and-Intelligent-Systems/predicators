"""Ball component for the domino environment.

This component handles:
- Ball object that can be blown by fans
- Ball target (goal position marker)
- Ball physics (mass, friction, damping)
- Related predicates (BallAtTarget)
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.pybullet_helpers.objects import create_pybullet_block, \
    create_pybullet_sphere, update_object
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, Type


class BallComponent(DominoEnvComponent):
    """Component for ball and ball target.

    Manages:
    - Ball that can be pushed by wind or collisions
    - Ball target marker for goal positions
    - BallAtTarget predicate for goal checking
    """

    # =========================================================================
    # BALL CONFIGURATION
    # =========================================================================
    ball_radius: ClassVar[float] = 0.05
    ball_mass: ClassVar[float] = 0.5
    ball_friction: ClassVar[float] = 0.5
    ball_restitution: ClassVar[float] = 0.3
    ball_linear_damping: ClassVar[float] = 0.5
    ball_angular_damping: ClassVar[float] = 0.3
    ball_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.0, 0.0, 1.0, 1.0)

    # =========================================================================
    # TARGET CONFIGURATION
    # =========================================================================
    target_thickness: ClassVar[float] = 0.00001
    target_mass: ClassVar[float] = 0.0
    target_friction: ClassVar[float] = 0.04
    target_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.0, 1.0, 0.0, 1.0)

    def __init__(self,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 table_height: float = 0.4,
                 position_tolerance: Optional[float] = None) -> None:
        """Initialize the ball component.

        Args:
            workspace_bounds: Dictionary with x_lb, x_ub, y_lb, y_ub.
            table_height: Height of the table surface.
            position_tolerance: Distance threshold for BallAtTarget predicate.
        """
        super().__init__()

        self.table_height = table_height
        self.ball_height_offset = self.ball_radius

        # Position tolerance for goal checking
        if position_tolerance is None:
            self.position_tolerance = CFG.domino_fan_ball_position_tolerance
        else:
            self.position_tolerance = position_tolerance

        # Workspace bounds
        if workspace_bounds is None:
            workspace_bounds = {
                "x_lb": 0.4,
                "x_ub": 1.1,
                "y_lb": 1.1,
                "y_ub": 1.6
            }
        self.x_lb = workspace_bounds["x_lb"]
        self.x_ub = workspace_bounds["x_ub"]
        self.y_lb = workspace_bounds["y_lb"]
        self.y_ub = workspace_bounds["y_ub"]

        # Create types
        self._ball_type = Type("ball", ["x", "y", "z"])
        self._ball_target_type = Type("ball_target", ["x", "y", "z", "is_hit"])

        # Create objects
        self._ball = Object("ball", self._ball_type)
        self._ball_target = Object("ball_target", self._ball_target_type)

        # Create predicates
        self._BallAtTarget = Predicate(
            "BallAtTarget", [self._ball_type, self._ball_target_type],
            self._BallAtTarget_holds)

        # Reference to current state for is_hit feature extraction
        self._current_state: Optional[State] = None

    # -------------------------------------------------------------------------
    # DominoEnvComponent interface implementation
    # -------------------------------------------------------------------------

    def get_types(self) -> Set[Type]:
        return {self._ball_type, self._ball_target_type}

    def get_predicates(self) -> Set[Predicate]:
        return {self._BallAtTarget}

    def get_goal_predicates(self) -> Set[Predicate]:
        return {self._BallAtTarget}

    def get_objects(self) -> List[Object]:
        return [self._ball, self._ball_target]

    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        """Create PyBullet bodies for ball and target."""
        self._physics_client_id = physics_client_id
        bodies: Dict[str, Any] = {}

        # Create ball
        ball_id = create_pybullet_sphere(
            color=self.ball_color,
            radius=self.ball_radius,
            mass=self.ball_mass,
            friction=self.ball_friction,
            # Match lateral with spinning to preserve the prior behavior
            # where the ball didn't pinwheel around the contact normal.
            spinning_friction=self.ball_friction,
            position=(0.75, 1.35, self.table_height + self.ball_height_offset),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)

        p.changeDynamics(ball_id,
                         -1,
                         linearDamping=self.ball_linear_damping,
                         angularDamping=self.ball_angular_damping,
                         restitution=self.ball_restitution,
                         ccdSweptSphereRadius=self.ball_radius * 0.9,
                         physicsClientId=physics_client_id)
        bodies["ball_id"] = ball_id

        # Create ball target (flat green marker)
        ball_target_id = create_pybullet_block(
            color=self.target_color,
            half_extents=(self.ball_radius, self.ball_radius,
                          self.target_thickness),
            mass=self.target_mass,
            friction=self.target_friction,
            position=(0, 0, self.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        bodies["ball_target_id"] = ball_target_id

        return bodies

    def store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store PyBullet body IDs on ball objects."""
        self._ball.id = pybullet_bodies["ball_id"]
        self._ball_target.id = pybullet_bodies["ball_target_id"]

    def reset_state(self, state: State) -> None:
        """Reset ball and target to match state."""
        assert self._physics_client_id is not None
        self._current_state = state

        # Position ball
        ball_x = state.get(self._ball, "x")
        ball_y = state.get(self._ball, "y")
        ball_z = state.get(self._ball, "z")
        update_object(self._ball.id,
                      position=(ball_x, ball_y, ball_z),
                      physics_client_id=self._physics_client_id)

        # Reset ball velocity
        p.resetBaseVelocity(self._ball.id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._physics_client_id)

        # Position ball target
        target_x = state.get(self._ball_target, "x")
        target_y = state.get(self._ball_target, "y")
        target_z = state.get(self._ball_target, "z")
        update_object(self._ball_target.id,
                      position=(target_x, target_y, target_z),
                      physics_client_id=self._physics_client_id)

    def extract_feature(self, obj: Object, feature: str) -> Optional[float]:
        """Extract feature for ball-related objects."""
        if obj.type == self._ball_type:
            if feature in ["x", "y", "z"]:
                pos, _ = p.getBasePositionAndOrientation(
                    obj.id, physicsClientId=self._physics_client_id)
                return pos[{"x": 0, "y": 1, "z": 2}[feature]]

        if obj.type == self._ball_target_type:
            if feature == "is_hit":
                # Need current state to check ball position
                if self._current_state is not None:
                    bx = self._current_state.get(self._ball, "x")
                    by = self._current_state.get(self._ball, "y")
                    tx = self._current_state.get(self._ball_target, "x")
                    ty = self._current_state.get(self._ball_target, "y")
                    dist = np.sqrt((bx - tx)**2 + (by - ty)**2)
                    return (1.0 if dist < self.position_tolerance else 0.0)
                return 0.0
            if feature in ["x", "y", "z"]:
                pos, _ = p.getBasePositionAndOrientation(
                    obj.id, physicsClientId=self._physics_client_id)
                return pos[{"x": 0, "y": 1, "z": 2}[feature]]

        return None

    def set_current_state(self, state: State) -> None:
        """Update reference to current state for feature extraction."""
        self._current_state = state

    # -------------------------------------------------------------------------
    # Predicate hold functions
    # -------------------------------------------------------------------------

    def _BallAtTarget_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        """Check if ball is at target position."""
        ball, target = objects
        bx, by = state.get(ball, "x"), state.get(ball, "y")
        tx, ty = state.get(target, "x"), state.get(target, "y")
        dist = np.sqrt((bx - tx)**2 + (by - ty)**2)
        return dist < self.position_tolerance

    # -------------------------------------------------------------------------
    # Initial state helpers
    # -------------------------------------------------------------------------

    def get_init_dict_entries(
        self,
        rng: "np.random.Generator",
        ball_position: Optional[Tuple[float, float]] = None,
        target_position: Optional[Tuple[float, float]] = None
    ) -> Dict[Object, Dict[str, Any]]:
        """Return initial state dict entries for ball and target.

        Args:
            rng: Random number generator.
            ball_position: Optional (x, y) for ball. If None, placed in corner.
            target_position: Optional (x, y) for target. If None, placed in
                           opposite corner.
        """
        init_dict: Dict[Object, Dict[str, Any]] = {}

        # Ball position
        if ball_position is None:
            ball_x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
            ball_y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)
        else:
            ball_x, ball_y = ball_position

        init_dict[self._ball] = {
            "x": ball_x,
            "y": ball_y,
            "z": self.table_height + self.ball_height_offset
        }

        # Target position
        if target_position is None:
            # Random target position (ensure far enough from ball)
            min_distance: float = 0.15
            target_x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
            target_y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)
            while np.sqrt((target_x - ball_x)**2 +
                          (target_y - ball_y)**2) < min_distance:
                target_x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
                target_y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)
        else:
            target_x, target_y = target_position

        init_dict[self._ball_target] = {
            "x": target_x,
            "y": target_y,
            "z": self.table_height,
            "is_hit": 0.0
        }

        return init_dict

    # -------------------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------------------

    @property
    def ball(self) -> Object:
        """Ball."""
        return self._ball

    @property
    def ball_target(self) -> Object:
        """Ball target."""
        return self._ball_target

    @property
    def ball_type(self) -> Type:
        """Ball type."""
        return self._ball_type

    @property
    def ball_target_type(self) -> Type:
        """Ball target type."""
        return self._ball_target_type

    @property
    def BallAtTarget(self) -> Predicate:
        """BallAtTarget."""
        return self._BallAtTarget

    @property
    def ball_id(self) -> int:
        """Return PyBullet ID of the ball (for wind targeting)."""
        return self._ball.id
