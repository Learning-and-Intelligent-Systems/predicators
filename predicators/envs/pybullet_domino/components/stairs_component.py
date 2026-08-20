"""Stairs component for the domino environment.

This component handles:
- Creating stairs (platforms) under dominoes with increasing height
- Dynamic stair creation based on domino positions
"""

from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.pybullet_helpers.objects import create_pybullet_block
from predicators.structs import Object, State, Type


class StairsComponent(DominoEnvComponent):
    """Component for stairs under dominoes.

    Manages:
    - Dynamic stair creation under each domino
    - Progressive height increase for each successive domino
    - Stair removal and recreation on state reset
    """

    # =========================================================================
    # STAIRS CONFIGURATION
    # =========================================================================
    stair_width: ClassVar[float] = 0.078  # Slightly larger than domino width
    stair_depth: ClassVar[float] = 0.078  # Square base
    base_stair_height: ClassVar[float] = 0.02  # Base height for first stair
    stair_height_increment: ClassVar[
        float] = 0.008  # Height increase per domino

    stair_mass: ClassVar[float] = 0.0  # Static
    stair_friction: ClassVar[float] = 0.5
    stair_color: ClassVar[Tuple[float, float, float,
                                float]] = (0.7, 0.6, 0.5, 1.0)

    def __init__(self,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 table_height: float = 0.4,
                 domino_type: Optional[Type] = None,
                 enabled: bool = True) -> None:
        """Initialize the stairs component.

        Args:
            workspace_bounds: Dictionary with x_lb, x_ub, y_lb, y_ub.
            table_height: Height of the table surface.
            domino_type: The domino type to reference for positioning.
            enabled: Whether stairs are enabled (can be toggled via CFG).
        """
        super().__init__()

        self.table_height = table_height
        self.enabled = enabled
        self._domino_type = domino_type

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

        # Storage for dynamically created stair bodies
        self._stair_ids: List[int] = []

    # -------------------------------------------------------------------------
    # DominoEnvComponent interface implementation
    # -------------------------------------------------------------------------

    def get_types(self) -> Set[Type]:
        # Stairs don't have their own type - they're dynamically created
        return set()

    def get_predicates(self) -> Set:
        return set()  # Stairs don't have predicates

    def get_goal_predicates(self) -> Set:
        return set()  # Stairs never appear in goals

    def get_objects(self) -> List[Object]:
        return []  # Stairs are dynamically created, not pre-defined objects

    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        """Initialize PyBullet - stairs created in reset_state."""
        self._physics_client_id = physics_client_id
        return {}  # No pre-created bodies

    def store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """No bodies to store - stairs are created dynamically."""

    def set_domino_type(self, domino_type: Type) -> None:
        """Set the domino type reference for positioning stairs."""
        self._domino_type = domino_type

    def reset_state(self, state: State) -> None:
        """Create stairs under dominoes based on current state."""
        assert self._physics_client_id is not None
        # Remove existing stairs
        for stair_id in self._stair_ids:
            if stair_id >= 0:
                p.removeBody(stair_id, physicsClientId=self._physics_client_id)
        self._stair_ids = []

        # Only create stairs if enabled and we have a domino type
        if not self.enabled or self._domino_type is None:
            return

        # Get domino objects from state
        domino_objs = state.get_objects(self._domino_type)
        if not domino_objs:
            return

        # Create stairs under each domino with progressively increasing height
        for i, domino_obj in enumerate(domino_objs):
            domino_x = state.get(domino_obj, "x")
            domino_y = state.get(domino_obj, "y")

            # Calculate stair height based on domino index
            stair_height = self.base_stair_height + (
                i * self.stair_height_increment)

            # Create stair block under the domino
            stair_id = create_pybullet_block(
                color=self.stair_color,
                half_extents=(self.stair_width / 2, self.stair_depth / 2,
                              stair_height / 2),
                mass=self.stair_mass,
                friction=self.stair_friction,
                position=(domino_x, domino_y,
                          self.table_height + stair_height / 2),
                orientation=p.getQuaternionFromEuler([0, 0, 0]),
                physics_client_id=self._physics_client_id)

            self._stair_ids.append(stair_id)

    def extract_feature(self, obj: Object, feature: str) -> Optional[float]:
        """Stairs don't have extractable features."""
        return None

    # -------------------------------------------------------------------------
    # Initial state helpers
    # -------------------------------------------------------------------------

    def get_init_dict_entries(
        self,
        rng: np.random.Generator,
    ) -> Dict[Object, Dict[str, Any]]:
        """Stairs don't add init dict entries - created dynamically."""
        return {}

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable stairs creation."""
        self.enabled = enabled

    @property
    def stair_ids(self) -> List[int]:
        """Return list of current stair PyBullet body IDs."""
        return self._stair_ids
