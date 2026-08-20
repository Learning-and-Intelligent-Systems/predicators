"""Ramp component for the domino environment.

This component handles:
- Ramp objects (platform + slope) for creating height transitions
- Ramp positioning and orientation
- Static ramp obstacles
"""

import os
import tempfile
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.structs import Object, State, Type


class RampComponent(DominoEnvComponent):
    """Component for ramps.

    Manages:
    - Static ramp obstacles with platform and slope
    - Ramp positioning in the workspace
    """

    # =========================================================================
    # RAMP CONFIGURATION
    # =========================================================================
    ramp_width: ClassVar[float] = 0.20  # Width perpendicular to slope
    ramp_length: ClassVar[float] = 0.25  # Length along slope direction
    ramp_height: ClassVar[float] = 0.08  # Height of the slope
    platform_height: ClassVar[float] = 0.02  # Base platform thickness

    ramp_mass: ClassVar[float] = 0.0  # Static (0 mass)
    ramp_friction: ClassVar[float] = 0.5
    ramp_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.8, 0.6, 0.4, 1.0)

    def __init__(self,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 table_height: float = 0.4,
                 max_ramps: int = 10) -> None:
        """Initialize the ramp component.

        Args:
            workspace_bounds: Dictionary with x_lb, x_ub, y_lb, y_ub.
            table_height: Height of the table surface.
            max_ramps: Maximum number of ramps to create.
        """
        super().__init__()

        self.table_height = table_height
        self.max_ramps = max_ramps

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

        # Create type
        self._ramp_type = Type("ramp", ["x", "y", "z", "yaw", "pitch", "roll"])

        # Create ramp objects
        self._ramps: List[Object] = []
        for i in range(max_ramps):
            ramp_obj = Object(f"ramp_{i}", self._ramp_type)
            self._ramps.append(ramp_obj)

        # No predicates for ramps (they're just obstacles)

    # -------------------------------------------------------------------------
    # DominoEnvComponent interface implementation
    # -------------------------------------------------------------------------

    def get_types(self) -> Set[Type]:
        return {self._ramp_type}

    def get_predicates(self) -> Set:
        return set()  # Ramps don't have predicates

    def get_goal_predicates(self) -> Set:
        return set()  # Ramps never appear in goals

    def get_objects(self) -> List[Object]:
        return self._ramps

    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        """Create PyBullet bodies for ramps."""
        self._physics_client_id = physics_client_id
        bodies: Dict[str, Any] = {}

        ramp_ids = []
        for _ in range(self.max_ramps):
            ramp_id = self._create_ramp(physics_client_id)
            ramp_ids.append(ramp_id)

        bodies["ramp_ids"] = ramp_ids
        return bodies

    def store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store PyBullet body IDs on ramp objects."""
        for ramp, ramp_id in zip(self._ramps, pybullet_bodies["ramp_ids"]):
            ramp.id = ramp_id

    def reset_state(self, state: State) -> None:
        """Reset ramps to match state."""
        # Position ramps that are in the state
        ramp_objs = state.get_objects(self._ramp_type)
        for ramp in ramp_objs:
            x = state.get(ramp, "x")
            y = state.get(ramp, "y")
            z = state.get(ramp, "z")
            yaw = state.get(ramp, "yaw")
            pitch = state.get(ramp, "pitch")
            roll = state.get(ramp, "roll")

            orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
            p.resetBasePositionAndOrientation(
                ramp.id,
                posObj=[x, y, z],
                ornObj=orientation,
                physicsClientId=self._physics_client_id)

        # Move unused ramps out of view
        for i in range(len(ramp_objs), len(self._ramps)):
            p.resetBasePositionAndOrientation(
                self._ramps[i].id,
                posObj=[-10.0 - i * 0.1, -10.0, 0.0],
                ornObj=[0, 0, 0, 1],
                physicsClientId=self._physics_client_id)

    def extract_feature(self, obj: Object, feature: str) -> Optional[float]:
        """Extract feature for ramp objects."""
        if obj.type == self._ramp_type:
            pos, orn = p.getBasePositionAndOrientation(
                obj.id, physicsClientId=self._physics_client_id)

            if feature == "x":
                return pos[0]
            if feature == "y":
                return pos[1]
            if feature == "z":
                return pos[2]
            if feature in ["yaw", "pitch", "roll"]:
                euler = p.getEulerFromQuaternion(orn)
                return euler[{"roll": 0, "pitch": 1, "yaw": 2}[feature]]

        return None

    # -------------------------------------------------------------------------
    # Ramp creation
    # -------------------------------------------------------------------------
    def _create_ramp(self,
                     physics_client_id: int,
                     height: float = 0.15,
                     ramp_length: float = 0.15,
                     platform_length: float = 0.1,
                     width: float = 0.1,
                     position: Optional[list[float]] = None) -> int:
        """Creates a ramp with a flat platform using a generated .obj file for
        visuals and a convex hull for physics."""

        if position is None:
            position = [0, 0, 0]

        # Half-width for centering
        w = width / 2.0

        # --- 1. Define Vertices ---
        # Coordinates: x (length), y (width), z (height)
        # L=Left (+y), R=Right (-y)

        # Back face (x=0)
        v_back_low_L = [0, w, 0]  # v1
        v_back_low_R = [0, -w, 0]  # v2
        v_back_high_L = [0, w, height]  # v3
        v_back_high_R = [0, -w, height]  # v4

        # Transition (Platform end / Slope start) (x=platform_length)
        v_trans_high_L = [platform_length, w, height]  # v5
        v_trans_high_R = [platform_length, -w, height]  # v6

        # Tip (Slope end) (x=platform_length + ramp_length)
        total_len = platform_length + ramp_length
        v_tip_low_L = [total_len, w, 0]  # v7
        v_tip_low_R = [total_len, -w, 0]  # v8

        # List for Collision (PyBullet will auto-compute hull from these)
        collision_vertices = [
            v_back_low_L, v_back_low_R, v_back_high_L, v_back_high_R,
            v_trans_high_L, v_trans_high_R, v_tip_low_L, v_tip_low_R
        ]

        # --- 2. Generate OBJ File for Visuals ---
        # We must define the faces (indices) for the visual mesh.
        # OBJ is 1-indexed.

        obj_content = f"""
        # Vertices
        v {v_back_low_L[0]} {v_back_low_L[1]} {v_back_low_L[2]}
        v {v_back_low_R[0]} {v_back_low_R[1]} {v_back_low_R[2]}
        v {v_back_high_L[0]} {v_back_high_L[1]} {v_back_high_L[2]}
        v {v_back_high_R[0]} {v_back_high_R[1]} {v_back_high_R[2]}
        v {v_trans_high_L[0]} {v_trans_high_L[1]} {v_trans_high_L[2]}
        v {v_trans_high_R[0]} {v_trans_high_R[1]} {v_trans_high_R[2]}
        v {v_tip_low_L[0]} {v_tip_low_L[1]} {v_tip_low_L[2]}
        v {v_tip_low_R[0]} {v_tip_low_R[1]} {v_tip_low_R[2]}

        # Faces (f v1 v2 v3 ...)
        f 1 2 4 3       # Back
        f 1 7 8 2       # Bottom
        f 3 4 6 5       # Platform Top
        f 5 6 8 7       # Slope Top
        f 1 3 5 7       # Left Side
        f 2 8 6 4       # Right Side
        """

        # Create a temporary file for the OBJ
        # We keep the file explicitly to pass the name to PyBullet
        temp_obj = tempfile.NamedTemporaryFile(suffix=".obj",
                                               delete=False,
                                               mode='w')
        temp_obj.write(obj_content)
        temp_obj.close()

        try:
            # --- 3. Create PyBullet Objects ---

            # Collision Shape (uses raw vertices)
            col_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                vertices=collision_vertices,
                meshScale=[1, 1, 1],
                physicsClientId=physics_client_id)

            # Visual Shape (uses the generated OBJ file)
            vis_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=temp_obj.name,
                meshScale=[1, 1, 1],
                rgbaColor=[0.7, 0.7, 0.7, 1],
                physicsClientId=physics_client_id)

            # MultiBody
            ramp_id = p.createMultiBody(
                baseMass=1,  # Static
                baseCollisionShapeIndex=col_shape_id,
                baseVisualShapeIndex=vis_shape_id,
                basePosition=position,
                physicsClientId=physics_client_id)

        finally:
            # Cleanup: Remove the temp file after loading
            if os.path.exists(temp_obj.name):
                os.remove(temp_obj.name)

        return ramp_id

    # -------------------------------------------------------------------------
    # Initial state helpers
    # -------------------------------------------------------------------------

    def get_init_dict_entries(
        self,
        rng: "np.random.Generator",
        num_ramps: int = 1,
        ramp_positions: Optional[List[Tuple[float, float]]] = None,
        ramp_orientations: Optional[List[float]] = None
    ) -> Dict[Object, Dict[str, Any]]:
        """Return initial state dict entries for ramps.

        Args:
            rng: Random number generator.
            num_ramps: Number of ramps to place (default 2).
            ramp_positions: Optional list of (x, y) positions. If None, random.
            ramp_orientations: Optional list of yaw angles. If None, random.
        """
        init_dict: Dict[Object, Dict[str, Any]] = {}

        if num_ramps == 0:
            return init_dict

        # Generate random positions if not provided
        if ramp_positions is None:
            ramp_positions = []
            for _ in range(num_ramps):
                x = rng.uniform(self.x_lb + 0.1, self.x_ub - 0.1)
                y = rng.uniform(self.y_lb + 0.1, self.y_ub - 0.1)
                ramp_positions.append((x, y))

        # Generate random orientations if not provided
        if ramp_orientations is None:
            ramp_orientations = [
                rng.uniform(0, 2 * np.pi) for _ in range(num_ramps)
            ]

        # Create init dict entries
        for i in range(min(num_ramps, self.max_ramps)):
            x, y = ramp_positions[i]
            yaw = ramp_orientations[i]

            init_dict[self._ramps[i]] = {
                "x": x,
                "y": y,
                "z": self.table_height,
                "yaw": yaw,
                "pitch": 0.0,
                "roll": 0.0
            }

        return init_dict

    # -------------------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------------------

    @property
    def ramps(self) -> List[Object]:
        """Ramps."""
        return self._ramps

    @property
    def ramp_type(self) -> Type:
        """Ramp type."""
        return self._ramp_type
