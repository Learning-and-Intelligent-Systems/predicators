"""Grid component for the domino environment.

This component handles:
- Grid-based positioning with discrete grid cells
- Discrete rotation angles (8-way: -135, -90, -45, 0, 45, 90, 135, 180)
- Grid predicates (DominoAtPos, DominoAtRot, PosClear, Connected, etc.)
- Grid coordinate generation and debug visualization
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.settings import CFG
from predicators.structs import DerivedPredicate, GroundAtom, Object, \
    Predicate, State, Type


class GridComponent(DominoEnvComponent):
    """Component for grid-based positioning and discrete rotations.

    Adds spatial discretization to the domino environment:
    - Position objects on a configurable grid
    - 8 discrete rotation angles
    - Predicates for spatial relationships between dominoes and grid cells
    """

    # Grid configuration
    debug_line_height: ClassVar[float] = 0.02

    def __init__(self,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 table_height: float = 0.4,
                 pos_gap: float = 0.098,
                 domino_type: Optional[Type] = None) -> None:
        super().__init__()

        if workspace_bounds is None:
            workspace_bounds = {
                "x_lb": 0.4,
                "x_ub": 1.1,
                "y_lb": 1.1,
                "y_ub": 1.6,
                "z_lb": 0.4,
                "z_ub": 0.95,
            }
        self.x_lb = workspace_bounds["x_lb"]
        self.x_ub = workspace_bounds["x_ub"]
        self.y_lb = workspace_bounds["y_lb"]
        self.y_ub = workspace_bounds["y_ub"]
        self.table_height = table_height
        self.pos_gap = pos_gap

        # Store domino type reference for predicate evaluation
        self._domino_type = domino_type

        # Create types
        self._position_type = Type("loc", ["xx", "yy"],
                                   sim_features=["id", "xx", "yy"])
        self._angle_type = Type("angle", ["angle"])
        self._direction_type = Type("direction", ["dir"])

        # Create rotation objects for 8 discrete angles
        self.rotations: List[Object] = []
        for angle in [-135, -90, -45, 0, 45, 90, 135, 180]:
            self.rotations.append(Object(f"ang_{angle}", self._angle_type))

        # Position objects are created per-task in get_init_dict_entries
        self.positions: List[Object] = []
        self.grid_pos: List[Tuple[float, float]] = []

        # Debug visualization
        self._debug_line_ids: List[int] = []

        # Create predicates (DerivedPredicates need domino_type set)
        self._create_predicates()

    def _create_predicates(self) -> None:
        """Create grid predicates."""
        if self._domino_type is None:
            # Can't create predicates without domino type
            return

        self._DominoAtPos = Predicate("DominoAtPos",
                                      [self._domino_type, self._position_type],
                                      self._DominoAtPos_holds)
        self._DominoAtRot = Predicate("DominoAtRot",
                                      [self._domino_type, self._angle_type],
                                      self._DominoAtRot_holds)
        self._Connected = Predicate("Connected",
                                    [self._position_type, self._position_type],
                                    self._Connected_holds)
        self._PosClear = Predicate("PosClear", [self._position_type],
                                   self._PosClear_holds)
        self._InFrontDirection = DerivedPredicate(
            "InFrontDirection",
            [self._domino_type, self._domino_type, self._direction_type],
            self._InFrontDirection_holds,
            auxiliary_predicates={self._DominoAtPos, self._DominoAtRot})
        self._InFront = DerivedPredicate(
            "InFront", [self._domino_type, self._domino_type],
            self._InFront_holds,
            auxiliary_predicates={self._InFrontDirection})
        self._AdjacentTo = DerivedPredicate(
            "AdjacentTo", [self._position_type, self._domino_type],
            self._AdjacentTo_holds,
            auxiliary_predicates={self._DominoAtPos})

    # -------------------------------------------------------------------------
    # DominoEnvComponent interface
    # -------------------------------------------------------------------------

    def get_types(self) -> Set[Type]:
        return {self._position_type, self._angle_type, self._direction_type}

    def get_predicates(self) -> Set[Predicate]:
        if self._domino_type is None:
            return set()
        preds = {
            self._DominoAtPos,
            self._DominoAtRot,
            self._PosClear,
            self._InFrontDirection,
            self._InFront,
        }
        if CFG.domino_include_connected_predicate:
            preds.add(self._Connected)
        else:
            preds.add(self._AdjacentTo)
        return preds

    def get_goal_predicates(self) -> Set[Predicate]:
        return set()

    def get_objects(self) -> List[Object]:
        return self.positions + self.rotations

    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        self._physics_client_id = physics_client_id
        return {}

    def store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        pass

    def reset_state(self, state: State) -> None:
        assert self._physics_client_id is not None
        # Clear existing debug lines
        for line_id in self._debug_line_ids:
            p.removeUserDebugItem(line_id,
                                  physicsClientId=self._physics_client_id)
        self._debug_line_ids = []

        # Draw debug lines at grid cell centers
        position_objs = state.get_objects(self._position_type)
        for pos_obj in position_objs:
            x = state.get(pos_obj, "xx")
            y = state.get(pos_obj, "yy")
            line_id = p.addUserDebugLine(
                [x, y, self.table_height],
                [x, y, self.table_height + self.debug_line_height], [1, 0, 0],
                parentObjectUniqueId=-1,
                parentLinkIndex=-1,
                physicsClientId=self._physics_client_id)
            self._debug_line_ids.append(line_id)

    def extract_feature(self, obj: Object, feature: str) -> Optional[float]:
        # Grid helper-object features (loc/angle/direction) are encoded in
        # their names; reuse the canonical name-based reconstruction.
        return self.reconstruct_feature_from_name(obj, feature)

    @staticmethod
    def reconstruct_feature_from_name(obj: Object,
                                      feature: str) -> Optional[float]:
        """Reconstruct a grid helper-object feature from its name.

        The grid helper objects (loc/angle/direction) are injected into
        tasks by the ground-truth models and carry no PyBullet body, so
        their feature values are encoded in their names (e.g.
        "loc_0.47_1.28", "ang_-90", "straight"). The composed env calls
        this during its _get_state round-trip, where there is no live
        GridComponent to query (these objects appear only inside oracle /
        process-planning, which requires the grid).

        Returns None for non-grid objects/features so the caller can fall
        through to its own error handling.
        """
        if obj.type.name == "loc" and feature in ("xx", "yy"):
            # Name format: "loc_<x>_<y>", e.g. "loc_0.47_1.28".
            _, x_str, y_str = obj.name.split("_")
            return float(x_str) if feature == "xx" else float(y_str)
        if obj.type.name == "angle" and feature == "angle":
            # Name format: "ang_<degrees>", e.g. "ang_-90".
            return float(obj.name.split("_")[1])
        if obj.type.name == "direction" and feature == "dir":
            return {"straight": 0.0, "left": 1.0, "right": 2.0}[obj.name]
        return None

    def get_init_dict_entries(
            self, rng: np.random.Generator) -> Dict[Object, Dict[str, Any]]:
        """Return initial state entries for rotation objects."""
        entries: Dict[Object, Dict[str, Any]] = {}
        for rot_obj in self.rotations:
            angle_str = rot_obj.name.split("_")[1]
            entries[rot_obj] = {"angle": float(angle_str)}
        return entries

    # -------------------------------------------------------------------------
    # Grid coordinate generation
    # -------------------------------------------------------------------------

    def generate_grid_coordinates(
            self, num_pos_x: int,
            num_pos_y: int) -> Tuple[List[float], List[float]]:
        """Generate grid coordinates centered within the workspace."""
        total_x_range = self.x_ub - self.x_lb
        total_y_range = self.y_ub - self.y_lb

        x_start = self.x_lb + (total_x_range -
                               (num_pos_x - 1) * self.pos_gap) / 2
        y_start = self.y_lb + (total_y_range -
                               (num_pos_y - 1) * self.pos_gap) / 2

        x_coords = [
            round(x_start + i * self.pos_gap, 5) for i in range(num_pos_x)
        ]
        y_coords = [
            round(y_start + i * self.pos_gap, 5) for i in range(num_pos_y)
        ]
        return x_coords, y_coords

    def create_position_objects(
            self, num_pos_x: int,
            num_pos_y: int) -> Tuple[List[Object], Dict[Object, Dict]]:
        """Create position objects and their initial state dicts for a task."""
        x_coords, y_coords = self.generate_grid_coordinates(
            num_pos_x, num_pos_y)
        self.grid_pos = [(x, y) for y in y_coords for x in x_coords]

        self.positions = []
        pos_dict: Dict[Object, Dict] = {}
        pos_index = 0
        for i in range(num_pos_y):
            for j in range(num_pos_x):
                name = f"loc_y{i}_x{j}"
                obj = Object(name, self._position_type)
                obj.xx = x_coords[j]
                obj.yy = y_coords[i]
                self.positions.append(obj)
                pos_dict[obj] = {"xx": x_coords[j], "yy": y_coords[i]}
                pos_index += 1

        return self.positions, pos_dict

    # -------------------------------------------------------------------------
    # Predicate implementations
    # -------------------------------------------------------------------------

    def _DominoAtPos_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is at a specific grid position (closest match)."""
        domino, position = objects
        if state.get(domino, "is_held"):
            return False

        domino_x = state.get(domino, "x")
        domino_y = state.get(domino, "y")

        closest_position = None
        closest_distance = float('inf')
        for pos in state.get_objects(self._position_type):
            pos_x = state.get(pos, "xx")
            pos_y = state.get(pos, "yy")
            distance = np.sqrt((domino_x - pos_x)**2 + (domino_y - pos_y)**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_position = pos
        return closest_position == position

    def _DominoAtRot_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is at a specific discrete rotation (15deg
        tolerance)."""
        domino, rotation = objects
        if state.get(domino, "is_held"):
            return False

        domino_rot = state.get(domino, "yaw")
        target_rot_radians = np.radians(state.get(rotation, "angle"))
        rotation_tolerance = np.radians(22)
        angle_diff = abs(utils.wrap_angle(domino_rot - target_rot_radians))
        return angle_diff <= rotation_tolerance

    def _Connected_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        """Check if two positions are adjacent in cardinal directions."""
        pos1, pos2 = objects
        if pos1.name == pos2.name:
            return False

        x1, y1 = state.get(pos1, "xx"), state.get(pos1, "yy")
        x2, y2 = state.get(pos2, "xx"), state.get(pos2, "yy")

        dx, dy = abs(x1 - x2), abs(y1 - y2)
        tolerance = self.pos_gap * 0.1

        x_adjacent = abs(dx - self.pos_gap) < tolerance and dy < tolerance
        y_adjacent = abs(dy - self.pos_gap) < tolerance and dx < tolerance
        return x_adjacent or y_adjacent

    @staticmethod
    def _PosClear_holds(state: State, objects: Sequence[Object]) -> bool:
        """Check if a position is clear (not occupied by any domino).

        A position is considered clear if no domino is currently at that
        position. The occupancy tolerance is derived from the grid
        spacing (half the smallest gap between location objects).
        """
        position, = objects

        target_x = state.get(position, "xx")
        target_y = state.get(position, "yy")

        # Calculate grid spacing (minimum distance between positions).
        position_type = position.type
        positions = list(state.get_objects(position_type))
        min_distance = float('inf')
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1:]:
                x1 = state.get(pos1, "xx")
                y1 = state.get(pos1, "yy")
                x2 = state.get(pos2, "xx")
                y2 = state.get(pos2, "yy")
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance > 1e-6:  # Skip identical positions
                    min_distance = min(min_distance, distance)
        position_tolerance = (min_distance *
                              0.5 if min_distance != float('inf') else 0.1)

        for obj in state:
            if obj.type.name == "domino":
                domino_x = state.get(obj, "x")
                domino_y = state.get(obj, "y")
                if (abs(domino_x - target_x) <= position_tolerance
                        and abs(domino_y - target_y) <= position_tolerance
                        and not state.get(obj, "is_held")):
                    return False
        return True

    @staticmethod
    def _InFrontDirection_holds(atoms: Set[GroundAtom],
                                objects: Sequence[Object]) -> bool:
        """Check if domino1 is in front of domino2 in the given direction.

        Uses decoupled positional and rotational checks for efficiency.
        """
        domino1, domino2, direction_obj = objects

        _pos_coord_cache: Dict[Object, Tuple[float, float]] = {}
        _rot_rad_cache: Dict[Object, float] = {}

        def extract_coords(pos_obj: Object) -> Tuple[float, float]:
            # Location names encode continuous coords, e.g. "loc_0.49_1.23".
            if pos_obj in _pos_coord_cache:
                return _pos_coord_cache[pos_obj]
            name_parts = pos_obj.name.split("_")
            result = (float(name_parts[1]), float(name_parts[2]))
            _pos_coord_cache[pos_obj] = result
            return result

        def extract_rotation_angle_rad(rot_obj: Object) -> float:
            if rot_obj in _rot_rad_cache:
                return _rot_rad_cache[rot_obj]
            angle_str = rot_obj.name.split("_")[1]
            result = np.radians(float(angle_str))
            _rot_rad_cache[rot_obj] = result
            return result

        d1_positions = {
            extract_coords(a.objects[1])
            for a in atoms
            if a.predicate.name == "DominoAtPos" and a.objects[0] == domino1
        }
        d1_rotations = {
            extract_rotation_angle_rad(a.objects[1])
            for a in atoms
            if a.predicate.name == "DominoAtRot" and a.objects[0] == domino1
        }
        d2_positions = {
            extract_coords(a.objects[1])
            for a in atoms
            if a.predicate.name == "DominoAtPos" and a.objects[0] == domino2
        }
        d2_rotations = {
            extract_rotation_angle_rad(a.objects[1])
            for a in atoms
            if a.predicate.name == "DominoAtRot" and a.objects[0] == domino2
        }

        def _check_case(front_pos: Set[Tuple[float, float]],
                        front_rot: Set[float],
                        back_pos: Set[Tuple[float, float]],
                        back_rot: Set[float],
                        direction_name: str,
                        tolerance: float = 1e-6) -> bool:
            if not all([front_pos, front_rot, back_pos, back_rot]):
                return False

            # pos_gap is the physical spacing between adjacent grid cells.
            from predicators.envs.pybullet_domino.env import \
                PyBulletDominoComposedEnv  # pylint: disable=import-outside-toplevel
            pos_gap = PyBulletDominoComposedEnv.pos_gap

            position_possible = False
            for (x_b, y_b) in back_pos:
                for rot_b in back_rot:
                    # Relationship only holds for cardinal rotations.
                    if not (abs(np.sin(rot_b)) < tolerance
                            or abs(np.cos(rot_b)) < tolerance):
                        continue
                    expected_x = x_b + pos_gap * np.sin(rot_b)
                    expected_y = y_b + pos_gap * np.cos(rot_b)
                    for (x_f, y_f) in front_pos:
                        if (abs(x_f - expected_x) < pos_gap * 0.3
                                and abs(y_f - expected_y) < pos_gap * 0.3):
                            position_possible = True
                            break
                    if position_possible:
                        break
                if position_possible:
                    break

            if not position_possible:
                return False

            if direction_name == "left":
                expected_diff = np.pi / 4
            elif direction_name == "straight":
                expected_diff = 0
            elif direction_name == "right":
                expected_diff = -np.pi / 4
            else:
                return False

            for rot_b in back_rot:
                for rot_f in front_rot:
                    diff = utils.wrap_angle(rot_f - rot_b)
                    if abs(diff - expected_diff) < tolerance:
                        return True
            return False

        dir_name = direction_obj.name
        opposite = {"left": "right", "right": "left"}.get(dir_name, dir_name)

        if _check_case(d1_positions, d1_rotations, d2_positions, d2_rotations,
                       dir_name):
            return True
        if _check_case(d2_positions, d2_rotations, d1_positions, d1_rotations,
                       opposite):
            return True
        return False

    @staticmethod
    def _InFront_holds(atoms: Set[GroundAtom],
                       objects: Sequence[Object]) -> bool:
        """Check if domino1 is in front of domino2 in any direction."""
        domino1, domino2 = objects
        for atom in atoms:
            if (atom.predicate.name == "InFrontDirection"
                    and len(atom.objects) == 3 and atom.objects[0] == domino1
                    and atom.objects[1] == domino2):
                return True
        return False

    @staticmethod
    def _AdjacentTo_holds(atoms: Set[GroundAtom],
                          objects: Sequence[Object]) -> bool:
        """Check if a position is adjacent to a domino in cardinal directions.

        Adjacent means about one ``pos_gap`` away in a cardinal
        direction (up/down/left/right) but not diagonal, over the
        continuous-coordinate location names (e.g. ``loc_0.49_1.23``).
        """
        position, domino = objects

        _pos_coord_cache: Dict[Object, Tuple[float, float]] = {}

        def extract_coords(pos_obj: Object) -> Tuple[float, float]:
            if pos_obj in _pos_coord_cache:
                return _pos_coord_cache[pos_obj]
            name_parts = pos_obj.name.split("_")
            result = (float(name_parts[1]), float(name_parts[2]))
            _pos_coord_cache[pos_obj] = result
            return result

        # pos_gap is the physical spacing between adjacent grid cells.
        from predicators.envs.pybullet_domino.env import \
            PyBulletDominoComposedEnv  # pylint: disable=import-outside-toplevel
        pos_gap = PyBulletDominoComposedEnv.pos_gap

        target_x, target_y = extract_coords(position)

        domino_positions = {
            extract_coords(a.objects[1])
            for a in atoms
            if a.predicate.name == "DominoAtPos" and a.objects[0] == domino
        }

        for domino_x, domino_y in domino_positions:
            dx = abs(target_x - domino_x)
            dy = abs(target_y - domino_y)
            if ((abs(dx - pos_gap) < pos_gap * 0.3 and dy < pos_gap * 0.3) or
                (abs(dy - pos_gap) < pos_gap * 0.3 and dx < pos_gap * 0.3)):
                return True
        return False

    # -------------------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------------------

    @property
    def position_type(self) -> Type:
        """Position type."""
        return self._position_type

    @property
    def angle_type(self) -> Type:
        """Angle type."""
        return self._angle_type
