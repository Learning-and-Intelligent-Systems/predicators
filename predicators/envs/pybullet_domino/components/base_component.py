"""Abstract base class for domino environment components.

Components are modular pieces of the domino environment that can be composed
together to create different environment variants. Each component is responsible
for:
- Defining its own types, predicates, and objects
- Creating and managing PyBullet bodies
- Extracting state features from objects it manages
- Resetting its state when the environment resets
- Optionally performing per-step simulation updates (e.g., physics)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import numpy as np

from predicators.structs import Object, Predicate, State, Type


class DominoEnvComponent(ABC):
    """Abstract base class for all domino environment components.

    Components encapsulate specific functionality (e.g., dominoes, fans,
    balls) and can be composed together to create different environment
    configurations.
    """

    def __init__(self) -> None:
        """Initialize the component.

        Subclasses should create their types, predicates, and objects
        here.
        """
        self._physics_client_id: Optional[int] = None

    # -------------------------------------------------------------------------
    # Abstract methods that must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_types(self) -> Set[Type]:
        """Return the types introduced by this component.

        These types will be added to the environment's type set.
        """
        raise NotImplementedError

    @abstractmethod
    def get_predicates(self) -> Set[Predicate]:
        """Return the predicates introduced by this component.

        These predicates will be added to the environment's predicate
        set.
        """
        raise NotImplementedError

    @abstractmethod
    def get_goal_predicates(self) -> Set[Predicate]:
        """Return predicates that can be used in goal specifications.

        Typically a subset of get_predicates().
        """
        raise NotImplementedError

    @abstractmethod
    def get_objects(self) -> List[Object]:
        """Return all objects managed by this component.

        These objects will be included in the environment's state.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        """Create PyBullet bodies for this component.

        Args:
            physics_client_id: The PyBullet physics client ID.

        Returns:
            Dictionary mapping string keys to PyBullet body IDs and other
            information that needs to be stored.
        """
        raise NotImplementedError

    @abstractmethod
    def store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to PyBullet body IDs on component objects.

        This is called after initialize_pybullet to associate PyBullet IDs
        with the component's Object instances.

        Args:
            pybullet_bodies: Dictionary returned by initialize_pybullet.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_state(self, state: State) -> None:
        """Reset the component to match the given state.

        This is called when the environment resets to a new task.

        Args:
            state: The target state to reset to.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_feature(self, obj: Object, feature: str) -> Optional[float]:
        """Extract a feature value for an object managed by this component.

        Args:
            obj: The object to extract the feature from.
            feature: The name of the feature to extract.

        Returns:
            The feature value, or None if this component doesn't handle
            this object/feature combination.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Optional methods that can be overridden by subclasses
    # -------------------------------------------------------------------------

    def set_physics_client_id(self, physics_client_id: int) -> None:
        """Set the physics client ID for this component.

        Called by the composed environment after PyBullet
        initialization.
        """
        self._physics_client_id = physics_client_id

    def step(self) -> None:
        """Called each simulation step.

        Override this method to add per-step physics updates (e.g., wind
        forces from fans). By default, does nothing.
        """

    def get_init_dict_entries(
            self, rng: np.random.Generator) -> Dict[Object, Dict[str, Any]]:
        """Return initial state dictionary entries for task generation.

        Override this method to provide default initial state values for
        objects managed by this component.

        Args:
            rng: Random number generator for stochastic initialization.

        Returns:
            Dictionary mapping objects to their initial feature values.
        """
        del rng  # unused in base implementation
        return {}

    def get_object_ids_for_held_check(self) -> List[int]:
        """Return PyBullet body IDs that should be checked for robot holding.

        Override this method if this component has objects that can be
        picked up by the robot.

        Returns:
            List of PyBullet body IDs.
        """
        return []

    @property
    def out_of_view_xy(self) -> tuple:
        """Return (x, y) coordinates for placing unused objects out of view."""
        return (10.0, 10.0)
