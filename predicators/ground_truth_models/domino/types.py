"""Helper types for the domino environment."""

from typing import Set

import numpy as np

from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.env import PyBulletDominoComposedEnv
from predicators.ground_truth_models import GroundTruthTypeFactory
from predicators.structs import Object, Task, Type
from predicators.utils import PyBulletState


class PyBulletDominoGroundTruthTypeFactory(GroundTruthTypeFactory):
    """Ground-truth helper types for the domino environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "pybullet_domino", "pybullet_domino_real",
            "pybullet_domino_real_geometry"
        }

    @classmethod
    def get_helper_types(cls, env_name: str) -> Set[Type]:
        """Get helper types for the domino environment.

        Returns position and rotation types used for grid-based
        planning.
        """
        del env_name  # unused

        # The grid types (loc/angle/direction) are defined canonically by
        # GridComponent; delegate so there is a single source of truth. Only
        # oracle / process-planning approaches request these helpers (and the
        # grid predicates and processes built on them); the oracle does so
        # unconditionally, so the grid is intrinsic to it and needs no flag.
        from predicators.envs.pybullet_domino.components.grid_component import \
            GridComponent  # pylint: disable=import-outside-toplevel
        return GridComponent().get_types()

    @classmethod
    def augment_task_with_helper_objects(cls, task: Task) -> Task:
        """Augment task with helper objects for positions, angles, directions.

        Creates grid of location objects based on start and target
        domino positions, discrete angle objects, and direction objects.
        """
        # Get the helper types
        helper_types = cls.get_helper_types(env_name="")
        type_dict = {t.name: t for t in helper_types}

        # Create helper objects with features
        helper_objects = {}

        # Grid configuration (from pybullet_domino environment)
        pos_gap = PyBulletDominoComposedEnv.pos_gap

        # Get domino type from task objects
        domino_type = None
        for obj in task.init:
            if obj.type.name == "domino":
                domino_type = obj.type
                break

        if domino_type is None:
            # No dominoes in task, return unchanged
            return task

        # Find start and target dominoes to determine grid bounds
        start_domino = None
        target_dominoes = []

        for obj in task.init:
            if obj.type != domino_type:
                continue

            # Check if start domino using predicate
            if DominoComponent._StartBlock_holds(task.init, [obj]):  # pylint: disable=protected-access
                start_domino = obj

            # Check if target domino using predicate
            elif DominoComponent._TargetDomino_holds(task.init, [obj]):  # pylint: disable=protected-access
                target_dominoes.append(obj)

        # Create direction objects (like in old pybullet_domino.py)
        if "direction" in type_dict:
            direction_type = type_dict["direction"]
            direction_names = ["straight", "left", "right"]
            for i, name in enumerate(direction_names):
                direction_obj = Object(name, direction_type)
                helper_objects[direction_obj] = np.array([float(i)])

        # Create angle objects for discrete rotations
        if "angle" in type_dict:
            angle_type = type_dict["angle"]
            # Create rotation objects for 8 discrete angles
            angle_values = [-135, -90, -45, 0, 45, 90, 135, 180]  # degrees
            for angle in angle_values:
                name = f"ang_{angle}"
                angle_obj = Object(name, angle_type)
                # Convert to radians for storage
                helper_objects[angle_obj] = np.array([float(angle)])

        # Create grid of location objects based on domino positions
        if "loc" in type_dict and start_domino is not None and target_dominoes:
            loc_type = type_dict["loc"]

            # Get positions of all relevant dominoes
            all_relevant_dominoes = [start_domino] + target_dominoes
            xs = [task.init.get(d, "x") for d in all_relevant_dominoes]
            ys = [task.init.get(d, "y") for d in all_relevant_dominoes]

            # Calculate grid bounds with some padding
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Create grid with pos_gap spacing
            num_x = int(np.ceil((max_x - min_x) / pos_gap)) + 1
            num_y = int(np.ceil((max_y - min_y) / pos_gap)) + 1

            for i in range(num_x):
                for j in range(num_y):
                    x = min_x + i * pos_gap
                    y = min_y + j * pos_gap
                    # Use exact x, y values with 2 decimal places in the name
                    loc_obj = Object(f"loc_{x:.2f}_{y:.2f}", loc_type)
                    helper_objects[loc_obj] = np.array([x, y])

            # Create location objects for all other dominos (not start or
            # target)
            other_dominos = []
            for obj in task.init:
                if obj.type != domino_type:
                    continue
                if obj != start_domino and obj not in target_dominoes:
                    other_dominos.append(obj)

            # Add exact location objects for other dominos
            for domino in other_dominos:
                x = task.init.get(domino, "x")
                y = task.init.get(domino, "y")

                # Check if a location object with these coordinates already
                # exists
                location_exists = False
                for existing_loc, existing_coords in helper_objects.items():
                    if existing_loc.type == loc_type:
                        # Check if coordinates match (with small tolerance for
                        # floating point)
                        if np.allclose(existing_coords, [x, y], atol=1e-3):
                            location_exists = True
                            break

                # Only create new location object if one doesn't already exist
                if not location_exists:
                    # Use exact x, y values with 2 decimal places in the name
                    loc_obj = Object(f"loc_{x:.2f}_{y:.2f}", loc_type)
                    helper_objects[loc_obj] = np.array([x, y])

        # If no helper objects were created, return the task unchanged
        if not helper_objects:
            return task

        # Create new state data with helper objects included
        new_state_data = dict(task.init.data)
        new_state_data.update(helper_objects)

        # Create the new initial state
        new_init = PyBulletState(new_state_data, task.init.simulator_state)

        # Return new task with augmented initial state (preserve goal_nl so an
        # agent-with-grid ablation still surfaces the NL goal to the LLM).
        return Task(new_init,
                    task.goal,
                    task.alt_goal,
                    goal_nl=task.goal_nl,
                    evaluator=task.evaluator)
