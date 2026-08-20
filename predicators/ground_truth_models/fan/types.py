"""Helper types for the fan environment.

Mirrors the domino design: the grid ``loc``/``side`` helper objects are
NOT part of the env's normal (agent-visible) state. They are injected
into a task at plan time by the oracle / process-planning approach via
``augment_task_with_helper_objects`` so that the agent runs grid-free
while the oracle gets the spatial scaffolding it needs.
"""

from typing import Dict, Optional, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthTypeFactory
from predicators.structs import GroundAtom, Object, Predicate, State, Task, \
    Type
from predicators.utils import PyBulletState

# The ``side`` helper objects and their ``side_idx`` values, reproducing
# what the fan env used to bake into every task (pybullet_fan.py). The name
# is the human-facing direction; the index is the value the grid predicates
# compare against ``fan.facing_side``.
_SIDE_NAME_TO_IDX = {"left": 1.0, "right": 0.0, "down": 3.0, "up": 2.0}


class PyBulletFanGroundTruthTypeFactory(GroundTruthTypeFactory):
    """Ground-truth helper types for the fan environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_fan"}

    @classmethod
    def get_helper_types(cls, env_name: str) -> Set[Type]:
        """The grid helper types (``loc`` positions, ``side`` directions).

        Delegates to the env's canonical type objects so there is a
        single source of truth. Only oracle / process-planning
        approaches request these; agent approaches run grid-free.
        """
        del env_name  # unused
        from predicators.envs.pybullet_fan import \
            PyBulletFanEnv  # pylint: disable=import-outside-toplevel
        return {PyBulletFanEnv._location_type, PyBulletFanEnv._side_type}  # pylint: disable=protected-access

    @classmethod
    def augment_state_with_helper_objects(cls, state: State) -> State:
        """Inject the grid ``loc``/``side`` objects into a state.

        Rebuilds the exact task grid (reusing the env helper that infers the
        train/test grid) and adds the four ``side`` direction objects.
        Coordinates are encoded in each ``loc`` cell's name so the env can
        reconstruct their features during its ``_get_state`` round-trip.

        This is used both to augment the task's initial state (for planning)
        and - crucially - to re-derive the grid on every execution state, so
        the oracle's closed-loop ``Wait`` options can keep tracking the ball's
        cell-by-cell ``BallAtLoc`` progress even though the agent-visible state
        is grid-free.

        The grid is inferred from the stationary TARGET (always on a grid
        cell), NOT the ball: during execution the ball moves between cells and
        would momentarily align to the other grid phase, flipping the whole
        injected grid step-to-step and desyncing the policy. Returns the state
        unchanged if there is no ball.
        """
        from predicators.envs.pybullet_fan import \
            PyBulletFanEnv  # pylint: disable=import-outside-toplevel

        helper_types = {t.name: t for t in cls.get_helper_types(env_name="")}
        loc_type = helper_types["loc"]
        side_type = helper_types["side"]

        ball_obj = next((o for o in state if o.type.name == "ball"), None)
        if ball_obj is None:
            return state
        # Grid objects may already be present (e.g. an already-augmented
        # state); nothing to do then.
        if any(o.type.name == "loc" for o in state):
            return state

        # Use the stationary target (on-grid) as the grid reference; fall back
        # to the ball only if there is no target.
        ref_obj = next((o for o in state if o.type.name == "target"), ball_obj)
        ref_x = state.get(ref_obj, "x")
        ref_y = state.get(ref_obj, "y")
        x_coords, y_coords = PyBulletFanEnv._grid_coords_for_point(  # pylint: disable=protected-access
            ref_x, ref_y)

        helper_objects: Dict[Object, np.ndarray] = {}
        # Location objects: encode coords in the name (loc_<x>_<y>).
        for x in x_coords:
            for y in y_coords:
                loc_obj = Object(f"loc_{x:.4f}_{y:.4f}", loc_type)
                helper_objects[loc_obj] = np.array([x, y], dtype=np.float32)
        # Side objects: name encodes the direction, feature is side_idx.
        for side_name, side_idx in _SIDE_NAME_TO_IDX.items():
            side_obj = Object(side_name, side_type)
            helper_objects[side_obj] = np.array([side_idx], dtype=np.float32)

        new_state_data = dict(state.data)
        new_state_data.update(helper_objects)
        return PyBulletState(new_state_data, state.simulator_state)

    @classmethod
    def augment_task_with_helper_objects(cls, task: Task) -> Task:
        """Inject the grid ``loc``/``side`` objects and re-express the goal.

        Injects the grid into the initial state (see
        ``augment_state_with_helper_objects``) and rewrites the physical
        goal ``BallAtTarget(ball, target)`` into the grid goal
        ``BallAtLoc(ball, target_loc)`` where ``target_loc`` is the
        injected cell nearest the physical target, so the oracle's grid
        processes can achieve it. Non-grid goal atoms (e.g. FanOff) are
        preserved.
        """
        # Locate the ball and target physical objects.
        ball_obj: Optional[Object] = None
        target_obj: Optional[Object] = None
        for obj in task.init:
            if obj.type.name == "ball":
                ball_obj = obj
            elif obj.type.name == "target":
                target_obj = obj
        if ball_obj is None or target_obj is None:
            # Nothing to scaffold; return unchanged.
            return task

        new_init = cls.augment_state_with_helper_objects(task.init)
        loc_type = {t.name: t for t in cls.get_helper_types("")}["loc"]

        # Rewrite the goal: BallAtTarget(ball, target) -> BallAtLoc(ball, loc).
        target_x = task.init.get(target_obj, "x")
        target_y = task.init.get(target_obj, "y")
        loc_objs = [o for o in new_init if o.type.name == "loc"]
        target_loc = min(loc_objs,
                         key=lambda o: (new_init.get(o, "xx") - target_x)**2 +
                         (new_init.get(o, "yy") - target_y)**2)
        ball_at_loc = cls._get_ball_at_loc_predicate(ball_obj.type, loc_type)
        new_goal = set()
        for atom in task.goal:
            if atom.predicate.name == "BallAtTarget":
                new_goal.add(GroundAtom(ball_at_loc, [ball_obj, target_loc]))
            else:
                new_goal.add(atom)

        # Preserve goal_nl: the agent-with-grid ablation surfaces it to the
        # LLM even though the symbolic goal is now the grid BallAtLoc.
        return Task(new_init,
                    new_goal,
                    task.alt_goal,
                    goal_nl=task.goal_nl,
                    evaluator=task.evaluator)

    @staticmethod
    def _get_ball_at_loc_predicate(ball_type: Type,
                                   loc_type: Type) -> Predicate:
        """Fetch the injected-grid ``BallAtLoc`` predicate for the goal."""
        from predicators.envs.pybullet_fan import \
            PyBulletFanEnv  # pylint: disable=import-outside-toplevel
        from predicators.ground_truth_models.fan.predicates import \
            PyBulletFanGroundTruthPredicateFactory  # pylint: disable=import-outside-toplevel
        types_dict = {
            "ball": ball_type,
            "fan": PyBulletFanEnv._fan_type,  # pylint: disable=protected-access
            "loc": loc_type,
            "side": PyBulletFanEnv._side_type,  # pylint: disable=protected-access
        }
        preds = PyBulletFanGroundTruthPredicateFactory.get_helper_predicates(
            "pybullet_fan", types_dict)
        return next(p for p in preds if p.name == "BallAtLoc")
