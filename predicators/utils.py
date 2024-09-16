"""General utility methods."""

from __future__ import annotations

import abc
import contextlib
import curses
import functools
import gc
import heapq as hq
import importlib
import io
import itertools
import logging
import os
import pkgutil
import re
import subprocess
import sys
import tempfile
import time
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Collection, Dict, \
    FrozenSet, Generator, Generic, Hashable, Iterator, List, Optional, \
    Sequence, Set, Tuple
from typing import Type as TypingType
from typing import TypeVar, Union, cast

try:  # pragma: no cover
    from gtts import gTTS
    from playsound import playsound
    _TTS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _TTS_AVAILABLE = False

import dill as pkl
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as mp
import PIL.Image
from bosdyn.client import math_helpers
from gym.spaces import Box
from matplotlib import patches
from numpy.typing import NDArray
from PIL import ImageDraw, ImageFont
from pyperplan.heuristics.heuristic_base import \
    Heuristic as _PyperplanBaseHeuristic
from pyperplan.planner import HEURISTICS as _PYPERPLAN_HEURISTICS
from scipy.stats import beta as BetaRV

from predicators.args import create_arg_parser
from predicators.pretrained_model_interface import GoogleGeminiLLM, \
    GoogleGeminiVLM, LargeLanguageModel, OpenAILLM, OpenAIVLM, \
    VisionLanguageModel
from predicators.pybullet_helpers.joint import JointPositions
from predicators.settings import CFG, GlobalSettings
from predicators.structs import NSRT, Action, Array, DummyOption, \
    EntToEntSub, GroundAtom, GroundAtomTrajectory, \
    GroundNSRTOrSTRIPSOperator, Image, LDLRule, LiftedAtom, \
    LiftedDecisionList, LiftedOrGroundAtom, LowLevelTrajectory, Metrics, \
    NSRTOrSTRIPSOperator, Object, ObjectOrVariable, Observation, OptionSpec, \
    ParameterizedOption, Predicate, Segment, SpotAction, SpotActionExtraInfo, \
    State, STRIPSOperator, Task, Type, Variable, VarToObjSub, Video, \
    VLMPredicate, _GroundLDLRule, _GroundNSRT, _GroundSTRIPSOperator, \
    _Option, _TypedEntity
from predicators.third_party.fast_downward_translator.translate import \
    main as downward_translate

if TYPE_CHECKING:
    from predicators.envs import BaseEnv

# NOTE: some spot utilities use plt.show(), which requires a GUI back-end.
# But testing in headless mode requires a non-GUI backend.
try:
    matplotlib.use("TkAgg")
except ImportError:  # pragma: no cover
    matplotlib.use("Agg")

# Unpickling CUDA models errs out if the device isn't recognized because of
# an unusual name, including in supercloud, but we can set it manually
if "CUDA_VISIBLE_DEVICES" in os.environ:  # pragma: no cover
    cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(cuda_visible_devices) and cuda_visible_devices[0] != "0":
        cuda_visible_devices[0] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)


def count_positives_for_ops(
    strips_ops: List[STRIPSOperator],
    option_specs: List[OptionSpec],
    segments: List[Segment],
    max_groundings: Optional[int] = None,
) -> Tuple[int, int, List[Set[int]], List[Set[int]]]:
    """Returns num true positives, num false positives, and for each strips op,
    lists of segment indices that contribute true or false positives.

    The lists of segment indices are useful only for debugging; they are
    otherwise redundant with num_true_positives/num_false_positives.
    """
    assert len(strips_ops) == len(option_specs)
    num_true_positives = 0
    num_false_positives = 0
    # The following two lists are just useful for debugging.
    true_positive_idxs: List[Set[int]] = [set() for _ in strips_ops]
    false_positive_idxs: List[Set[int]] = [set() for _ in strips_ops]
    for seg_idx, segment in enumerate(segments):
        objects = set(segment.states[0])
        segment_option = segment.get_option()
        option_objects = segment_option.objects
        covered_by_some_op = False
        # Ground only the operators with a matching option spec.
        for op_idx, (op,
                     option_spec) in enumerate(zip(strips_ops, option_specs)):
            # If the parameterized options are different, not relevant.
            if option_spec[0] != segment_option.parent:
                continue
            option_vars = option_spec[1]
            assert len(option_vars) == len(option_objects)
            option_var_to_obj = dict(zip(option_vars, option_objects))
            # We want to get all ground operators whose corresponding
            # substitution is consistent with the option vars for this
            # segment. So, determine all of the operator variables
            # that are not in the option vars, and consider all
            # groundings of them.
            for grounding_idx, ground_op in enumerate(
                    all_ground_operators_given_partial(op, objects,
                                                       option_var_to_obj)):
                if max_groundings is not None and \
                    grounding_idx > max_groundings:
                    break
                # Check the ground_op against the segment.
                if not ground_op.preconditions.issubset(segment.init_atoms):
                    continue
                if ground_op.add_effects == segment.add_effects and \
                    ground_op.delete_effects == segment.delete_effects:
                    covered_by_some_op = True
                    true_positive_idxs[op_idx].add(seg_idx)
                else:
                    false_positive_idxs[op_idx].add(seg_idx)
                    num_false_positives += 1
        if covered_by_some_op:
            num_true_positives += 1
    return num_true_positives, num_false_positives, \
        true_positive_idxs, false_positive_idxs


def count_branching_factor(strips_ops: List[STRIPSOperator],
                           segments: List[Segment]) -> int:
    """Returns the total branching factor for all states in the segments."""
    total_branching_factor = 0
    for segment in segments:
        atoms = segment.init_atoms
        objects = set(segment.states[0])
        ground_ops = {
            ground_op
            for op in strips_ops
            for ground_op in all_ground_operators(op, objects)
        }
        for _ in get_applicable_operators(ground_ops, atoms):
            total_branching_factor += 1
    return total_branching_factor


def segment_trajectory_to_start_end_state_sequence(
        seg_traj: List[Segment]) -> List[State]:
    """Convert a trajectory of segments into a trajectory of states, made up of
    only the initial/final states of the segments.

    The length of the return value will always be one greater than the
    length of the given seg_traj.
    """
    assert len(seg_traj) >= 1
    states = []
    for i, seg in enumerate(seg_traj):
        states.append(seg.states[0])
        if i < len(seg_traj) - 1:
            assert seg.states[-1].allclose(seg_traj[i + 1].states[0])
    states.append(seg_traj[-1].states[-1])
    assert len(states) == len(seg_traj) + 1
    return states


def segment_trajectory_to_atoms_sequence(
        seg_traj: List[Segment]) -> List[Set[GroundAtom]]:
    """Convert a trajectory of segments into a trajectory of ground atoms.

    The length of the return value will always be one greater than the
    length of the given seg_traj.
    """
    assert len(seg_traj) >= 1
    atoms_seq = []
    for i, seg in enumerate(seg_traj):
        atoms_seq.append(seg.init_atoms)
        if i < len(seg_traj) - 1:
            assert seg.final_atoms == seg_traj[i + 1].init_atoms
    atoms_seq.append(seg_traj[-1].final_atoms)
    assert len(atoms_seq) == len(seg_traj) + 1
    return atoms_seq


def num_options_in_action_sequence(actions: Sequence[Action]) -> int:
    """Given a sequence of actions with options included, get the number of
    options that are encountered."""
    num_options = 0
    last_option = None
    for action in actions:
        current_option = action.get_option()
        if not current_option is last_option:
            last_option = current_option
            num_options += 1
    return num_options


def entropy(p: float) -> float:
    """Entropy of a Bernoulli variable with parameter p."""
    assert 0.0 <= p <= 1.0
    if p in {0.0, 1.0}:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def create_state_from_dict(data: Dict[Object, Dict[str, float]],
                           simulator_state: Optional[Any] = None) -> State:
    """Small utility to generate a state from a dictionary `data` of individual
    feature values for each object.

    A simulator_state for the outputted State may optionally be
    provided.
    """
    state_dict = {}
    for obj, obj_data in data.items():
        obj_vec = []
        for feat in obj.type.feature_names:
            obj_vec.append(obj_data[feat])
        state_dict[obj] = np.array(obj_vec)
    return State(state_dict, simulator_state)


def create_json_dict_from_ground_atoms(
        ground_atoms: Collection[GroundAtom]) -> Dict[str, List[List[str]]]:
    """Saves a set of ground atoms in a JSON-compatible dict.

    Helper for creating the goal dict in create_json_dict_from_task().
    """
    predicate_to_argument_lists = defaultdict(list)
    for atom in sorted(ground_atoms):
        argument_list = [o.name for o in atom.objects]
        predicate_to_argument_lists[atom.predicate.name].append(argument_list)
    return dict(predicate_to_argument_lists)


def create_json_dict_from_task(task: Task) -> Dict[str, Any]:
    """Create a JSON-compatible dict from a task.

    The format of the dict is:

    {
        "objects": {
            <object name>: <type name>
        }
        "init": {
            <object name>: {
                <feature name>: <value>
            }
        }
        "goal": {
            <predicate name> : [
                [<object name>]
            ]
        }
    }

    The dict can be loaded with BaseEnv._load_task_from_json(). This is
    helpful for testing and designing standalone tasks.
    """
    object_dict = {o.name: o.type.name for o in task.init}
    init_dict = {
        o.name: dict(zip(o.type.feature_names, task.init.data[o]))
        for o in task.init
    }
    goal_dict = create_json_dict_from_ground_atoms(task.goal)
    return {"objects": object_dict, "init": init_dict, "goal": goal_dict}


def prompt_user(prompt: str) -> str:  # pragma: no cover
    """Ask the user for input with voice and text."""
    if _TTS_AVAILABLE:
        with tempfile.NamedTemporaryFile() as voice:
            gTTS(text=prompt, lang="en").write_to_fp(voice)
            playsound(voice.name)
    return input(prompt)


def wait_for_any_button_press(msg: str) -> None:  # pragma: no cover
    """Print some text and wait for the user to press any button."""
    stdscr = curses.initscr()
    curses.noecho()
    stdscr.addstr(msg)
    stdscr.getkey()
    curses.flushinp()
    curses.endwin()


def construct_active_sampler_input(state: State, objects: Sequence[Object],
                                   params: Array,
                                   param_option: ParameterizedOption) -> Array:
    """Helper function for active sampler learning and explorer."""

    assert not CFG.sampler_learning_use_goals
    sampler_input_lst = [1.0]  # start with bias term
    if CFG.active_sampler_learning_feature_selection == "all":
        for obj in objects:
            sampler_input_lst.extend(state[obj])
        sampler_input_lst.extend(params)

    else:
        assert CFG.active_sampler_learning_feature_selection == "oracle"
        if CFG.env == "bumpy_cover":
            if param_option.name == "Pick":
                # In this case, the x-data should be
                # [block_bumpy, relative_pick_loc]
                assert len(objects) == 1
                block = objects[0]
                block_pos = state[block][3]
                block_bumpy = state[block][5]
                sampler_input_lst.append(block_bumpy)
                assert len(params) == 1
                sampler_input_lst.append(params[0] - block_pos)
            else:
                assert param_option.name == "Place"
                assert len(objects) == 2
                block, target = objects
                target_pos = state[target][3]
                grasp = state[block][4]
                target_width = state[target][2]
                sampler_input_lst.extend([grasp, target_width])
                assert len(params) == 1
                sampler_input_lst.append(params[0] - target_pos)
        elif CFG.env == "ball_and_cup_sticky_table":
            if "PlaceCup" in param_option.name and "Table" in param_option.name:
                _, _, _, table = objects
                table_y = state.get(table, "y")
                table_x = state.get(table, "x")
                sticky = state.get(table, "sticky")
                sticky_region_x = state.get(table, "sticky_region_x_offset")
                sticky_region_y = state.get(table, "sticky_region_y_offset")
                sticky_region_radius = state.get(table, "sticky_region_radius")
                table_radius = state.get(table, "radius")
                _, _, _, param_x, param_y = params
                sampler_input_lst.append(table_radius)
                sampler_input_lst.append(sticky)
                sampler_input_lst.append(sticky_region_x)
                sampler_input_lst.append(sticky_region_y)
                sampler_input_lst.append(sticky_region_radius)
                sampler_input_lst.append(table_x)
                sampler_input_lst.append(table_y)
                sampler_input_lst.append(param_x)
                sampler_input_lst.append(param_y)
            else:  # Use all features.
                for obj in objects:
                    sampler_input_lst.extend(state[obj])
                sampler_input_lst.extend(params)
        elif "spot" in CFG.env:  # pragma: no cover
            if "SweepIntoContainer" in param_option.name:
                _, _, target, _, container = objects
                for obj in [target, container]:
                    sampler_input_lst.append(state.get(obj, "x"))
                    sampler_input_lst.append(state.get(obj, "y"))
                sampler_input_lst.extend(params)
            elif "SweepTwoObjects" in param_option.name:
                _, _, target1, target2, _, container = objects
                for obj in [target1, target2, container]:
                    sampler_input_lst.append(state.get(obj, "x"))
                    sampler_input_lst.append(state.get(obj, "y"))
                sampler_input_lst.extend(params)
            elif "Pick" in param_option.name:
                if not CFG.active_sampler_learning_object_specific_samplers:
                    target_obj = objects[1]
                    object_id = state.get(target_obj, "object_id")
                    sampler_input_lst.append(object_id)
                sampler_input_lst.extend(params)
            elif "Place" in param_option.name and "OnTop" in param_option.name:
                surface_obj = objects[2]
                if not CFG.active_sampler_learning_object_specific_samplers:
                    held_obj = objects[1]
                    sampler_input_lst.append(state.get(held_obj, "object_id"))
                    sampler_input_lst.append(
                        state.get(surface_obj, "object_id"))
                if surface_obj.type.name == "drafting_table":
                    sampler_input_lst.extend([
                        state.get(surface_obj, "sticky-region-x"),
                        state.get(surface_obj, "sticky-region-y")
                    ])
                else:
                    sampler_input_lst.extend([0.0, 0.0])
                # Samples are relative dx, dy, dz, and we only need
                # dx and dy for the table!
                sampler_input_lst.extend(params[:2])
            elif param_option.name.startswith("DropObjectInside") and \
                len(objects) == 3:
                if not CFG.active_sampler_learning_object_specific_samplers:
                    _, held_obj, container_obj = objects
                    held_object_id = state.get(held_obj, "object_id")
                    sampler_input_lst.append(held_object_id)
                    container_obj_id = state.get(container_obj, "object_id")
                    sampler_input_lst.append(container_obj_id)
                sampler_input_lst.extend(params)
            else:
                base_feat_names = [
                    "x",
                    "y",
                    "z",
                    "qw",
                    "qx",
                    "qy",
                    "qz",
                    "shape",
                    "height",
                    "width",
                    "length",
                ]
                if not CFG.active_sampler_learning_object_specific_samplers:
                    base_feat_names.append("object_id")
                for obj in objects:
                    if obj.type.name == "robot":
                        sampler_input_lst.extend(state[obj])
                    else:
                        for feat in base_feat_names:
                            sampler_input_lst.append(state.get(obj, feat))
                sampler_input_lst.extend(params)
        else:
            raise NotImplementedError("Oracle feature selection not "
                                      f"implemented for {CFG.env}")

    return np.array(sampler_input_lst)


class _Geom2D(abc.ABC):
    """A 2D shape that contains some points."""

    @abc.abstractmethod
    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        """Plot the shape on a given pyplot axis."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def contains_point(self, x: float, y: float) -> bool:
        """Checks if a point is contained in the shape."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def sample_random_point(
            self,
            rng: np.random.Generator,
            min_dist_from_edge: float = 0.0) -> Tuple[float, float]:
        """Samples a random point inside the 2D shape."""
        raise NotImplementedError("Override me!")

    def intersects(self, other: _Geom2D) -> bool:
        """Checks if this shape intersects with another one."""
        return geom2ds_intersect(self, other)


@dataclass(frozen=True)
class LineSegment(_Geom2D):
    """A helper class for visualizing and collision checking line segments."""
    x1: float
    y1: float
    x2: float
    y2: float

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        ax.plot([self.x1, self.x2], [self.y1, self.y2], **kwargs)

    @staticmethod
    def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
        return np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

    def contains_point(self, x: float, y: float) -> bool:
        # https://stackoverflow.com/questions/328107
        a = (self.x1, self.y1)
        b = (self.x2, self.y2)
        c = (x, y)
        # Need to use an epsilon for numerical stability. But we are checking
        # if the distance from a to b is (approximately) equal to the distance
        # from a to c and the distance from c to b.
        eps = 1e-6
        return -eps < self._dist(a, c) + self._dist(c, b) - self._dist(a,
                                                                       b) < eps

    def sample_random_point(
            self,
            rng: np.random.Generator,
            min_dist_from_edge: float = 0.0) -> Tuple[float, float]:
        assert min_dist_from_edge == 0.0, "not yet implemented" + \
            " non-infinite min_dist_from_edge"
        line_slope = (self.y2 - self.y1) / (self.x2 - self.x1)
        y_intercept = self.y2 - (line_slope * self.x2)
        random_x_point = rng.uniform(self.x1, self.x2)
        random_y_point_on_line = line_slope * random_x_point + y_intercept
        assert self.contains_point(random_x_point, random_y_point_on_line)
        return (random_x_point, random_y_point_on_line)


@dataclass(frozen=True)
class Circle(_Geom2D):
    """A helper class for visualizing and collision checking circles."""
    x: float
    y: float
    radius: float

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        patch = patches.Circle((self.x, self.y), self.radius, **kwargs)
        ax.add_patch(patch)

    def contains_point(self, x: float, y: float) -> bool:
        return (x - self.x)**2 + (y - self.y)**2 <= self.radius**2

    def contains_circle(self, other_circle: Circle) -> bool:
        """Check whether this circle wholly contains another one."""
        dist_between_centers = np.sqrt((other_circle.x - self.x)**2 +
                                       (other_circle.y - self.y)**2)
        return (dist_between_centers + other_circle.radius) <= self.radius

    def sample_random_point(
            self,
            rng: np.random.Generator,
            min_dist_from_edge: float = 0.0) -> Tuple[float, float]:
        assert min_dist_from_edge < self.radius, "min_dist_from_edge is " + \
            "greater than radius"
        rand_mag = rng.uniform(0, self.radius - min_dist_from_edge)
        rand_theta = rng.uniform(0, 2 * np.pi)
        x_point = self.x + rand_mag * np.cos(rand_theta)
        y_point = self.y + rand_mag * np.sin(rand_theta)
        assert self.contains_point(x_point, y_point)
        return (x_point, y_point)


@dataclass(frozen=True)
class Triangle(_Geom2D):
    """A helper class for visualizing and collision checking triangles."""
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        patch = patches.Polygon(
            [[self.x1, self.y1], [self.x2, self.y2], [self.x3, self.y3]],
            **kwargs)
        ax.add_patch(patch)

    def __post_init__(self) -> None:
        dist1 = np.sqrt((self.x1 - self.x2)**2 + (self.y1 - self.y2)**2)
        dist2 = np.sqrt((self.x2 - self.x3)**2 + (self.y2 - self.y3)**2)
        dist3 = np.sqrt((self.x3 - self.x1)**2 + (self.y3 - self.y1)**2)
        dists = sorted([dist1, dist2, dist3])
        assert dists[0] + dists[1] >= dists[2]
        if dists[0] + dists[1] == dists[2]:
            raise ValueError("Degenerate triangle!")

    def contains_point(self, x: float, y: float) -> bool:
        # Adapted from https://stackoverflow.com/questions/2049582/.
        sign1 = ((x - self.x2) * (self.y1 - self.y2) - (self.x1 - self.x2) *
                 (y - self.y2)) > 0
        sign2 = ((x - self.x3) * (self.y2 - self.y3) - (self.x2 - self.x3) *
                 (y - self.y3)) > 0
        sign3 = ((x - self.x1) * (self.y3 - self.y1) - (self.x3 - self.x1) *
                 (y - self.y1)) > 0
        has_neg = (not sign1) or (not sign2) or (not sign3)
        has_pos = sign1 or sign2 or sign3
        return not has_neg or not has_pos

    def sample_random_point(
            self,
            rng: np.random.Generator,
            min_dist_from_edge: float = 0.0) -> Tuple[float, float]:
        assert min_dist_from_edge == 0.0, "not yet implemented" + \
            " non-zero min_dist_from_edge"
        a = np.array([self.x2 - self.x1, self.y2 - self.y1])
        b = np.array([self.x3 - self.x1, self.y3 - self.y1])
        u1 = rng.uniform(0, 1)
        u2 = rng.uniform(0, 1)
        if u1 + u2 > 1.0:
            u1 = 1 - u1
            u2 = 1 - u2
        point_in_triangle = (u1 * a + u2 * b) + np.array([self.x1, self.y1])
        assert self.contains_point(point_in_triangle[0], point_in_triangle[1])
        return (point_in_triangle[0], point_in_triangle[1])


@dataclass(frozen=True)
class Rectangle(_Geom2D):
    """A helper class for visualizing and collision checking rectangles.

    Following the convention in plt.Rectangle, the origin is at the
    bottom left corner, and rotation is anti-clockwise about that point.

    Unlike plt.Rectangle, the angle is in radians.
    """
    x: float
    y: float
    width: float
    height: float
    theta: float  # in radians, between -np.pi and np.pi

    def __post_init__(self) -> None:
        assert -np.pi <= self.theta <= np.pi, "Expecting angle in [-pi, pi]."

    @staticmethod
    def from_center(center_x: float, center_y: float, width: float,
                    height: float, rotation_about_center: float) -> Rectangle:
        """Create a rectangle given an (x, y) for the center, with theta
        rotating about that center point."""
        x = center_x - width / 2
        y = center_y - height / 2
        norm_rect = Rectangle(x, y, width, height, 0.0)
        assert np.isclose(norm_rect.center[0], center_x)
        assert np.isclose(norm_rect.center[1], center_y)
        return norm_rect.rotate_about_point(center_x, center_y,
                                            rotation_about_center)

    @functools.cached_property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Get the rotation matrix."""
        return np.array([[np.cos(self.theta), -np.sin(self.theta)],
                         [np.sin(self.theta),
                          np.cos(self.theta)]])

    @functools.cached_property
    def inverse_rotation_matrix(self) -> NDArray[np.float64]:
        """Get the inverse rotation matrix."""
        return np.array([[np.cos(self.theta),
                          np.sin(self.theta)],
                         [-np.sin(self.theta),
                          np.cos(self.theta)]])

    @functools.cached_property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get the four vertices for the rectangle."""
        scale_matrix = np.array([
            [self.width, 0],
            [0, self.height],
        ])
        translate_vector = np.array([self.x, self.y])
        vertices = np.array([
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
        ])
        vertices = vertices @ scale_matrix.T
        vertices = vertices @ self.rotation_matrix.T
        vertices = translate_vector + vertices
        # Convert to a list of tuples. Slightly complicated to appease both
        # type checking and linting.
        return list(map(lambda p: (p[0], p[1]), vertices))

    @functools.cached_property
    def line_segments(self) -> List[LineSegment]:
        """Get the four line segments for the rectangle."""
        vs = list(zip(self.vertices, self.vertices[1:] + [self.vertices[0]]))
        line_segments = []
        for ((x1, y1), (x2, y2)) in vs:
            line_segments.append(LineSegment(x1, y1, x2, y2))
        return line_segments

    @functools.cached_property
    def center(self) -> Tuple[float, float]:
        """Get the point at the center of the rectangle."""
        x, y = np.mean(self.vertices, axis=0)
        return (x, y)

    @functools.cached_property
    def circumscribed_circle(self) -> Circle:
        """Returns x, y, radius."""
        x, y = self.center
        radius = np.sqrt((self.width / 2)**2 + (self.height / 2)**2)
        return Circle(x, y, radius)

    def contains_point(self, x: float, y: float) -> bool:
        # First invert translation, then invert rotation.
        rx, ry = np.array([x - self.x, y - self.y
                           ]) @ self.inverse_rotation_matrix.T
        return 0 <= rx <= self.width and \
               0 <= ry <= self.height

    def sample_random_point(
            self,
            rng: np.random.Generator,
            min_dist_from_edge: float = 0.0) -> Tuple[float, float]:
        assert min_dist_from_edge <= self.width / 2 and \
            min_dist_from_edge <= self.height / 2, \
                "cannot achieve sample with min_dist_from_edge"
        rand_width = rng.uniform(min_dist_from_edge,
                                 self.width - min_dist_from_edge)
        rand_height = rng.uniform(min_dist_from_edge,
                                  self.height - min_dist_from_edge)
        # First rotate, then translate.
        rx, ry = np.array([rand_width, rand_height]) @ self.rotation_matrix.T
        x = rx + self.x
        y = ry + self.y
        assert self.contains_point(x, y)
        return (x, y)

    def rotate_about_point(self, x: float, y: float, rot: float) -> Rectangle:
        """Create a new rectangle that is this rectangle, but rotated CCW by
        the given rotation (in radians), relative to the (x, y) origin.

        Rotates the vertices first, then uses them to recompute the new
        theta.
        """
        vertices = np.array(self.vertices)
        origin = np.array([x, y])
        # Translate the vertices so that they become the "origin".
        vertices = vertices - origin
        # Rotate.
        rotate_matrix = np.array([[np.cos(rot), -np.sin(rot)],
                                  [np.sin(rot), np.cos(rot)]])
        vertices = vertices @ rotate_matrix.T
        # Translate the vertices back.
        vertices = vertices + origin
        # Recompute theta.
        (lx, ly), _, _, (rx, ry) = vertices
        theta = np.arctan2(ry - ly, rx - lx)
        rect = Rectangle(lx, ly, self.width, self.height, theta)
        assert np.allclose(rect.vertices, vertices)
        return rect

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        angle = self.theta * 180 / np.pi
        patch = patches.Rectangle((self.x, self.y),
                                  self.width,
                                  self.height,
                                  angle=angle,
                                  **kwargs)
        ax.add_patch(patch)


def line_segments_intersect(seg1: LineSegment, seg2: LineSegment) -> bool:
    """Checks if two line segments intersect.

    This method, which works by checking relative orientation, allows
    for collinearity, and only checks if each segment straddles the line
    containing the other.
    """

    def _subtract(a: Tuple[float, float], b: Tuple[float, float]) \
        -> Tuple[float, float]:
        x1, y1 = a
        x2, y2 = b
        return (x1 - x2), (y1 - y2)

    def _cross_product(a: Tuple[float, float], b: Tuple[float, float]) \
        -> float:
        x1, y1 = b
        x2, y2 = a
        return x1 * y2 - x2 * y1

    def _direction(a: Tuple[float, float], b: Tuple[float, float],
                   c: Tuple[float, float]) -> float:
        return _cross_product(_subtract(a, c), _subtract(a, b))

    p1 = (seg1.x1, seg1.y1)
    p2 = (seg1.x2, seg1.y2)
    p3 = (seg2.x1, seg2.y1)
    p4 = (seg2.x2, seg2.y2)
    d1 = _direction(p3, p4, p1)
    d2 = _direction(p3, p4, p2)
    d3 = _direction(p1, p2, p3)
    d4 = _direction(p1, p2, p4)

    return ((d2 < 0 < d1) or (d1 < 0 < d2)) and ((d4 < 0 < d3) or
                                                 (d3 < 0 < d4))


def circles_intersect(circ1: Circle, circ2: Circle) -> bool:
    """Checks if two circles intersect."""
    x1, y1, r1 = circ1.x, circ1.y, circ1.radius
    x2, y2, r2 = circ2.x, circ2.y, circ2.radius
    return (x1 - x2)**2 + (y1 - y2)**2 < (r1 + r2)**2


def rectangles_intersect(rect1: Rectangle, rect2: Rectangle) -> bool:
    """Checks if two rectangles intersect."""
    # Optimization: if the circumscribed circles don't intersect, then
    # the rectangles also don't intersect.
    if not circles_intersect(rect1.circumscribed_circle,
                             rect2.circumscribed_circle):
        return False
    # Case 1: line segments intersect.
    if any(
            line_segments_intersect(seg1, seg2) for seg1 in rect1.line_segments
            for seg2 in rect2.line_segments):
        return True
    # Case 2: rect1 inside rect2.
    if rect1.contains_point(rect2.center[0], rect2.center[1]):
        return True
    # Case 3: rect2 inside rect1.
    if rect2.contains_point(rect1.center[0], rect1.center[1]):
        return True
    # Not intersecting.
    return False


def line_segment_intersects_circle(seg: LineSegment,
                                   circ: Circle,
                                   ax: Optional[plt.Axes] = None) -> bool:
    """Checks if a line segment intersects a circle.

    If ax is not None, a diagram is plotted on the axis to illustrate
    the computations, which is useful for checking correctness.
    """
    # First check if the end points of the segment are in the circle.
    if circ.contains_point(seg.x1, seg.y1):
        return True
    if circ.contains_point(seg.x2, seg.y2):
        return True
    # Project the circle radius onto the extended line.
    c = (circ.x, circ.y)
    # Project (a, c) onto (a, b).
    a = (seg.x1, seg.y1)
    b = (seg.x2, seg.y2)
    ba = np.subtract(b, a)
    ca = np.subtract(c, a)
    da = ba * np.dot(ca, ba) / np.dot(ba, ba)
    # The point on the extended line that is the closest to the center.
    d = dx, dy = (a[0] + da[0], a[1] + da[1])
    # Optionally plot the important points.
    if ax is not None:
        circ.plot(ax, color="red", alpha=0.5)
        seg.plot(ax, color="black", linewidth=2)
        ax.annotate("A", a)
        ax.annotate("B", b)
        ax.annotate("C", c)
        ax.annotate("D", d)
    # Check if the point is on the line. If it's not, there is no intersection,
    # because we already checked that the circle does not contain the end
    # points of the line segment.
    if not seg.contains_point(dx, dy):
        return False
    # So d is on the segment. Check if it's in the circle.
    return circ.contains_point(dx, dy)


def line_segment_intersects_rectangle(seg: LineSegment,
                                      rect: Rectangle) -> bool:
    """Checks if a line segment intersects a rectangle."""
    # Case 1: one of the end points of the segment is in the rectangle.
    if rect.contains_point(seg.x1, seg.y1) or \
        rect.contains_point(seg.x2, seg.y2):
        return True
    # Case 2: the segment intersects with one of the rectangle sides.
    return any(line_segments_intersect(s, seg) for s in rect.line_segments)


def rectangle_intersects_circle(rect: Rectangle, circ: Circle) -> bool:
    """Checks if a rectangle intersects a circle."""
    # Optimization: if the circumscribed circle of the rectangle doesn't
    # intersect with the circle, then there can't be an intersection.
    if not circles_intersect(rect.circumscribed_circle, circ):
        return False
    # Case 1: the circle's center is in the rectangle.
    if rect.contains_point(circ.x, circ.y):
        return True
    # Case 2: one of the sides of the rectangle intersects the circle.
    for seg in rect.line_segments:
        if line_segment_intersects_circle(seg, circ):
            return True
    return False


def geom2ds_intersect(geom1: _Geom2D, geom2: _Geom2D) -> bool:
    """Check if two 2D bodies intersect."""
    if isinstance(geom1, LineSegment) and isinstance(geom2, LineSegment):
        return line_segments_intersect(geom1, geom2)
    if isinstance(geom1, LineSegment) and isinstance(geom2, Circle):
        return line_segment_intersects_circle(geom1, geom2)
    if isinstance(geom1, LineSegment) and isinstance(geom2, Rectangle):
        return line_segment_intersects_rectangle(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, LineSegment):
        return line_segment_intersects_rectangle(geom2, geom1)
    if isinstance(geom1, Circle) and isinstance(geom2, LineSegment):
        return line_segment_intersects_circle(geom2, geom1)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Rectangle):
        return rectangles_intersect(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Circle):
        return rectangle_intersects_circle(geom1, geom2)
    if isinstance(geom1, Circle) and isinstance(geom2, Rectangle):
        return rectangle_intersects_circle(geom2, geom1)
    if isinstance(geom1, Circle) and isinstance(geom2, Circle):
        return circles_intersect(geom1, geom2)
    raise NotImplementedError("Intersection not implemented for geoms "
                              f"{geom1} and {geom2}")


@functools.lru_cache(maxsize=None)
def unify(atoms1: FrozenSet[LiftedOrGroundAtom],
          atoms2: FrozenSet[LiftedOrGroundAtom]) -> Tuple[bool, EntToEntSub]:
    """Return whether the given two sets of atoms can be unified.

    Also return the mapping between variables/objects in these atom
    sets. This mapping is empty if the first return value is False.
    """
    atoms_lst1 = sorted(atoms1)
    atoms_lst2 = sorted(atoms2)

    # Terminate quickly if there is a mismatch between predicates
    preds1 = [atom.predicate for atom in atoms_lst1]
    preds2 = [atom.predicate for atom in atoms_lst2]
    if preds1 != preds2:
        return False, {}

    # Terminate quickly if there is a mismatch between numbers
    num1 = len({o for atom in atoms_lst1 for o in atom.entities})
    num2 = len({o for atom in atoms_lst2 for o in atom.entities})
    if num1 != num2:
        return False, {}

    # Try to get lucky with a one-to-one mapping
    subs12: EntToEntSub = {}
    subs21 = {}
    success = True
    for atom1, atom2 in zip(atoms_lst1, atoms_lst2):
        if not success:
            break
        for v1, v2 in zip(atom1.entities, atom2.entities):
            if v1 in subs12 and subs12[v1] != v2:
                success = False
                break
            if v2 in subs21:
                success = False
                break
            subs12[v1] = v2
            subs21[v2] = v1
    if success:
        return True, subs12

    # If all else fails, use search
    solved, sub = find_substitution(atoms_lst1, atoms_lst2)
    rev_sub = {v: k for k, v in sub.items()}
    return solved, rev_sub


@functools.lru_cache(maxsize=None)
def unify_preconds_effects_options(
        preconds1: FrozenSet[LiftedOrGroundAtom],
        preconds2: FrozenSet[LiftedOrGroundAtom],
        add_effects1: FrozenSet[LiftedOrGroundAtom],
        add_effects2: FrozenSet[LiftedOrGroundAtom],
        delete_effects1: FrozenSet[LiftedOrGroundAtom],
        delete_effects2: FrozenSet[LiftedOrGroundAtom],
        param_option1: ParameterizedOption, param_option2: ParameterizedOption,
        option_args1: Tuple[_TypedEntity, ...],
        option_args2: Tuple[_TypedEntity, ...]) -> Tuple[bool, EntToEntSub]:
    """Wrapper around unify() that handles option arguments, preconditions, add
    effects, and delete effects.

    Changes predicate names so that all are treated differently by
    unify().
    """
    if param_option1 != param_option2:
        # Can't unify if the parameterized options are different.
        return False, {}
    opt_arg_pred1 = Predicate("OPT-ARGS", [a.type for a in option_args1],
                              _classifier=lambda s, o: False)  # dummy
    f_option_args1 = frozenset({GroundAtom(opt_arg_pred1, option_args1)})
    new_preconds1 = wrap_atom_predicates(preconds1, "PRE-")
    f_new_preconds1 = frozenset(new_preconds1)
    new_add_effects1 = wrap_atom_predicates(add_effects1, "ADD-")
    f_new_add_effects1 = frozenset(new_add_effects1)
    new_delete_effects1 = wrap_atom_predicates(delete_effects1, "DEL-")
    f_new_delete_effects1 = frozenset(new_delete_effects1)

    opt_arg_pred2 = Predicate("OPT-ARGS", [a.type for a in option_args2],
                              _classifier=lambda s, o: False)  # dummy
    f_option_args2 = frozenset({LiftedAtom(opt_arg_pred2, option_args2)})
    new_preconds2 = wrap_atom_predicates(preconds2, "PRE-")
    f_new_preconds2 = frozenset(new_preconds2)
    new_add_effects2 = wrap_atom_predicates(add_effects2, "ADD-")
    f_new_add_effects2 = frozenset(new_add_effects2)
    new_delete_effects2 = wrap_atom_predicates(delete_effects2, "DEL-")
    f_new_delete_effects2 = frozenset(new_delete_effects2)

    all_atoms1 = (f_option_args1 | f_new_preconds1 | f_new_add_effects1
                  | f_new_delete_effects1)
    all_atoms2 = (f_option_args2 | f_new_preconds2 | f_new_add_effects2
                  | f_new_delete_effects2)
    return unify(all_atoms1, all_atoms2)


def wrap_predicate(predicate: Predicate, prefix: str) -> Predicate:
    """Return a new predicate which adds the given prefix string to the name.

    NOTE: the classifier is removed.
    """
    new_predicate = Predicate(prefix + predicate.name,
                              predicate.types,
                              _classifier=lambda s, o: False)  # dummy
    return new_predicate


def wrap_atom_predicates(atoms: Collection[LiftedOrGroundAtom],
                         prefix: str) -> Set[LiftedOrGroundAtom]:
    """Return a new set of atoms which adds the given prefix string to the name
    of every atom's predicate.

    NOTE: all the classifiers are removed.
    """
    new_atoms = set()
    for atom in atoms:
        new_predicate = wrap_predicate(atom.predicate, prefix)
        new_atoms.add(atom.__class__(new_predicate, atom.entities))
    return new_atoms


class LinearChainParameterizedOption(ParameterizedOption):
    """A parameterized option implemented via a sequence of "child"
    parameterized options.

    This class is meant to help ParameterizedOption manual design.

    The children are executed in order starting with the first in the sequence
    and transitioning when the terminal function of each child is hit.

    The children are assumed to chain together, so the initiable of the next
    child should always be True when the previous child terminates. If this
    is not the case, an AssertionError is raised.

    The children must all have the same types and params_space, which in turn
    become the types and params_space for this ParameterizedOption.

    The LinearChainParameterizedOption has memory, which stores the current
    child index.
    """

    def __init__(self, name: str,
                 children: Sequence[ParameterizedOption]) -> None:
        assert len(children) > 0
        self._children = children

        # Make sure that the types and params spaces are consistent.
        types = children[0].types
        params_space = children[0].params_space
        for i in range(1, len(self._children)):
            child = self._children[i]
            assert types == child.types
            assert np.allclose(params_space.low, child.params_space.low)
            assert np.allclose(params_space.high, child.params_space.high)

        super().__init__(name,
                         types,
                         params_space,
                         policy=self._policy,
                         initiable=self._initiable,
                         terminal=self._terminal)

    def _initiable(self, state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        # Initialize the current child to the first one.
        memory["current_child_index"] = 0
        # Create memory dicts for each child to avoid key collisions. One
        # example of a failure that arises without this is when using
        # multiple SingletonParameterizedOption instances, each of those
        # options would be referencing the same start_state in memory.
        memory["child_memory"] = [{} for _ in self._children]
        current_child = self._children[0]
        child_memory = memory["child_memory"][0]
        return current_child.initiable(state, child_memory, objects, params)

    def _policy(self, state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        # Check if the current child has terminated.
        current_index = memory["current_child_index"]
        current_child = self._children[current_index]
        child_memory = memory["child_memory"][current_index]
        if current_child.terminal(state, child_memory, objects, params):
            # Move on to the next child.
            current_index += 1
            memory["current_child_index"] = current_index
            current_child = self._children[current_index]
            child_memory = memory["child_memory"][current_index]
            assert current_child.initiable(state, child_memory, objects,
                                           params)
        return current_child.policy(state, child_memory, objects, params)

    def _terminal(self, state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        # Check if the last child has terminated.
        current_index = memory["current_child_index"]
        if current_index < len(self._children) - 1:
            return False
        current_child = self._children[current_index]
        child_memory = memory["child_memory"][current_index]
        return current_child.terminal(state, child_memory, objects, params)


class SingletonParameterizedOption(ParameterizedOption):
    """A parameterized option that takes a single action and stops.

    For convenience:
        * Initiable defaults to always True.
        * Types defaults to [].
        * Params space defaults to Box(0, 1, (0, )).
    """

    def __init__(
        self,
        name: str,
        policy: Callable[[State, Dict, Sequence[Object], Array], Action],
        types: Optional[Sequence[Type]] = None,
        params_space: Optional[Box] = None,
        initiable: Optional[Callable[[State, Dict, Sequence[Object], Array],
                                     bool]] = None
    ) -> None:
        if types is None:
            types = []
        if params_space is None:
            params_space = Box(0, 1, (0, ))
        if initiable is None:
            initiable = lambda _1, _2, _3, _4: True

        # Wrap the given initiable so that we can track whether the action
        # has been executed yet.
        def _initiable(state: State, memory: Dict, objects: Sequence[Object],
                       params: Array) -> bool:
            # Always update the memory dict due to the "is" check in _terminal.
            memory["start_state"] = state
            assert initiable is not None
            return initiable(state, memory, objects, params)

        def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del objects, params  # unused
            assert "start_state" in memory, \
                "Must call initiable() before terminal()."
            return state is not memory["start_state"]

        super().__init__(name,
                         types,
                         params_space,
                         policy=policy,
                         initiable=_initiable,
                         terminal=_terminal)


class PyBulletState(State):
    """A PyBullet state that stores the robot joint positions in addition to
    the features that are exposed in the object-centric state."""

    @property
    def joint_positions(self) -> JointPositions:
        """Expose the current joints state in the simulator_state."""
        return cast(JointPositions, self.simulator_state)

    def allclose(self, other: State) -> bool:
        # Ignores the simulator state.
        return State(self.data).allclose(State(other.data))

    def copy(self) -> State:
        state_dict_copy = super().copy().data
        simulator_state_copy = list(self.joint_positions)
        return PyBulletState(state_dict_copy, simulator_state_copy)


class StateWithCache(State):
    """A state with a cache stored in the simulator state that is ignored for
    state equality checks.

    The cache is deliberately not copied.
    """

    @property
    def cache(self) -> Dict[str, Dict]:
        """Expose the cache in the simulator_state."""
        return cast(Dict[str, Dict], self.simulator_state)

    def allclose(self, other: State) -> bool:
        # Ignores the simulator state.
        return State(self.data).allclose(State(other.data))

    def copy(self) -> State:
        state_dict_copy = super().copy().data
        return StateWithCache(state_dict_copy, self.cache)


class LoggingMonitor(abc.ABC):
    """Observes states and actions during environment interaction."""

    @abc.abstractmethod
    def reset(self, train_or_test: str, task_idx: int) -> None:
        """Called when the monitor starts a new episode."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def observe(self, obs: Observation, action: Optional[Action]) -> None:
        """Record an observation and the action that is about to be taken.

        On the last timestep of a trajectory, no action is taken, so
        action is None.
        """
        raise NotImplementedError("Override me!")


def run_policy(
    policy: Callable[[State], Action],
    env: BaseEnv,
    train_or_test: str,
    task_idx: int,
    termination_function: Callable[[State], bool],
    max_num_steps: int,
    do_env_reset: bool = True,
    exceptions_to_break_on: Optional[Set[TypingType[Exception]]] = None,
    monitor: Optional[LoggingMonitor] = None
) -> Tuple[LowLevelTrajectory, Metrics]:
    """Execute a policy starting from the initial state of a train or test task
    in the environment. The task's goal is not used.

    Note that the environment internal state is updated.

    Terminates when any of these conditions hold:
    (1) the termination_function returns True
    (2) max_num_steps is reached
    (3) policy() or step() raise an exception of type in exceptions_to_break_on

    Note that in the case where the exception is raised in step, we exclude the
    last action from the returned trajectory to maintain the invariant that
    the trajectory states are of length one greater than the actions.

    NOTE: this may be deprecated in the future in favor of run_episode defined
    in cogman.py. Ideally, we should consolidate both run_policy and
    run_policy_with_simulator below into run_episode.
    """
    if do_env_reset:
        env.reset(train_or_test, task_idx)
        if monitor is not None:
            monitor.reset(train_or_test, task_idx)
    obs = env.get_observation()
    assert isinstance(obs, State)
    state = obs
    states = [state]
    actions: List[Action] = []
    metrics: Metrics = defaultdict(float)
    metrics["policy_call_time"] = 0.0
    exception_raised_in_step = False
    if not termination_function(state):
        for _ in range(max_num_steps):
            monitor_observed = False
            exception_raised_in_step = False
            try:
                start_time = time.perf_counter()
                act = policy(state)
                metrics["policy_call_time"] += time.perf_counter() - start_time
                # Note: it's important to call monitor.observe() before
                # env.step(), because the monitor may use the environment's
                # internal state.
                if monitor is not None:
                    monitor.observe(state, act)
                    monitor_observed = True
                state = env.step(act)
                actions.append(act)
                states.append(state)
            except Exception as e:
                if exceptions_to_break_on is not None and \
                    type(e) in exceptions_to_break_on:
                    if monitor_observed:
                        exception_raised_in_step = True
                    break
                if monitor is not None and not monitor_observed:
                    monitor.observe(state, None)
                raise e
            if termination_function(state):
                break
    if monitor is not None and not exception_raised_in_step:
        monitor.observe(state, None)
    traj = LowLevelTrajectory(states, actions)
    return traj, metrics


def run_policy_with_simulator(
        policy: Callable[[State], Action],
        simulator: Callable[[State, Action], State],
        init_state: State,
        termination_function: Callable[[State], bool],
        max_num_steps: int,
        exceptions_to_break_on: Optional[Set[TypingType[Exception]]] = None,
        monitor: Optional[LoggingMonitor] = None) -> LowLevelTrajectory:
    """Execute a policy from a given initial state, using a simulator.

    *** This function should not be used with any core code, because we want
    to avoid the assumption of a simulator when possible. ***

    This is similar to run_policy, with three major differences:
    (1) The initial state `init_state` can be any state, not just the initial
    state of a train or test task. (2) A simulator (function that takes state
    as input) is assumed. (3) Metrics are not returned.

    Note that the environment internal state is NOT updated.

    Terminates when any of these conditions hold:
    (1) the termination_function returns True
    (2) max_num_steps is reached
    (3) policy() or step() raise an exception of type in exceptions_to_break_on

    Note that in the case where the exception is raised in step, we exclude the
    last action from the returned trajectory to maintain the invariant that
    the trajectory states are of length one greater than the actions.
    """
    state = init_state
    states = [state]
    actions: List[Action] = []
    exception_raised_in_step = False
    if not termination_function(state):
        for _ in range(max_num_steps):
            monitor_observed = False
            exception_raised_in_step = False
            try:
                act = policy(state)
                if monitor is not None:
                    monitor.observe(state, act)
                    monitor_observed = True
                state = simulator(state, act)
                actions.append(act)
                states.append(state)
            except Exception as e:
                if exceptions_to_break_on is not None and \
                    type(e) in exceptions_to_break_on:
                    if monitor_observed:
                        exception_raised_in_step = True
                    break
                if monitor is not None and not monitor_observed:
                    monitor.observe(state, None)
                raise e
            if termination_function(state):
                break
    if monitor is not None and not exception_raised_in_step:
        monitor.observe(state, None)
    traj = LowLevelTrajectory(states, actions)
    return traj


class ExceptionWithInfo(Exception):
    """An exception with an optional info dictionary that is initially
    empty."""

    def __init__(self, message: str, info: Optional[Dict] = None) -> None:
        super().__init__(message)
        if info is None:
            info = {}
        assert isinstance(info, dict)
        self.info = info


class OptionExecutionFailure(ExceptionWithInfo):
    """An exception raised by an option policy in the course of execution."""


class OptionTimeoutFailure(OptionExecutionFailure):
    """A special kind of option execution failure due to an exceeded budget."""


class RequestActPolicyFailure(ExceptionWithInfo):
    """An exception raised by an acting policy in a request when it fails to
    produce an action, which terminates the interaction."""


class HumanDemonstrationFailure(ExceptionWithInfo):
    """An exception raised when CFG.demonstrator == "human" and the human gives
    a bad input."""


class EnvironmentFailure(ExceptionWithInfo):
    """Exception raised when any type of failure occurs in an environment.

    The info dictionary must contain a key "offending_objects", which
    maps to a set of objects responsible for the failure.
    """

    def __repr__(self) -> str:
        return f"{super().__repr__()}: {self.info}"

    def __str__(self) -> str:
        return repr(self)


def option_policy_to_policy(
    option_policy: Callable[[State], _Option],
    max_option_steps: Optional[int] = None,
    raise_error_on_repeated_state: bool = False,
) -> Callable[[State], Action]:
    """Create a policy that executes a policy over options."""
    cur_option = DummyOption
    num_cur_option_steps = 0
    last_state: Optional[State] = None

    def _policy(state: State) -> Action:
        nonlocal cur_option, num_cur_option_steps, last_state

        if cur_option is DummyOption:
            last_option: Optional[_Option] = None
        else:
            last_option = cur_option

        if max_option_steps is not None and \
            num_cur_option_steps >= max_option_steps:
            raise OptionTimeoutFailure(
                "Exceeded max option steps.",
                info={"last_failed_option": last_option})

        if last_state is not None and \
            raise_error_on_repeated_state and state.allclose(last_state):
            raise OptionTimeoutFailure(
                "Encountered repeated state.",
                info={"last_failed_option": last_option})
        last_state = state

        if cur_option is DummyOption or cur_option.terminal(state):
            try:
                cur_option = option_policy(state)
            except OptionExecutionFailure as e:
                e.info["last_failed_option"] = last_option
                raise e
            if not cur_option.initiable(state):
                raise OptionExecutionFailure(
                    "Unsound option policy.",
                    info={"last_failed_option": last_option})
            num_cur_option_steps = 0

        num_cur_option_steps += 1

        return cur_option.policy(state)

    return _policy


def option_plan_to_policy(
        plan: Sequence[_Option],
        max_option_steps: Optional[int] = None,
        raise_error_on_repeated_state: bool = False
) -> Callable[[State], Action]:
    """Create a policy that executes a sequence of options in order."""
    queue = list(plan)  # don't modify plan, just in case

    def _option_policy(state: State) -> _Option:
        del state  # not used
        if not queue:
            raise OptionExecutionFailure("Option plan exhausted!")
        return queue.pop(0)

    return option_policy_to_policy(
        _option_policy,
        max_option_steps=max_option_steps,
        raise_error_on_repeated_state=raise_error_on_repeated_state)


def nsrt_plan_to_greedy_option_policy(
    nsrt_plan: Sequence[_GroundNSRT],
    goal: Set[GroundAtom],
    rng: np.random.Generator,
    necessary_atoms_seq: Optional[Sequence[Set[GroundAtom]]] = None
) -> Callable[[State], _Option]:
    """Greedily execute an NSRT plan, assuming downward refinability and that
    any sample will work.

    If an option is not initiable or if the plan runs out, an
    OptionExecutionFailure is raised.
    """
    cur_nsrt: Optional[_GroundNSRT] = None
    nsrt_queue = list(nsrt_plan)
    if necessary_atoms_seq is None:
        empty_atoms: Set[GroundAtom] = set()
        necessary_atoms_seq = [empty_atoms for _ in range(len(nsrt_plan) + 1)]
    assert len(necessary_atoms_seq) == len(nsrt_plan) + 1
    necessary_atoms_queue = list(necessary_atoms_seq)

    def _option_policy(state: State) -> _Option:
        nonlocal cur_nsrt
        if not nsrt_queue:
            raise OptionExecutionFailure("NSRT plan exhausted.")
        expected_atoms = necessary_atoms_queue.pop(0)
        if not all(a.holds(state) for a in expected_atoms):
            raise OptionExecutionFailure(
                "Executing the NSRT failed to achieve the necessary atoms.")
        cur_nsrt = nsrt_queue.pop(0)
        cur_option = cur_nsrt.sample_option(state, goal, rng)
        logging.debug(f"Using option {cur_option.name}{cur_option.objects} "
                      "from NSRT plan.")
        return cur_option

    return _option_policy


def nsrt_plan_to_greedy_policy(
    nsrt_plan: Sequence[_GroundNSRT],
    goal: Set[GroundAtom],
    rng: np.random.Generator,
    necessary_atoms_seq: Optional[Sequence[Set[GroundAtom]]] = None
) -> Callable[[State], Action]:
    """Greedily execute an NSRT plan, assuming downward refinability and that
    any sample will work.

    If an option is not initiable or if the plan runs out, an
    OptionExecutionFailure is raised.
    """
    option_policy = nsrt_plan_to_greedy_option_policy(
        nsrt_plan, goal, rng, necessary_atoms_seq=necessary_atoms_seq)
    return option_policy_to_policy(option_policy)


def sample_applicable_option(param_options: List[ParameterizedOption],
                             state: State,
                             rng: np.random.Generator) -> Optional[_Option]:
    """Sample an applicable option."""
    for _ in range(CFG.random_options_max_tries):
        param_opt = param_options[rng.choice(len(param_options))]
        objs = get_random_object_combination(list(state), param_opt.types, rng)
        if objs is None:
            continue
        params = param_opt.params_space.sample()
        opt = param_opt.ground(objs, params)
        if opt.initiable(state):
            return opt
    return None


def create_random_option_policy(
        options: Collection[ParameterizedOption], rng: np.random.Generator,
        fallback_policy: Callable[[State],
                                  Action]) -> Callable[[State], Action]:
    """Create a policy that executes random initiable options.

    If no applicable option can be found, query the fallback policy.
    """
    sorted_options = sorted(options, key=lambda o: o.name)
    cur_option = DummyOption

    def _policy(state: State) -> Action:
        nonlocal cur_option
        if cur_option is DummyOption or cur_option.terminal(state):
            cur_option = DummyOption
            sample = sample_applicable_option(sorted_options, state, rng)
            if sample is not None:
                cur_option = sample
            else:
                return fallback_policy(state)
        act = cur_option.policy(state)
        return act

    return _policy


def sample_applicable_ground_nsrt(
        state: State, ground_nsrts: Sequence[_GroundNSRT],
        predicates: Set[Predicate],
        rng: np.random.Generator) -> Optional[_GroundNSRT]:
    """Choose uniformly among the ground NSRTs that are applicable in the
    state."""
    atoms = abstract(state, predicates)
    applicable_nsrts = sorted(get_applicable_operators(ground_nsrts, atoms))
    if len(applicable_nsrts) == 0:
        return None
    idx = rng.choice(len(applicable_nsrts))
    return applicable_nsrts[idx]


def action_arrs_to_policy(
        action_arrs: Sequence[Array]) -> Callable[[State], Action]:
    """Create a policy that executes action arrays in sequence."""

    queue = list(action_arrs)  # don't modify original, just in case

    def _policy(s: State) -> Action:
        del s  # unused
        return Action(queue.pop(0))

    return _policy


def _get_entity_combinations(
        entities: Collection[ObjectOrVariable],
        types: Sequence[Type]) -> Iterator[List[ObjectOrVariable]]:
    """Get all combinations of entities satisfying the given types sequence."""
    sorted_entities = sorted(entities)
    choices = []
    for vt in types:
        this_choices = []
        for ent in sorted_entities:
            if ent.is_instance(vt):
                this_choices.append(ent)
        choices.append(this_choices)
    for choice in itertools.product(*choices):
        yield list(choice)


def get_object_combinations(objects: Collection[Object],
                            types: Sequence[Type]) -> Iterator[List[Object]]:
    """Get all combinations of objects satisfying the given types sequence."""
    return _get_entity_combinations(objects, types)


def get_variable_combinations(
        variables: Collection[Variable],
        types: Sequence[Type]) -> Iterator[List[Variable]]:
    """Get all combinations of variables satisfying the given types
    sequence."""
    return _get_entity_combinations(variables, types)


def get_all_ground_atoms_for_predicate(
        predicate: Predicate, objects: Collection[Object]) -> Set[GroundAtom]:
    """Get all groundings of the predicate given objects.

    Note: we don't want lru_cache() on this function because we might want
    to call it with stripped predicates, and we wouldn't want it to return
    cached values.
    """
    ground_atoms = set()
    for args in get_object_combinations(objects, predicate.types):
        ground_atom = GroundAtom(predicate, args)
        ground_atoms.add(ground_atom)
    return ground_atoms


def get_all_lifted_atoms_for_predicate(
        predicate: Predicate,
        variables: FrozenSet[Variable]) -> Set[LiftedAtom]:
    """Get all groundings of the predicate given variables.

    Note: we don't want lru_cache() on this function because we might want
    to call it with stripped predicates, and we wouldn't want it to return
    cached values.
    """
    lifted_atoms = set()
    for args in get_variable_combinations(variables, predicate.types):
        lifted_atom = LiftedAtom(predicate, args)
        lifted_atoms.add(lifted_atom)
    return lifted_atoms


def get_random_object_combination(
        objects: Collection[Object], types: Sequence[Type],
        rng: np.random.Generator) -> Optional[List[Object]]:
    """Get a random list of objects from the given collection that satisfy the
    given sequence of types.

    Duplicates are always allowed. If a particular type has no object,
    return None.
    """
    types_to_objs = defaultdict(list)
    for obj in objects:
        types_to_objs[obj.type].append(obj)
    result = []
    for t in types:
        t_objs = types_to_objs[t]
        if not t_objs:
            return None
        result.append(t_objs[rng.choice(len(t_objs))])
    return result


def find_substitution(
    super_atoms: Collection[LiftedOrGroundAtom],
    sub_atoms: Collection[LiftedOrGroundAtom],
    allow_redundant: bool = False,
) -> Tuple[bool, EntToEntSub]:
    """Find a substitution from the entities in super_atoms to the entities in
    sub_atoms s.t. sub_atoms is a subset of super_atoms.

    If allow_redundant is True, then multiple entities in sub_atoms can
    refer to the same single entity in super_atoms.

    If no substitution exists, return (False, {}).
    """
    super_entities_by_type: Dict[Type, List[_TypedEntity]] = defaultdict(list)
    super_pred_to_tuples = defaultdict(set)
    for atom in super_atoms:
        for obj in atom.entities:
            if obj not in super_entities_by_type[obj.type]:
                super_entities_by_type[obj.type].append(obj)
        super_pred_to_tuples[atom.predicate].add(tuple(atom.entities))
    sub_variables = sorted({e for atom in sub_atoms for e in atom.entities})
    return _find_substitution_helper(sub_atoms, super_entities_by_type,
                                     sub_variables, super_pred_to_tuples, {},
                                     allow_redundant)


def _find_substitution_helper(
        sub_atoms: Collection[LiftedOrGroundAtom],
        super_entities_by_type: Dict[Type, List[_TypedEntity]],
        remaining_sub_variables: List[_TypedEntity],
        super_pred_to_tuples: Dict[Predicate,
                                   Set[Tuple[_TypedEntity,
                                             ...]]], partial_sub: EntToEntSub,
        allow_redundant: bool) -> Tuple[bool, EntToEntSub]:
    """Helper for find_substitution."""
    # Base case: check if all assigned
    if not remaining_sub_variables:
        return True, partial_sub
    # Find next variable to assign
    remaining_sub_variables = remaining_sub_variables.copy()
    next_sub_var = remaining_sub_variables.pop(0)
    # Consider possible assignments
    for super_obj in super_entities_by_type[next_sub_var.type]:
        if not allow_redundant and super_obj in partial_sub.values():
            continue
        new_sub = partial_sub.copy()
        new_sub[next_sub_var] = super_obj
        # Check if consistent
        if not _substitution_consistent(new_sub, super_pred_to_tuples,
                                        sub_atoms):
            continue
        # Backtracking search
        solved, final_sub = _find_substitution_helper(sub_atoms,
                                                      super_entities_by_type,
                                                      remaining_sub_variables,
                                                      super_pred_to_tuples,
                                                      new_sub, allow_redundant)
        if solved:
            return solved, final_sub
    # Failure
    return False, {}


def _substitution_consistent(
        partial_sub: EntToEntSub,
        super_pred_to_tuples: Dict[Predicate, Set[Tuple[_TypedEntity, ...]]],
        sub_atoms: Collection[LiftedOrGroundAtom]) -> bool:
    """Helper for _find_substitution_helper."""
    for sub_atom in sub_atoms:
        if not set(sub_atom.entities).issubset(partial_sub.keys()):
            continue
        substituted_vars = tuple(partial_sub[e] for e in sub_atom.entities)
        if substituted_vars not in super_pred_to_tuples[sub_atom.predicate]:
            return False
    return True


def create_new_variables(
    types: Sequence[Type],
    existing_vars: Optional[Collection[Variable]] = None,
    var_prefix: str = "?x",
) -> List[Variable]:
    """Create new variables of the given types, avoiding name collisions with
    existing variables.

    By convention, all new variables are of the form
    <var_prefix><number>.
    """
    pre_len = len(var_prefix)
    existing_var_nums = set()
    if existing_vars:
        for v in existing_vars:
            if v.name.startswith(var_prefix) and v.name[pre_len:].isdigit():
                existing_var_nums.add(int(v.name[pre_len:]))
    if existing_var_nums:
        counter = itertools.count(max(existing_var_nums) + 1)
    else:
        counter = itertools.count(0)
    new_vars = []
    for t in types:
        new_var_name = f"{var_prefix}{next(counter)}"
        new_var = Variable(new_var_name, t)
        new_vars.append(new_var)
    return new_vars


def param_option_to_nsrt(param_option: ParameterizedOption,
                         nsrts: Set[NSRT]) -> NSRT:
    """If options and NSRTs are 1:1, then map an option to an NSRT."""
    nsrt_matches = [n for n in nsrts if n.option == param_option]
    assert len(nsrt_matches) == 1
    nsrt = nsrt_matches[0]
    return nsrt


def option_to_ground_nsrt(option: _Option, nsrts: Set[NSRT]) -> _GroundNSRT:
    """If options and NSRTs are 1:1, then map an option to an NSRT."""
    nsrt = param_option_to_nsrt(option.parent, nsrts)
    return nsrt.ground(option.objects)


_S = TypeVar("_S", bound=Hashable)  # state in heuristic search
_A = TypeVar("_A")  # action in heuristic search


@dataclass(frozen=True)
class _HeuristicSearchNode(Generic[_S, _A]):
    state: _S
    edge_cost: float
    cumulative_cost: float
    parent: Optional[_HeuristicSearchNode[_S, _A]] = None
    action: Optional[_A] = None


def _run_heuristic_search(
        initial_state: _S,
        check_goal: Callable[[_S], bool],
        get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
        get_priority: Callable[[_HeuristicSearchNode[_S, _A]], Any],
        max_expansions: int = 10000000,
        max_evals: int = 10000000,
        timeout: int = 10000000,
        lazy_expansion: bool = False) -> Tuple[List[_S], List[_A]]:
    """A generic heuristic search implementation.

    Depending on get_priority, can implement A*, GBFS, or UCS.

    If no goal is found, returns the state with the best priority.
    """
    queue: List[Tuple[Any, int, _HeuristicSearchNode[_S, _A]]] = []
    state_to_best_path_cost: Dict[_S, float] = \
        defaultdict(lambda: float("inf"))

    root_node: _HeuristicSearchNode[_S, _A] = _HeuristicSearchNode(
        initial_state, 0, 0)
    root_priority = get_priority(root_node)
    best_node = root_node
    best_node_priority = root_priority
    tiebreak = itertools.count()
    hq.heappush(queue, (root_priority, next(tiebreak), root_node))
    num_expansions = 0
    num_evals = 1
    start_time = time.perf_counter()

    while len(queue) > 0 and time.perf_counter() - start_time < timeout and \
            num_expansions < max_expansions and num_evals < max_evals:
        _, _, node = hq.heappop(queue)
        # If we already found a better path here, don't bother.
        if state_to_best_path_cost[node.state] < node.cumulative_cost:
            continue
        # If the goal holds, return.
        if check_goal(node.state):
            return _finish_plan(node)
        num_expansions += 1
        # Generate successors.
        for action, child_state, cost in get_successors(node.state):
            if time.perf_counter() - start_time >= timeout:
                break
            child_path_cost = node.cumulative_cost + cost
            # If we already found a better path to this child, don't bother.
            if state_to_best_path_cost[child_state] <= child_path_cost:
                continue
            # Add new node.
            child_node = _HeuristicSearchNode(state=child_state,
                                              edge_cost=cost,
                                              cumulative_cost=child_path_cost,
                                              parent=node,
                                              action=action)
            priority = get_priority(child_node)
            num_evals += 1
            hq.heappush(queue, (priority, next(tiebreak), child_node))
            state_to_best_path_cost[child_state] = child_path_cost
            if priority < best_node_priority:
                best_node_priority = priority
                best_node = child_node
                # Optimization: if we've found a better child, immediately
                # explore the child without expanding the rest of the children.
                # Accomplish this by putting the parent node back on the queue.
                if lazy_expansion:
                    hq.heappush(queue, (priority, next(tiebreak), node))
                    break
            if num_evals >= max_evals:
                break

    # Did not find path to goal; return best path seen.
    return _finish_plan(best_node)


def _finish_plan(
        node: _HeuristicSearchNode[_S, _A]) -> Tuple[List[_S], List[_A]]:
    """Helper for _run_heuristic_search and run_hill_climbing."""
    rev_state_sequence: List[_S] = []
    rev_action_sequence: List[_A] = []

    while node.parent is not None:
        action = cast(_A, node.action)
        rev_action_sequence.append(action)
        rev_state_sequence.append(node.state)
        node = node.parent
    rev_state_sequence.append(node.state)

    return rev_state_sequence[::-1], rev_action_sequence[::-1]


def run_gbfs(initial_state: _S,
             check_goal: Callable[[_S], bool],
             get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
             heuristic: Callable[[_S], float],
             max_expansions: int = 10000000,
             max_evals: int = 10000000,
             timeout: int = 10000000,
             lazy_expansion: bool = False) -> Tuple[List[_S], List[_A]]:
    """Greedy best-first search."""
    get_priority = lambda n: heuristic(n.state)
    return _run_heuristic_search(initial_state, check_goal, get_successors,
                                 get_priority, max_expansions, max_evals,
                                 timeout, lazy_expansion)


def run_astar(initial_state: _S,
              check_goal: Callable[[_S], bool],
              get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
              heuristic: Callable[[_S], float],
              max_expansions: int = 10000000,
              max_evals: int = 10000000,
              timeout: int = 10000000,
              lazy_expansion: bool = False) -> Tuple[List[_S], List[_A]]:
    """A* search."""
    get_priority = lambda n: heuristic(n.state) + n.cumulative_cost
    return _run_heuristic_search(initial_state, check_goal, get_successors,
                                 get_priority, max_expansions, max_evals,
                                 timeout, lazy_expansion)


def run_hill_climbing(
    initial_state: _S,
    check_goal: Callable[[_S], bool],
    get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
    heuristic: Callable[[_S], float],
    early_termination_heuristic_thresh: Optional[float] = None,
    enforced_depth: int = 0,
    parallelize: bool = False,
    verbose: bool = True,
    timeout: float = float('inf')
) -> Tuple[List[_S], List[_A], List[float]]:
    """Enforced hill climbing local search.

    For each node, the best child node is always selected, if that child is
    an improvement over the node. If no children improve on the node, look
    at the children's children, etc., up to enforced_depth, where enforced_depth
    0 corresponds to simple hill climbing. Terminate when no improvement can
    be found. early_termination_heuristic_thresh allows for searching until
    heuristic reaches a specified value.

    Lower heuristic is better.
    """
    assert enforced_depth >= 0
    cur_node: _HeuristicSearchNode[_S, _A] = _HeuristicSearchNode(
        initial_state, 0, 0)
    last_heuristic = heuristic(cur_node.state)
    heuristics = [last_heuristic]
    visited = {initial_state}
    if verbose:
        logging.info(f"\n\nStarting hill climbing at state {cur_node.state} "
                     f"with heuristic {last_heuristic}")
    start_time = time.perf_counter()
    while True:

        # Stops when heuristic reaches specified value.
        if early_termination_heuristic_thresh is not None \
            and last_heuristic <= early_termination_heuristic_thresh:
            break

        if check_goal(cur_node.state):
            if verbose:
                logging.info("\nTerminating hill climbing, achieved goal")
            break
        best_heuristic = float("inf")
        best_child_node = None
        current_depth_nodes = [cur_node]
        all_best_heuristics = []
        for depth in range(0, enforced_depth + 1):
            if verbose:
                logging.info(f"Searching for an improvement at depth {depth}")
            # This is a list to ensure determinism. Note that duplicates are
            # filtered out in the `child_state in visited` check.
            successors_at_depth = []
            for parent in current_depth_nodes:
                for action, child_state, cost in get_successors(parent.state):
                    # Raise error if timeout gets hit.
                    if time.perf_counter() - start_time > timeout:
                        raise TimeoutError()
                    if child_state in visited:
                        continue
                    visited.add(child_state)
                    child_path_cost = parent.cumulative_cost + cost
                    child_node = _HeuristicSearchNode(
                        state=child_state,
                        edge_cost=cost,
                        cumulative_cost=child_path_cost,
                        parent=parent,
                        action=action)
                    successors_at_depth.append(child_node)
                    if parallelize:
                        continue  # heuristic computation is parallelized later
                    child_heuristic = heuristic(child_node.state)
                    if child_heuristic < best_heuristic:
                        best_heuristic = child_heuristic
                        best_child_node = child_node
            if parallelize:
                # Parallelize the expensive part (heuristic computation).
                num_cpus = mp.cpu_count()
                fn = lambda n: (heuristic(n.state), n)
                with mp.Pool(processes=num_cpus) as p:
                    for child_heuristic, child_node in p.map(
                            fn, successors_at_depth):
                        if child_heuristic < best_heuristic:
                            best_heuristic = child_heuristic
                            best_child_node = child_node
            all_best_heuristics.append(best_heuristic)
            if last_heuristic > best_heuristic:
                # Some improvement found.
                if verbose:
                    logging.info(f"Found an improvement at depth {depth}")
                break
            # Continue on to the next depth.
            current_depth_nodes = successors_at_depth
            if verbose:
                logging.info(f"No improvement found at depth {depth}")
        if best_child_node is None:
            if verbose:
                logging.info("\nTerminating hill climbing, no more successors")
            break
        if last_heuristic <= best_heuristic:
            if verbose:
                logging.info(
                    "\nTerminating hill climbing, could not improve score")
            break
        heuristics.extend(all_best_heuristics)
        cur_node = best_child_node
        last_heuristic = best_heuristic
        if verbose:
            logging.info(f"\nHill climbing reached new state {cur_node.state} "
                         f"with heuristic {last_heuristic}")
    states, actions = _finish_plan(cur_node)
    assert len(states) == len(heuristics)
    return states, actions, heuristics


def run_policy_guided_astar(
        initial_state: _S,
        check_goal: Callable[[_S], bool],
        get_valid_actions: Callable[[_S], Iterator[Tuple[_A, float]]],
        get_next_state: Callable[[_S, _A], _S],
        heuristic: Callable[[_S], float],
        policy: Callable[[_S], Optional[_A]],
        num_rollout_steps: int,
        rollout_step_cost: float,
        max_expansions: int = 10000000,
        max_evals: int = 10000000,
        timeout: int = 10000000,
        lazy_expansion: bool = False) -> Tuple[List[_S], List[_A]]:
    """Perform A* search, but at each node, roll out a given policy for a given
    number of timesteps, creating new successors at each step.

    Stop the rollout prematurely if the policy returns None.

    Note that unlike the other search functions, which take get_successors as
    input, this function takes get_valid_actions and get_next_state as two
    separate inputs. This is necessary because we need to anticipate the next
    state conditioned on the action output by the policy.

    The get_valid_actions generates (action, cost) tuples. For policy-generated
    transitions, the costs are ignored, and rollout_step_cost is used instead.
    """

    # Create a new successor function that rolls out the policy first.
    # A successor here means: from this state, if you take this sequence of
    # actions in order, you'll end up at this final state.
    def get_successors(state: _S) -> Iterator[Tuple[List[_A], _S, float]]:
        # Get policy-based successors.
        policy_state = state
        policy_action_seq = []
        policy_cost = 0.0
        for _ in range(num_rollout_steps):
            action = policy(policy_state)
            valid_actions = {a for a, _ in get_valid_actions(policy_state)}
            if action is None or action not in valid_actions:
                break
            policy_state = get_next_state(policy_state, action)
            policy_action_seq.append(action)
            policy_cost += rollout_step_cost
            yield (list(policy_action_seq), policy_state, policy_cost)

        # Get primitive successors.
        for action, cost in get_valid_actions(state):
            next_state = get_next_state(state, action)
            yield ([action], next_state, cost)

    _, action_subseqs = run_astar(initial_state=initial_state,
                                  check_goal=check_goal,
                                  get_successors=get_successors,
                                  heuristic=heuristic,
                                  max_expansions=max_expansions,
                                  max_evals=max_evals,
                                  timeout=timeout,
                                  lazy_expansion=lazy_expansion)

    # The states are "jumpy", so we need to reconstruct the dense state
    # sequence from the action subsequences. We also need to construct a
    # flat action sequence.
    state = initial_state
    state_seq = [state]
    action_seq = []
    for action_subseq in action_subseqs:
        for action in action_subseq:
            action_seq.append(action)
            state = get_next_state(state, action)
            state_seq.append(state)

    return state_seq, action_seq


_RRTState = TypeVar("_RRTState")


class RRT(Generic[_RRTState]):
    """Rapidly-exploring random tree."""

    def __init__(self, sample_fn: Callable[[_RRTState], _RRTState],
                 extend_fn: Callable[[_RRTState, _RRTState],
                                     Iterator[_RRTState]],
                 collision_fn: Callable[[_RRTState], bool],
                 distance_fn: Callable[[_RRTState, _RRTState],
                                       float], rng: np.random.Generator,
                 num_attempts: int, num_iters: int, smooth_amt: int):
        self._sample_fn = sample_fn
        self._extend_fn = extend_fn
        self._collision_fn = collision_fn
        self._distance_fn = distance_fn
        self._rng = rng
        self._num_attempts = num_attempts
        self._num_iters = num_iters
        self._smooth_amt = smooth_amt

    def query(self,
              pt1: _RRTState,
              pt2: _RRTState,
              sample_goal_eps: float = 0.0) -> Optional[List[_RRTState]]:
        """Query the RRT, to get a collision-free path from pt1 to pt2.

        If none is found, returns None.
        """
        if self._collision_fn(pt1) or self._collision_fn(pt2):
            return None
        direct_path = self._try_direct_path(pt1, pt2)
        if direct_path is not None:
            return direct_path
        for _ in range(self._num_attempts):
            path = self._rrt_connect(pt1,
                                     goal_sampler=lambda: pt2,
                                     sample_goal_eps=sample_goal_eps)
            if path is not None:
                return self._smooth_path(path)
        return None

    def query_to_goal_fn(
            self,
            start: _RRTState,
            goal_sampler: Callable[[], _RRTState],
            goal_fn: Callable[[_RRTState], bool],
            sample_goal_eps: float = 0.0) -> Optional[List[_RRTState]]:
        """Query the RRT, to get a collision-free path from start to a point
        such that goal_fn(point) is True. Uses goal_sampler to sample a target
        for a direct path or with probability sample_goal_eps.

        If none is found, returns None.
        """
        if self._collision_fn(start):
            return None
        direct_path = self._try_direct_path(start, goal_sampler())
        if direct_path is not None:
            return direct_path
        for _ in range(self._num_attempts):
            path = self._rrt_connect(start,
                                     goal_sampler,
                                     goal_fn,
                                     sample_goal_eps=sample_goal_eps)
            if path is not None:
                return self._smooth_path(path)
        return None

    def _try_direct_path(self, pt1: _RRTState,
                         pt2: _RRTState) -> Optional[List[_RRTState]]:
        path = [pt1]
        for newpt in self._extend_fn(pt1, pt2):
            if self._collision_fn(newpt):
                return None
            path.append(newpt)
        return path

    def _rrt_connect(
        self,
        pt1: _RRTState,
        goal_sampler: Callable[[], _RRTState],
        goal_fn: Optional[Callable[[_RRTState], bool]] = None,
        sample_goal_eps: float = 0.0,
    ) -> Optional[List[_RRTState]]:
        root = _RRTNode(pt1)
        nodes = [root]

        for _ in range(self._num_iters):
            # Sample the goal with a small probability, otherwise randomly
            # choose a point.
            sample_goal = self._rng.random() < sample_goal_eps
            samp = goal_sampler() if sample_goal else self._sample_fn(pt1)
            min_key = functools.partial(self._get_pt_dist_to_node, samp)
            nearest = min(nodes, key=min_key)
            reached_goal = False
            for newpt in self._extend_fn(nearest.data, samp):
                if self._collision_fn(newpt):
                    break
                nearest = _RRTNode(newpt, parent=nearest)
                nodes.append(nearest)
            else:
                reached_goal = sample_goal
            # Check goal_fn if defined
            if reached_goal or goal_fn is not None and goal_fn(nearest.data):
                path = nearest.path_from_root()
                return [node.data for node in path]
        return None

    def _get_pt_dist_to_node(self, pt: _RRTState,
                             node: _RRTNode[_RRTState]) -> float:
        return self._distance_fn(pt, node.data)

    def _smooth_path(self, path: List[_RRTState]) -> List[_RRTState]:
        assert len(path) > 2
        for _ in range(self._smooth_amt):
            i = self._rng.integers(0, len(path) - 1)
            j = self._rng.integers(0, len(path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = list(self._extend_fn(path[i], path[j]))
            if len(shortcut) < j - i and \
                    all(not self._collision_fn(pt) for pt in shortcut):
                path = path[:i + 1] + shortcut + path[j + 1:]
        return path


class BiRRT(RRT[_RRTState]):
    """Bidirectional rapidly-exploring random tree."""

    def query_to_goal_fn(
            self,
            start: _RRTState,
            goal_sampler: Callable[[], _RRTState],
            goal_fn: Callable[[_RRTState], bool],
            sample_goal_eps: float = 0.0) -> Optional[List[_RRTState]]:
        raise NotImplementedError("Can't query to goal function using BiRRT")

    def _rrt_connect(
        self,
        pt1: _RRTState,
        goal_sampler: Callable[[], _RRTState],
        goal_fn: Optional[Callable[[_RRTState], bool]] = None,
        sample_goal_eps: float = 0.0,
    ) -> Optional[List[_RRTState]]:
        # goal_fn and sample_goal_eps are unused
        pt2 = goal_sampler()
        root1, root2 = _RRTNode(pt1), _RRTNode(pt2)
        nodes1, nodes2 = [root1], [root2]

        for _ in range(self._num_iters):
            if len(nodes1) > len(nodes2):
                nodes1, nodes2 = nodes2, nodes1
            samp = self._sample_fn(pt1)
            min_key1 = functools.partial(self._get_pt_dist_to_node, samp)
            nearest1 = min(nodes1, key=min_key1)
            for newpt in self._extend_fn(nearest1.data, samp):
                if self._collision_fn(newpt):
                    break
                nearest1 = _RRTNode(newpt, parent=nearest1)
                nodes1.append(nearest1)
            min_key2 = functools.partial(self._get_pt_dist_to_node,
                                         nearest1.data)
            nearest2 = min(nodes2, key=min_key2)
            for newpt in self._extend_fn(nearest2.data, nearest1.data):
                if self._collision_fn(newpt):
                    break
                nearest2 = _RRTNode(newpt, parent=nearest2)
                nodes2.append(nearest2)
            else:
                path1 = nearest1.path_from_root()
                path2 = nearest2.path_from_root()
                # This is a tricky case to cover.
                if path1[0] != root1:  # pragma: no cover
                    path1, path2 = path2, path1
                assert path1[0] == root1
                path = path1[:-1] + path2[::-1]
                return [node.data for node in path]
        return None


class _RRTNode(Generic[_RRTState]):
    """A node for RRT."""

    def __init__(self,
                 data: _RRTState,
                 parent: Optional[_RRTNode[_RRTState]] = None) -> None:
        self.data = data
        self.parent = parent

    def path_from_root(self) -> List[_RRTNode[_RRTState]]:
        """Return the path from the root to this node."""
        sequence = []
        node: Optional[_RRTNode[_RRTState]] = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]


def strip_predicate(predicate: Predicate) -> Predicate:
    """Remove the classifier from the given predicate to make a new Predicate.

    Implement this by replacing the classifier with one that errors.
    """

    def _stripped_classifier(state: State, objects: Sequence[Object]) -> bool:
        raise Exception("Stripped classifier should never be called!")

    return Predicate(predicate.name, predicate.types, _stripped_classifier)


def strip_task(task: Task, included_predicates: Set[Predicate]) -> Task:
    """Create a new task where any excluded goal predicates have their
    classifiers removed."""
    stripped_goal: Set[GroundAtom] = set()
    for atom in task.goal:
        if atom.predicate in included_predicates:
            stripped_goal.add(atom)
            continue
        stripped_pred = strip_predicate(atom.predicate)
        stripped_atom = GroundAtom(stripped_pred, atom.objects)
        stripped_goal.add(stripped_atom)
    return Task(task.init, stripped_goal, alt_goal=task.alt_goal)


def create_vlm_predicate(
        name: str, types: Sequence[Type],
        get_vlm_query_str: Callable[[Sequence[Object]], str]) -> VLMPredicate:
    """Simple function that creates VLMPredicates with dummy classifiers, which
    is the most-common way these need to be created."""

    def _stripped_classifier(
            state: State,
            objects: Sequence[Object]) -> bool:  # pragma: no cover.
        raise Exception("VLM predicate classifier should never be called!")

    return VLMPredicate(name, types, _stripped_classifier, get_vlm_query_str)


def create_llm_by_name(
        model_name: str) -> LargeLanguageModel:  # pragma: no cover
    """Create particular llm using a provided name."""
    if "gemini" in model_name:
        return GoogleGeminiLLM(model_name)
    return OpenAILLM(model_name)


def create_vlm_by_name(
        model_name: str) -> VisionLanguageModel:  # pragma: no cover
    """Create particular vlm using a provided name."""
    if "gemini" in model_name:
        return GoogleGeminiVLM(model_name)
    return OpenAIVLM(model_name)


def parse_model_output_into_option_plan(
    model_prediction: str, objects: Collection[Object],
    types: Collection[Type], options: Collection[ParameterizedOption],
    parse_continuous_params: bool
) -> List[Tuple[ParameterizedOption, Sequence[Object], Sequence[float]]]:
    """Assuming text for an option plan that is predicted as text by a large
    model, parse it into a sequence of ParameterizedOptions coupled with a list
    of objects and continuous parameters that will be used to ground the
    ParameterizedOption.

    We assume the model's output is such that each line is formatted as
    option_name(obj0:type0, obj1:type1,...)[continuous_param0,
    continuous_param1, ...].
    """
    option_plan: List[Tuple[ParameterizedOption, Sequence[Object],
                            Sequence[float]]] = []
    # Setup dictionaries enabling us to easily map names to specific
    # Python objects during parsing.
    option_name_to_option = {op.name: op for op in options}
    type_name_to_type = {typ.name: typ for typ in types}
    obj_name_to_obj = {o.name: o for o in objects}
    options_str_list = model_prediction.split('\n')
    for option_str in options_str_list:
        option_str_stripped = option_str.strip()
        option_name = option_str_stripped.split('(')[0]
        # Skip empty option strs.
        if not option_str:
            continue
        if option_name not in option_name_to_option.keys() or \
            "(" not in option_str:
            logging.info(
                f"Line {option_str} output by model doesn't "
                "contain a valid option name. Terminating option plan "
                "parsing.")
            break
        if parse_continuous_params and "[" not in option_str:
            logging.info(
                f"Line {option_str} output by model doesn't contain a "
                "'[' and is thus improperly formatted.")
            break
        option = option_name_to_option[option_name]
        # Now that we have the option, we need to parse out the objects
        # along with specified types.
        try:
            start_index = option_str_stripped.index('(') + 1
            end_index = option_str_stripped.index(')', start_index)
        except ValueError:
            logging.info(
                f"Line {option_str} output by model is improperly formatted.")
            break
        typed_objects_str_list = option_str_stripped[
            start_index:end_index].split(',')
        objs_list = []
        continuous_params_list = []
        malformed = False
        for i, type_object_string in enumerate(typed_objects_str_list):
            object_type_str_list = type_object_string.strip().split(':')
            # We expect this list to be [object_name, type_name].
            if len(object_type_str_list) != 2:
                logging.info(f"Line {option_str} output by model has a "
                             "malformed object-type list.")
                malformed = True
                break
            object_name = object_type_str_list[0]
            type_name = object_type_str_list[1]
            if object_name not in obj_name_to_obj.keys():
                logging.info(f"Line {option_str} output by model has an "
                             "invalid object name.")
                malformed = True
                break
            obj = obj_name_to_obj[object_name]
            # Check that the type of this object agrees
            # with what's expected given the ParameterizedOption.
            if type_name not in type_name_to_type:
                logging.info(f"Line {option_str} output by model has an "
                             "invalid type name.")
                malformed = True
                break
            try:
                if option.types[i] not in type_name_to_type[
                        type_name].get_ancestors():
                    logging.info(
                        f"Line {option_str} output by model has an "
                        "invalid type that doesn't agree with the option"
                        f"{option}")
                    malformed = True
                    break
            except IndexError:
                # In this case, there's more supplied arguments than the
                # option has.
                logging.info(f"Line {option_str} output by model has an "
                             "too many object arguments for option"
                             f"{option}")
                malformed = True
                break
            objs_list.append(obj)
        # The types of the objects match, but we haven't yet checked if
        # all arguments of the option have an associated object.
        if len(objs_list) != len(option.types):
            malformed = True
        # Now, we attempt to parse out the continuous parameters.
        if parse_continuous_params:
            params_str_list = option_str_stripped.split('[')[1].strip(
                ']').split(',')
            for i, continuous_params_str in enumerate(params_str_list):
                stripped_continuous_param_str = continuous_params_str.strip()
                if len(stripped_continuous_param_str) == 0:
                    continue
                try:
                    curr_cont_param = float(stripped_continuous_param_str)
                except ValueError:
                    logging.info(f"Line {option_str} output by model has an "
                                 "invalid continouous parameter that can't be"
                                 "converted to a float.")
                    malformed = True
                    break
                continuous_params_list.append(curr_cont_param)
            if len(continuous_params_list) != option.params_space.shape[0]:
                logging.info(f"Line {option_str} output by model has "
                             "invalid continouous parameter(s) that don't "
                             f"agree with {option}{option.params_space}.")
                malformed = True
                break
        if not malformed:
            option_plan.append((option, objs_list, continuous_params_list))
    return option_plan


def get_prompt_for_vlm_state_labelling(
        prompt_type: str, atoms_list: List[str], label_history: List[str],
        imgs_history: List[List[PIL.Image.Image]],
        cropped_imgs_history: List[List[PIL.Image.Image]],
        skill_history: List[Action]) -> Tuple[str, List[PIL.Image.Image]]:
    """Prompt for generating labels for an entire trajectory. Similar to the
    above prompting method, this outputs a list of prompts to label the state
    at each timestep of traj with atom values).

    Note that all our prompts are saved as separate txt files under the
    'vlm_input_data_prompts/atom_labelling' folder.
    """
    # Load the pre-specified prompt.
    filepath_prefix = get_path_to_predicators_root() + \
        "/predicators/datasets/vlm_input_data_prompts/atom_proposal/"
    try:
        with open(filepath_prefix +
                  CFG.grammar_search_vlm_atom_label_prompt_type + ".txt",
                  "r",
                  encoding="utf-8") as f:
            prompt = f.read()
    except FileNotFoundError:
        raise ValueError("Unknown VLM prompting option " +
                         f"{CFG.grammar_search_vlm_atom_label_prompt_type}")
    # The prompt ends with a section for 'Predicates', so list these.
    for atom_str in atoms_list:
        prompt += f"\n{atom_str}"

    if "img_option_diffs" in prompt_type:
        # In this case, we need to load the 'per_scene_naive' prompt as well
        # for the first timestep.
        with open(filepath_prefix + "per_scene_naive.txt",
                  "r",
                  encoding="utf-8") as f:
            init_prompt = f.read()
        for atom_str in atoms_list:
            init_prompt += f"\n{atom_str}"
        if len(label_history) == 0:
            return (init_prompt, imgs_history[0])
        # Now, we use actual difference-based prompting for the second timestep
        # and beyond.
        curr_prompt = prompt[:]
        curr_prompt_imgs = [
            imgs_timestep for imgs_timestep in imgs_history[-1]
        ]
        if CFG.vlm_include_cropped_images:
            if CFG.env in ["burger", "burger_no_move"]:  # pragma: no cover
                curr_prompt_imgs.extend(
                    [cropped_imgs_history[-1][1], cropped_imgs_history[-1][0]])
            else:
                raise NotImplementedError(
                    f"Cropped images not implemented for {CFG.env}.")
        curr_prompt += "\n\nSkill executed between states: "
        skill_name = skill_history[-1].name + str(skill_history[-1].objects)
        curr_prompt += skill_name
        if "label_history" in prompt_type:
            curr_prompt += "\n\nPredicate values in the first scene, " \
            "before the skill was executed: \n"
            curr_prompt += label_history[-1]
        return (curr_prompt, curr_prompt_imgs)
    else:
        # NOTE: we rip out only the first image from each trajectory
        # which is fine for most domains, but will be problematic for
        # situations in which there is more than one image per state.
        return (prompt, [imgs_history[-1][0]])


def query_vlm_for_atom_vals(
        vlm_atoms: Collection[GroundAtom],
        state: State,
        vlm: Optional[VisionLanguageModel] = None) -> Set[GroundAtom]:
    """Given a set of ground atoms, queries a VLM and gets the subset of these
    atoms that are true."""
    true_atoms: Set[GroundAtom] = set()
    # This only works if state.simulator_state is some list of images that the
    # vlm can be called on.
    assert state.simulator_state is not None
    assert isinstance(state.simulator_state["images"], List)
    if "vlm_atoms_history" not in state.simulator_state:
        state.simulator_state["vlm_atoms_history"] = []
    imgs = state.simulator_state["images"]
    previous_states = []
    # We assume the state.simulator_state contains a list of previous states.
    if "state_history" in state.simulator_state:
        previous_states = state.simulator_state["state_history"]
    state_imgs_history = [state.simulator_state["images"] for state in previous_states]
    vlm_atoms = sorted(vlm_atoms)
    atom_queries_str = [atom.get_vlm_query_str() for atom in vlm_atoms]
    vlm_query_str, imgs = get_prompt_for_vlm_state_labelling(CFG.vlm_test_time_atom_label_prompt_type, atom_queries_str, state.simulator_state["vlm_atoms_history"], state_imgs_history, [], state.simulator_state["skill_history"])
    if vlm is None:
        vlm = create_vlm_by_name(CFG.vlm_model_name)  # pragma: no cover.
    vlm_input_imgs = \
        [PIL.Image.fromarray(img_arr) for img_arr in imgs] # type: ignore
    vlm_output = vlm.sample_completions(vlm_query_str,
                                        vlm_input_imgs,
                                        0.0,
                                        seed=CFG.seed,
                                        num_completions=1)
    assert len(vlm_output) == 1
    vlm_output_str = vlm_output[0]
    print(f"VLM output: {vlm_output_str}")
    all_atom_queries = atom_queries_str.strip().split("\n")
    all_vlm_responses = vlm_output_str.strip().split("\n")
    # NOTE: this assumption is likely too brittle; if this is breaking, feel
    # free to remove/adjust this and change the below parsing loop accordingly!
    assert len(all_atom_queries) == len(all_vlm_responses)
    for i, (atom_query, curr_vlm_output_line) in enumerate(
            zip(all_atom_queries, all_vlm_responses)):
        assert atom_query + ":" in curr_vlm_output_line
        assert "." in curr_vlm_output_line
        period_idx = curr_vlm_output_line.find(".")
        if curr_vlm_output_line[len(atom_query +
                                    ":"):period_idx].lower().strip() == "true":
            true_atoms.add(vlm_atoms[i])
    # Add the text of the VLM's response to the state, to be used in the future!
    state.simulator_state["vlm_atoms_history"].append(all_vlm_responses)
    return true_atoms


def abstract(state: State,
             preds: Collection[Predicate],
             vlm: Optional[VisionLanguageModel] = None) -> Set[GroundAtom]:
    """Get the atomic representation of the given state (i.e., a set of ground
    atoms), using the given set of predicates.

    Duplicate arguments in predicates are allowed.
    """
    # Start by pulling out all VLM predicates.
    vlm_preds = set(pred for pred in preds if isinstance(pred, VLMPredicate))
    # Next, classify all non-VLM predicates.
    atoms = set()
    for pred in preds:
        if pred not in vlm_preds:
            for choice in get_object_combinations(list(state), pred.types):
                if pred.holds(state, choice):
                    atoms.add(GroundAtom(pred, choice))
    if len(vlm_preds) > 0:
        # Now, aggregate all the VLM predicates and make a single call to a
        # VLM to get their values.
        vlm_atoms = set()
        for pred in vlm_preds:
            for choice in get_object_combinations(list(state), pred.types):
                vlm_atoms.add(GroundAtom(pred, choice))
        true_vlm_atoms = query_vlm_for_atom_vals(vlm_atoms, state, vlm)
        atoms |= true_vlm_atoms
    return atoms


def all_ground_operators(
        operator: STRIPSOperator,
        objects: Collection[Object]) -> Iterator[_GroundSTRIPSOperator]:
    """Get all possible groundings of the given operator with the given
    objects."""
    types = [p.type for p in operator.parameters]
    for choice in get_object_combinations(objects, types):
        yield operator.ground(tuple(choice))


def all_ground_operators_given_partial(
        operator: STRIPSOperator, objects: Collection[Object],
        sub: VarToObjSub) -> Iterator[_GroundSTRIPSOperator]:
    """Get all possible groundings of the given operator with the given objects
    such that the parameters are consistent with the given substitution."""
    assert set(sub).issubset(set(operator.parameters))
    types = [p.type for p in operator.parameters if p not in sub]
    for choice in get_object_combinations(objects, types):
        # Complete the choice with the args that are determined from the sub.
        choice_lst = list(choice)
        choice_lst.reverse()
        completed_choice = []
        for p in operator.parameters:
            if p in sub:
                completed_choice.append(sub[p])
            else:
                completed_choice.append(choice_lst.pop())
        assert not choice_lst
        ground_op = operator.ground(tuple(completed_choice))
        yield ground_op


def all_ground_nsrts(nsrt: NSRT,
                     objects: Collection[Object]) -> Iterator[_GroundNSRT]:
    """Get all possible groundings of the given NSRT with the given objects."""
    types = [p.type for p in nsrt.parameters]
    for choice in get_object_combinations(objects, types):
        yield nsrt.ground(tuple(choice))


def all_ground_nsrts_fd_translator(
        nsrts: Set[NSRT], objects: Collection[Object],
        predicates: Set[Predicate], types: Set[Type],
        init_atoms: Set[GroundAtom],
        goal: Set[GroundAtom]) -> Iterator[_GroundNSRT]:
    """Get all possible groundings of the given set of NSRTs with the given
    objects, using Fast Downward's translator for efficiency."""
    nsrt_name_to_nsrt = {nsrt.name.lower(): nsrt for nsrt in nsrts}
    obj_name_to_obj = {obj.name.lower(): obj for obj in objects}
    dom_str = create_pddl_domain(nsrts, predicates, types, "mydomain")
    prob_str = create_pddl_problem(objects, init_atoms, goal, "mydomain",
                                   "myproblem")
    with nostdout():
        sas_task = downward_translate(dom_str, prob_str)  # type: ignore
    for operator in sas_task.operators:
        split_name = operator.name[1:-1].split()  # strip out ( and )
        nsrt = nsrt_name_to_nsrt[split_name[0]]
        objs = [obj_name_to_obj[name] for name in split_name[1:]]
        yield nsrt.ground(objs)


def all_possible_ground_atoms(state: State,
                              preds: Set[Predicate]) -> List[GroundAtom]:
    """Get a sorted list of all possible ground atoms in a state given the
    predicates.

    Ignores the predicates' classifiers.
    """
    objects = frozenset(state)
    ground_atoms = set()
    for pred in preds:
        ground_atoms |= get_all_ground_atoms_for_predicate(pred, objects)
    return sorted(ground_atoms)


def all_ground_ldl_rules(
    rule: LDLRule,
    objects: Collection[Object],
    static_predicates: Optional[Collection[Predicate]] = None,
    init_atoms: Optional[Collection[GroundAtom]] = None
) -> List[_GroundLDLRule]:
    """Get all possible groundings of the given rule with the given objects.

    If provided, use the static predicates and init_atoms to avoid
    grounding rules that will never have satisfied preconditions in any
    state.
    """
    if static_predicates is None:
        static_predicates = set()
    if init_atoms is None:
        init_atoms = set()
    return _cached_all_ground_ldl_rules(rule, frozenset(objects),
                                        frozenset(static_predicates),
                                        frozenset(init_atoms))


@functools.lru_cache(maxsize=None)
def _cached_all_ground_ldl_rules(
        rule: LDLRule, objects: FrozenSet[Object],
        static_predicates: FrozenSet[Predicate],
        init_atoms: FrozenSet[GroundAtom]) -> List[_GroundLDLRule]:
    """Helper for all_ground_ldl_rules() that caches the outputs."""
    ground_rules = []
    # Use static preconds to reduce the map of parameters to possible objects.
    # For example, if IsBall(?x) is a positive state precondition, then only
    # the objects that appear in init_atoms with IsBall could bind to ?x.
    # For now, we just check unary static predicates, since that covers the
    # common case where such predicates are used in place of types.
    # Create map from each param to unary static predicates.
    param_to_pos_preds: Dict[Variable, Set[Predicate]] = {
        p: set()
        for p in rule.parameters
    }
    param_to_neg_preds: Dict[Variable, Set[Predicate]] = {
        p: set()
        for p in rule.parameters
    }
    for (preconditions, param_to_preds) in [
        (rule.pos_state_preconditions, param_to_pos_preds),
        (rule.neg_state_preconditions, param_to_neg_preds),
    ]:
        for atom in preconditions:
            pred = atom.predicate
            if pred in static_predicates and pred.arity == 1:
                param = atom.variables[0]
                param_to_preds[param].add(pred)
    # Create the param choices, filtering based on the unary static atoms.
    param_choices = []  # list of lists of possible objects for each param
    # Preprocess the atom sets for faster lookups.
    init_atom_tups = {(a.predicate, tuple(a.objects)) for a in init_atoms}
    for param in rule.parameters:
        choices = []
        for obj in objects:
            # Types must match, as usual.
            if obj.type != param.type:
                continue
            # Check the static conditions.
            binding_valid = True
            for pred in param_to_pos_preds[param]:
                if (pred, (obj, )) not in init_atom_tups:
                    binding_valid = False
                    break
            for pred in param_to_neg_preds[param]:
                if (pred, (obj, )) in init_atom_tups:
                    binding_valid = False
                    break
            if binding_valid:
                choices.append(obj)
        # Must be sorted for consistency with other grounding code.
        param_choices.append(sorted(choices))
    for choice in itertools.product(*param_choices):
        ground_rule = rule.ground(choice)
        ground_rules.append(ground_rule)
    return ground_rules


def parse_ldl_from_str(ldl_str: str, types: Collection[Type],
                       predicates: Collection[Predicate],
                       nsrts: Collection[NSRT]) -> LiftedDecisionList:
    """Parse a lifted decision list from a string representation of it."""
    parser = _LDLParser(types, predicates, nsrts)
    return parser.parse(ldl_str)


class _LDLParser:
    """Parser for lifted decision lists from strings."""

    def __init__(self, types: Collection[Type],
                 predicates: Collection[Predicate],
                 nsrts: Collection[NSRT]) -> None:
        self._nsrt_name_to_nsrt = {nsrt.name.lower(): nsrt for nsrt in nsrts}
        self._type_name_to_type = {t.name.lower(): t for t in types}
        self._predicate_name_to_predicate = {
            p.name.lower(): p
            for p in predicates
        }

    def parse(self, ldl_str: str) -> LiftedDecisionList:
        """Run parsing."""
        ldl_str = ldl_str.lower()  # ignore case during parsing
        rules = []
        rule_matches = re.finditer(r"\(:rule", ldl_str)
        for start in rule_matches:
            rule_str = find_balanced_expression(ldl_str, start.start())
            rule = self._parse_rule(rule_str)
            rules.append(rule)
        return LiftedDecisionList(rules)

    def _parse_rule(self, rule_str: str) -> LDLRule:
        rule_pattern = r"\(:rule(.*):parameters(.*):preconditions(.*)" + \
                       r":goals(.*):action(.*)\)"
        match_result = re.match(rule_pattern, rule_str, re.DOTALL)
        assert match_result is not None
        # Remove white spaces.
        matches = [m.strip().rstrip() for m in match_result.groups()]
        # Unpack the matches.
        rule_name, params_str, preconds_str, goals_str, nsrt_str = matches
        # Handle the parameters.
        assert "?" in params_str, "Assuming all rules have parameters."
        variable_name_to_variable = {}
        assert params_str.endswith(")")
        for param_str in params_str[:-1].split("?")[1:]:
            param_name, param_type_str = param_str.split("-")
            param_name = param_name.strip()
            param_type_str = param_type_str.strip()
            variable_name = "?" + param_name
            param_type = self._type_name_to_type[param_type_str]
            variable = Variable(variable_name, param_type)
            variable_name_to_variable[variable_name] = variable
        # Handle the preconditions.
        pos_preconds, neg_preconds = self._parse_lifted_atoms(
            preconds_str, variable_name_to_variable)
        # Handle the goals.
        pos_goals, neg_goals = self._parse_lifted_atoms(
            goals_str, variable_name_to_variable)
        assert not neg_goals, "Negative LDL goals not currently supported"
        # Handle the NSRT.
        nsrt = self._parse_into_nsrt(nsrt_str, variable_name_to_variable)
        # Finalize the rule.
        params = sorted(variable_name_to_variable.values())
        return LDLRule(rule_name, params, pos_preconds, neg_preconds,
                       pos_goals, nsrt)

    def _parse_lifted_atoms(
        self, atoms_str: str, variable_name_to_variable: Dict[str, Variable]
    ) -> Tuple[Set[LiftedAtom], Set[LiftedAtom]]:
        """Parse the given string (representing either preconditions or
        effects) into a set of positive lifted atoms and a set of negative
        lifted atoms.

        Check against params to make sure typing is correct.
        """
        assert atoms_str[0] == "("
        assert atoms_str[-1] == ")"

        # Handle conjunctions.
        if atoms_str.startswith("(and") and atoms_str[4] in (" ", "\n", "("):
            clauses = find_all_balanced_expressions(atoms_str[4:-1].strip())
            pos_atoms, neg_atoms = set(), set()
            for clause in clauses:
                clause_pos_atoms, clause_neg_atoms = self._parse_lifted_atoms(
                    clause, variable_name_to_variable)
                pos_atoms |= clause_pos_atoms
                neg_atoms |= clause_neg_atoms
            return pos_atoms, neg_atoms

        # Handle negations.
        if atoms_str.startswith("(not") and atoms_str[4] in (" ", "\n", "("):
            # Only contains a single literal inside not.
            split_strs = atoms_str[4:-1].strip()[1:-1].strip().split()
            pred = self._predicate_name_to_predicate[split_strs[0]]
            args = [variable_name_to_variable[arg] for arg in split_strs[1:]]
            lifted_atom = LiftedAtom(pred, args)
            return set(), {lifted_atom}

        # Handle single positive atoms.
        split_strs = atoms_str[1:-1].split()
        # Empty conjunction.
        if not split_strs:
            return set(), set()
        pred = self._predicate_name_to_predicate[split_strs[0]]
        args = [variable_name_to_variable[arg] for arg in split_strs[1:]]
        lifted_atom = LiftedAtom(pred, args)
        return {lifted_atom}, set()

    def _parse_into_nsrt(
            self, nsrt_str: str,
            variable_name_to_variable: Dict[str, Variable]) -> NSRT:
        """Parse the given string into an NSRT."""
        assert nsrt_str[0] == "("
        assert nsrt_str[-1] == ")"
        nsrt_str = nsrt_str[1:-1].split()[0]
        nsrt = self._nsrt_name_to_nsrt[nsrt_str]
        # Validate parameters.
        variables = variable_name_to_variable.values()
        for v in nsrt.parameters:
            assert v in variables, f"NSRT parameter {v} missing from LDL rule"
        return nsrt


_T = TypeVar("_T")  # element of a set


def sample_subsets(universe: Sequence[_T], num_samples: int, min_set_size: int,
                   max_set_size: int,
                   rng: np.random.Generator) -> Iterator[Set[_T]]:
    """Sample multiple subsets from a universe."""
    assert min_set_size <= max_set_size
    assert max_set_size <= len(universe), "Not enough elements in universe"
    for _ in range(num_samples):
        set_size = rng.integers(min_set_size, max_set_size + 1)
        idxs = rng.choice(np.arange(len(universe)),
                          size=set_size,
                          replace=False)
        sample = {universe[i] for i in idxs}
        yield sample


def create_dataset_filename_str(
        saving_ground_atoms: bool,
        online_learning_cycle: Optional[int] = None) -> Tuple[str, str]:
    """Generate strings to be used for the filename for a dataset file that is
    about to be saved.

    Returns a tuple of strings where the first element is the dataset
    filename itself and the second is a template string used to generate
    it. If saving_ground_atoms is True, then we will name the file with
    a "_ground_atoms" suffix.
    """
    # Setup the dataset filename for saving/loading GroundAtoms.
    regex = r"(\d+)"
    suffix_str = ""
    suffix_str += f"__{online_learning_cycle}"
    if saving_ground_atoms:
        suffix_str += "__ground_atoms"
    suffix_str += ".data"
    dataset_fname_template = (
        f"{CFG.env}__{CFG.offline_data_method}__{CFG.demonstrator}__"
        f"{regex}__{CFG.included_options}__{CFG.seed}" + suffix_str)
    dataset_fname = os.path.join(
        CFG.data_dir,
        dataset_fname_template.replace(regex, str(CFG.num_train_tasks)))
    return dataset_fname, dataset_fname_template


def create_ground_atom_dataset(
        trajectories: Sequence[LowLevelTrajectory],
        predicates: Set[Predicate]) -> List[GroundAtomTrajectory]:
    """Apply all predicates to all trajectories in the dataset."""
    ground_atom_dataset = []
    for traj in trajectories:
        atoms = [abstract(s, predicates) for s in traj.states]
        ground_atom_dataset.append((traj, atoms))
    return ground_atom_dataset


def prune_ground_atom_dataset(
        ground_atom_dataset: List[GroundAtomTrajectory],
        kept_predicates: Collection[Predicate]) -> List[GroundAtomTrajectory]:
    """Create a new ground atom dataset by keeping only some predicates."""
    new_ground_atom_dataset = []
    for traj, atoms in ground_atom_dataset:
        assert len(traj.states) == len(atoms)
        kept_atoms = [{a
                       for a in sa if a.predicate in kept_predicates}
                      for sa in atoms]
        new_ground_atom_dataset.append((traj, kept_atoms))
    return new_ground_atom_dataset


def load_ground_atom_dataset(
        dataset_fname: str,
        trajectories: List[LowLevelTrajectory]) -> List[GroundAtomTrajectory]:
    """Load a previously-saved ground atom dataset.

    Note importantly that we only save atoms themselves, we don't save
    the low-level trajectory information that's necessary to make
    GroundAtomTrajectories given series of ground atoms (that info can
    be saved separately, in case one wants to just load trajectories and
    not also load ground atoms). Thus, this function needs to take these
    trajectories as input.
    """
    os.makedirs(CFG.data_dir, exist_ok=True)
    # Check that the dataset file was previously saved.
    ground_atom_dataset_atoms: Optional[List[List[Set[GroundAtom]]]] = []
    if os.path.exists(dataset_fname):
        # Load the ground atoms dataset.
        with open(dataset_fname, "rb") as f:
            ground_atom_dataset_atoms = pkl.load(f)
        assert ground_atom_dataset_atoms is not None
        assert len(trajectories) == len(ground_atom_dataset_atoms)
        logging.info("\n\nLOADED GROUND ATOM DATASET")

        # The saved ground atom dataset consists only of sequences
        # of sets of GroundAtoms, we need to recombine this with
        # the LowLevelTrajectories to create a GroundAtomTrajectory.
        ground_atom_dataset = []
        for i, traj in enumerate(trajectories):
            ground_atom_seq = ground_atom_dataset_atoms[i]
            ground_atom_dataset.append(
                (traj, [set(atoms) for atoms in ground_atom_seq]))
    else:
        raise ValueError(f"Cannot load ground atoms: {dataset_fname}")
    return ground_atom_dataset


def save_ground_atom_dataset(ground_atom_dataset: List[GroundAtomTrajectory],
                             dataset_fname: str) -> None:
    """Saves a given ground atom dataset so it can be loaded in the future."""
    # Save ground atoms dataset to file. Note that a
    # GroundAtomTrajectory contains a normal LowLevelTrajectory and a
    # list of sets of GroundAtoms, so we only save the list of
    # GroundAtoms (the LowLevelTrajectories are saved separately).
    ground_atom_dataset_to_pkl = []
    for gt_traj in ground_atom_dataset:
        trajectory = []
        for ground_atom_set in gt_traj[1]:
            trajectory.append(ground_atom_set)
        ground_atom_dataset_to_pkl.append(trajectory)
    with open(dataset_fname, "wb") as f:
        pkl.dump(ground_atom_dataset_to_pkl, f)


def merge_ground_atom_datasets(
        gad1: List[GroundAtomTrajectory],
        gad2: List[GroundAtomTrajectory]) -> List[GroundAtomTrajectory]:
    """Merges two ground atom datasets sharing the same underlying low-level
    trajectory via the union of ground atoms at each state."""
    assert len(gad1) == len(
        gad2), "Ground atom datasets must be of the same length to merge them."
    merged_ground_atom_dataset = []
    for ground_atom_traj1, ground_atom_traj2 in zip(gad1, gad2):
        ll_traj1, ga_list1 = ground_atom_traj1
        ll_traj2, ga_list2 = ground_atom_traj2
        assert ll_traj1 == ll_traj2, "Ground atom trajectories must share " \
            "the same low-level trajectory to be able to merge them."
        merged_ga_list = [ga1 | ga2 for ga1, ga2 in zip(ga_list1, ga_list2)]
        merged_ground_atom_dataset.append((ll_traj1, merged_ga_list))
    return merged_ground_atom_dataset


def extract_preds_and_types(
    ops: Collection[NSRTOrSTRIPSOperator]
) -> Tuple[Dict[str, Predicate], Dict[str, Type]]:
    """Extract the predicates and types used in the given operators."""
    preds = {}
    types = {}
    for op in ops:
        for atom in op.preconditions | op.add_effects | op.delete_effects:
            for var_type in atom.predicate.types:
                types[var_type.name] = var_type
            preds[atom.predicate.name] = atom.predicate
    return preds, types


def get_static_preds(ops: Collection[NSRTOrSTRIPSOperator],
                     predicates: Collection[Predicate]) -> Set[Predicate]:
    """Get the subset of predicates from the given set that are static with
    respect to the given lifted operators."""
    static_preds = set()
    for pred in predicates:
        # This predicate is not static if it appears in any op's effects.
        if any(
                any(atom.predicate == pred for atom in op.add_effects) or any(
                    atom.predicate == pred for atom in op.delete_effects)
                for op in ops):
            continue
        static_preds.add(pred)
    return static_preds


def get_static_atoms(ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
                     atoms: Collection[GroundAtom]) -> Set[GroundAtom]:
    """Get the subset of atoms from the given set that are static with respect
    to the given ground operators.

    Note that this can include MORE than simply the set of atoms whose
    predicates are static, because now we have ground operators.
    """
    static_atoms = set()
    for atom in atoms:
        # This atom is not static if it appears in any op's effects.
        if any(
                any(atom == eff for eff in op.add_effects) or any(
                    atom == eff for eff in op.delete_effects)
                for op in ground_ops):
            continue
        static_atoms.add(atom)
    return static_atoms


def get_reachable_atoms(ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
                        atoms: Collection[GroundAtom]) -> Set[GroundAtom]:
    """Get all atoms that are reachable from the init atoms."""
    reachables = set(atoms)
    while True:
        fixed_point_reached = True
        for op in ground_ops:
            if op.preconditions.issubset(reachables):
                for new_reachable_atom in op.add_effects - reachables:
                    fixed_point_reached = False
                    reachables.add(new_reachable_atom)
        if fixed_point_reached:
            break
    return reachables


def get_applicable_operators(
        ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
        atoms: Collection[GroundAtom]) -> Iterator[GroundNSRTOrSTRIPSOperator]:
    """Iterate over ground operators whose preconditions are satisfied.

    Note: the order may be nondeterministic. Users should be invariant.
    """
    for op in ground_ops:
        applicable = op.preconditions.issubset(atoms)
        if applicable:
            yield op


def apply_operator(op: GroundNSRTOrSTRIPSOperator,
                   atoms: Set[GroundAtom]) -> Set[GroundAtom]:
    """Get a next set of atoms given a current set and a ground operator."""
    # Note that we are removing the ignore effects before the
    # application of the operator, because if the ignore effect
    # appears in the effects, we still know that the effects
    # will be true, so we don't want to remove them.
    new_atoms = {a for a in atoms if a.predicate not in op.ignore_effects}
    for atom in op.delete_effects:
        new_atoms.discard(atom)
    for atom in op.add_effects:
        new_atoms.add(atom)
    return new_atoms


def compute_necessary_atoms_seq(
        skeleton: List[_GroundNSRT], atoms_seq: List[Set[GroundAtom]],
        goal: Set[GroundAtom]) -> List[Set[GroundAtom]]:
    """Given a skeleton and a corresponding atoms sequence, return a
    'necessary' atoms sequence that includes only the necessary image at each
    step."""
    necessary_atoms_seq = [set(goal)]
    necessary_image = set(goal)
    for t in range(len(atoms_seq) - 2, -1, -1):
        curr_nsrt = skeleton[t]
        necessary_image -= set(curr_nsrt.add_effects)
        necessary_image |= set(curr_nsrt.preconditions)
        necessary_atoms_seq = [set(necessary_image)] + necessary_atoms_seq
    return necessary_atoms_seq


def compute_atoms_seq_from_plan(
        skeleton: List[_GroundNSRT],
        init_atoms: Set[GroundAtom]) -> List[Set[GroundAtom]]:
    """Compute a sequence of atoms by applying ground NSRTs from a plan in
    sequence."""
    atoms_sequence = [init_atoms]
    for ground_nsrt in skeleton:
        atoms_sequence.append(apply_operator(ground_nsrt, atoms_sequence[-1]))
    return atoms_sequence


def get_successors_from_ground_ops(
        atoms: Set[GroundAtom],
        ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
        unique: bool = True) -> Iterator[Set[GroundAtom]]:
    """Get all next atoms from ground operators.

    If unique is true, only yield each unique successor once.
    """
    seen_successors = set()
    for ground_op in get_applicable_operators(ground_ops, atoms):
        next_atoms = apply_operator(ground_op, atoms)
        if unique:
            frozen_next_atoms = frozenset(next_atoms)
            if frozen_next_atoms in seen_successors:
                continue
            seen_successors.add(frozen_next_atoms)
        yield next_atoms


def ops_and_specs_to_dummy_nsrts(
        strips_ops: Sequence[STRIPSOperator],
        option_specs: Sequence[OptionSpec]) -> Set[NSRT]:
    """Create NSRTs from strips operators and option specs with dummy
    samplers."""
    assert len(strips_ops) == len(option_specs)
    nsrts = set()
    for op, (param_option, option_vars) in zip(strips_ops, option_specs):
        nsrt = op.make_nsrt(
            param_option,
            option_vars,  # dummy sampler
            lambda s, g, rng, o: np.zeros(1, dtype=np.float32))
        nsrts.add(nsrt)
    return nsrts


# Note: create separate `heuristics.py` module if we need to add new
#  heuristics in the future.


def create_task_planning_heuristic(
    heuristic_name: str,
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
) -> _TaskPlanningHeuristic:
    """Create a task planning heuristic that consumes ground atoms and
    estimates the cost-to-go."""
    if heuristic_name in _PYPERPLAN_HEURISTICS:
        return _create_pyperplan_heuristic(heuristic_name, init_atoms, goal,
                                           ground_ops, predicates, objects)
    if heuristic_name == GoalCountHeuristic.HEURISTIC_NAME:
        return GoalCountHeuristic(heuristic_name, init_atoms, goal, ground_ops)
    raise ValueError(f"Unrecognized heuristic name: {heuristic_name}.")


@dataclass(frozen=True)
class _TaskPlanningHeuristic:
    """A task planning heuristic."""
    name: str
    init_atoms: Collection[GroundAtom]
    goal: Set[GroundAtom]
    ground_ops: Collection[Union[_GroundNSRT, _GroundSTRIPSOperator]]

    def __call__(self, atoms: Collection[GroundAtom]) -> float:
        raise NotImplementedError("Override me!")


class GoalCountHeuristic(_TaskPlanningHeuristic):
    """The number of goal atoms that are not in the current state."""
    HEURISTIC_NAME: ClassVar[str] = "goal_count"

    def __call__(self, atoms: Collection[GroundAtom]) -> float:
        return len(self.goal.difference(atoms))


############################### Pyperplan Glue ###############################


def _create_pyperplan_heuristic(
    heuristic_name: str,
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
) -> _PyperplanHeuristicWrapper:
    """Create a pyperplan heuristic that inherits from
    _TaskPlanningHeuristic."""
    assert heuristic_name in _PYPERPLAN_HEURISTICS
    static_atoms = get_static_atoms(ground_ops, init_atoms)
    pyperplan_heuristic_cls = _PYPERPLAN_HEURISTICS[heuristic_name]
    pyperplan_task = _create_pyperplan_task(init_atoms, goal, ground_ops,
                                            predicates, objects, static_atoms)
    pyperplan_heuristic = pyperplan_heuristic_cls(pyperplan_task)
    pyperplan_goal = _atoms_to_pyperplan_facts(goal - static_atoms)
    return _PyperplanHeuristicWrapper(heuristic_name, init_atoms, goal,
                                      ground_ops, static_atoms,
                                      pyperplan_heuristic, pyperplan_goal)


_PyperplanFacts = FrozenSet[str]


@dataclass(frozen=True)
class _PyperplanNode:
    """Container glue for pyperplan heuristics."""
    state: _PyperplanFacts
    goal: _PyperplanFacts


@dataclass(frozen=True)
class _PyperplanOperator:
    """Container glue for pyperplan heuristics."""
    name: str
    preconditions: _PyperplanFacts
    add_effects: _PyperplanFacts
    del_effects: _PyperplanFacts


@dataclass(frozen=True)
class _PyperplanTask:
    """Container glue for pyperplan heuristics."""
    facts: _PyperplanFacts
    initial_state: _PyperplanFacts
    goals: _PyperplanFacts
    operators: Collection[_PyperplanOperator]


@dataclass(frozen=True)
class _PyperplanHeuristicWrapper(_TaskPlanningHeuristic):
    """A light wrapper around pyperplan's heuristics."""
    _static_atoms: Set[GroundAtom]
    _pyperplan_heuristic: _PyperplanBaseHeuristic
    _pyperplan_goal: _PyperplanFacts

    def __call__(self, atoms: Collection[GroundAtom]) -> float:
        # Note: filtering out static atoms.
        pyperplan_facts = _atoms_to_pyperplan_facts(set(atoms) \
                                                    - self._static_atoms)
        return self._evaluate(pyperplan_facts, self._pyperplan_goal,
                              self._pyperplan_heuristic)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _evaluate(pyperplan_facts: _PyperplanFacts,
                  pyperplan_goal: _PyperplanFacts,
                  pyperplan_heuristic: _PyperplanBaseHeuristic) -> float:
        pyperplan_node = _PyperplanNode(pyperplan_facts, pyperplan_goal)
        logging.disable(logging.DEBUG)
        result = pyperplan_heuristic(pyperplan_node)
        logging.disable(logging.NOTSET)
        return result


def _create_pyperplan_task(
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
    static_atoms: Set[GroundAtom],
) -> _PyperplanTask:
    """Helper glue for pyperplan heuristics."""
    all_atoms = set()
    for predicate in predicates:
        all_atoms.update(
            get_all_ground_atoms_for_predicate(predicate, frozenset(objects)))
    # Note: removing static atoms.
    pyperplan_facts = _atoms_to_pyperplan_facts(all_atoms - static_atoms)
    pyperplan_state = _atoms_to_pyperplan_facts(init_atoms - static_atoms)
    pyperplan_goal = _atoms_to_pyperplan_facts(goal - static_atoms)
    pyperplan_operators = set()
    for op in ground_ops:
        # Note: the pyperplan operator must include the objects, because hFF
        # uses the operator name in constructing the relaxed plan, and the
        # relaxed plan is a set. If we instead just used op.name, there would
        # be a very nasty bug where two ground operators in the relaxed plan
        # that have different objects are counted as just one.
        name = op.name + "-".join(o.name for o in op.objects)
        pyperplan_operator = _PyperplanOperator(
            name,
            # Note: removing static atoms from preconditions.
            _atoms_to_pyperplan_facts(op.preconditions - static_atoms),
            _atoms_to_pyperplan_facts(op.add_effects),
            _atoms_to_pyperplan_facts(op.delete_effects))
        pyperplan_operators.add(pyperplan_operator)
    return _PyperplanTask(pyperplan_facts, pyperplan_state, pyperplan_goal,
                          pyperplan_operators)


@functools.lru_cache(maxsize=None)
def _atom_to_pyperplan_fact(atom: GroundAtom) -> str:
    """Convert atom to tuple for interface with pyperplan."""
    arg_str = " ".join(o.name for o in atom.objects)
    return f"({atom.predicate.name} {arg_str})"


def _atoms_to_pyperplan_facts(
        atoms: Collection[GroundAtom]) -> _PyperplanFacts:
    """Light wrapper around _atom_to_pyperplan_fact() that operates on a
    collection of atoms."""
    return frozenset({_atom_to_pyperplan_fact(atom) for atom in atoms})


############################## End Pyperplan Glue ##############################


def create_pddl_types_str(types: Collection[Type]) -> str:
    """Create a PDDL-style types string that handles hierarchy correctly."""
    # Case 1: no type hierarchy.
    if all(t.parent is None for t in types):
        types_str = " ".join(t.name for t in sorted(types))
    # Case 2: type hierarchy.
    else:
        parent_to_children_types: Dict[Type,
                                       List[Type]] = {t: []
                                                      for t in types}
        for t in sorted(types):
            if t.parent:
                parent_to_children_types[t.parent].append(t)
        types_str = ""
        for parent_type in sorted(parent_to_children_types):
            child_types = parent_to_children_types[parent_type]
            if not child_types:
                # Special case: type has no children and also does not appear
                # as a child of another type.
                is_child_type = any(
                    parent_type in children
                    for children in parent_to_children_types.values())
                if not is_child_type:
                    types_str += f"\n    {parent_type.name}"
                # Otherwise, the type will appear as a child elsewhere.
            else:
                child_type_str = " ".join(t.name for t in child_types)
                types_str += f"\n    {child_type_str} - {parent_type.name}"
    return types_str


def create_pddl_domain(operators: Collection[NSRTOrSTRIPSOperator],
                       predicates: Collection[Predicate],
                       types: Collection[Type], domain_name: str) -> str:
    """Create a PDDL domain str from STRIPSOperators or NSRTs."""
    # Sort everything to ensure determinism.
    preds_lst = sorted(predicates)
    types_str = create_pddl_types_str(types)
    ops_lst = sorted(operators)
    preds_str = "\n    ".join(pred.pddl_str() for pred in preds_lst)
    ops_strs = "\n\n  ".join(op.pddl_str() for op in ops_lst)
    return f"""(define (domain {domain_name})
  (:requirements :typing)
  (:types {types_str})

  (:predicates\n    {preds_str}
  )

  {ops_strs}
)"""


def create_pddl_problem(objects: Collection[Object],
                        init_atoms: Collection[GroundAtom],
                        goal: Set[GroundAtom], domain_name: str,
                        problem_name: str) -> str:
    """Create a PDDL problem str."""
    # Sort everything to ensure determinism.
    objects_lst = sorted(objects)
    init_atoms_lst = sorted(init_atoms)
    goal_lst = sorted(goal)
    objects_str = "\n    ".join(f"{o.name} - {o.type.name}"
                                for o in objects_lst)
    init_str = "\n    ".join(atom.pddl_str() for atom in init_atoms_lst)
    goal_str = "\n    ".join(atom.pddl_str() for atom in goal_lst)
    return f"""(define (problem {problem_name}) (:domain {domain_name})
  (:objects\n    {objects_str}
  )
  (:init\n    {init_str}
  )
  (:goal (and {goal_str}))
)
"""


@functools.lru_cache(maxsize=None)
def get_failure_predicate(option: ParameterizedOption,
                          idxs: Tuple[int]) -> Predicate:
    """Create a Failure predicate for a parameterized option."""
    idx_str = ",".join(map(str, idxs))
    arg_types = [option.types[i] for i in idxs]
    return Predicate(f"{option.name}Failed_arg{idx_str}",
                     arg_types,
                     _classifier=lambda s, o: False)


def _get_idxs_to_failure_predicate(
        option: ParameterizedOption,
        max_arity: int = 1) -> Dict[Tuple[int, ...], Predicate]:
    """Helper for get_all_failure_predicates() and get_failure_atoms()."""
    idxs_to_failure_predicate: Dict[Tuple[int, ...], Predicate] = {}
    num_types = len(option.types)
    max_num_idxs = min(max_arity, num_types)
    all_idxs = list(range(num_types))
    for arity in range(1, max_num_idxs + 1):
        for idxs in itertools.combinations(all_idxs, arity):
            pred = get_failure_predicate(option, idxs)
            idxs_to_failure_predicate[idxs] = pred
    return idxs_to_failure_predicate


def get_all_failure_predicates(options: Set[ParameterizedOption],
                               max_arity: int = 1) -> Set[Predicate]:
    """Get all possible failure predicates."""
    failure_preds: Set[Predicate] = set()
    for param_opt in options:
        preds = _get_idxs_to_failure_predicate(param_opt, max_arity=max_arity)
        failure_preds.update(preds.values())
    return failure_preds


def get_failure_atoms(failed_options: Collection[_Option],
                      max_arity: int = 1) -> Set[GroundAtom]:
    """Get ground failure atoms for the collection of failure options."""
    failure_atoms: Set[GroundAtom] = set()
    failed_option_specs = {(o.parent, tuple(o.objects))
                           for o in failed_options}
    for (param_opt, objs) in failed_option_specs:
        preds = _get_idxs_to_failure_predicate(param_opt, max_arity=max_arity)
        for idxs, pred in preds.items():
            obj_for_idxs = [objs[i] for i in idxs]
            failure_atom = GroundAtom(pred, obj_for_idxs)
            failure_atoms.add(failure_atom)
    return failure_atoms


@dataclass
class VideoMonitor(LoggingMonitor):
    """A monitor that renders each state and action encountered.

    The render_fn is generally env.render. Note that the state is unused
    because the environment should use its current internal state to
    render.
    """
    _render_fn: Callable[[Optional[Action], Optional[str]], Video]
    _video: Video = field(init=False, default_factory=list)

    def reset(self, train_or_test: str, task_idx: int) -> None:
        self._video = []

    def observe(self, obs: Observation, action: Optional[Action]) -> None:
        del obs  # unused
        self._video.extend(self._render_fn(action, None))

    def get_video(self) -> Video:
        """Return the video."""
        return self._video


@dataclass
class SimulateVideoMonitor(LoggingMonitor):
    """A monitor that calls render_state on each state and action seen.

    This monitor is meant for use with run_policy_with_simulator, as
    opposed to VideoMonitor, which is meant for use with run_policy.
    """
    _task: Task
    _render_state_fn: Callable[[State, Task, Optional[Action]], Video]
    _video: Video = field(init=False, default_factory=list)

    def reset(self, train_or_test: str, task_idx: int) -> None:
        self._video = []

    def observe(self, obs: Observation, action: Optional[Action]) -> None:
        assert isinstance(obs, State)
        self._video.extend(self._render_state_fn(obs, self._task, action))

    def get_video(self) -> Video:
        """Return the video."""
        return self._video


def create_video_from_partial_refinements(
    partial_refinements: Sequence[Tuple[Sequence[_GroundNSRT],
                                        Sequence[_Option]]],
    env: BaseEnv,
    train_or_test: str,
    task_idx: int,
    max_num_steps: int,
) -> Video:
    """Create a video from a list of skeletons and partial refinements.

    Note that the environment internal state is updated.
    """
    # Right now, the video is created by finding the longest partial
    # refinement. One could also implement an "all_skeletons" mode
    # that would create one video per skeleton.
    if CFG.failure_video_mode == "longest_only":
        # Visualize only the overall longest failed plan.
        _, plan = max(partial_refinements, key=lambda x: len(x[1]))
        policy = option_plan_to_policy(plan)
        video: Video = []
        state = env.reset(train_or_test, task_idx)
        for _ in range(max_num_steps):
            try:
                act = policy(state)
            except OptionExecutionFailure:
                video.extend(env.render())
                break
            video.extend(env.render(act))
            try:
                state = env.step(act)
            except EnvironmentFailure:
                break
        return video
    raise NotImplementedError("Unrecognized failure video mode: "
                              f"{CFG.failure_video_mode}.")


def fig2data(fig: matplotlib.figure.Figure, dpi: int) -> Image:
    """Convert matplotlib figure into Image."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).copy()
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4, ))
    data[..., [0, 1, 2, 3]] = data[..., [1, 2, 3, 0]]
    return data


def save_video(outfile: str, video: Video) -> None:
    """Save the video to video_dir/outfile."""
    outdir = CFG.video_dir
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile)
    imageio.mimwrite(outpath, video, fps=CFG.video_fps)  # type: ignore
    logging.info(f"Wrote out to {outpath}")


def save_images(outfile_prefix: str, video: Video) -> None:
    """Save the video as individual images to image_dir."""
    outdir = CFG.image_dir
    os.makedirs(outdir, exist_ok=True)
    width = len(str(len(video)))
    for i, image in enumerate(video):
        image_number = str(i).zfill(width)
        outfile = outfile_prefix + f"_image_{image_number}.png"
        outpath = os.path.join(outdir, outfile)
        imageio.imwrite(outpath, image)
        logging.info(f"Wrote out to {outpath}")


def get_env_asset_path(asset_name: str, assert_exists: bool = True) -> str:
    """Return the absolute path to env asset."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, "envs", "assets")
    path = os.path.join(asset_dir_path, asset_name)
    if assert_exists:
        assert os.path.exists(path), f"Env asset not found: {asset_name}."
    return path


def get_third_party_path() -> str:
    """Return the absolute path to the third party directory."""
    third_party_dir_path = os.path.join(get_path_to_predicators_root(),
                                        "predicators/third_party")
    return third_party_dir_path


def get_path_to_predicators_root() -> str:
    """Return the absolute path to the predicators root directory.

    Specifically, this returns something that looks like:
    '<installation-path>/predicators'. Note there is no '/' at the end.
    """
    module_path = Path(__file__)
    predicators_dir = module_path.parent.parent
    return str(predicators_dir)


def import_submodules(path: List[str], name: str) -> None:
    """Load all submodules on the given path.

    Useful for finding subclasses of an abstract base class
    automatically.
    """
    if not TYPE_CHECKING:
        for _, module_name, _ in pkgutil.walk_packages(path):
            if "__init__" not in module_name:
                # Important! We use an absolute import here to avoid issues
                # with isinstance checking when using relative imports.
                importlib.import_module(f"{name}.{module_name}")


def update_config(args: Dict[str, Any]) -> None:
    """Args is a dictionary of new arguments to add to the config CFG."""
    parser = create_arg_parser()
    update_config_with_parser(parser, args)


def update_config_with_parser(parser: ArgumentParser, args: Dict[str,
                                                                 Any]) -> None:
    """Helper function for update_config() that accepts a parser argument."""
    arg_specific_settings = GlobalSettings.get_arg_specific_settings(args)
    # Only override attributes, don't create new ones.
    allowed_args = set(CFG.__dict__) | set(arg_specific_settings)
    # Unfortunately, can't figure out any other way to do this.
    for parser_action in parser._actions:  # pylint: disable=protected-access
        allowed_args.add(parser_action.dest)
    for k in args:
        if k not in allowed_args:
            raise ValueError(f"Unrecognized arg: {k}")
    for k in ("env", "approach", "seed", "experiment_id"):
        if k not in args and hasattr(CFG, k):
            # For env, approach, seed, and experiment_id, if we don't
            # pass in a value and this key is already in the
            # configuration dict, add the current value to args.
            args[k] = getattr(CFG, k)
    for d in [arg_specific_settings, args]:
        for k, v in d.items():
            setattr(CFG, k, v)


def reset_config(args: Optional[Dict[str, Any]] = None,
                 default_seed: int = 123,
                 default_render_state_dpi: int = 10) -> None:
    """Reset to the default CFG, overriding with anything in args.

    This utility is meant for use in testing only.
    """
    parser = create_arg_parser()
    reset_config_with_parser(parser, args, default_seed,
                             default_render_state_dpi)


def reset_config_with_parser(parser: ArgumentParser,
                             args: Optional[Dict[str, Any]] = None,
                             default_seed: int = 123,
                             default_render_state_dpi: int = 10) -> None:
    """Helper function for reset_config that accepts a parser argument."""
    default_args = parser.parse_args([
        "--env",
        "default env placeholder",
        "--seed",
        str(default_seed),
        "--approach",
        "default approach placeholder",
    ])
    arg_dict = {
        k: v
        for k, v in GlobalSettings.__dict__.items() if not k.startswith("_")
    }
    arg_dict.update(vars(default_args))
    if args is not None:
        arg_dict.update(args)
    if args is None or "render_state_dpi" not in args:
        # By default, use a small value for the rendering DPI, to avoid
        # expensive rendering during testing.
        arg_dict["render_state_dpi"] = default_render_state_dpi
    update_config_with_parser(parser, arg_dict)


def get_config_path_str(experiment_id: Optional[str] = None) -> str:
    """Get a filename prefix for configuration based on the current CFG.

    If experiment_id is supplied, it is used in place of
    CFG.experiment_id.
    """
    if experiment_id is None:
        experiment_id = CFG.experiment_id
    return (f"{CFG.env}__{CFG.approach}__{CFG.seed}__{CFG.excluded_predicates}"
            f"__{CFG.included_options}__{experiment_id}")


def get_approach_save_path_str() -> str:
    """Get a path for saving approaches."""
    os.makedirs(CFG.approach_dir, exist_ok=True)
    return f"{CFG.approach_dir}/{get_config_path_str()}.saved"


def get_approach_load_path_str() -> str:
    """Get a path for loading approaches."""
    if not CFG.load_experiment_id:
        experiment_id = CFG.experiment_id
    else:
        experiment_id = CFG.load_experiment_id
    return f"{CFG.approach_dir}/{get_config_path_str(experiment_id)}.saved"


def parse_args(env_required: bool = True,
               approach_required: bool = True,
               seed_required: bool = True) -> Dict[str, Any]:
    """Parses command line arguments."""
    parser = create_arg_parser(env_required=env_required,
                               approach_required=approach_required,
                               seed_required=seed_required)
    return parse_args_with_parser(parser)


def parse_args_with_parser(parser: ArgumentParser) -> Dict[str, Any]:
    """Helper function for parse_args that accepts a parser argument."""
    args, overrides = parser.parse_known_args()
    arg_dict = vars(args)
    if len(overrides) == 0:
        return arg_dict
    # Update initial settings to make sure we're overriding
    # existing flags only
    update_config_with_parser(parser, arg_dict)
    # Override global settings
    assert len(overrides) >= 2
    assert len(overrides) % 2 == 0
    for flag, value in zip(overrides[:-1:2], overrides[1::2]):
        assert flag.startswith("--")
        setting_name = flag[2:]
        if setting_name not in CFG.__dict__:
            raise ValueError(f"Unrecognized flag: {setting_name}")
        arg_dict[setting_name] = string_to_python_object(value)
    return arg_dict


def string_to_python_object(value: str) -> Any:
    """Return the Python object corresponding to the given string value."""
    if value in ("None", "none"):
        return None
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    if value.isdigit() or value.startswith("lambda"):
        return eval(value)
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("["):
        assert value.endswith("]")
        inner_strs = value[1:-1].split(",")
        return [string_to_python_object(s) for s in inner_strs]
    if value.startswith("("):
        assert value.endswith(")")
        inner_strs = value[1:-1].split(",")
        return tuple(string_to_python_object(s) for s in inner_strs)
    return value


def flush_cache() -> None:
    """Clear all lru caches."""
    gc.collect()
    wrappers = [
        a for a in gc.get_objects()
        if isinstance(a, functools._lru_cache_wrapper)  # pylint: disable=protected-access
    ]

    for wrapper in wrappers:
        wrapper.cache_clear()


def parse_config_excluded_predicates(
        env: BaseEnv) -> Tuple[Set[Predicate], Set[Predicate]]:
    """Parse the CFG.excluded_predicates string, given an environment.

    Return a tuple of (included predicate set, excluded predicate set).
    """
    if CFG.excluded_predicates:
        if CFG.excluded_predicates == "all":
            excluded_names = {
                pred.name
                for pred in env.predicates if pred not in env.goal_predicates
            }
            logging.info(f"All non-goal predicates excluded: {excluded_names}")
            included = env.goal_predicates
        else:
            excluded_names = set(CFG.excluded_predicates.split(","))
            assert excluded_names.issubset(
                {pred.name for pred in env.predicates}), \
                "Unrecognized predicate in excluded_predicates!"
            included = {
                pred
                for pred in env.predicates if pred.name not in excluded_names
            }
            if CFG.offline_data_method != "demo+ground_atoms":
                if CFG.allow_exclude_goal_predicates:
                    if not env.goal_predicates.issubset(included):
                        logging.info("Note: excluding goal predicates!")
                else:
                    assert env.goal_predicates.issubset(included), \
                    "Can't exclude a goal predicate!"
    else:
        excluded_names = set()
        included = env.predicates
    excluded = {pred for pred in env.predicates if pred.name in excluded_names}
    return included, excluded


def replace_goals_with_agent_specific_goals(
        included_predicates: Set[Predicate],
        excluded_predicates: Set[Predicate], env: BaseEnv) -> Set[Predicate]:
    """Replace original goal predicates with agent-specific goal predicates if
    the environment defines them."""
    preds = included_predicates - env.goal_predicates \
        | env.agent_goal_predicates - excluded_predicates
    return preds


def null_sampler(state: State, goal: Set[GroundAtom], rng: np.random.Generator,
                 objs: Sequence[Object]) -> Array:
    """A sampler for an NSRT with no continuous parameters."""
    del state, goal, rng, objs  # unused
    return np.array([], dtype=np.float32)  # no continuous parameters


@functools.lru_cache(maxsize=None)
def get_git_commit_hash() -> str:
    """Return the hash of the current git commit."""
    out = subprocess.check_output(["git", "rev-parse", "HEAD"])
    return out.decode("ascii").strip()


def get_all_subclasses(cls: Any) -> Set[Any]:
    """Get all subclasses of the given class."""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)])


class _DummyFile(io.StringIO):
    """Dummy file object used by nostdout()."""

    def write(self, _: Any) -> int:
        """Mock write() method."""
        return 0

    def flush(self) -> None:
        """Mock flush() method."""


@contextlib.contextmanager
def nostdout() -> Generator[None, None, None]:
    """Suppress output for a block of code.

    To use, wrap code in the statement `with utils.nostdout():`. Note
    that calls to the logging library, which this codebase uses
    primarily, are unaffected. So, this utility is mostly helpful when
    calling third-party code.
    """
    save_stdout = sys.stdout
    sys.stdout = _DummyFile()
    yield
    sys.stdout = save_stdout


def query_ldl(
    ldl: LiftedDecisionList,
    atoms: Set[GroundAtom],
    objects: Set[Object],
    goal: Set[GroundAtom],
    static_predicates: Optional[Set[Predicate]] = None,
    init_atoms: Optional[Collection[GroundAtom]] = None
) -> Optional[_GroundNSRT]:
    """Queries a lifted decision list representing a goal-conditioned policy.

    Given an abstract state and goal, the rules are grounded in order. The
    first applicable ground rule is used to return a ground NSRT.

    If static_predicates is provided, it is used to avoid grounding rules with
    nonsense preconditions like IsBall(robot).

    If no rule is applicable, returns None.
    """
    for rule in ldl.rules:
        for ground_rule in all_ground_ldl_rules(
                rule,
                objects,
                static_predicates=static_predicates,
                init_atoms=init_atoms):
            if ground_rule.pos_state_preconditions.issubset(atoms) and \
                    not ground_rule.neg_state_preconditions & atoms and \
                    ground_rule.goal_preconditions.issubset(goal):
                return ground_rule.ground_nsrt
    return None


def generate_random_string(length: int, alphabet: Sequence[str],
                           rng: np.random.Generator) -> str:
    """Generates a random string of the given length using the provided set of
    characters (alphabet)."""
    assert all(len(c) == 1 for c in alphabet)
    return "".join(rng.choice(alphabet, size=length))


def find_balanced_expression(s: str, index: int) -> str:
    """Find balanced expression in string starting from given index."""
    assert s[index] == "("
    start_index = index
    balance = 1
    while balance != 0:
        index += 1
        symbol = s[index]
        if symbol == "(":
            balance += 1
        elif symbol == ")":
            balance -= 1
    return s[start_index:index + 1]


def find_all_balanced_expressions(s: str) -> List[str]:
    """Return a list of all balanced expressions in a string, starting from the
    beginning."""
    assert s[0] == "("
    assert s[-1] == ")"
    exprs = []
    index = 0
    start_index = index
    balance = 1
    while index < len(s) - 1:
        index += 1
        if balance == 0:
            exprs.append(s[start_index:index])
            # Jump to next "(".
            while True:
                if s[index] == "(":
                    break
                index += 1
            start_index = index
            balance = 1
            continue
        symbol = s[index]
        if symbol == "(":
            balance += 1
        elif symbol == ")":
            balance -= 1
    assert balance == 0
    exprs.append(s[start_index:index + 1])
    return exprs


def range_intersection(lb1: float, ub1: float, lb2: float, ub2: float) -> bool:
    """Given upper and lower bounds for two ranges, returns True iff the ranges
    intersect."""
    return (lb1 <= lb2 <= ub1) or (lb2 <= lb1 <= ub2)


def compute_abs_range_given_two_ranges(lb1: float, ub1: float, lb2: float,
                                       ub2: float) -> Tuple[float, float]:
    """Given upper and lower bounds of two feature ranges, returns the upper.

    and lower bound of |f1 - f2|.
    """
    # Now, we must compute the upper and lower bounds of
    # the expression |t1.f1 - t2.f2|. If the intervals
    # [lb1, ub1] and [lb2, ub2] overlap, then the lower
    # bound of the expression is just 0. Otherwise, if
    # lb2 > ub1, the lower bound is |ub1 - lb2|, and if
    # ub2 < lb1, the lower bound is |lb1 - ub2|.
    if range_intersection(lb1, ub1, lb2, ub2):
        lb = 0.0
    else:
        lb = min(abs(lb2 - ub1), abs(lb1 - ub2))
    # The upper bound for the expression can be
    # computed in a similar fashion.
    ub = max(abs(ub2 - lb1), abs(ub1 - lb2))
    return (lb, ub)


def roundrobin(iterables: Sequence[Iterator]) -> Iterator:
    """roundrobin(['ABC...', 'D...', 'EF...']) --> A D E B F C..."""
    # Recipe credited to George Sakkis, code adapted slightly from
    # from https://docs.python.org/3/library/itertools.html
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        for nxt in nexts:
            yield nxt()


def get_task_seed(train_or_test: str, task_idx: int) -> int:
    """Parses task seed from CFG.test_env_seed_offset."""
    assert task_idx < CFG.test_env_seed_offset
    # SeedSequence generates a sequence of random values given an integer
    # "entropy". We use CFG.seed to define the "entropy" and then get the
    # n^th generated random value and use that to seed the gym environment.
    # This is all to avoid unintentional dependence between experiments
    # that are conducted with consecutive random seeds. For example, if
    # we used CFG.seed + task_idx to seed the gym environment, there would
    # be overlap between experiments when CFG.seed = 1, CFG.seed = 2, etc.
    seed_entropy = CFG.seed
    if train_or_test == "test":
        seed_entropy += CFG.test_env_seed_offset
    seed_sequence = np.random.SeedSequence(seed_entropy)
    # Need to cast to int because generate_state() returns a numpy int.
    task_seed = int(seed_sequence.generate_state(task_idx + 1)[-1])
    return task_seed


def _beta_bernoulli_posterior_alpha_beta(
        success_history: List[bool],
        alpha: float = 1.0,
        beta: float = 1.0) -> Tuple[float, float]:
    """See https://gregorygundersen.com/blog/2020/08/19/bernoulli-beta/"""
    n = len(success_history)
    s = sum(success_history)
    alpha_n = alpha + s
    beta_n = n - s + beta
    return (alpha_n, beta_n)


def beta_bernoulli_posterior(success_history: List[bool],
                             alpha: float = 1.0,
                             beta: float = 1.0) -> BetaRV:
    """Returns the RV."""
    alpha_n, beta_n = _beta_bernoulli_posterior_alpha_beta(
        success_history, alpha, beta)
    return BetaRV(alpha_n, beta_n)


def beta_bernoulli_posterior_mean(success_history: List[bool],
                                  alpha: float = 1.0,
                                  beta: float = 1.0) -> float:
    """Faster computation to avoid instantiating BetaRV when not needed."""
    alpha_n, beta_n = _beta_bernoulli_posterior_alpha_beta(
        success_history, alpha, beta)
    return alpha_n / (alpha_n + beta_n)


def beta_from_mean_and_variance(mean: float,
                                variance: float,
                                variance_lower_pad: float = 1e-6,
                                variance_upper_pad: float = 1e-3) -> BetaRV:
    """Recover a beta distribution given a mean and a variance.

    See https://stats.stackexchange.com/questions/12232/ for derivation.
    """
    # Clip variance.
    variance = max(min(variance,
                       mean * (1 - mean) - variance_upper_pad),
                   variance_lower_pad)
    alpha = ((1 - mean) / variance - 1 / mean) * (mean**2)
    beta = alpha * (1 / mean - 1)
    assert alpha > 0
    assert beta > 0
    rv = BetaRV(alpha, beta)
    assert abs(rv.mean() - mean) < 1e-6
    return rv


def rotate_point_in_image(r: float, c: float, rot_degrees: float, height: int,
                          width: int) -> Tuple[int, int]:
    """If an image has been rotated using ndimage.rotate, this computes the
    location of a pixel (r, c) in that image following the same rotation.

    The rotation is expected in degrees, following ndimage.rotate.
    """
    rotation_radians = np.radians(rot_degrees)
    transform_matrix = np.array(
        [[np.cos(rotation_radians), -np.sin(rotation_radians)],
         [np.sin(rotation_radians),
          np.cos(rotation_radians)]])
    # Subtract the center of the image from the pixel location to
    # translate the rotation to the origin.
    center = np.array([(height - 1) / 2., (width - 1) / 2])
    centered_pt = np.subtract([r, c], center)
    # Apply the rotation.
    rotated_pt_centered = np.matmul(transform_matrix, centered_pt)
    # Add the center of the image back to the pixel location to
    # translate the rotation back from the origin.
    rotated_pt = rotated_pt_centered + center
    return rotated_pt[0], rotated_pt[1]


def get_se3_pose_from_state(
        state: State, obj: Object) -> math_helpers.SE3Pose:  # pragma: no cover
    """Helper for spot environments."""
    return math_helpers.SE3Pose(
        state.get(obj, "x"), state.get(obj, "y"), state.get(obj, "z"),
        math_helpers.Quat(state.get(obj, "qw"), state.get(obj, "qx"),
                          state.get(obj, "qy"), state.get(obj, "qz")))


def create_spot_env_action(
        action_extra_info: SpotActionExtraInfo) -> Action:  # pragma: no cover
    """Helper for spot environments."""
    return SpotAction(np.array([], dtype=np.float32),
                      extra_info=action_extra_info)


def _obs_to_state_pass_through(obs: Observation) -> State:
    """Helper for run_ground_nsrt_with_assertions."""
    assert isinstance(obs, State)
    return obs


def run_ground_nsrt_with_assertions(ground_nsrt: _GroundNSRT,
                                    state: State,
                                    env: BaseEnv,
                                    rng: np.random.Generator,
                                    override_params: Optional[Array] = None,
                                    obs_to_state: Callable[
                                        [Observation],
                                        State] = _obs_to_state_pass_through,
                                    assert_add_effects: bool = True,
                                    assert_delete_effects: bool = True,
                                    max_steps: int = 400) -> State:
    """Utility for tests.

    NOTE: assumes that the internal state of env corresponds to state.
    """
    ground_nsrt_str = f"{ground_nsrt.name}{ground_nsrt.objects}"
    for atom in ground_nsrt.preconditions:
        assert atom.holds(state), \
            f"Precondition for {ground_nsrt_str} failed: {atom}"
    option = ground_nsrt.sample_option(state, set(), rng)
    if override_params is not None:
        option = option.parent.ground(option.objects,
                                      override_params)  # pragma: no cover
    assert option.initiable(state)
    for _ in range(max_steps):
        act = option.policy(state)
        obs = env.step(act)
        state = obs_to_state(obs)
        if option.terminal(state):
            break
    if assert_add_effects:
        for atom in ground_nsrt.add_effects:
            assert atom.holds(state), \
                f"Add effect for {ground_nsrt_str} failed: {atom}"
    if assert_delete_effects:
        for atom in ground_nsrt.delete_effects:
            assert not atom.holds(state), \
                f"Delete effect for {ground_nsrt_str} failed: {atom}"
    return state


def get_scaled_default_font(
        draw: ImageDraw.ImageDraw,
        size: int) -> ImageFont.FreeTypeFont:  # pragma: no cover
    """Method that modifies the size of some provided PIL ImageDraw font.

    Useful for scaling up font sizes when using PIL to insert text
    directly into images.
    """
    # Determine the scaling factor
    base_font = ImageFont.load_default()
    width, height = draw.textbbox((0, 0), "A", font=base_font)[:2]
    scale_factor = size / max(width, height)
    # Scale the font using the factor
    return base_font.font_variant(size=int(scale_factor *  # type: ignore
                                           base_font.size))  # type: ignore


def add_text_to_draw_img(
        draw: ImageDraw.ImageDraw, position: Tuple[int, int], text: str,
        font: ImageFont.FreeTypeFont
) -> ImageDraw.ImageDraw:  # pragma: no cover
    """Method that adds some text with a particular font at a particular pixel
    position in an input PIL.ImageDraw.ImageDraw image.

    Returns the modified ImageDraw.ImageDraw with the added text.
    """
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    background_position = (position[0] - 5, position[1] - 5
                           )  # Slightly larger than text
    background_size = (text_width + 10, text_height + 10)
    # Draw the background rectangle
    draw.rectangle(
        (background_position, (background_position[0] + background_size[0],
                               background_position[1] + background_size[1])),
        fill="black")
    # Add the text to the image
    draw.text(position, text, fill="red", font=font)
    return draw
