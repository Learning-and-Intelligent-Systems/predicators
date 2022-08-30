"""General utility methods."""

from __future__ import annotations

import abc
import contextlib
import functools
import gc
import heapq as hq
import io
import itertools
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Collection, Dict, \
    FrozenSet, Generator, Generic, Hashable, Iterator, List, Optional, \
    Sequence, Set, Tuple
from typing import Type as TypingType
from typing import TypeVar, Union, cast

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as mp
from gym.spaces import Box
from matplotlib import patches
from pyperplan.heuristics.heuristic_base import \
    Heuristic as _PyperplanBaseHeuristic
from pyperplan.planner import HEURISTICS as _PYPERPLAN_HEURISTICS

from predicators.args import create_arg_parser
from predicators.pybullet_helpers.joint import JointPositions
from predicators.settings import CFG, GlobalSettings
from predicators.structs import NSRT, Action, Array, DummyOption, \
    EntToEntSub, GroundAtom, GroundAtomTrajectory, \
    GroundNSRTOrSTRIPSOperator, Image, LDLRule, LiftedAtom, \
    LiftedDecisionList, LiftedOrGroundAtom, LowLevelTrajectory, Metrics, \
    NSRTOrSTRIPSOperator, Object, ObjectOrVariable, OptionSpec, \
    ParameterizedOption, Predicate, Segment, State, STRIPSOperator, Task, \
    Type, Variable, VarToObjSub, Video, _GroundLDLRule, _GroundNSRT, \
    _GroundSTRIPSOperator, _Option, _TypedEntity
from predicators.third_party.fast_downward_translator.translate import \
    main as downward_translate

if TYPE_CHECKING:
    from predicators.envs import BaseEnv

matplotlib.use("Agg")


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


def segment_trajectory_to_state_sequence(
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

    def contains_point(self, x: float, y: float) -> bool:
        # https://stackoverflow.com/questions/328107
        a = (self.x1, self.y1)
        b = (self.x2, self.y2)
        c = (x, y)
        # Need to use an epsilon for numerical stability. But we are checking
        # if the distance from a to b is (approximately) equal to the distance
        # from a to c and the distance from c to b.
        eps = 1e-6

        def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
            return np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

        return -eps < _dist(a, c) + _dist(c, b) - _dist(a, b) < eps


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

    @functools.cached_property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get the four vertices for the rectangle."""
        scale_matrix = np.array([
            [self.width, 0],
            [0, self.height],
        ])
        rotate_matrix = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                                  [np.sin(self.theta),
                                   np.cos(self.theta)]])
        translate_vector = np.array([self.x, self.y])
        vertices = np.array([
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
        ])
        vertices = vertices @ scale_matrix.T
        vertices = vertices @ rotate_matrix.T
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
        rotate_matrix = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                                  [np.sin(self.theta),
                                   np.cos(self.theta)]])
        rx, ry = np.array([x, y]) @ rotate_matrix.T
        return self.x <= rx <= self.x + self.width and \
               self.y <= ry <= self.y + self.height

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
        patch = patches.Rectangle((self.x, self.y), self.width, self.height,
                                  angle, **kwargs)
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
            if "start_state" in memory:
                assert state.allclose(memory["start_state"])
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


class BehaviorState(State):
    """A BEHAVIOR state that stores the index of the temporary BEHAVIOR state
    folder in addition to the features that are exposed in the object-centric
    state."""

    def allclose(self, other: State) -> bool:
        # Ignores the simulator state.
        return State(self.data).allclose(State(other.data))


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


class Monitor(abc.ABC):
    """Observes states and actions during environment interaction."""

    @abc.abstractmethod
    def observe(self, state: State, action: Optional[Action]) -> None:
        """Record a state and the action that is about to be taken.

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
        exceptions_to_break_on: Optional[Set[TypingType[Exception]]] = None,
        monitor: Optional[Monitor] = None
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
    """
    state = env.reset(train_or_test, task_idx)
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
                start_time = time.time()
                act = policy(state)
                metrics["policy_call_time"] += time.time() - start_time
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
        monitor: Optional[Monitor] = None) -> LowLevelTrajectory:
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


def option_plan_to_policy(
        plan: Sequence[_Option]) -> Callable[[State], Action]:
    """Create a policy that executes a sequence of options in order."""
    queue = list(plan)  # don't modify plan, just in case
    cur_option = DummyOption

    def _policy(state: State) -> Action:
        nonlocal cur_option
        if cur_option.terminal(state):
            if not queue:
                raise OptionExecutionFailure("Option plan exhausted!")
            cur_option = queue.pop(0)
            assert cur_option.initiable(state), "Unsound option plan"
        return cur_option.policy(state)

    return _policy


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
            for _ in range(CFG.random_options_max_tries):
                param_opt = sorted_options[rng.choice(len(sorted_options))]
                objs = get_random_object_combination(list(state),
                                                     param_opt.types, rng)
                if objs is None:
                    continue
                params = param_opt.params_space.sample()
                opt = param_opt.ground(objs, params)
                if opt.initiable(state):
                    cur_option = opt
                    break
            else:
                return fallback_policy(state)
        act = cur_option.policy(state)
        return act

    return _policy


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
    """Get all combinations of objects satisfying the given types sequence."""
    return _get_entity_combinations(variables, types)


def get_all_ground_atoms_for_predicate(
        predicate: Predicate, objects: FrozenSet[Object]) -> Set[GroundAtom]:
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
        defaultdict(lambda : float("inf"))

    root_node: _HeuristicSearchNode[_S, _A] = _HeuristicSearchNode(
        initial_state, 0, 0)
    root_priority = get_priority(root_node)
    best_node = root_node
    best_node_priority = root_priority
    tiebreak = itertools.count()
    hq.heappush(queue, (root_priority, next(tiebreak), root_node))
    num_expansions = 0
    num_evals = 1
    start_time = time.time()

    while len(queue) > 0 and time.time() - start_time < timeout and \
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
            if time.time() - start_time >= timeout:
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
        parallelize: bool = False) -> Tuple[List[_S], List[_A], List[float]]:
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
    logging.info(f"\n\nStarting hill climbing at state {cur_node.state} "
                 f"with heuristic {last_heuristic}")
    while True:

        # Stops when heuristic reaches specified value.
        if early_termination_heuristic_thresh is not None \
            and last_heuristic <= early_termination_heuristic_thresh:
            break

        if check_goal(cur_node.state):
            logging.info("\nTerminating hill climbing, achieved goal")
            break
        best_heuristic = float("inf")
        best_child_node = None
        current_depth_nodes = [cur_node]
        all_best_heuristics = []
        for depth in range(0, enforced_depth + 1):
            logging.info(f"Searching for an improvement at depth {depth}")
            # This is a list to ensure determinism. Note that duplicates are
            # filtered out in the `child_state in visited` check.
            successors_at_depth = []
            for parent in current_depth_nodes:
                for action, child_state, cost in get_successors(parent.state):
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
                logging.info(f"Found an improvement at depth {depth}")
                break
            # Continue on to the next depth.
            current_depth_nodes = successors_at_depth
            logging.info(f"No improvement found at depth {depth}")
        if best_child_node is None:
            logging.info("\nTerminating hill climbing, no more successors")
            break
        if last_heuristic <= best_heuristic:
            logging.info(
                "\nTerminating hill climbing, could not improve score")
            break
        heuristics.extend(all_best_heuristics)
        cur_node = best_child_node
        last_heuristic = best_heuristic
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


class BiRRT(Generic[_S]):
    """Bidirectional rapidly-exploring random tree."""

    def __init__(self, sample_fn: Callable[[_S], _S],
                 extend_fn: Callable[[_S, _S], Iterator[_S]],
                 collision_fn: Callable[[_S], bool],
                 distance_fn: Callable[[_S, _S],
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

    def query(self, pt1: _S, pt2: _S) -> Optional[List[_S]]:
        """Query the BiRRT, to get a collision-free path from pt1 to pt2.

        If none is found, returns None.
        """
        if self._collision_fn(pt1) or self._collision_fn(pt2):
            return None
        direct_path = self._try_direct_path(pt1, pt2)
        if direct_path is not None:
            return direct_path
        for _ in range(self._num_attempts):
            path = self._rrt_connect(pt1, pt2)
            if path is not None:
                return self._smooth_path(path)
        return None

    def _try_direct_path(self, pt1: _S, pt2: _S) -> Optional[List[_S]]:
        path = [pt1]
        for newpt in self._extend_fn(pt1, pt2):
            if self._collision_fn(newpt):
                return None
            path.append(newpt)
        return path

    def _rrt_connect(self, pt1: _S, pt2: _S) -> Optional[List[_S]]:
        root1, root2 = _BiRRTNode(pt1), _BiRRTNode(pt2)
        nodes1, nodes2 = [root1], [root2]

        def _get_pt_dist_to_node(pt: _S, node: _BiRRTNode[_S]) -> float:
            return self._distance_fn(pt, node.data)

        for _ in range(self._num_iters):
            if len(nodes1) > len(nodes2):
                nodes1, nodes2 = nodes2, nodes1
            samp = self._sample_fn(pt1)
            min_key1 = functools.partial(_get_pt_dist_to_node, samp)
            nearest1 = min(nodes1, key=min_key1)
            for newpt in self._extend_fn(nearest1.data, samp):
                if self._collision_fn(newpt):
                    break
                nearest1 = _BiRRTNode(newpt, parent=nearest1)
                nodes1.append(nearest1)
            min_key2 = functools.partial(_get_pt_dist_to_node, nearest1.data)
            nearest2 = min(nodes2, key=min_key2)
            for newpt in self._extend_fn(nearest2.data, nearest1.data):
                if self._collision_fn(newpt):
                    break
                nearest2 = _BiRRTNode(newpt, parent=nearest2)
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

    def _smooth_path(self, path: List[_S]) -> List[_S]:
        assert len(path) > 2
        for _ in range(self._smooth_amt):
            i = self._rng.integers(0, len(path) - 1)
            j = self._rng.integers(0, len(path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = list(self._extend_fn(path[i], path[j]))
            if len(shortcut) < j-i and \
               all(not self._collision_fn(pt) for pt in shortcut):
                path = path[:i + 1] + shortcut + path[j + 1:]
        return path


class _BiRRTNode(Generic[_S]):
    """A node for BiRRT."""

    def __init__(self,
                 data: _S,
                 parent: Optional[_BiRRTNode[_S]] = None) -> None:
        self.data = data
        self.parent = parent

    def path_from_root(self) -> List[_BiRRTNode[_S]]:
        """Return the path from the root to this node."""
        sequence = []
        node: Optional[_BiRRTNode[_S]] = self
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
    """Create a new task where any excluded predicates have their classifiers
    removed."""
    stripped_goal: Set[GroundAtom] = set()
    for atom in task.goal:
        # The atom's goal is known.
        if atom.predicate in included_predicates:
            stripped_goal.add(atom)
            continue
        # The atom's goal is unknown.
        stripped_pred = strip_predicate(atom.predicate)
        stripped_atom = GroundAtom(stripped_pred, atom.objects)
        stripped_goal.add(stripped_atom)
    return Task(task.init, stripped_goal)


def abstract(state: State, preds: Collection[Predicate]) -> Set[GroundAtom]:
    """Get the atomic representation of the given state (i.e., a set of ground
    atoms), using the given set of predicates.

    Duplicate arguments in predicates are allowed.
    """
    atoms = set()
    for pred in preds:
        for choice in get_object_combinations(list(state), pred.types):
            if pred.holds(state, choice):
                atoms.add(GroundAtom(pred, choice))
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
        yield nsrt.ground(choice)


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
        sas_task = downward_translate(dom_str, prob_str)
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


def all_ground_ldl_rules(rule: LDLRule,
                         objects: Collection[Object]) -> List[_GroundLDLRule]:
    """Get all possible groundings of the given rule with the given objects."""
    return _cached_all_ground_ldl_rules(rule, frozenset(objects))


@functools.lru_cache(maxsize=None)
def _cached_all_ground_ldl_rules(
        rule: LDLRule,
        frozen_objects: FrozenSet[Object]) -> List[_GroundLDLRule]:
    """Helper for all_ground_ldl_rules() that caches the outputs."""
    ground_rules = []
    types = [p.type for p in rule.parameters]
    for choice in get_object_combinations(frozen_objects, types):
        ground_rule = rule.ground(tuple(choice))
        ground_rules.append(ground_rule)
    return ground_rules


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
    if CFG.env == "behavior":  # pragma: no cover
        dataset_fname_template = (
            f"{CFG.env}__{CFG.behavior_scene_name}__{CFG.behavior_task_name}" +
            f"__{CFG.offline_data_method}__{CFG.demonstrator}__"
            f"{regex}__{CFG.included_options}__{CFG.seed}" + suffix_str)
    else:
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


def create_pddl_domain(operators: Collection[NSRTOrSTRIPSOperator],
                       predicates: Collection[Predicate],
                       types: Collection[Type], domain_name: str) -> str:
    """Create a PDDL domain str from STRIPSOperators or NSRTs."""
    # Sort everything to ensure determinism.
    preds_lst = sorted(predicates)
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
                continue
            child_type_str = " ".join(t.name for t in child_types)
            types_str += f"\n    {child_type_str} - {parent_type.name}"
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


@dataclass
class VideoMonitor(Monitor):
    """A monitor that renders each state and action encountered.

    The render_fn is generally env.render. Note that the state is unused
    because the environment should use its current internal state to
    render.
    """
    _render_fn: Callable[[Optional[Action], Optional[str]], Video]
    _video: Video = field(init=False, default_factory=list)

    def observe(self, state: State, action: Optional[Action]) -> None:
        del state  # unused
        self._video.extend(self._render_fn(action, None))

    def get_video(self) -> Video:
        """Return the video."""
        return self._video


@dataclass
class SimulateVideoMonitor(Monitor):
    """A monitor that calls render_state on each state and action seen.

    This monitor is meant for use with run_policy_with_simulator, as
    opposed to VideoMonitor, which is meant for use with run_policy.
    """
    _task: Task
    _render_state_fn: Callable[[State, Task, Optional[Action]], Video]
    _video: Video = field(init=False, default_factory=list)

    def observe(self, state: State, action: Optional[Action]) -> None:
        self._video.extend(self._render_state_fn(state, self._task, action))

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


def get_env_asset_path(asset_name: str, assert_exists: bool = True) -> str:
    """Return the absolute path to env asset."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, "envs", "assets")
    path = os.path.join(asset_dir_path, asset_name)
    if assert_exists:
        assert os.path.exists(path), f"Env asset not found: {asset_name}."
    return path


def update_config(args: Dict[str, Any]) -> None:
    """Args is a dictionary of new arguments to add to the config CFG."""
    arg_specific_settings = GlobalSettings.get_arg_specific_settings(args)
    # Only override attributes, don't create new ones.
    allowed_args = set(CFG.__dict__) | set(arg_specific_settings)
    parser = create_arg_parser()
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
    update_config(arg_dict)


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
    args, overrides = parser.parse_known_args()
    arg_dict = vars(args)
    if len(overrides) == 0:
        return arg_dict
    # Update initial settings to make sure we're overriding
    # existing flags only
    update_config(arg_dict)
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
                assert env.goal_predicates.issubset(included), \
                    "Can't exclude a goal predicate!"
    else:
        excluded_names = set()
        included = env.predicates
    excluded = {pred for pred in env.predicates if pred.name in excluded_names}
    return included, excluded


def parse_config_included_options(env: BaseEnv) -> Set[ParameterizedOption]:
    """Parse the CFG.included_options string, given an environment.

    Return the set of included oracle options.

    Note that "all" is not implemented because setting the option_learner flag
    to "no_learning" is the preferred way to include all options.
    """
    if not CFG.included_options:
        return set()
    included_names = set(CFG.included_options.split(","))
    assert included_names.issubset({option.name for option in env.options}), \
        "Unrecognized option in included_options!"
    included_options = {o for o in env.options if o.name in included_names}
    return included_options


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


def query_ldl(ldl: LiftedDecisionList, atoms: Set[GroundAtom],
              objects: Set[Object],
              goal: Set[GroundAtom]) -> Optional[_GroundNSRT]:
    """Queries a lifted decision list representing a goal-conditioned policy.

    Given an abstract state and goal, the rules are grounded in order. The
    first applicable ground rule is used to return a ground NSRT.

    If no rule is applicable, returns None.
    """
    for rule in ldl.rules:
        for ground_rule in all_ground_ldl_rules(rule, objects):
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
