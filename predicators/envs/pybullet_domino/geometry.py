"""Shared 2-D geometry for the domino environment.

Single source of truth for the plan-view conventions that were previously
re-derived inline across the task generators (``task_generators/``) and the
cascade certificate (``cascade_certificate.py``): the yaw <-> direction
mapping, oriented-rectangle corners, and the separating-axis (SAT) overlap /
signed-gap test.

Conventions (state ``yaw`` is a CCW z-rotation, ``getQuaternionFromEuler``):

* a block's **travel/heading** direction (the way a chain steps and the way
  the Push skill shoves) is ``(sin yaw, cos yaw)`` - see :func:`travel_dir`;
* a block's **fall / thin** axis (the way it topples) is
  ``(-sin yaw, cos yaw)`` - see :func:`fall_axis`;
* ``yaw_along(dx, dy)`` returns the yaw whose fall axis points along
  ``(dx, dy)``; ``heading_yaw(dx, dy)`` the yaw whose travel direction does.

Pure Python (``math`` only, no numpy) so the certificate stays a lightweight,
picklable pure function; the values are IEEE doubles identical to the numpy
scalar results the generators used, to within a rounding-invisible ULP.
"""

import math
from typing import List, Tuple

Point = Tuple[float, float]
Rect = List[Point]


def travel_dir(yaw: float) -> Point:
    """Chain travel / heading unit direction for ``yaw``: ``(sin, cos)``."""
    return math.sin(yaw), math.cos(yaw)


def fall_axis(yaw: float, sign: float = 1.0) -> Point:
    """Fall (thin) axis for ``yaw``: ``(-sin, cos)`` times ``sign``.

    A negative roll falls along ``(-sin yaw, cos yaw)``; pass
    ``sign=-1`` for the positive-roll direction.
    """
    return -math.sin(yaw) * sign, math.cos(yaw) * sign


def yaw_along(dx: float, dy: float) -> float:
    """Yaw whose fall axis ``(-sin, cos)`` points along ``(dx, dy)``.

    ``arctan2(-dx, dy)`` (``arctan2(dx, dy)`` would mirror it across the
    y-axis for non-axis-aligned directions).
    """
    return math.atan2(-dx, dy)


def heading_yaw(dx: float, dy: float) -> float:
    """Yaw whose travel direction ``(sin, cos)`` points along ``(dx, dy)``."""
    return math.atan2(dx, dy)


def wrap_angle(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def rect_corners(cx: float, cy: float, ax: float, ay: float, half_len: float,
                 half_wid: float) -> Rect:
    """Corners of an oriented rectangle centered at ``(cx, cy)``.

    The long axis is the unit vector ``(ax, ay)`` (half-extent
    ``half_len``) and the short axis is its left normal (half-extent
    ``half_wid``).
    """
    px, py = -ay, ax
    return [(cx + sl * half_len * ax + sw * half_wid * px,
             cy + sl * half_len * ay + sw * half_wid * py)
            for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def domino_footprint(cx: float, cy: float, yaw: float, half_width: float,
                     half_depth: float) -> Rect:
    """Plan-view rectangle of a domino-like body at ``yaw``.

    The block's width axis is ``(cos yaw, sin yaw)`` (half-extent
    ``half_width``) and its depth/thin axis is ``(-sin yaw, cos yaw)``
    (half-extent ``half_depth``) - the convention shared by the certificate's
    footprints and the generator's placement collision check.
    """
    return rect_corners(cx, cy, math.cos(yaw), math.sin(yaw), half_width,
                        half_depth)


def rect_gap(rect_a: Rect, rect_b: Rect) -> float:
    """Signed separation between two oriented rectangles (SAT over their edge
    normals): positive is the plan-view clearance between them, negative means
    they overlap."""
    best = -math.inf
    for rect, other in ((rect_a, rect_b), (rect_b, rect_a)):
        for i in range(2):
            ex = rect[i + 1][0] - rect[i][0]
            ey = rect[i + 1][1] - rect[i][1]
            norm = math.hypot(ex, ey)
            nx, ny = -ey / norm, ex / norm
            proj = [c[0] * nx + c[1] * ny for c in rect]
            proj_other = [c[0] * nx + c[1] * ny for c in other]
            sep = max(min(proj_other) - max(proj), min(proj) - max(proj_other))
            best = max(best, sep)
    return best


def rects_overlap(rect_a: Rect, rect_b: Rect) -> bool:
    """Whether two oriented rectangles overlap in plan view (touching edges do
    not count as overlapping)."""
    return rect_gap(rect_a, rect_b) < 0.0
