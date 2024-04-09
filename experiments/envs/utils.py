from shapely import Geometry
from shapely.geometry import box as Box, Point, Polygon, MultiPolygon
from matplotlib import patches
import numpy as np
import numpy.typing as npt
import logging

def BoxWH(x: float, y: float, w: float, h: float) -> Polygon:
    return Box(x, y, x + w, y + h)

def plot_geometry(poly: Geometry, **kwargs) -> patches.Patch:
    if isinstance(poly, Point):
        return patches.Wedge((poly.x, poly.y), 0.3, 60, 120, **kwargs)
    elif isinstance(poly, Polygon):
        return patches.Polygon(np.array(poly.exterior.xy).T, closed=True, **kwargs)
    else:
        raise ValueError("Unsupported geometry type")

def construct_subcell_box(cells: npt.NDArray[np.bool_], subcell: float, diff_margin: float = 1e-5) -> Polygon:
    h, w = cells.shape
    poly = BoxWH(0, 0, w * subcell, h * subcell)
    diff = MultiPolygon([])
    for y, row in enumerate(cells[::-1]):
        for x, cell in enumerate(row):
            if not cell:
                diff = diff.union(BoxWH(x * subcell, y * subcell, subcell, subcell))
    return poly.difference(diff.buffer(diff_margin, cap_style='square', join_style='mitre'))