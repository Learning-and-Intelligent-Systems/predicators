from shapely import Geometry
from shapely.geometry import box as Box, Point, Polygon
from matplotlib import patches
import numpy as np
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