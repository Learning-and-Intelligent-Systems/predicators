from shapely.geometry import box as Box, Point, Polygon as ShapelyPolygon, MultiPolygon, LineString, MultiLineString
from matplotlib.patches import Polygon as PLTPolygon
import numpy as np

def BoxWH(x: float, y: float, w: float, h: float) -> ShapelyPolygon:
    return Box(x, y, x + w, y + h)

def plot_polygon(poly: ShapelyPolygon, **kwargs) -> PLTPolygon:
    return PLTPolygon(np.array(poly.exterior.xy).T, closed=True, **kwargs)