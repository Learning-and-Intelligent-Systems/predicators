from shapely.geometry import box as Box, Point, Polygon, MultiPolygon, LineString, MultiLineString

def BoxWH(x: float, y: float, w: float, h: float) -> Polygon:
    return Box(x, y, x + w, y + h)