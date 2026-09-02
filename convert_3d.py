"""
convert_3d.py (solid-first, tube-fallback)
-------------------------------------------
Two ways to turn your 2D stroke into a 3D mesh:

1. SOLID EXTRUSION (tried first): treats your stroke as the outline of a
   flat shape, fills it in, and pushes it straight up into a solid block
   -- e.g. draw a square, get an actual solid cube/box. This is what you
   want for clean, simple, closed shapes.

2. TUBE FALLBACK: if the drawing is too tangled/messy for step 1 to turn
   into a clean solid (self-crossing lines, stray strokes, an open path),
   we fall back to tracing the literal path as a 3D tube instead of
   failing outright.

WHY YOUR CUBE LOOKED LIKE A MESSY FLAT SHAPE:
Hand-tracking sometimes keeps the "draw" gesture active for a split
second while your hand moves between corners, adding small stray lines
across the middle of your shape. The old tube-only approach traced
every one of those stray lines as its own 3D tube. Solid extrusion
mostly ignores that noise, because it only cares about the OUTER
boundary of everything you drew, not each individual line segment.
"""

import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon


def points_to_polygon(points, canvas_width, canvas_height, epsilon_ratio=0.002):
    """
    Convert an ordered list of (x, y) pixel points into a cleaned-up
    Shapely polygon (the filled outer boundary of the stroke).
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to form a polygon.")

    mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    pts_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [pts_array], isClosed=True, color=255, thickness=8)
    cv2.fillPoly(mask, [pts_array], color=255)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour could be found from the drawn points.")

    largest_contour = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(largest_contour, True)
    epsilon = epsilon_ratio * perimeter
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

    contour_points = [tuple(pt[0]) for pt in simplified]
    if len(contour_points) < 3:
        raise ValueError("Simplified contour has fewer than 3 points; draw a larger shape.")

    polygon = Polygon(contour_points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # fixes minor self-intersections

    # buffer(0) can turn a self-intersecting shape into a MultiPolygon
    # (several disconnected pieces). Keep only the biggest piece.
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda p: p.area)

    if polygon.is_empty or polygon.area < 1:
        raise ValueError("Cleaned-up polygon is empty or too small.")

    return polygon


def extrude_polygon(polygon, extrude_height=50.0):
    """Extrude a 2D Shapely polygon into a solid 3D trimesh block."""
    return trimesh.creation.extrude_polygon(polygon, height=extrude_height)


# ---------------------------------------------------------------------
# Tube fallback (same as before) -- used only if solid extrusion fails
# ---------------------------------------------------------------------

def stroke_to_tube_mesh(points, canvas_height, tube_radius=6.0, z_scale=0.15,
                         smooth_joints=True):
    if len(points) < 2:
        raise ValueError("Need at least 2 points to build a tube.")

    pts3d = np.array(
        [(x * z_scale, (canvas_height - y) * z_scale, 0.0) for (x, y) in points]
    )
    scaled_radius = tube_radius * z_scale

    segments = []
    for i in range(len(pts3d) - 1):
        p0, p1 = pts3d[i], pts3d[i + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue

        cyl = trimesh.creation.cylinder(radius=scaled_radius, height=seg_len, sections=8)
        seg_dir = seg_vec / seg_len
        rotation = trimesh.geometry.align_vectors(np.array([0, 0, 1]), seg_dir)
        cyl.apply_transform(rotation)
        cyl.apply_translation((p0 + p1) / 2.0)
        segments.append(cyl)

        if smooth_joints:
            joint = trimesh.creation.icosphere(radius=scaled_radius, subdivisions=1)
            joint.apply_translation(p0)
            segments.append(joint)

    if smooth_joints and len(pts3d) > 0:
        joint = trimesh.creation.icosphere(radius=scaled_radius, subdivisions=1)
        joint.apply_translation(pts3d[-1])
        segments.append(joint)

    if not segments:
        raise ValueError("Could not build any tube segments from the given points.")

    return trimesh.util.concatenate(segments)


# ---------------------------------------------------------------------
# Public entry point used by main.py
# ---------------------------------------------------------------------

def convert_points_to_mesh(points, canvas_width, canvas_height, extrude_height=50.0,
                            tube_radius=6.0):
    """
    Try to build a clean SOLID block first. Only fall back to a tube
    trace if the drawing is too messy/open for that to work.
    """
    try:
        polygon = points_to_polygon(points, canvas_width, canvas_height)
        mesh = extrude_polygon(polygon, extrude_height=extrude_height)
        print("Built a solid extruded block.")
        return mesh
    except Exception as solid_error:
        print(f"Solid extrusion failed ({solid_error}); falling back to tube trace...")
        return stroke_to_tube_mesh(points, canvas_height, tube_radius=tube_radius)


def export_mesh(mesh, filepath="airsketch_model.obj"):
    """Export the mesh to .obj or .stl (format is inferred from the extension)."""
    mesh.export(filepath)
    return filepath
