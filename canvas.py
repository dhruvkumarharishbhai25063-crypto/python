"""
canvas.py
---------
Handles the virtual drawing canvas that gets overlaid on the webcam feed:
- Stores the ordered list of stroke points (the path the user draws).
- Draws lines onto a transparent-able canvas layer.
- Renders a color palette + eraser bar at the top of the screen and
  detects when the fingertip is hovering over one of those buttons.
"""

import cv2
import numpy as np


class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # The drawing layer: a blank (black) image the same size as the
        # webcam frame. Black pixels are treated as "empty" when we
        # composite this on top of the live video.
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)

        # Ordered list of every point in the current stroke: [(x, y), ...]
        # Kept separately from the pixel layer so we can later convert
        # this exact path into a 2D polygon for 3D extrusion.
        self.stroke_points = []

        # All finished strokes (in case the user draws multiple disconnected
        # shapes before switching to 3D mode). Each entry is a point list.
        self.all_strokes = []

        self.prev_point = None
        self.draw_color = (0, 0, 255)   # BGR - default red
        self.brush_thickness = 6
        self.eraser_thickness = 40
        self.is_erasing = False

        # --- Palette / toolbar setup ---
        # Each entry: (label, BGR color or None for eraser, x_start, x_end)
        self.toolbar_height = 60
        self.palette = [
            ("Red",    (0, 0, 255)),
            ("Green",  (0, 255, 0)),
            ("Blue",   (255, 0, 0)),
            ("Yellow", (0, 255, 255)),
            ("Eraser", None),
        ]
        self._build_palette_zones()

    def _build_palette_zones(self):
        """Compute clickable x-ranges for each palette button across the top bar."""
        zone_width = self.width // len(self.palette)
        self.palette_zones = []
        for i, (label, color) in enumerate(self.palette):
            x_start = i * zone_width
            x_end = x_start + zone_width
            self.palette_zones.append((label, color, x_start, x_end))

    def draw_toolbar(self, frame):
        """Draw the palette/eraser bar onto the given frame (in place)."""
        for label, color, x_start, x_end in self.palette_zones:
            swatch_color = color if color is not None else (50, 50, 50)
            cv2.rectangle(frame, (x_start, 0), (x_end, self.toolbar_height), swatch_color, -1)
            cv2.putText(
                frame, label, (x_start + 10, self.toolbar_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
            # Highlight the currently active tool
            active = (color == self.draw_color and not self.is_erasing) or \
                     (color is None and self.is_erasing)
            if active:
                cv2.rectangle(frame, (x_start, 0), (x_end, self.toolbar_height), (255, 255, 255), 3)
        return frame

    def check_toolbar_hover(self, x, y):
        """
        If (x, y) — the fingertip position — is inside the toolbar area,
        select that color/eraser and return True. Otherwise return False.
        """
        if y > self.toolbar_height:
            return False

        for label, color, x_start, x_end in self.palette_zones:
            if x_start <= x <= x_end:
                if color is None:
                    self.is_erasing = True
                else:
                    self.is_erasing = False
                    self.draw_color = color
                return True
        return False

    def start_or_continue_stroke(self, x, y):
        """
        Called every frame while in 'draw' gesture mode. Draws a line
        segment from the previous fingertip position to the current one,
        and records the point in the ordered stroke list.
        """
        if y <= self.toolbar_height:
            # Don't draw while hovering the toolbar
            self.prev_point = None
            return

        if self.prev_point is None:
            self.prev_point = (x, y)

        color = (0, 0, 0) if self.is_erasing else self.draw_color
        thickness = self.eraser_thickness if self.is_erasing else self.brush_thickness

        cv2.line(self.layer, self.prev_point, (x, y), color, thickness)

        if not self.is_erasing:
            self.stroke_points.append((x, y))

        self.prev_point = (x, y)

    def end_stroke(self):
        """Call when the drawing gesture stops, to close off the current stroke."""
        if self.stroke_points:
            self.all_strokes.append(self.stroke_points)
            self.stroke_points = []
        self.prev_point = None

    def composite(self, frame):
        """Overlay the drawing layer on top of the live webcam frame."""
        gray = cv2.cvtColor(self.layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        layer_fg = cv2.bitwise_and(self.layer, self.layer, mask=mask)
        combined = cv2.add(frame_bg, layer_fg)
        return combined

    def clear(self):
        """Wipe the canvas and all stored stroke data ('c' key)."""
        self.layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.stroke_points = []
        self.all_strokes = []
        self.prev_point = None

    def get_all_points(self):
        """
        Return every drawn point across all strokes as one flat ordered list.
        Used when converting the drawing into a 2D polygon for 3D extrusion.
        """
        # End any in-progress stroke first so it's included
        if self.stroke_points:
            self.end_stroke()

        flat_points = []
        for stroke in self.all_strokes:
            flat_points.extend(stroke)
        return flat_points

    def save(self, path="drawing.png"):
        """Save just the drawing layer (not the webcam feed) as a PNG ('s' key)."""
        cv2.imwrite(path, self.layer)
        return path