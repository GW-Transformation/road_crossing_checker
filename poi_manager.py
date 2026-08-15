import cv2
import numpy as np
import json
import os

class POIManager:
    """
    Manages Point of Interest (POI) / Region of Interest (ROI) zone for Road Crossing.
    Supports:
      - Polygon coordinates (normalized 0.0 - 1.0)
      - Checking if point or bounding box is inside POI
      - Drawing POI overlay on frame
      - Saving and loading POI configuration
    """
    def __init__(self, config_path="poi_config.json", default_polygon=None):
        self.config_path = config_path
        self.polygon = []
        if default_polygon:
            self.polygon = default_polygon
        elif os.path.exists(config_path):
            self.load_config(config_path)
        else:
            # Default POI zone covering central crossing corridor (normalized coordinates)
            self.polygon = [
                {"x": 0.05, "y": 0.05},
                {"x": 0.95, "y": 0.05},
                {"x": 0.95, "y": 0.95},
                {"x": 0.05, "y": 0.95}
            ]
            self.save_config(config_path)

    def load_config(self, path=None):
        path = path or self.config_path
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    self.polygon = data.get("poi_polygon", self.polygon)
            except Exception as e:
                print(f"[WARN] Failed to load POI config: {e}")

    def save_config(self, path=None):
        path = path or self.config_path
        try:
            with open(path, "w") as f:
                json.dump({"poi_polygon": self.polygon}, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save POI config: {e}")

    def set_polygon_normalized(self, points):
        """points: list of dicts [{'x': 0.1, 'y': 0.2}, ...] or list of tuples [(0.1, 0.2), ...]"""
        new_poly = []
        for p in points:
            if isinstance(p, dict):
                new_poly.append({"x": float(p["x"]), "y": float(p["y"])})
            elif isinstance(p, (list, tuple)):
                new_poly.append({"x": float(p[0]), "y": float(p[1])})
        self.polygon = new_poly
        self.save_config()

    def get_pixel_polygon(self, frame_w, frame_h):
        pts = []
        for p in self.polygon:
            px = int(p["x"] * frame_w)
            py = int(p["y"] * frame_h)
            pts.append([px, py])
        return np.array(pts, dtype=np.int32)

    def is_point_inside(self, norm_x, norm_y):
        """Checks if a normalized point (x, y) is inside the POI polygon"""
        if len(self.polygon) < 3:
            return True
        pts = np.array([[p["x"], p["y"]] for p in self.polygon], dtype=np.float32)
        dist = cv2.pointPolygonTest(pts, (float(norm_x), float(norm_y)), False)
        return dist >= 0

    def is_bbox_inside_or_intersect(self, bbox_norm):
        """bbox_norm: (x1, y1, x2, y2) in 0..1 range"""
        x1, y1, x2, y2 = bbox_norm
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # Check center and bottom center (pedestrian foot location)
        return self.is_point_inside(cx, cy) or self.is_point_inside(cx, y2)

    def draw_poi_overlay(self, frame, active=False):
        """Draws POI boundary with neon accent and label"""
        h, w, _ = frame.shape
        pts = self.get_pixel_polygon(w, h)
        if len(pts) < 3:
            return frame
            
        overlay = frame.copy()
        fill_color = (0, 100, 255) if active else (50, 150, 50)
        border_color = (0, 200, 255) if active else (0, 255, 120)
        
        cv2.fillPoly(overlay, [pts], fill_color)
        alpha = 0.12 if not active else 0.22
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)
        
        # POI Label at first point
        label_x, label_y = pts[0][0] + 8, pts[0][1] + 20
        cv2.putText(frame, "POI: ACTIVE SAFETY ZONE" if active else "POI: ROAD CROSSING ZONE", 
                    (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1, cv2.LINE_AA)
        return frame
