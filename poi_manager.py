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

    def find_nearest_vertex(self, px, py, frame_w, frame_h, max_dist_px=25):
        """Finds index of nearest vertex within threshold, returns -1 if none"""
        best_idx = -1
        best_dist = max_dist_px
        for i, p in enumerate(self.polygon):
            vx = int(p["x"] * frame_w)
            vy = int(p["y"] * frame_h)
            dist = np.sqrt((px - vx)**2 + (py - vy)**2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def update_vertex(self, idx, px, py, frame_w, frame_h):
        """Updates vertex position using pixel coordinates, clamped to 0..1"""
        if 0 <= idx < len(self.polygon):
            norm_x = max(0.0, min(1.0, float(px) / float(frame_w)))
            norm_y = max(0.0, min(1.0, float(py) / float(frame_h)))
            self.polygon[idx] = {"x": round(norm_x, 4), "y": round(norm_y, 4)}

    def add_vertex(self, px, py, frame_w, frame_h):
        """Adds a new vertex at normalized position"""
        norm_x = max(0.0, min(1.0, float(px) / float(frame_w)))
        norm_y = max(0.0, min(1.0, float(py) / float(frame_h)))
        self.polygon.append({"x": round(norm_x, 4), "y": round(norm_y, 4)})

    def remove_last_vertex(self):
        """Removes the last vertex if more than 3 remain"""
        if len(self.polygon) > 3:
            self.polygon.pop()

    def reset_to_default(self):
        """Resets POI zone to default corridor box"""
        self.polygon = [
            {"x": 0.05, "y": 0.05},
            {"x": 0.95, "y": 0.05},
            {"x": 0.95, "y": 0.95},
            {"x": 0.05, "y": 0.95}
        ]
        self.save_config()

    def draw_poi_overlay(self, frame, active=False, edit_mode=False, selected_vertex_idx=-1):
        """Draws POI boundary with neon accent, labels, and interactive edit handles"""
        h, w, _ = frame.shape
        pts = self.get_pixel_polygon(w, h)
        if len(pts) < 3:
            return frame
            
        overlay = frame.copy()
        if edit_mode:
            fill_color = (180, 100, 20)
            border_color = (0, 220, 255)
            alpha = 0.20
        else:
            fill_color = (0, 100, 255) if active else (50, 150, 50)
            border_color = (0, 200, 255) if active else (0, 255, 120)
            alpha = 0.12 if not active else 0.22
        
        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2 if not edit_mode else 3, lineType=cv2.LINE_AA)
        
        # POI Label at first point
        label_x, label_y = max(10, min(w - 240, pts[0][0] + 8)), max(25, pts[0][1] + 20)
        if edit_mode:
            cv2.putText(frame, "POI: [EDIT MODE - DRAG HANDLES]", 
                        (label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "POI: ACTIVE SAFETY ZONE" if active else "POI: ROAD CROSSING ZONE", 
                        (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1, cv2.LINE_AA)
        
        # Draw interactive vertex handles in Edit Mode
        if edit_mode:
            for idx, pt in enumerate(pts):
                is_selected = (idx == selected_vertex_idx)
                handle_color = (0, 255, 255) if is_selected else (0, 180, 255)
                center_color = (255, 255, 255) if is_selected else (40, 40, 40)
                radius = 11 if is_selected else 8
                
                # Outer glow and ring
                cv2.circle(frame, (pt[0], pt[1]), radius + 3, (0, 0, 0), -1)
                cv2.circle(frame, (pt[0], pt[1]), radius, handle_color, -1)
                cv2.circle(frame, (pt[0], pt[1]), radius - 3, center_color, -1)
                
                # Vertex numbering
                cv2.putText(frame, f"P{idx+1}", (pt[0] + 12, pt[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                            
        return frame
