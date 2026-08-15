import cv2
import numpy as np
import time
import uuid
import os
from collections import deque

class TrackedPerson:
    """
    Independent state container for an individual tracked person.
    """
    def __init__(self, track_id, bbox_norm, entry_time=None):
        self.track_id = track_id
        self.bbox_norm = bbox_norm # (x1, y1, x2, y2) normalized [0..1]
        self.centroid_norm = ((bbox_norm[0] + bbox_norm[2])/2.0, (bbox_norm[1] + bbox_norm[3])/2.0)
        
        self.entry_time = entry_time if entry_time is not None else time.time()
        self.last_seen_time = self.entry_time
        
        self.in_poi = False
        self.previously_in_poi = False
        self.entered_poi_once = False
        self.frames_in_poi = 0
        self.exited_poi = False
        self.active_monitor = False
        
        # 3-Step Safety Standards
        self.step1_ok = False
        self.step2_ok = False
        self.step3_ok = False
        self.total_ok = False
        self.is_ng = False
        self.judged = False
        
        self.step1_hold_count = 0
        self.step2_hold_count = 0
        self.step3_hold_count = 0
        
        self.step1_time = None
        self.step2_time = None
        self.step3_time = None
        
        # Exponential moving averages for smooth landmark tracking
        self.smooth_yaw = 0.0
        self.smooth_arm_dx = 0.0
        
        # Quality Face Crop
        self.best_face_image = None
        self.best_face_score = 0.0
        
        # Evidence Recording Buffer (stores up to ~180 frames = 6 seconds)
        self.evidence_buffer = deque(maxlen=180)
        
        # Alarms & DB Status
        self.alarm_triggered = False
        self.db_logged = False
        self.event_uuid = str(uuid.uuid4())[:8]
        
        self.trajectory = []

    def update_position(self, bbox_norm, current_time):
        self.bbox_norm = bbox_norm
        cx = (bbox_norm[0] + bbox_norm[2]) / 2.0
        cy = (bbox_norm[1] + bbox_norm[3]) / 2.0
        
        # Smooth centroid update
        self.centroid_norm = (0.7 * self.centroid_norm[0] + 0.3 * cx, 0.7 * self.centroid_norm[1] + 0.3 * cy)
        self.last_seen_time = current_time
        self.trajectory.append((self.centroid_norm[0], self.centroid_norm[1], current_time))
        if len(self.trajectory) > 60:
            self.trajectory.pop(0)

    def set_poi_state(self, current_in_poi):
        self.previously_in_poi = self.in_poi
        self.in_poi = current_in_poi
        
        if current_in_poi:
            self.frames_in_poi += 1
            if self.frames_in_poi >= 5:
                self.entered_poi_once = True
            self.active_monitor = True
        else:
            if self.previously_in_poi and self.entered_poi_once:
                self.exited_poi = True

    def update_face(self, face_crop, score=1.0):
        if face_crop is not None and face_crop.size > 0 and score >= self.best_face_score:
            self.best_face_image = face_crop.copy()
            self.best_face_score = score

    def add_evidence_frame(self, frame):
        self.evidence_buffer.append(frame.copy())

    def save_evidence_video(self, output_dir="data/evidence", fps=25.0):
        os.makedirs(output_dir, exist_ok=True)
        if len(self.evidence_buffer) == 0:
            return ""
        out_filename = f"evidence_{self.event_uuid}_track_{self.track_id}.mp4"
        out_path = os.path.join(output_dir, out_filename)
        h, w, _ = self.evidence_buffer[0].shape
        
        writer = None
        for codec in ['avc1', 'H264', 'mp4v']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                w_test = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                if w_test.isOpened():
                    writer = w_test
                    break
            except Exception:
                continue
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in self.evidence_buffer:
            writer.write(f)
        writer.release()
        return out_path

    def save_face_image(self, output_dir="data/faces"):
        os.makedirs(output_dir, exist_ok=True)
        if self.best_face_image is None or self.best_face_image.size == 0:
            return ""
        out_filename = f"face_{self.event_uuid}_track_{self.track_id}.jpg"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, self.best_face_image)
        return out_path

class SimpleMultiPersonTracker:
    """
    Robust 1-to-1 Spatial Multi-Person Tracker with Hungarian/Global Minimum Matching.
    Prevents track cross-talk or mistaken step updates when multiple people are in frame.
    """
    def __init__(self, max_disappeared_sec=2.0, max_dist_threshold=0.45):
        self.next_track_id = 1
        self.tracked_persons = {} # track_id -> TrackedPerson
        self.max_disappeared_sec = max_disappeared_sec
        self.max_dist_threshold = max_dist_threshold
        self.recently_removed_persons = []

    def update(self, detected_bboxes_norm, current_time):
        self.recently_removed_persons = []
        
        if len(detected_bboxes_norm) == 0:
            removed_ids = []
            for tid, person in self.tracked_persons.items():
                if (current_time - person.last_seen_time) > self.max_disappeared_sec:
                    removed_ids.append(tid)
                    self.recently_removed_persons.append(person)
            for tid in removed_ids:
                del self.tracked_persons[tid]
            return list(self.tracked_persons.values())

        if len(self.tracked_persons) == 0:
            for bbox in detected_bboxes_norm:
                person = TrackedPerson(self.next_track_id, bbox, entry_time=current_time)
                self.tracked_persons[self.next_track_id] = person
                self.next_track_id += 1
            return list(self.tracked_persons.values())

        track_ids = list(self.tracked_persons.keys())
        track_centroids = [self.tracked_persons[tid].centroid_norm for tid in track_ids]
        det_centroids = [((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for b in detected_bboxes_norm]

        # Build full distance matrix
        num_tracks = len(track_ids)
        num_dets = len(detected_bboxes_norm)
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        
        for t_idx, t_cen in enumerate(track_centroids):
            for d_idx, d_cen in enumerate(det_centroids):
                # Combined distance + size differential penalty
                dx = t_cen[0] - d_cen[0]
                dy = t_cen[1] - d_cen[1]
                dist = np.sqrt(dx*dx + dy*dy)
                cost_matrix[t_idx, d_idx] = dist

        # Strict 1-to-1 Global Greedy Minimum Assignment
        matched_tracks = set()
        matched_dets = set()
        
        flat_indices = np.argsort(cost_matrix, axis=None)
        for idx in flat_indices:
            t_idx = idx // num_dets
            d_idx = idx % num_dets
            
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            if cost_matrix[t_idx, d_idx] > self.max_dist_threshold:
                continue
                
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            tid = track_ids[t_idx]
            self.tracked_persons[tid].update_position(detected_bboxes_norm[d_idx], current_time)

        # Unmatched detections -> new track IDs
        for d_idx, bbox in enumerate(detected_bboxes_norm):
            if d_idx not in matched_dets:
                person = TrackedPerson(self.next_track_id, bbox, entry_time=current_time)
                self.tracked_persons[self.next_track_id] = person
                self.next_track_id += 1

        # Remove timed-out tracks
        removed_ids = []
        for tid, person in self.tracked_persons.items():
            if (current_time - person.last_seen_time) > self.max_disappeared_sec:
                removed_ids.append(tid)
                self.recently_removed_persons.append(person)
        for tid in removed_ids:
            del self.tracked_persons[tid]

        return list(self.tracked_persons.values())
