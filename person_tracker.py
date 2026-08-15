import cv2
import numpy as np
import time
import uuid
import os
from collections import deque

def compute_bbox_iou(box1, box2):
    """Computes Intersection over Union (IOU) between two (x1, y1, x2, y2) bboxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class TrackedPerson:
    """
    Robust State container for a single verified person.
    Ensures ghost blips (< 4 frames) are suppressed before rendering or tracking steps.
    """
    def __init__(self, track_id, bbox_norm, entry_time=None):
        self.track_id = track_id
        self.bbox_norm = bbox_norm
        self.centroid_norm = ((bbox_norm[0] + bbox_norm[2])/2.0, (bbox_norm[1] + bbox_norm[3])/2.0)
        
        self.entry_time = entry_time if entry_time is not None else time.time()
        self.last_seen_time = self.entry_time
        
        self.in_poi = False
        self.previously_in_poi = False
        self.frames_in_poi = 0
        self.total_frames_tracked = 1
        self.consecutive_misses = 0
        
        # Temporal Confirmation (Confirmed Real Person after >= 4 frames)
        self.is_verified = False
        self.entered_poi_once = False
        self.exited_poi = False
        self.active_monitor = False
        
        # 3-Step Safety Standards
        self.checked_left = False
        self.checked_right = False
        self.checked_front = False
        
        self.left_hold_count = 0
        self.right_hold_count = 0
        self.front_hold_count = 0
        
        self.step1_ok = False
        self.step2_ok = False
        self.step3_ok = False
        self.total_ok = False
        self.is_ng = False
        self.judged = False
        
        # Face / Head Avatar Crop
        self.best_face_image = None
        self.best_face_score = 0.0
        
        # Subsampled Evidence Buffer (~15 fps, max 90 frames = 6 sec)
        self.evidence_buffer = deque(maxlen=90)
        self.frame_subsample_counter = 0
        
        self.alarm_triggered = False
        self.db_logged = False
        self.event_uuid = str(uuid.uuid4())[:8]
        
        self.trajectory = []

    def update_position(self, bbox_norm, current_time):
        self.bbox_norm = bbox_norm
        cx = (bbox_norm[0] + bbox_norm[2]) / 2.0
        cy = (bbox_norm[1] + bbox_norm[3]) / 2.0
        self.centroid_norm = (0.75 * self.centroid_norm[0] + 0.25 * cx, 0.75 * self.centroid_norm[1] + 0.25 * cy)
        self.last_seen_time = current_time
        self.total_frames_tracked += 1
        self.consecutive_misses = 0
        
        # Confirm person identity after 4 consistent frames
        if self.total_frames_tracked >= 4:
            self.is_verified = True
            
        self.trajectory.append((self.centroid_norm[0], self.centroid_norm[1], current_time))
        if len(self.trajectory) > 60:
            self.trajectory.pop(0)

    def set_poi_state(self, current_in_poi):
        self.previously_in_poi = self.in_poi
        self.in_poi = current_in_poi
        
        if current_in_poi:
            self.frames_in_poi += 1
            if self.is_verified:
                self.active_monitor = True
            if self.frames_in_poi >= 4 and self.is_verified:
                self.entered_poi_once = True
        else:
            if self.previously_in_poi and self.entered_poi_once:
                self.exited_poi = True

    def update_face(self, face_crop, score=1.0):
        if face_crop is not None and face_crop.size > 0 and score >= self.best_face_score:
            self.best_face_image = face_crop.copy()
            self.best_face_score = score

    def add_evidence_frame(self, frame):
        self.frame_subsample_counter += 1
        if self.frame_subsample_counter % 3 == 0:
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / w
                frame_to_store = cv2.resize(frame, (640, int(h * scale)))
            else:
                frame_to_store = frame.copy()
            self.evidence_buffer.append(frame_to_store)

    def save_evidence_video(self, output_dir="data/evidence", fps=15.0):
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
            if f.shape[0] != h or f.shape[1] != w:
                f = cv2.resize(f, (w, h))
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
    Robust Multi-Person Tracker with NMS Duplicate Suppression & Strict Association Gating.
    """
    def __init__(self, max_disappeared_sec=1.5, max_dist_threshold=0.32):
        self.next_track_id = 1
        self.tracked_persons = {}
        self.max_disappeared_sec = max_disappeared_sec
        self.max_dist_threshold = max_dist_threshold
        self.recently_removed_persons = []

    def update(self, detected_bboxes_norm, current_time):
        self.recently_removed_persons = []
        
        # 1. Non-Maximum Suppression (NMS) on duplicate detections of the same person
        filtered_bboxes = []
        for bbox in detected_bboxes_norm:
            duplicate = False
            for kept in filtered_bboxes:
                if compute_bbox_iou(bbox, kept) > 0.40:
                    duplicate = True
                    break
            if not duplicate:
                filtered_bboxes.append(bbox)

        if len(filtered_bboxes) == 0:
            removed_ids = []
            for tid, person in self.tracked_persons.items():
                if (current_time - person.last_seen_time) > self.max_disappeared_sec:
                    removed_ids.append(tid)
                    if person.is_verified:
                        self.recently_removed_persons.append(person)
            for tid in removed_ids:
                del self.tracked_persons[tid]
            return list(self.tracked_persons.values())

        if len(self.tracked_persons) == 0:
            for bbox in filtered_bboxes:
                person = TrackedPerson(self.next_track_id, bbox, entry_time=current_time)
                self.tracked_persons[self.next_track_id] = person
                self.next_track_id += 1
            return list(self.tracked_persons.values())

        track_ids = list(self.tracked_persons.keys())
        track_centroids = [self.tracked_persons[tid].centroid_norm for tid in track_ids]
        det_centroids = [((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for b in filtered_bboxes]

        num_tracks = len(track_ids)
        num_dets = len(filtered_bboxes)
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        
        for t_idx, t_cen in enumerate(track_centroids):
            for d_idx, d_cen in enumerate(det_centroids):
                dx = t_cen[0] - d_cen[0]
                dy = t_cen[1] - d_cen[1]
                dist = np.sqrt(dx*dx + dy*dy)
                # IOU overlap bonus to lock onto existing tracks
                iou = compute_bbox_iou(self.tracked_persons[track_ids[t_idx]].bbox_norm, filtered_bboxes[d_idx])
                cost_matrix[t_idx, d_idx] = dist - (iou * 0.15)

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
            self.tracked_persons[tid].update_position(filtered_bboxes[d_idx], current_time)

        for d_idx, bbox in enumerate(filtered_bboxes):
            if d_idx not in matched_dets:
                person = TrackedPerson(self.next_track_id, bbox, entry_time=current_time)
                self.tracked_persons[self.next_track_id] = person
                self.next_track_id += 1

        removed_ids = []
        for tid, person in self.tracked_persons.items():
            if (current_time - person.last_seen_time) > self.max_disappeared_sec:
                removed_ids.append(tid)
                if person.is_verified:
                    self.recently_removed_persons.append(person)
        for tid in removed_ids:
            del self.tracked_persons[tid]

        return list(self.tracked_persons.values())
