import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import json
from poi_manager import POIManager
from person_tracker import SimpleMultiPersonTracker, TrackedPerson
from alarm_player import AlarmWarningPlayer
from db_manager import CrossingDatabase
from face_recognition_client import ExternalFaceRecognitionClient

class RoadCrossingSafetyChecker:
    """
    Advanced Pedestrian Road Crossing Standard Safety Checker
    =========================================================
    Features:
      - Point of Interest (POI) safety corridor detection
      - Active Monitoring when person/face is detected in POI
      - Individual pedestrian tracking & history
      - 3-Step Safety Verification:
          Step 1: Look Left AND Point Left
          Step 2: Look Right AND Point Right
          Step 3: Look Forward AND Point Forward
      - Automatic NG detection with Voice Alarm: "กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง"
      - Snaps best face crop & saves evidence video
      - Dispatches External Face Recognition API to identify person
      - Full DB logging to SQLite
    """
    def __init__(
        self,
        model_path=None,
        min_hold_frames=6,
        poi_config_path="poi_config.json",
        db_path="data/crossing_records.db",
        face_api_url=None,
        enable_alarm=True,
        max_crossing_wait_sec=5.0
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "pose_landmarker_lite.task")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
            num_poses=2
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.min_hold_frames = min_hold_frames
        self.max_crossing_wait_sec = max_crossing_wait_sec
        
        # Subsystems
        self.poi_manager = POIManager(config_path=os.path.join(base_dir, poi_config_path))
        self.tracker = SimpleMultiPersonTracker()
        self.db = CrossingDatabase(db_path=os.path.join(base_dir, db_path))
        self.alarm = AlarmWarningPlayer(enabled=enable_alarm)
        self.face_client = ExternalFaceRecognitionClient(
            api_url=face_api_url,
            on_match_callback=self._on_face_matched
        )
        
        # Global stats / active list
        self.active_tracked_persons = []
        self.last_ng_event = None

    def _on_face_matched(self, record_id, result):
        """Callback when external face recognition server returns identification"""
        if result and result.get("matched"):
            self.db.update_external_recognition(
                record_id=record_id,
                person_id=result.get("person_id"),
                name=result.get("name"),
                confidence=result.get("confidence", 0.0)
            )

    def _classify_head_look(self, nose, l_ear, r_ear, l_eye, r_eye):
        ear_mid_x = (l_ear.x + r_ear.x) / 2.0
        ear_width = abs(l_ear.x - r_ear.x) + 1e-6
        head_yaw_ratio = (nose.x - ear_mid_x) / ear_width
        
        if head_yaw_ratio < -0.40:
            return "LEFT", head_yaw_ratio
        elif head_yaw_ratio > 0.25:
            return "RIGHT", head_yaw_ratio
        elif -0.38 <= head_yaw_ratio <= 0.22:
            return "FORWARD", head_yaw_ratio
        else:
            return "CENTER", head_yaw_ratio

    def _classify_arm_point(self, lm):
        l_sh, r_sh = lm[11], lm[12]
        l_wr, r_wr = lm[15], lm[16]
        
        r_raised = (r_wr.y < (r_sh.y + 0.18))
        l_raised = (l_wr.y < (l_sh.y + 0.18))
        
        r_dx, r_dy, r_dz = r_wr.x - r_sh.x, r_wr.y - r_sh.y, r_wr.z - r_sh.z
        l_dx, l_dy, l_dz = l_wr.x - l_sh.x, l_wr.y - l_sh.y, l_wr.z - l_sh.z
        
        if r_raised and not l_raised:
            active_arm, dx, dy, dz = "RIGHT_ARM", r_dx, r_dy, r_dz
        elif l_raised and not r_raised:
            active_arm, dx, dy, dz = "LEFT_ARM", l_dx, l_dy, l_dz
        elif r_raised and l_raised:
            if abs(r_dx) >= abs(l_dx):
                active_arm, dx, dy, dz = "RIGHT_ARM", r_dx, r_dy, r_dz
            else:
                active_arm, dx, dy, dz = "LEFT_ARM", l_dx, l_dy, l_dz
        else:
            return "NONE", None, (0.0, 0.0, 0.0)

        if dx < -0.10:
            return "LEFT", active_arm, (dx, dy, dz)
        elif dx > 0.12:
            return "RIGHT", active_arm, (dx, dy, dz)
        elif abs(dx) <= 0.12 and dy < 0.20:
            return "FORWARD", active_arm, (dx, dy, dz)
        else:
            return "UNKNOWN", active_arm, (dx, dy, dz)

    def _extract_face_crop(self, frame, lm):
        h, w, _ = frame.shape
        # Key head landmarks: nose(0), ears(7,8), eyes(2,5), mouth(9,10)
        xs = [lm[i].x for i in [0, 2, 5, 7, 8, 9, 10] if i < len(lm)]
        ys = [lm[i].y for i in [0, 2, 5, 7, 8, 9, 10] if i < len(lm)]
        if not xs or not ys:
            return None, (0, 0, 0, 0)
            
        min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
        min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
        
        # Margin
        pad_x = (max_x - min_x) * 0.45 + 0.02
        pad_y = (max_y - min_y) * 0.45 + 0.02
        
        x1 = max(0, int((min_x - pad_x) * w))
        y1 = max(0, int((min_y - pad_y) * h))
        x2 = min(w, int((max_x + pad_x) * w))
        y2 = min(h, int((max_y + pad_y) * h))
        
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            face_crop = frame[y1:y2, x1:x2].copy()
            return face_crop, (x1, y1, x2, y2)
        return None, (0, 0, 0, 0)

    def process_frame(self, frame, timestamp_sec=None):
        if timestamp_sec is None:
            timestamp_sec = time.time()
            
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        detected_poses = []
        detected_bboxes = []
        
        if detection_result.pose_landmarks:
            for lm in detection_result.pose_landmarks:
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                bbox = (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))
                detected_poses.append((lm, bbox))
                detected_bboxes.append(bbox)

        # Update Multi-Person Tracker
        tracked_persons = self.tracker.update(detected_bboxes, timestamp_sec)
        self.active_tracked_persons = tracked_persons

        # Associate poses to tracked persons
        for person in tracked_persons:
            person.add_evidence_frame(frame)
            
            # Find best matching pose
            matched_lm = None
            for lm, p_bbox in detected_poses:
                p_cen = ((p_bbox[0]+p_bbox[2])/2.0, (p_bbox[1]+p_bbox[3])/2.0)
                if np.sqrt((p_cen[0]-person.centroid_norm[0])**2 + (p_cen[1]-person.centroid_norm[1])**2) < 0.25:
                    matched_lm = lm
                    break
                    
            # Check POI Containment
            in_poi = self.poi_manager.is_bbox_inside_or_intersect(person.bbox_norm)
            person.in_poi = in_poi
            if in_poi:
                person.active_monitor = True

            if matched_lm and person.active_monitor:
                # Snap high quality face crop
                face_crop, face_coords = self._extract_face_crop(frame, matched_lm)
                if face_crop is not None:
                    person.update_face(face_crop, score=1.0)
                    
                # Classify Look & Point
                head_look, head_yaw = self._classify_head_look(
                    matched_lm[0], matched_lm[7], matched_lm[8], matched_lm[2], matched_lm[5]
                )
                point_dir, active_arm, arm_vec = self._classify_arm_point(matched_lm)
                
                # Step 1: Look Left + Point Left
                if not person.step1_ok:
                    if head_look == "LEFT" and point_dir == "LEFT":
                        person.step1_hold_count += 1
                        if person.step1_hold_count >= self.min_hold_frames:
                            person.step1_ok = True
                            person.step1_time = timestamp_sec
                    else:
                        person.step1_hold_count = max(0, person.step1_hold_count - 1)
                        
                # Step 2: Look Right + Point Right
                elif person.step1_ok and not person.step2_ok:
                    if head_look == "RIGHT" and point_dir == "RIGHT":
                        person.step2_hold_count += 1
                        if person.step2_hold_count >= self.min_hold_frames:
                            person.step2_ok = True
                            person.step2_time = timestamp_sec
                    else:
                        person.step2_hold_count = max(0, person.step2_hold_count - 1)
                        
                # Step 3: Look Forward + Point Forward
                elif person.step2_ok and not person.step3_ok:
                    if head_look in ["FORWARD", "CENTER"] and point_dir == "FORWARD":
                        person.step3_hold_count += 1
                        if person.step3_hold_count >= self.min_hold_frames:
                            person.step3_ok = True
                            person.step3_time = timestamp_sec
                            person.total_ok = True
                            person.is_ng = False
                    else:
                        person.step3_hold_count = max(0, person.step3_hold_count - 1)
                        
                # Check for NG Condition:
                # If person has been in POI crossing area for more than max_crossing_wait_sec
                # or is moving across without completing the 3 steps -> Trigger NG!
                time_in_poi = timestamp_sec - person.entry_time
                if not person.total_ok and time_in_poi >= self.max_crossing_wait_sec:
                    person.is_ng = True
                    self._trigger_ng_actions(person, timestamp_sec)

                # If person finished all 3 steps successfully -> Trigger TOTAL_OK Log
                elif person.total_ok and not person.db_logged:
                    self._trigger_total_ok_actions(person, timestamp_sec)

                # Draw skeleton on person
                self._draw_person_skeleton(frame, matched_lm, w, h, person.total_ok, person.is_ng)

        # Draw POI Overlay
        poi_active = any(p.active_monitor for p in tracked_persons)
        frame = self.poi_manager.draw_poi_overlay(frame, active=poi_active)
        
        # Draw Comprehensive Multi-Person HUD Overlay
        annotated_frame = self._draw_multi_hud(frame, tracked_persons)
        
        info = {
            "timestamp": timestamp_sec,
            "tracked_count": len(tracked_persons),
            "persons": [
                {
                    "track_id": p.track_id,
                    "in_poi": p.in_poi,
                    "step1_ok": p.step1_ok,
                    "step2_ok": p.step2_ok,
                    "step3_ok": p.step3_ok,
                    "total_ok": p.total_ok,
                    "is_ng": p.is_ng
                } for p in tracked_persons
            ]
        }
        return annotated_frame, info

    def _trigger_ng_actions(self, person, timestamp_sec):
        if person.db_logged:
            return
            
        # 1. Voice Alarm Warning
        if not person.alarm_triggered:
            self.alarm.play_alarm(reason="Safety Checklist Incomplete")
            person.alarm_triggered = True
            
        # 2. Save Evidence Video & Cropped Face
        face_path = person.save_face_image()
        evidence_path = person.save_evidence_video()
        
        # 3. Log to SQLite Database
        record_data = {
            "event_uuid": person.event_uuid,
            "track_id": person.track_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp_sec)),
            "status": "NG",
            "step1_status": "OK" if person.step1_ok else "NG",
            "step2_status": "OK" if person.step2_ok else "NG",
            "step3_status": "OK" if person.step3_ok else "NG",
            "duration_sec": timestamp_sec - person.entry_time,
            "face_image_path": face_path,
            "evidence_video_path": evidence_path,
            "metadata": {"in_poi": person.in_poi, "trajectory_len": len(person.trajectory)}
        }
        record_id = self.db.insert_record(record_data)
        person.db_logged = True
        self.last_ng_event = record_data
        print(f"\n[ALERT NG] Person #{person.track_id} failed standard! Logged to DB (ID: {record_id})")

        # 4. Dispatch Async External Face Recognition API
        if person.best_face_image is not None:
            self.face_client.recognize_face_async(person.best_face_image, record_id, metadata=record_data)

    def _trigger_total_ok_actions(self, person, timestamp_sec):
        if person.db_logged:
            return
        face_path = person.save_face_image()
        evidence_path = person.save_evidence_video()
        record_data = {
            "event_uuid": person.event_uuid,
            "track_id": person.track_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp_sec)),
            "status": "TOTAL_OK",
            "step1_status": "OK",
            "step2_status": "OK",
            "step3_status": "OK",
            "duration_sec": timestamp_sec - person.entry_time,
            "face_image_path": face_path,
            "evidence_video_path": evidence_path,
            "metadata": {"in_poi": person.in_poi}
        }
        record_id = self.db.insert_record(record_data)
        person.db_logged = True
        print(f"\n[TOTAL OK] Person #{person.track_id} completed standard! Logged to DB (ID: {record_id})")

    def _draw_person_skeleton(self, frame, lm, w, h, total_ok, is_ng):
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (15, 19), (16, 20)
        ]
        color = (0, 0, 255) if is_ng else ((0, 255, 0) if total_ok else (0, 220, 255))
        for p1, p2 in connections:
            if p1 < len(lm) and p2 < len(lm):
                pt1 = (int(lm[p1].x * w), int(lm[p1].y * h))
                pt2 = (int(lm[p2].x * w), int(lm[p2].y * h))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    def _draw_multi_hud(self, frame, tracked_persons):
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Top HUD Dashboard Bar
        hud_h = 145
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (18, 18, 24), -1)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (18, 18, 24), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.putText(frame, "PEDESTRIAN CROSSING SAFETY MONITOR", (12, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                    
        active_count = len([p for p in tracked_persons if p.active_monitor])
        cv2.putText(frame, f"Active Pedestrians in POI: {active_count}", (w - 240, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
                    
        # If there are tracked persons, render active checklist for first/primary person
        if tracked_persons:
            primary = tracked_persons[0]
            
            s1_c = (50, 240, 50) if primary.step1_ok else ((0, 0, 255) if primary.is_ng else (140, 140, 140))
            s1_s = "OK" if primary.step1_ok else ("NG" if primary.is_ng else "--")
            cv2.putText(frame, f"Person #{primary.track_id} Step 1: Look & Point LEFT   [{s1_s}]", (14, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, s1_c, 1, cv2.LINE_AA)
                        
            s2_c = (50, 240, 50) if primary.step2_ok else ((0, 0, 255) if (primary.is_ng and not primary.step2_ok) else (140, 140, 140))
            s2_s = "OK" if primary.step2_ok else ("NG" if primary.is_ng else "--")
            cv2.putText(frame, f"Person #{primary.track_id} Step 2: Look & Point RIGHT  [{s2_s}]", (14, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, s2_c, 1, cv2.LINE_AA)
                        
            s3_c = (50, 240, 50) if primary.step3_ok else ((0, 0, 255) if (primary.is_ng and not primary.step3_ok) else (140, 140, 140))
            s3_s = "OK" if primary.step3_ok else ("NG" if primary.is_ng else "--")
            cv2.putText(frame, f"Person #{primary.track_id} Step 3: Look & Point FRONT  [{s3_s}]", (14, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, s3_c, 1, cv2.LINE_AA)
                        
            # Total Card
            if primary.total_ok:
                cv2.rectangle(frame, (10, 102), (w - 10, 134), (30, 180, 30), -1)
                cv2.putText(frame, f"PERSON #{primary.track_id}: TOTAL OK (PASS - SAFE TO CROSS)", (18, 124),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            elif primary.is_ng:
                cv2.rectangle(frame, (10, 102), (w - 10, 134), (0, 0, 200), -1)
                cv2.putText(frame, f"PERSON #{primary.track_id}: ALARM NG - SAFETY PROTOCOL VIOLATION!", (18, 124),
                            cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (10, 102), (w - 10, 134), (50, 120, 200), -1)
                cv2.putText(frame, f"PERSON #{primary.track_id}: MONITORING SAFETY SEQUENCE...", (18, 124),
                            cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Waiting for pedestrian to enter Point of Interest (POI)...", (14, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)
                        
        # Bottom Bar: System Status
        status_msg = "ACTIVE MONITOR ON | VOICE ALARM READY | DB LOGGING ENABLED"
        cv2.putText(frame, status_msg, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)
        return frame

    def generate_final_report(self):
        """Returns structured JSON summary of check session"""
        persons_report = []
        for p in self.tracker.tracked_persons.values():
            persons_report.append({
                "track_id": p.track_id,
                "step1": "OK" if p.step1_ok else "NG",
                "step2": "OK" if p.step2_ok else "NG",
                "step3": "OK" if p.step3_ok else "NG",
                "total": "TOTAL OK" if p.total_ok else "TOTAL NG",
                "is_standard_compliant": bool(p.total_ok)
            })
            
        all_passed = (len(persons_report) > 0 and all(pr["is_standard_compliant"] for pr in persons_report))
        
        # Primary person fallback
        p1 = self.active_tracked_persons[0] if self.active_tracked_persons else None
        
        return {
            "step1_look_point_left": "OK" if (p1 and p1.step1_ok) else "NG",
            "step2_look_point_right": "OK" if (p1 and p1.step2_ok) else "NG",
            "step3_look_point_forward": "OK" if (p1 and p1.step3_ok) else "NG",
            "total_result": "TOTAL OK" if all_passed else "TOTAL NG",
            "is_standard_compliant": all_passed,
            "persons": persons_report
        }
