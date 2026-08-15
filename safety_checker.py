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
    High-Performance & Resource-Optimized Road Crossing Safety Checker
    ==================================================================
    - Optimized 360p/480p inference downscaling for embedded & low-spec hardware
    - Strict Qualified Pedestrian verification (prevents false alarms on empty frames)
    - Alarms ONLY when a confirmed pedestrian leaves POI without completing 3 steps (1 time only)
    - Multi-person 1-to-1 matching with zero step cross-talk
    """
    def __init__(
        self,
        model_path=None,
        min_hold_frames=5,
        poi_config_path="poi_config.json",
        db_path="data/crossing_records.db",
        face_api_url=None,
        enable_alarm=True,
        max_crossing_wait_sec=5.0,
        inference_max_width=480,
        **kwargs
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "pose_landmarker_lite.task")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            num_poses=2
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.min_hold_frames = min_hold_frames
        self.max_crossing_wait_sec = max_crossing_wait_sec
        self.inference_max_width = inference_max_width
        
        # Subsystems
        self.poi_manager = POIManager(config_path=os.path.join(base_dir, poi_config_path))
        self.tracker = SimpleMultiPersonTracker()
        self.db = CrossingDatabase(db_path=os.path.join(base_dir, db_path))
        self.alarm = AlarmWarningPlayer(enabled=enable_alarm)
        self.face_client = ExternalFaceRecognitionClient(
            api_url=face_api_url,
            on_match_callback=self._on_face_matched
        )
        
        self.active_tracked_persons = []
        self.last_ng_event = None

    def _on_face_matched(self, record_id, result):
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
        
        if head_yaw_ratio < -0.38:
            return "LEFT", head_yaw_ratio
        elif head_yaw_ratio > 0.24:
            return "RIGHT", head_yaw_ratio
        elif -0.36 <= head_yaw_ratio <= 0.22:
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
        elif abs(dx) <= 0.12 and dy < 0.22:
            return "FORWARD", active_arm, (dx, dy, dz)
        else:
            return "UNKNOWN", active_arm, (dx, dy, dz)

    def _extract_face_crop(self, frame, lm):
        h, w, _ = frame.shape
        xs = [lm[i].x for i in [0, 2, 5, 7, 8, 9, 10] if i < len(lm)]
        ys = [lm[i].y for i in [0, 2, 5, 7, 8, 9, 10] if i < len(lm)]
        if not xs or not ys:
            return None, (0, 0, 0, 0)
            
        min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
        min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
        
        pad_x = (max_x - min_x) * 0.40 + 0.02
        pad_y = (max_y - min_y) * 0.40 + 0.02
        
        x1 = max(0, int((min_x - pad_x) * w))
        y1 = max(0, int((min_y - pad_y) * h))
        x2 = min(w, int((max_x + pad_x) * w))
        y2 = min(h, int((max_y + pad_y) * h))
        
        if (x2 - x1) > 12 and (y2 - y1) > 12:
            face_crop = frame[y1:y2, x1:x2].copy()
            return face_crop, (x1, y1, x2, y2)
        return None, (0, 0, 0, 0)

    def process_frame(self, frame, timestamp_sec=None):
        if timestamp_sec is None:
            timestamp_sec = time.time()
            
        h, w, _ = frame.shape
        
        # Performance Optimization: Downscale frame for MediaPipe inference
        if w > self.inference_max_width:
            scale = self.inference_max_width / float(w)
            inf_w = self.inference_max_width
            inf_h = int(h * scale)
            small_frame = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_LINEAR)
        else:
            small_frame = frame

        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        detected_poses = []
        detected_bboxes = []
        
        if detection_result.pose_landmarks:
            for lm in detection_result.pose_landmarks:
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                bw = max(xs) - min(xs)
                bh = max(ys) - min(ys)
                # Filter out noisy, tiny, or incomplete detections
                if bh < 0.15 or bw < 0.06:
                    continue
                bbox = (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))
                detected_poses.append((lm, bbox))
                detected_bboxes.append(bbox)

        tracked_persons = self.tracker.update(detected_bboxes, timestamp_sec)
        self.active_tracked_persons = tracked_persons

        # Strict 1-to-1 Spatial Match
        assigned_poses = {}
        if tracked_persons and detected_poses:
            num_persons = len(tracked_persons)
            num_poses = len(detected_poses)
            cost_mat = np.zeros((num_persons, num_poses), dtype=np.float32)
            
            for p_idx, person in enumerate(tracked_persons):
                for pose_idx, (lm, p_bbox) in enumerate(detected_poses):
                    p_cen = ((p_bbox[0]+p_bbox[2])/2.0, (p_bbox[1]+p_bbox[3])/2.0)
                    cost_mat[p_idx, pose_idx] = np.sqrt((p_cen[0]-person.centroid_norm[0])**2 + (p_cen[1]-person.centroid_norm[1])**2)
                    
            matched_p = set()
            matched_pose = set()
            flat_indices = np.argsort(cost_mat, axis=None)
            
            for idx in flat_indices:
                p_idx = idx // num_poses
                pose_idx = idx % num_poses
                if p_idx in matched_p or pose_idx in matched_pose:
                    continue
                if cost_mat[p_idx, pose_idx] > 0.40:
                    continue
                matched_p.add(p_idx)
                matched_pose.add(pose_idx)
                assigned_poses[tracked_persons[p_idx].track_id] = detected_poses[pose_idx][0]

        # Process each tracked person independently
        for person in tracked_persons:
            matched_lm = assigned_poses.get(person.track_id)
            if matched_lm:
                # Snap face image
                face_crop, face_coords = self._extract_face_crop(frame, matched_lm)
                if face_crop is not None:
                    person.update_face(face_crop, score=1.0)
                    
                # Classify 3-step safety standard
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
                            self.alarm.play_step_ok(step_num=1)
                    else:
                        person.step1_hold_count = max(0, person.step1_hold_count - 1)
                        
                # Step 2: Look Right + Point Right
                elif person.step1_ok and not person.step2_ok:
                    if head_look == "RIGHT" and point_dir == "RIGHT":
                        person.step2_hold_count += 1
                        if person.step2_hold_count >= self.min_hold_frames:
                            person.step2_ok = True
                            person.step2_time = timestamp_sec
                            self.alarm.play_step_ok(step_num=2)
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
                            self.alarm.play_step_ok(step_num=3) # Level-Up Victory Sound!
                    else:
                        person.step3_hold_count = max(0, person.step3_hold_count - 1)

            # Check POI containment
            in_poi = self.poi_manager.is_bbox_inside_or_intersect(person.bbox_norm)
            person.set_poi_state(in_poi)

            if person.active_monitor:
                person.add_evidence_frame(frame)

            # If person stepped out of POI while still in frame:
            if person.exited_poi and person.is_qualified_pedestrian and not person.judged:
                self._judge_person_outcome(person, timestamp_sec)

        # Process any persons that disappeared/left the camera frame
        for person in self.tracker.recently_removed_persons:
            if person.is_qualified_pedestrian and not person.judged:
                self._judge_person_outcome(person, timestamp_sec)

        # Draw POI Overlay
        poi_active = any(p.active_monitor for p in tracked_persons)
        frame = self.poi_manager.draw_poi_overlay(frame, active=poi_active)
        
        # Draw Dedicated Active People Panel (Faces + Step 1, 2, 3 Status)
        annotated_frame = self._draw_active_people_panel(frame, tracked_persons)
        
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
                } for p in tracked_persons if p.is_qualified_pedestrian or p.frames_in_poi >= 5
            ]
        }
        return annotated_frame, info

    def _judge_person_outcome(self, person, timestamp_sec):
        """
        Judges a qualified person once upon disappearing or leaving POI:
        - If total_ok == True ➔ TOTAL OK (Logged to DB, 0 voice alarms)
        - If total_ok == False ➔ NG (Voice alarm plays EXACTLY 1 TIME, logged to DB, Face API called)
        """
        person.judged = True
        
        if person.total_ok:
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
            print(f"\n[JUDGMENT: TOTAL OK] Person #{person.track_id} crossed with complete standard! Logged to DB (ID: {record_id})")
        else:
            person.is_ng = True
            
            # Voice Alarm: EXACTLY 1 TIME
            if not person.alarm_triggered:
                self.alarm.play_alarm(reason="Disappeared without completing safety steps")
                person.alarm_triggered = True
                
            face_path = person.save_face_image()
            evidence_path = person.save_evidence_video()
            
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
                "metadata": {"in_poi": person.in_poi, "reason": "Left POI without completing standard"}
            }
            record_id = self.db.insert_record(record_data)
            person.db_logged = True
            self.last_ng_event = record_data
            print(f"\n[JUDGMENT: NG ALERT] 🚨 Person #{person.track_id} left POI without standard! Voice alarm played (1 time). Logged to DB (ID: {record_id})")
            
            if person.best_face_image is not None:
                self.face_client.recognize_face_async(person.best_face_image, record_id, metadata=record_data)

    def _draw_active_people_panel(self, frame, tracked_persons):
        h, w, _ = frame.shape
        panel_w = 260
        panel_x = w - panel_w - 10
        panel_y = 10
        
        # Only display confirmed pedestrians in the panel
        active_list = [p for p in tracked_persons if p.is_qualified_pedestrian or p.frames_in_poi >= 5]
        if not active_list:
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, panel_y + 36), (20, 20, 26), -1)
            cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
            cv2.rectangle(frame, (panel_x, panel_y), (w - 10, panel_y + 36), (60, 60, 70), 1)
            cv2.putText(frame, "ACTIVE MONITOR: IDLE", (panel_x + 12, panel_y + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
            return frame

        card_h = 75
        total_panel_h = min(h - 20, len(active_list) * (card_h + 8) + 36)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, panel_y + total_panel_h), (18, 20, 28), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (w - 10, panel_y + total_panel_h), (0, 200, 255), 1)
        
        cv2.putText(frame, f"ACTIVE PEOPLE IN POI ({len(active_list)})", (panel_x + 10, panel_y + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        curr_y = panel_y + 30
        for person in active_list:
            if curr_y + card_h > h - 10:
                break
                
            card_bg = (30, 34, 46)
            cv2.rectangle(frame, (panel_x + 6, curr_y), (w - 16, curr_y + card_h), card_bg, -1)
            cv2.rectangle(frame, (panel_x + 6, curr_y), (w - 16, curr_y + card_h), (50, 60, 80), 1)
            
            face_size = 54
            face_x = panel_x + 12
            face_y = curr_y + 10
            
            if person.best_face_image is not None and person.best_face_image.size > 0:
                try:
                    resized_face = cv2.resize(person.best_face_image, (face_size, face_size))
                    frame[face_y:face_y+face_size, face_x:face_x+face_size] = resized_face
                except Exception:
                    cv2.rectangle(frame, (face_x, face_y), (face_x+face_size, face_y+face_size), (80, 80, 90), -1)
            else:
                cv2.rectangle(frame, (face_x, face_y), (face_x+face_size, face_y+face_size), (40, 45, 60), -1)
                cv2.putText(frame, "FACE", (face_x + 12, face_y + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 150), 1, cv2.LINE_AA)
                            
            cv2.rectangle(frame, (face_x, face_y), (face_x+face_size, face_y+face_size), (0, 200, 255), 1)

            info_x = face_x + face_size + 10
            cv2.putText(frame, f"Person #{person.track_id}", (info_x, curr_y + 18),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
                        
            s1_col = (0, 240, 0) if person.step1_ok else ((0, 0, 255) if person.is_ng else (140, 140, 140))
            s1_txt = "S1:OK" if person.step1_ok else "S1:--"
            cv2.putText(frame, s1_txt, (info_x, curr_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, s1_col, 1, cv2.LINE_AA)
            
            s2_col = (0, 240, 0) if person.step2_ok else ((0, 0, 255) if person.is_ng else (140, 140, 140))
            s2_txt = "S2:OK" if person.step2_ok else "S2:--"
            cv2.putText(frame, s2_txt, (info_x + 56, curr_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, s2_col, 1, cv2.LINE_AA)
            
            s3_col = (0, 240, 0) if person.step3_ok else ((0, 0, 255) if person.is_ng else (140, 140, 140))
            s3_txt = "S3:OK" if person.step3_ok else "S3:--"
            cv2.putText(frame, s3_txt, (info_x + 112, curr_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, s3_col, 1, cv2.LINE_AA)
            
            if person.total_ok:
                status_str = "STATUS: TOTAL OK (SAFE)"
                st_color = (0, 255, 120)
            elif person.is_ng:
                status_str = "STATUS: NG (VIOLATION)"
                st_color = (0, 70, 255)
            else:
                status_str = "STATUS: IN PROGRESS..."
                st_color = (0, 200, 255)
                
            cv2.putText(frame, status_str, (info_x, curr_y + 54),
                        cv2.FONT_HERSHEY_DUPLEX, 0.36, st_color, 1, cv2.LINE_AA)
                        
            curr_y += card_h + 8
            
        return frame

    def generate_final_report(self):
        now = time.time()
        for p in self.tracker.tracked_persons.values():
            if p.is_qualified_pedestrian and not p.judged:
                self._judge_person_outcome(p, now)
                
        qualified_persons = [p for p in self.tracker.tracked_persons.values() if p.is_qualified_pedestrian]
        persons_report = []
        all_passed = True if qualified_persons else False
        for p in qualified_persons:
            persons_report.append({
                "track_id": p.track_id,
                "step1": "OK" if p.step1_ok else "NG",
                "step2": "OK" if p.step2_ok else "NG",
                "step3": "OK" if p.step3_ok else "NG",
                "total": "TOTAL OK" if p.total_ok else "TOTAL NG",
                "is_standard_compliant": bool(p.total_ok)
            })
            if not p.total_ok:
                all_passed = False
                
        p1 = qualified_persons[0] if qualified_persons else (self.active_tracked_persons[0] if self.active_tracked_persons else None)
        return {
            "step1_look_point_left": "OK" if (p1 and p1.step1_ok) else "NG",
            "step2_look_point_right": "OK" if (p1 and p1.step2_ok) else "NG",
            "step3_look_point_forward": "OK" if (p1 and p1.step3_ok) else "NG",
            "total_result": "TOTAL OK" if all_passed else "TOTAL NG",
            "is_standard_compliant": all_passed,
            "persons": persons_report
        }
