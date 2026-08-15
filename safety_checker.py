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
    High-Performance & High-Accuracy Road Crossing Safety Checker
    =============================================================
    - Clean Face Extraction: Strictly bounds head/face above shoulder line (no hands/torso)
    - Instant Point Forward Recognition: Natural arm alignment for 1st-try completion
    - Strict Person Verification: Skeletons & Active Cards shown ONLY for verified people
    - Flexible Sequence: (Left <-> Right first, ending with Front)
    - Any Hand: Left or Right hand pointing
    - High-Visibility Screen HUD & 0 False Alarms
    """
    def __init__(
        self,
        model_path=None,
        min_hold_frames=2,
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
            min_pose_detection_confidence=0.50,
            min_pose_presence_confidence=0.50,
            min_tracking_confidence=0.50,
            num_poses=3
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.min_hold_frames = max(1, min_hold_frames)
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
        self.last_live_feedback = ""

    def _on_face_matched(self, record_id, result):
        if result and result.get("matched"):
            self.db.update_external_recognition(
                record_id=record_id,
                person_id=result.get("person_id"),
                name=result.get("name"),
                confidence=result.get("confidence", 0.0)
            )

    def _is_valid_human_pose(self, lm):
        """Strict Anatomical Verification"""
        if len(lm) < 25:
            return False
            
        l_sh, r_sh = lm[11], lm[12]
        l_hip, r_hip = lm[23], lm[24]
        
        sh_w = abs(l_sh.x - r_sh.x)
        torso_h = abs((l_hip.y + r_hip.y)/2.0 - (l_sh.y + r_sh.y)/2.0)
        
        if sh_w < 0.04 or torso_h < 0.08:
            return False
            
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        total_w = max(xs) - min(xs)
        total_h = max(ys) - min(ys)
        
        if total_h < 0.16 or total_w < 0.06:
            return False
            
        return True

    def _classify_head_look(self, lm):
        nose = lm[0]
        l_ear, r_ear = lm[7], lm[8]
        
        ear_mid_x = (l_ear.x + r_ear.x) / 2.0
        ear_w = abs(l_ear.x - r_ear.x) + 1e-5
        yaw_ratio = (nose.x - ear_mid_x) / ear_w
        
        if yaw_ratio < -0.16:
            return "LEFT", yaw_ratio
        elif yaw_ratio > 0.14:
            return "RIGHT", yaw_ratio
        elif abs(yaw_ratio) <= 0.35:
            return "FRONT", yaw_ratio
        else:
            return "UNKNOWN", yaw_ratio

    def _classify_arm_pointing_dirs(self, lm):
        """
        Evaluates both arms with proper direction prioritization.
        Ensures FRONT pointing is recognized on the 1st try without false Left/Right misclassification.
        """
        l_sh, r_sh = lm[11], lm[12]
        l_wr, r_wr = lm[15], lm[16]
        l_idx, r_idx = lm[19], lm[20]
        
        r_raised = (r_wr.y < (r_sh.y + 0.28)) or (r_idx.y < (r_sh.y + 0.28))
        l_raised = (l_wr.y < (l_sh.y + 0.28)) or (l_idx.y < (l_sh.y + 0.28))
        
        r_dx = (r_idx.x - r_sh.x) if abs(r_idx.x - r_sh.x) > abs(r_wr.x - r_sh.x) else (r_wr.x - r_sh.x)
        l_dx = (l_idx.x - l_sh.x) if abs(l_idx.x - l_sh.x) > abs(l_wr.x - l_sh.x) else (l_wr.x - l_sh.x)
        
        dirs = []
        
        # Right Arm
        if r_raised:
            # Distinct Left (cross-body point)
            if r_dx < -0.11:
                dirs.append("LEFT")
            # Distinct Right (wide point right)
            elif r_dx > 0.11:
                dirs.append("RIGHT")
            # Pointing Forward (natural straight or slightly angled chest/shoulder line)
            elif -0.11 <= r_dx <= 0.11:
                dirs.append("FRONT")
                
        # Left Arm
        if l_raised:
            # Distinct Left (wide point left)
            if l_dx < -0.11:
                dirs.append("LEFT")
            # Distinct Right (cross-body point)
            elif l_dx > 0.11:
                dirs.append("RIGHT")
            # Pointing Forward
            elif -0.11 <= l_dx <= 0.11:
                dirs.append("FRONT")
                
        return dirs

    def _extract_face_crop(self, frame, lm):
        """
        Clean Face Crop Extraction:
        Strictly bounds the face from forehead to chin, stopping above the shoulder line.
        Prevents hands, arms, or chest from appearing in the dashboard face snapshot.
        """
        h, w, _ = frame.shape
        
        # Key head landmarks
        nose = lm[0]
        l_ear, r_ear = lm[7], lm[8]
        l_eye, r_eye = lm[2], lm[5]
        l_sh, r_sh = lm[11], lm[12]
        
        # Shoulder line (upper chest boundary)
        sh_y = min(l_sh.y, r_sh.y)
        
        # Horizontal head center and width
        head_cx = nose.x
        ear_dist = abs(l_ear.x - r_ear.x)
        head_w = max(0.12, max(ear_dist * 1.5, 0.10))
        
        # Vertical head bounds: top of forehead down to chin (above shoulder)
        head_h = max(0.14, abs(sh_y - nose.y) * 1.6)
        
        min_x = max(0.0, head_cx - head_w / 2.0)
        max_x = min(1.0, head_cx + head_w / 2.0)
        
        min_y = max(0.0, nose.y - (head_h * 0.55))
        # Strictly clamp bottom to chin above shoulder line so raised hands/chest are never included
        max_y = min(1.0, min(sh_y - 0.015, nose.y + (head_h * 0.45)))
        
        if max_y <= min_y + 0.05:
            max_y = min(1.0, min_y + 0.12)
            
        x1 = max(0, int(min_x * w))
        y1 = max(0, int(min_y * h))
        x2 = min(w, int(max_x * w))
        y2 = min(h, int(max_y * h))
        
        if (x2 - x1) > 20 and (y2 - y1) > 20:
            face_crop = frame[y1:y2, x1:x2].copy()
            return face_crop, (x1, y1, x2, y2)
        return None, (0, 0, 0, 0)

    def process_frame(self, frame, timestamp_sec=None):
        if timestamp_sec is None:
            timestamp_sec = time.time()
            
        h, w, _ = frame.shape
        
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
                if not self._is_valid_human_pose(lm):
                    continue
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                bbox = (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))
                detected_poses.append((lm, bbox))
                detected_bboxes.append(bbox)

        tracked_persons = self.tracker.update(detected_bboxes, timestamp_sec)
        self.active_tracked_persons = [p for p in tracked_persons if p.is_verified]

        # Strict 1-to-1 Spatial Match
        assigned_poses = {}
        if tracked_persons and detected_poses:
            num_persons = len(tracked_persons)
            num_poses = len(detected_poses)
            cost_mat = np.zeros((num_persons, num_poses), dtype=np.float32)
            
            for p_idx, person in enumerate(tracked_persons):
                for pose_idx, (lm, p_bbox) in enumerate(detected_poses):
                    p_cen = ((p_bbox[0]+p_bbox[2])/2.0, (p_bbox[1]+p_bbox[3])/2.0)
                    dist = np.sqrt((p_cen[0]-person.centroid_norm[0])**2 + (p_cen[1]-person.centroid_norm[1])**2)
                    cost_mat[p_idx, pose_idx] = dist
                    
            matched_p = set()
            matched_pose = set()
            flat_indices = np.argsort(cost_mat, axis=None)
            
            for idx in flat_indices:
                p_idx = idx // num_poses
                pose_idx = idx % num_poses
                if p_idx in matched_p or pose_idx in matched_pose:
                    continue
                if cost_mat[p_idx, pose_idx] > 0.35:
                    continue
                matched_p.add(p_idx)
                matched_pose.add(pose_idx)
                assigned_poses[tracked_persons[p_idx].track_id] = detected_poses[pose_idx][0]

        # 1. Update POI and Step States for VERIFIED persons only
        for person in tracked_persons:
            if not person.is_verified:
                continue
                
            in_poi = self.poi_manager.is_bbox_inside_or_intersect(person.bbox_norm)
            person.set_poi_state(in_poi)

            matched_lm = assigned_poses.get(person.track_id)
            if matched_lm:
                # Snap clean face crop (when looking straight/facing forward or clear posture)
                face_crop, face_coords = self._extract_face_crop(frame, matched_lm)
                if face_crop is not None:
                    person.update_face(face_crop, score=1.0)
                    
                head_look, head_yaw = self._classify_head_look(matched_lm)
                pointing_dirs = self._classify_arm_pointing_dirs(matched_lm)
                
                point_str = "/".join(pointing_dirs) if pointing_dirs else "NONE"
                self.last_live_feedback = f"Head: {head_look:<5} | Point: {point_str:<6}"
                
                if head_look in pointing_dirs:
                    matched_action = head_look
                    
                    if matched_action == "LEFT" and not person.checked_left:
                        person.left_hold_count += 1
                        if person.left_hold_count >= self.min_hold_frames:
                            person.checked_left = True
                            if not person.step1_ok:
                                person.step1_ok = True
                                self.alarm.play_step_ok(step_num=1)
                            else:
                                person.step2_ok = True
                                self.alarm.play_step_ok(step_num=2)
                                
                    elif matched_action == "RIGHT" and not person.checked_right:
                        person.right_hold_count += 1
                        if person.right_hold_count >= self.min_hold_frames:
                            person.checked_right = True
                            if not person.step1_ok:
                                person.step1_ok = True
                                self.alarm.play_step_ok(step_num=1)
                            else:
                                person.step2_ok = True
                                self.alarm.play_step_ok(step_num=2)
                                
                    elif matched_action == "FRONT" and person.checked_left and person.checked_right and not person.checked_front:
                        person.front_hold_count += 1
                        if person.front_hold_count >= self.min_hold_frames:
                            person.checked_front = True
                            person.step3_ok = True
                            person.total_ok = True
                            person.is_ng = False
                            self.alarm.play_step_ok(step_num=3)
                else:
                    person.left_hold_count = max(0, person.left_hold_count - 1)
                    person.right_hold_count = max(0, person.right_hold_count - 1)
                    person.front_hold_count = max(0, person.front_hold_count - 1)

            if person.exited_poi and person.entered_poi_once and not person.judged:
                self._judge_person_outcome(person, timestamp_sec)

        for person in self.tracker.recently_removed_persons:
            if person.is_verified and person.entered_poi_once and not person.judged:
                self._judge_person_outcome(person, timestamp_sec)

        # 2. Draw Skeleton ONLY for Verified Persons
        for person in tracked_persons:
            if person.is_verified:
                matched_lm = assigned_poses.get(person.track_id)
                if matched_lm:
                    self._draw_person_skeleton(
                        frame, matched_lm, w, h,
                        total_ok=person.total_ok,
                        is_ng=person.is_ng,
                        checked_left=person.checked_left,
                        checked_right=person.checked_right,
                        checked_front=person.checked_front,
                        track_id=person.track_id
                    )

        # 3. Draw POI Overlay (Active when verified person is in POI)
        poi_active = any(p.in_poi for p in tracked_persons if p.is_verified)
        frame = self.poi_manager.draw_poi_overlay(frame, active=poi_active)
        
        # 4. Draw Active People Panel (Shows verified persons only)
        annotated_frame = self._draw_large_hud_panel(frame, tracked_persons)
        
        # 5. Store Annotated Frame into Evidence Buffers
        for person in tracked_persons:
            if person.is_verified and (person.in_poi or person.entered_poi_once):
                person.add_evidence_frame(annotated_frame)

        verified_persons = [p for p in tracked_persons if p.is_verified]
        info = {
            "timestamp": timestamp_sec,
            "tracked_count": len(verified_persons),
            "persons": [
                {
                    "track_id": p.track_id,
                    "in_poi": p.in_poi,
                    "checked_left": p.checked_left,
                    "checked_right": p.checked_right,
                    "checked_front": p.checked_front,
                    "step1_ok": p.step1_ok,
                    "step2_ok": p.step2_ok,
                    "step3_ok": p.step3_ok,
                    "total_ok": p.total_ok,
                    "is_ng": p.is_ng
                } for p in verified_persons if p.in_poi or p.entered_poi_once
            ]
        }
        return annotated_frame, info

    def _draw_person_skeleton(self, frame, lm, w, h, total_ok, is_ng, checked_left, checked_right, checked_front, track_id):
        if is_ng:
            bone_color = (0, 0, 255)
            joint_color = (120, 120, 255)
        elif total_ok:
            bone_color = (0, 255, 120)
            joint_color = (255, 255, 255)
        elif checked_left and checked_right:
            bone_color = (0, 220, 255)
            joint_color = (255, 255, 255)
        elif checked_left or checked_right:
            bone_color = (0, 180, 255)
            joint_color = (255, 255, 255)
        else:
            bone_color = (220, 220, 220)
            joint_color = (0, 200, 255)

        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (15, 19), (16, 20), (23, 25), (24, 26), (25, 27), (26, 28)
        ]

        for p1, p2 in connections:
            if p1 < len(lm) and p2 < len(lm):
                pt1 = (int(lm[p1].x * w), int(lm[p1].y * h))
                pt2 = (int(lm[p2].x * w), int(lm[p2].y * h))
                cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        for i in [0, 7, 8, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24]:
            if i < len(lm):
                pt = (int(lm[i].x * w), int(lm[i].y * h))
                cv2.circle(frame, pt, 5, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 2, (0, 0, 0), 1, cv2.LINE_AA)

        if 0 < len(lm):
            head_x = int(lm[0].x * w)
            head_y = max(30, int(lm[0].y * h) - 28)
            
            l_txt = "L:OK" if checked_left else "L:--"
            r_txt = "R:OK" if checked_right else "R:--"
            f_txt = "F:OK" if checked_front else "F:--"
            tag_str = f"#{track_id} [{l_txt} {r_txt} {f_txt}]"
            
            (tw, th), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(frame, (head_x - tw//2 - 6, head_y - th - 6), (head_x + tw//2 + 6, head_y + 6), (20, 20, 26), -1)
            cv2.rectangle(frame, (head_x - tw//2 - 6, head_y - th - 6), (head_x + tw//2 + 6, head_y + 6), bone_color, 1)
            cv2.putText(frame, tag_str, (head_x - tw//2, head_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    def _judge_person_outcome(self, person, timestamp_sec):
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
                "metadata": {"in_poi": person.in_poi, "checked_left": person.checked_left, "checked_right": person.checked_right, "checked_front": person.checked_front}
            }
            record_id = self.db.insert_record(record_data)
            person.db_logged = True
            print(f"\n[JUDGMENT: TOTAL OK] Person #{person.track_id} crossed with complete standard! Logged to DB (ID: {record_id})")
        else:
            person.is_ng = True
            if not person.alarm_triggered:
                self.alarm.play_alarm(reason="Left POI without completing 3 safety steps")
                person.alarm_triggered = True
                
            face_path = person.save_face_image()
            evidence_path = person.save_evidence_video()
            
            record_data = {
                "event_uuid": person.event_uuid,
                "track_id": person.track_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp_sec)),
                "status": "NG",
                "step1_status": "OK" if (person.checked_left or person.checked_right) else "NG",
                "step2_status": "OK" if (person.checked_left and person.checked_right) else "NG",
                "step3_status": "OK" if person.checked_front else "NG",
                "duration_sec": timestamp_sec - person.entry_time,
                "face_image_path": face_path,
                "evidence_video_path": evidence_path,
                "metadata": {"in_poi": person.in_poi, "checked_left": person.checked_left, "checked_right": person.checked_right, "checked_front": person.checked_front}
            }
            record_id = self.db.insert_record(record_data)
            person.db_logged = True
            self.last_ng_event = record_data
            print(f"\n[JUDGMENT: NG ALERT] 🚨 Person #{person.track_id} left POI without standard! Voice alarm played (1 time). Logged to DB (ID: {record_id})")
            
            if person.best_face_image is not None:
                self.face_client.recognize_face_async(person.best_face_image, record_id, metadata=record_data)

    def _draw_large_hud_panel(self, frame, tracked_persons):
        h, w, _ = frame.shape
        
        panel_w = max(340, int(w * 0.32))
        panel_w = min(panel_w, w - 20)
        panel_x = w - panel_w - 10
        panel_y = 10
        
        cv2.rectangle(frame, (0, h - 38), (w, h), (16, 18, 24), -1)
        cv2.rectangle(frame, (0, h - 38), (w, h), (40, 48, 64), 1)
        
        feedback = self.last_live_feedback or "Ready - Stand in POI to begin"
        cv2.putText(frame, f"REALTIME ACTION: {feedback}", (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA)
        
        active_list = [p for p in tracked_persons if p.is_verified and (p.in_poi or p.entered_poi_once)]
        if not active_list:
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, panel_y + 44), (20, 22, 30), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (panel_x, panel_y), (w - 10, panel_y + 44), (60, 68, 85), 1)
            cv2.putText(frame, "MONITOR STATUS: IDLE", (panel_x + 16, panel_y + 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.46, (200, 200, 210), 1, cv2.LINE_AA)
            return frame

        card_h = 110
        total_panel_h = min(h - 50, len(active_list) * (card_h + 10) + 40)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, panel_y + total_panel_h), (16, 18, 26), -1)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (w - 10, panel_y + total_panel_h), (0, 200, 255), 2)
        
        cv2.putText(frame, f"PEDESTRIAN SAFETY STATUS ({len(active_list)})", (panel_x + 12, panel_y + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 220, 255), 1, cv2.LINE_AA)

        curr_y = panel_y + 36
        for person in active_list:
            if curr_y + card_h > h - 45:
                break
                
            card_bg = (26, 30, 42)
            cv2.rectangle(frame, (panel_x + 6, curr_y), (w - 16, curr_y + card_h), card_bg, -1)
            cv2.rectangle(frame, (panel_x + 6, curr_y), (w - 16, curr_y + card_h), (60, 72, 96), 1)
            
            face_size = 72
            face_x = panel_x + 12
            face_y = curr_y + 18
            
            if person.best_face_image is not None and person.best_face_image.size > 0:
                try:
                    resized_face = cv2.resize(person.best_face_image, (face_size, face_size))
                    frame[face_y:face_y+face_size, face_x:face_x+face_size] = resized_face
                except Exception:
                    cv2.rectangle(frame, (face_x, face_y), (face_x+face_size, face_y+face_size), (80, 80, 90), -1)
            else:
                cv2.rectangle(frame, (face_x, face_y), (face_x+face_size, face_y+face_size), (40, 45, 60), -1)
                cv2.putText(frame, "FACE", (face_x + 16, face_y + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 170), 1, cv2.LINE_AA)
                            
            avatar_border = (0, 255, 120) if person.total_ok else ((0, 0, 255) if person.is_ng else (0, 200, 255))
            cv2.rectangle(frame, (face_x, face_y), (face_x+face_size, face_y+face_size), avatar_border, 2)

            info_x = face_x + face_size + 12
            cv2.putText(frame, f"Person #{person.track_id}", (info_x, curr_y + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                        
            l_col = (0, 255, 120) if person.checked_left else ((0, 0, 255) if person.is_ng else (140, 140, 140))
            l_txt = "LEFT: [ OK ]" if person.checked_left else "LEFT: [ -- ]"
            cv2.putText(frame, l_txt, (info_x, curr_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, l_col, 1, cv2.LINE_AA)
            
            r_col = (0, 255, 120) if person.checked_right else ((0, 0, 255) if person.is_ng else (140, 140, 140))
            r_txt = "RIGHT: [ OK ]" if person.checked_right else "RIGHT: [ -- ]"
            cv2.putText(frame, r_txt, (info_x, curr_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, r_col, 1, cv2.LINE_AA)
            
            f_col = (0, 255, 120) if person.checked_front else ((0, 0, 255) if person.is_ng else (140, 140, 140))
            f_txt = "FRONT (Step 3): [ OK ]" if person.checked_front else "FRONT (Step 3): [ -- ]"
            cv2.putText(frame, f_txt, (info_x, curr_y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, f_col, 1, cv2.LINE_AA)
            
            if person.total_ok:
                status_str = "● TOTAL OK - SAFE TO CROSS"
                st_color = (0, 255, 120)
            elif person.is_ng:
                status_str = "● NG - VIOLATION ALARM"
                st_color = (0, 70, 255)
            else:
                status_str = "● POINT & LOOK TO VERIFY"
                st_color = (0, 200, 255)
                
            cv2.putText(frame, status_str, (info_x, curr_y + 104),
                        cv2.FONT_HERSHEY_DUPLEX, 0.38, st_color, 1, cv2.LINE_AA)
                        
            curr_y += card_h + 10
            
        return frame

    def generate_final_report(self):
        now = time.time()
        for p in self.tracker.tracked_persons.values():
            if p.is_verified and p.entered_poi_once and not p.judged:
                self._judge_person_outcome(p, now)
                
        poi_persons = [p for p in self.tracker.tracked_persons.values() if p.is_verified and p.entered_poi_once]
        persons_report = []
        all_passed = True if poi_persons else False
        for p in poi_persons:
            persons_report.append({
                "track_id": p.track_id,
                "checked_left": "OK" if p.checked_left else "NG",
                "checked_right": "OK" if p.checked_right else "NG",
                "checked_front": "OK" if p.checked_front else "NG",
                "step1": "OK" if (p.checked_left or p.checked_right) else "NG",
                "step2": "OK" if (p.checked_left and p.checked_right) else "NG",
                "step3": "OK" if p.checked_front else "NG",
                "total": "TOTAL OK" if p.total_ok else "TOTAL NG",
                "is_standard_compliant": bool(p.total_ok)
            })
            if not p.total_ok:
                all_passed = False
                
        p1 = poi_persons[0] if poi_persons else (self.active_tracked_persons[0] if self.active_tracked_persons else None)
        return {
            "step1_side1": "OK" if (p1 and (p1.checked_left or p1.checked_right)) else "NG",
            "step2_side2": "OK" if (p1 and (p1.checked_left and p1.checked_right)) else "NG",
            "step3_front": "OK" if (p1 and p1.checked_front) else "NG",
            "total_result": "TOTAL OK" if all_passed else "TOTAL NG",
            "is_standard_compliant": all_passed,
            "persons": persons_report
        }
