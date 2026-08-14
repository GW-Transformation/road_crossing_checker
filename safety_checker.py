import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import argparse
import json

class RoadCrossingSafetyChecker:
    """
    Pedestrian Road Crossing Standard Safety Checker
    Checks:
      Step 1: Look Left AND Point Left simultaneously
      Step 2: Look Right AND Point Right simultaneously
      Step 3: Look Forward AND Point Forward simultaneously
      Total: All 3 steps completed in order -> Total OK (PASS)
    """
    def __init__(self, model_path=None, min_hold_frames=8, reset_timeout_sec=8.0):
        if model_path is None:
            # Default model path in same directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "pose_landmarker_lite.task")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.min_hold_frames = min_hold_frames
        self.reset_timeout_sec = reset_timeout_sec
        self.reset()

    def reset(self):
        self.step1_ok = False
        self.step2_ok = False
        self.step3_ok = False
        self.total_ok = False
        
        self.step1_time = None
        self.step2_time = None
        self.step3_time = None
        
        self.step1_hold_count = 0
        self.step2_hold_count = 0
        self.step3_hold_count = 0
        
        self.last_action_timestamp = time.time()
        self.status_message = "Waiting for pedestrian..."

    def _classify_head_look(self, nose, l_ear, r_ear, l_eye, r_eye):
        """
        Classifies head orientation into: 'LEFT', 'RIGHT', 'FORWARD', or 'UNKNOWN'
        Coordinate system (normalized 0 to 1):
          x: 0 (camera left) -> 1 (camera right)
        """
        ear_mid_x = (l_ear.x + r_ear.x) / 2.0
        ear_width = abs(l_ear.x - r_ear.x) + 1e-6
        head_yaw_ratio = (nose.x - ear_mid_x) / ear_width
        
        # When looking camera-left: nose is noticeably left of ear center (head_yaw_ratio < -0.4)
        # When looking camera-right: nose is noticeably right of ear center (head_yaw_ratio > +0.25)
        # When looking forward: nose is between ears (-0.35 <= head_yaw_ratio <= 0.20)
        if head_yaw_ratio < -0.40:
            return "LEFT", head_yaw_ratio
        elif head_yaw_ratio > 0.25:
            return "RIGHT", head_yaw_ratio
        elif -0.38 <= head_yaw_ratio <= 0.22:
            return "FORWARD", head_yaw_ratio
        else:
            return "CENTER", head_yaw_ratio

    def _classify_arm_point(self, lm):
        """
        Classifies pointing gesture for active arm (Right or Left).
        Landmarks:
          11: L_Shoulder, 12: R_Shoulder
          13: L_Elbow, 14: R_Elbow
          15: L_Wrist, 16: R_Wrist
          19: L_Index, 20: R_Index
        """
        l_sh, r_sh = lm[11], lm[12]
        l_wr, r_wr = lm[15], lm[16]
        l_idx, r_idx = lm[19], lm[20]
        
        # Check if right arm is raised / active
        # Hand is active if wrist is near or above shoulder level
        r_raised = (r_wr.y < (r_sh.y + 0.18))
        l_raised = (l_wr.y < (l_sh.y + 0.18))
        
        # Direction from shoulder to wrist/index
        r_dx = r_wr.x - r_sh.x
        r_dy = r_wr.y - r_sh.y
        r_dz = r_wr.z - r_sh.z
        
        l_dx = l_wr.x - l_sh.x
        l_dy = l_wr.y - l_sh.y
        l_dz = l_wr.z - l_sh.z
        
        # Priority to the arm that is clearly active/extended
        active_arm = None
        dx, dy, dz = 0.0, 0.0, 0.0
        
        if r_raised and not l_raised:
            active_arm = "RIGHT_ARM"
            dx, dy, dz = r_dx, r_dy, r_dz
        elif l_raised and not r_raised:
            active_arm = "LEFT_ARM"
            dx, dy, dz = l_dx, l_dy, l_dz
        elif r_raised and l_raised:
            # Pick arm with larger horizontal extension or forward projection
            if abs(r_dx) >= abs(l_dx):
                active_arm = "RIGHT_ARM"
                dx, dy, dz = r_dx, r_dy, r_dz
            else:
                active_arm = "LEFT_ARM"
                dx, dy, dz = l_dx, l_dy, l_dz
        else:
            return "NONE", None, (0.0, 0.0, 0.0)

        # Classification based on dx (horizontal vector) and dz (forward depth)
        # Point Left: dx < -0.10
        # Point Right: dx > +0.12
        # Point Forward: |dx| <= 0.12 and arm is raised pointing towards front
        if dx < -0.10:
            return "LEFT", active_arm, (dx, dy, dz)
        elif dx > 0.12:
            return "RIGHT", active_arm, (dx, dy, dz)
        elif abs(dx) <= 0.12 and dy < 0.20:
            return "FORWARD", active_arm, (dx, dy, dz)
        else:
            return "UNKNOWN", active_arm, (dx, dy, dz)

    def process_frame(self, frame, timestamp_sec=None):
        """
        Process a single BGR image frame.
        Returns:
          annotated_frame: Frame with drawn overlays and safety checklist
          info_dict: Detection details and current step states
        """
        if timestamp_sec is None:
            timestamp_sec = time.time()
            
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        head_look = "NONE"
        point_dir = "NONE"
        head_yaw = 0.0
        arm_vec = (0.0, 0.0, 0.0)
        active_arm = "NONE"
        person_detected = False
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            person_detected = True
            lm = detection_result.pose_landmarks[0]
            
            nose = lm[0]
            l_eye, r_eye = lm[2], lm[5]
            l_ear, r_ear = lm[7], lm[8]
            
            head_look, head_yaw = self._classify_head_look(nose, l_ear, r_ear, l_eye, r_eye)
            point_dir, active_arm, arm_vec = self._classify_arm_point(lm)
            
            # Sequence State Machine
            # Step 1: Look Left AND Point Left
            if not self.step1_ok:
                if head_look == "LEFT" and point_dir == "LEFT":
                    self.step1_hold_count += 1
                    if self.step1_hold_count >= self.min_hold_frames:
                        self.step1_ok = True
                        self.step1_time = timestamp_sec
                        self.status_message = "Step 1 OK (Look Left + Point Left)"
                else:
                    self.step1_hold_count = max(0, self.step1_hold_count - 1)
            
            # Step 2: Look Right AND Point Right (after Step 1)
            elif self.step1_ok and not self.step2_ok:
                if head_look == "RIGHT" and point_dir == "RIGHT":
                    self.step2_hold_count += 1
                    if self.step2_hold_count >= self.min_hold_frames:
                        self.step2_ok = True
                        self.step2_time = timestamp_sec
                        self.status_message = "Step 2 OK (Look Right + Point Right)"
                else:
                    self.step2_hold_count = max(0, self.step2_hold_count - 1)
                    
            # Step 3: Look Forward AND Point Forward (after Step 2)
            elif self.step2_ok and not self.step3_ok:
                if head_look in ["FORWARD", "CENTER"] and point_dir == "FORWARD":
                    self.step3_hold_count += 1
                    if self.step3_hold_count >= self.min_hold_frames:
                        self.step3_ok = True
                        self.step3_time = timestamp_sec
                        self.total_ok = True
                        self.status_message = "STANDARD COMPLETE - TOTAL OK! SAFE TO CROSS"
                else:
                    self.step3_hold_count = max(0, self.step3_hold_count - 1)
                    
            # Draw Skeleton Landmarks
            self._draw_skeleton(frame, lm, w, h)
        else:
            self.status_message = "Searching for pedestrian..."

        # Render Professional UI Overlay
        annotated_frame = self._draw_ui_overlay(frame, head_look, point_dir, head_yaw, arm_vec, person_detected)
        
        info = {
            "timestamp": timestamp_sec,
            "person_detected": person_detected,
            "head_look": head_look,
            "point_dir": point_dir,
            "head_yaw": head_yaw,
            "active_arm": active_arm,
            "step1_ok": self.step1_ok,
            "step2_ok": self.step2_ok,
            "step3_ok": self.step3_ok,
            "total_ok": self.total_ok,
            "status_message": self.status_message
        }
        return annotated_frame, info

    def _draw_skeleton(self, frame, lm, w, h):
        connections = [
            (11, 12), # shoulders
            (11, 13), (13, 15), # left arm
            (12, 14), (14, 16), # right arm
            (11, 23), (12, 24), (23, 24), # torso
            (0, 1), (1, 2), (2, 3), (3, 7), # face left
            (0, 4), (4, 5), (5, 6), (6, 8), # face right
            (15, 19), (16, 20) # hands
        ]
        
        color = (0, 255, 0) if self.total_ok else (0, 220, 255)
        
        for p1_idx, p2_idx in connections:
            if p1_idx < len(lm) and p2_idx < len(lm):
                pt1 = (int(lm[p1_idx].x * w), int(lm[p1_idx].y * h))
                pt2 = (int(lm[p2_idx].x * w), int(lm[p2_idx].y * h))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
                
        for i in [0, 7, 8, 11, 12, 13, 14, 15, 16, 19, 20]:
            if i < len(lm):
                pt = (int(lm[i].x * w), int(lm[i].y * h))
                cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 2, (0, 100, 255), -1, cv2.LINE_AA)

    def _draw_ui_overlay(self, frame, head_look, point_dir, head_yaw, arm_vec, person_detected):
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Semi-transparent top bar
        bar_height = 140
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (20, 20, 25), -1)
        
        # Semi-transparent bottom status bar
        cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 25), -1)
        
        alpha = 0.82
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Header text
        cv2.putText(frame, "PEDESTRIAN CROSSING SAFETY CHECK", (12, 22), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Step Badges
        # Step 1
        s1_color = (50, 220, 50) if self.step1_ok else (100, 100, 100)
        s1_text = "Step 1: Look & Point LEFT   [OK]" if self.step1_ok else "Step 1: Look & Point LEFT   [--]"
        cv2.putText(frame, s1_text, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, s1_color, 1, cv2.LINE_AA)
        
        # Step 2
        s2_color = (50, 220, 50) if self.step2_ok else (100, 100, 100)
        s2_text = "Step 2: Look & Point RIGHT  [OK]" if self.step2_ok else "Step 2: Look & Point RIGHT  [--]"
        cv2.putText(frame, s2_text, (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.44, s2_color, 1, cv2.LINE_AA)
        
        # Step 3
        s3_color = (50, 220, 50) if self.step3_ok else (100, 100, 100)
        s3_text = "Step 3: Look & Point FRONT  [OK]" if self.step3_ok else "Step 3: Look & Point FRONT  [--]"
        cv2.putText(frame, s3_text, (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.44, s3_color, 1, cv2.LINE_AA)
        
        # Total Result Box
        if self.total_ok:
            total_bg = (30, 180, 30)
            total_label = "TOTAL: OK (SAFE TO CROSS)"
            text_color = (255, 255, 255)
        elif self.step1_ok or self.step2_ok:
            total_bg = (0, 140, 240)
            total_label = "TOTAL: IN PROGRESS..."
            text_color = (255, 255, 255)
        else:
            total_bg = (50, 50, 60)
            total_label = "TOTAL: PENDING CHECK"
            text_color = (200, 200, 200)
            
        cv2.rectangle(frame, (10, 100), (w - 10, 130), total_bg, -1)
        cv2.rectangle(frame, (10, 100), (w - 10, 130), (255, 255, 255), 1)
        cv2.putText(frame, total_label, (20, 121), cv2.FONT_HERSHEY_DUPLEX, 0.50, text_color, 1, cv2.LINE_AA)
        
        # Live HUD Status at Bottom
        live_str = f"Look: {head_look:<7} | Point: {point_dir:<7}"
        cv2.putText(frame, live_str, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)
        
        return frame

    def generate_final_report(self):
        """Returns structured JSON summary of check session"""
        return {
            "step1_look_point_left": "OK" if self.step1_ok else "NG",
            "step2_look_point_right": "OK" if self.step2_ok else "NG",
            "step3_look_point_forward": "OK" if self.step3_ok else "NG",
            "total_result": "TOTAL OK" if self.total_ok else "TOTAL NG",
            "is_standard_compliant": bool(self.total_ok),
            "step1_timestamp": self.step1_time,
            "step2_timestamp": self.step2_time,
            "step3_timestamp": self.step3_time
        }
