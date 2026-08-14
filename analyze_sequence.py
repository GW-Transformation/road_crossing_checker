import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

model_path = "/Users/m4ck/projects/road_crossing_checker/pose_landmarker_lite.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

for vid_name in ["1.mov", "2.mov"]:
    vid_path = f"/Users/m4ck/Desktop/{vid_name}"
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n================ FULL ANALYSIS: {vid_name} ================")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            lm = detection_result.pose_landmarks[0]
            # Key landmarks:
            # 0: nose, 2: left_eye, 5: right_eye, 7: left_ear, 8: right_ear
            # 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist
            nose = lm[0]
            l_ear, r_ear = lm[7], lm[8]
            l_eye, r_eye = lm[2], lm[5]
            l_sh, r_sh = lm[11], lm[12]
            l_el, r_el = lm[13], lm[14]
            l_wr, r_wr = lm[15], lm[16]
            l_idx, r_idx = lm[19], lm[20]
            
            # Head yaw calculation:
            # Ear midpoint
            ear_mid_x = (l_ear.x + r_ear.x) / 2.0
            ear_width = abs(l_ear.x - r_ear.x) + 1e-6
            # Ratio of nose offset from ear midpoint relative to ear width
            head_yaw_ratio = (nose.x - ear_mid_x) / ear_width
            
            # Eye yaw ratio
            eye_mid_x = (l_eye.x + r_eye.x) / 2.0
            eye_width = abs(l_eye.x - r_eye.x) + 1e-6
            eye_yaw_ratio = (nose.x - eye_mid_x) / eye_width

            # Arm pointing direction:
            # Right arm vector (shoulder to wrist / index)
            r_arm_dx = r_wr.x - r_sh.x
            r_arm_dy = r_wr.y - r_sh.y
            r_arm_dz = r_wr.z - r_sh.z
            
            # Left arm vector
            l_arm_dx = l_wr.x - l_sh.x
            l_arm_dy = l_wr.y - l_sh.y
            l_arm_dz = l_wr.z - l_sh.z
            
            # Active arm: pick the one that is raised (wrist significantly higher than resting or extended)
            # Resting wrist is around y > 0.6. Raised arm has wrist y < 0.5
            r_raised = r_wr.y < (r_sh.y + 0.15)
            l_raised = l_wr.y < (l_sh.y + 0.15)
            
            t = frame_idx / fps
            if frame_idx % 8 == 0:
                print(f"t={t:4.2f}s | HeadYaw: {head_yaw_ratio:+5.2f} (Nose.x={nose.x:.2f}) | "
                      f"R_Arm: dx={r_arm_dx:+5.2f}, dy={r_arm_dy:+5.2f}, dz={r_arm_dz:+5.2f}, raised={r_raised} | "
                      f"L_Arm: dx={l_arm_dx:+5.2f}, dy={l_arm_dy:+5.2f}, dz={l_arm_dz:+5.2f}, raised={l_raised}")
                
        frame_idx += 1
    cap.release()
