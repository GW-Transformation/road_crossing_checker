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
    print(f"\n================ Processing {vid_name} (fps={fps:.1f}, frames={total_frames}) ================")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            landmarks = detection_result.pose_landmarks[0]
            # 0: nose, 7: left_ear, 8: right_ear, 11: left_shoulder, 12: right_shoulder
            # 13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist, 19: left_index, 20: right_index
            nose = landmarks[0]
            l_ear = landmarks[7]
            r_ear = landmarks[8]
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            l_wrist = landmarks[15]
            r_wrist = landmarks[16]
            l_index = landmarks[19]
            r_index = landmarks[20]
            
            # Print sample every 0.3 seconds
            if frame_idx % int(fps * 0.3) == 0:
                t = frame_idx / fps
                ear_dx = l_ear.x - r_ear.x
                nose_x = nose.x
                
                # Head look indicator: relative position of nose between ears
                # When looking left/right:
                print(f"t={t:.2f}s | Nose: ({nose.x:.2f}, {nose.y:.2f}) | Ears(L/R): ({l_ear.x:.2f} / {r_ear.x:.2f}, dx={ear_dx:.2f}) | "
                      f"R_Wrist: ({r_wrist.x:.2f}, {r_wrist.y:.2f}) | L_Wrist: ({l_wrist.x:.2f}, {l_wrist.y:.2f})")
        frame_idx += 1
    cap.release()
