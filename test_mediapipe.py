import cv2
import mediapipe as mp
import numpy as np

print("MediaPipe version:", mp.__version__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1, # 0 for super fast (Raspberry Pi/low spec), 1 for balanced
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture("/Users/m4ck/Desktop/1.mov")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

print("Testing video 1.mov with MediaPipe Pose...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # Nose (0), Left Eye (2), Right Eye (5), Left Ear (7), Right Ear (8)
        # Left Shoulder (11), Right Shoulder (12), Left Elbow (13), Right Elbow (14)
        # Left Wrist (15), Right Wrist (16), Left Index (19), Right Index (20)
        nose = lm[mp_pose.PoseLandmark.NOSE]
        l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
        r_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]
        l_eye = lm[mp_pose.PoseLandmark.LEFT_EYE]
        r_eye = lm[mp_pose.PoseLandmark.RIGHT_EYE]
        
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
        r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        l_index = lm[mp_pose.PoseLandmark.LEFT_INDEX]
        r_index = lm[mp_pose.PoseLandmark.RIGHT_INDEX]
        
        # Calculate yaw / head direction
        # In image coordinates: x increases to the right (screen right = subject's left if facing camera)
        # Head rotation: nose.x relative to mid_ear or eye distances
        ear_mid_x = (l_ear.x + r_ear.x) / 2.0
        ear_mid_z = (l_ear.z + r_ear.z) / 2.0
        # When facing camera: nose is roughly between ears.
        # If looking camera left (subject right): nose moves left, r_ear moves back (z higher or lower), etc.
        # Let's inspect raw values at 0.5s intervals
        if frame_idx % int(fps * 0.4) == 0:
            t = frame_idx / fps
            print(f"t={t:.2f}s | Nose: ({nose.x:.2f}, {nose.y:.2f}, {nose.z:.2f}) | L_Ear: {l_ear.x:.2f}, R_Ear: {r_ear.x:.2f} | "
                  f"R_Wrist: ({r_wrist.x:.2f}, {r_wrist.y:.2f}) | L_Wrist: ({l_wrist.x:.2f}, {l_wrist.y:.2f})")
            
    frame_idx += 1

cap.release()
pose.close()
