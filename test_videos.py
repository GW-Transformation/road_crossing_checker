import cv2
import json
import time
import os
from safety_checker import RoadCrossingSafetyChecker

videos = [
    ("/Users/m4ck/Desktop/1.mov", "/Users/m4ck/projects/road_crossing_checker/output_1.mp4"),
    ("/Users/m4ck/Desktop/2.mov", "/Users/m4ck/projects/road_crossing_checker/output_2.mp4")
]

results = {}

for vid_in, vid_out in videos:
    vid_name = os.path.basename(vid_in)
    print(f"\n=======================================================")
    print(f"▶ TESTING VIDEO: {vid_name}")
    print(f"=======================================================")
    
    cap = cv2.VideoCapture(vid_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(vid_out, fourcc, fps, (width, height))
    
    checker = RoadCrossingSafetyChecker(min_hold_frames=6)
    
    frame_idx = 0
    start_t = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        t_sec = frame_idx / fps
        annotated_frame, info = checker.process_frame(frame, timestamp_sec=t_sec)
        out_writer.write(annotated_frame)
        
        if frame_idx % int(fps * 0.5) == 0:
            print(f"[{vid_name} t={t_sec:4.2f}s] Look: {info['head_look']:<7} | Point: {info['point_dir']:<7} | "
                  f"S1: {'OK' if info['step1_ok'] else '--'} | "
                  f"S2: {'OK' if info['step2_ok'] else '--'} | "
                  f"S3: {'OK' if info['step3_ok'] else '--'} | "
                  f"Total: {'OK' if info['total_ok'] else '--'}")
                  
        frame_idx += 1
        
    cap.release()
    out_writer.release()
    
    report = checker.generate_final_report()
    results[vid_name] = report
    
    print(f"\n--- RESULTS FOR {vid_name} ---")
    print(f"  Step 1 (Look & Point Left)   : {report['step1_look_point_left']} (at {report['step1_timestamp']}s)")
    print(f"  Step 2 (Look & Point Right)  : {report['step2_look_point_right']} (at {report['step2_timestamp']}s)")
    print(f"  Step 3 (Look & Point Forward): {report['step3_look_point_forward']} (at {report['step3_timestamp']}s)")
    print(f"  TOTAL RESULT                 : {report['total_result']}")
    print(f"  Annotated Video Saved        : {vid_out}")

print("\n=======================================================")
print("FINAL SUMMARY REPORT:")
print(json.dumps(results, indent=2))
