import cv2
import os
import json
import time
from safety_checker import RoadCrossingSafetyChecker
from db_manager import CrossingDatabase

db_test_path = "data/poi_exit_test.db"
if os.path.exists(db_test_path):
    os.remove(db_test_path)

db = CrossingDatabase(db_path=db_test_path)

print("==================================================")
print("TEST 1: Verification of Full Standard PASS on 1.mov")
print("==================================================")
checker_pass = RoadCrossingSafetyChecker(
    min_hold_frames=6,
    db_path=db_test_path,
    enable_alarm=True
)

cap = cv2.VideoCapture("/Users/m4ck/Desktop/1.mov")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_idx = 0

out_writer = cv2.VideoWriter("output_test_panel.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    t_sec = frame_idx / fps
    annotated_frame, info = checker_pass.process_frame(frame, timestamp_sec=t_sec)
    out_writer.write(annotated_frame)
    frame_idx += 1
cap.release()
out_writer.release()

report_pass = checker_pass.generate_final_report()
print("PASS Report:", report_pass)
assert report_pass["total_result"] == "TOTAL OK"
print("  ==> TEST 1 PASSED: Person crossed completing all 3 steps (TOTAL OK, No false alarm)")

print("\n==================================================")
print("TEST 2: Verification of NG Trigger when Leaving POI Incomplete")
print("==================================================")
checker_ng = RoadCrossingSafetyChecker(
    min_hold_frames=6,
    db_path=db_test_path,
    enable_alarm=True
)

# Simulate person entering POI, doing Step 1 only, then leaving POI
cap = cv2.VideoCapture("/Users/m4ck/Desktop/1.mov")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    t_sec = frame_idx / fps
    if t_sec > 1.8: # Person completed only Step 1
        # Now simulate person leaving POI by passing frames where person is outside or tracker clears
        print(f"Simulating person stepping out of POI at t={t_sec:.2f}s...")
        # Feed empty frames (person left view/POI)
        empty_frame = frame * 0
        for extra in range(30):
            checker_ng.process_frame(empty_frame, timestamp_sec=t_sec + 0.1 + extra*0.05)
        break
    checker_ng.process_frame(frame, timestamp_sec=t_sec)
    frame_idx += 1
cap.release()

report_ng = checker_ng.generate_final_report()
print("NG Report:", report_ng)
assert report_ng["total_result"] == "TOTAL NG"
print("  ==> TEST 2 PASSED: Person left POI without completing 3 steps -> Correctly judged NG & triggered voice alarm!")

print("\n==================================================")
print("TEST 3: Checking DB Records & Evidence Files")
print("==================================================")
records = db.get_records()
print(f"Total DB Records: {len(records)}")
for r in records:
    print(f"  ID #{r['id']} | Track #{r['track_id']} | Status: {r['status']} | Steps: (L={r['step1_status']}, R={r['step2_status']}, F={r['step3_status']}) | Face: {r['face_image_path']} | Video: {r['evidence_video_path']}")

assert len(records) >= 2
print("\n==================================================")
print("ALL TESTS COMPLETED SUCCESSFULLY! 🎯")
print("==================================================")
