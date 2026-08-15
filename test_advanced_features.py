import cv2
import os
import json
import time
from safety_checker import RoadCrossingSafetyChecker
from db_manager import CrossingDatabase
from poi_manager import POIManager

db_path = "data/test_records.db"
if os.path.exists(db_path):
    os.remove(db_path)

db = CrossingDatabase(db_path=db_path)
poi = POIManager(config_path="poi_config.json")

print("==================================================")
print("TEST 1: POI Point & Bounding Box Check")
print("==================================================")
# Test center point
assert poi.is_point_inside(0.5, 0.5) == True
# Test bbox in center
assert poi.is_bbox_inside_or_intersect((0.2, 0.2, 0.8, 0.8)) == True
print("POI test PASSED!")

print("\n==================================================")
print("TEST 2: Running Safety Checker on 1.mov (PASS case)")
print("==================================================")
checker = RoadCrossingSafetyChecker(
    min_hold_frames=6,
    db_path=db_path,
    enable_alarm=False,
    max_crossing_wait_sec=5.0
)

cap = cv2.VideoCapture("/Users/m4ck/Desktop/1.mov")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    t_sec = frame_idx / fps
    checker.process_frame(frame, timestamp_sec=t_sec)
    frame_idx += 1
cap.release()

report = checker.generate_final_report()
print("Final Report 1.mov:", report)
assert report["total_result"] == "TOTAL OK"
print("Test 2 PASSED (1.mov completed with TOTAL OK)!")

print("\n==================================================")
print("TEST 3: Running Incomplete / NG Case (Stopped early)")
print("==================================================")
checker_ng = RoadCrossingSafetyChecker(
    min_hold_frames=6,
    db_path=db_path,
    enable_alarm=True,
    max_crossing_wait_sec=1.5 # trigger NG quickly when not passing
)

# Process only first 2.5 seconds of 1.mov (Step 1 done, Step 2 & 3 not completed)
cap = cv2.VideoCapture("/Users/m4ck/Desktop/1.mov")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    t_sec = frame_idx / fps
    if t_sec > 2.2: # stop before Step 2 completes
        # Send a few extra frames to trigger NG timeout
        for extra in range(10):
            checker_ng.process_frame(frame, timestamp_sec=t_sec + 2.0 + extra*0.1)
        break
    checker_ng.process_frame(frame, timestamp_sec=t_sec)
    frame_idx += 1
cap.release()

print("\n==================================================")
print("TEST 4: Inspecting SQLite Database Records & Files")
print("==================================================")
records = db.get_records()
print(f"Total Database Records Logged: {len(records)}")
for r in records:
    print(f"  Record ID {r['id']} | Track #{r['track_id']} | Status: {r['status']} | Steps: (L={r['step1_status']}, R={r['step2_status']}, F={r['step3_status']}) | Face: {r['face_image_path']} | Video: {r['evidence_video_path']} | ExtID: {r['external_person_id']}")

# Check that face snapshots and evidence videos exist
faces = [f for f in os.listdir("data/faces") if f.endswith(".jpg")]
evidence = [f for f in os.listdir("data/evidence") if f.endswith(".mp4")]
print(f"\nSnapped Face Images: {len(faces)} files -> {faces}")
print(f"Evidence Video Clips: {len(evidence)} files -> {evidence}")

assert len(records) >= 1
assert len(faces) >= 1
print("\n==================================================")
print("ALL ADVANCED FEATURE TESTS PASSED SUCCESSFULLY! 🎯")
print("==================================================")
