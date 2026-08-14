import cv2
import json
import os
from safety_checker import RoadCrossingSafetyChecker

def test_full_video(vid_path):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    checker = RoadCrossingSafetyChecker(min_hold_frames=6)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        checker.process_frame(frame, timestamp_sec=frame_idx / fps)
        frame_idx += 1
    cap.release()
    return checker.generate_final_report()

def test_partial_video(vid_path, stop_at_sec):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    checker = RoadCrossingSafetyChecker(min_hold_frames=6)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = frame_idx / fps
        if t_sec > stop_at_sec:
            break
        checker.process_frame(frame, timestamp_sec=t_sec)
        frame_idx += 1
    cap.release()
    return checker.generate_final_report()

print("==================================================")
print("RUNNING EDGE-CASE AND NG/OK VERIFICATION TESTS")
print("==================================================")

# Test 1: Full 1.mov -> All OK
r1 = test_full_video("/Users/m4ck/Desktop/1.mov")
print("[TEST 1] Full 1.mov:")
print(" ", r1)
assert r1["step1_look_point_left"] == "OK"
assert r1["step2_look_point_right"] == "OK"
assert r1["step3_look_point_forward"] == "OK"
assert r1["total_result"] == "TOTAL OK"
print("  ==> TEST 1 PASSED (All Steps OK)")

# Test 2: Partial 1.mov (stopped at 1.8s, only step 1 done) -> Step 1 OK, Step 2 NG, Step 3 NG, Total NG
r2 = test_partial_video("/Users/m4ck/Desktop/1.mov", stop_at_sec=1.8)
print("\n[TEST 2] Partial 1.mov (Stopped at 1.8s - only Step 1 done):")
print(" ", r2)
assert r2["step1_look_point_left"] == "OK"
assert r2["step2_look_point_right"] == "NG"
assert r2["step3_look_point_forward"] == "NG"
assert r2["total_result"] == "TOTAL NG"
print("  ==> TEST 2 PASSED (Correctly shows Step 2 NG, Step 3 NG, Total NG)")

# Test 3: Partial 1.mov (stopped at 3.0s, Step 1 and 2 done) -> Step 1 OK, Step 2 OK, Step 3 NG, Total NG
r3 = test_partial_video("/Users/m4ck/Desktop/1.mov", stop_at_sec=3.0)
print("\n[TEST 3] Partial 1.mov (Stopped at 3.0s - Step 1 & 2 done, Step 3 missing):")
print(" ", r3)
assert r3["step1_look_point_left"] == "OK"
assert r3["step2_look_point_right"] == "OK"
assert r3["step3_look_point_forward"] == "NG"
assert r3["total_result"] == "TOTAL NG"
print("  ==> TEST 3 PASSED (Correctly shows Step 3 NG, Total NG)")

# Test 4: Full 2.mov -> All OK
r4 = test_full_video("/Users/m4ck/Desktop/2.mov")
print("\n[TEST 4] Full 2.mov:")
print(" ", r4)
assert r4["step1_look_point_left"] == "OK"
assert r4["step2_look_point_right"] == "OK"
assert r4["step3_look_point_forward"] == "OK"
assert r4["total_result"] == "TOTAL OK"
print("  ==> TEST 4 PASSED (All Steps OK)")

print("\n==================================================")
print("ALL TESTS PASSED WITH 100% SPECIFICATION MATCH!")
print("==================================================")
