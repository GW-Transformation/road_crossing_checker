#!/usr/bin/env python3
"""
Pedestrian Road Crossing Safety Standard Checker
================================================
Checks if a pedestrian complies with the safety standard before crossing a road:
  - Step 1: Look Left AND Point Finger/Hand to Left simultaneously
  - Step 2: Look Right AND Point Finger/Hand to Right simultaneously
  - Step 3: Look Forward AND Point Finger/Hand Forward simultaneously
  - Result: All 3 OK -> TOTAL OK (Safe to Cross). If any step is missing or wrong -> Step NG / Total NG.

Engineered for:
  - Raspberry Pi 4 / 5 / Embedded Linux / Low-spec Computers
  - Webcams, CCTV (RTSP/HTTP), and pre-recorded Video files
"""

import cv2
import numpy as np
import math
import argparse
import sys
import os
import time
import json
import csv
from datetime import datetime
from safety_checker import RoadCrossingSafetyChecker

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pedestrian Crossing Safety Standard Checker")
    parser.add_argument(
        "--source", "-s",
        default="0",
        help="Video source: '0' for default webcam, RTSP URL (e.g. rtsp://192.168.1.100/live), or path to video file (e.g. /path/to/video.mov)"
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Optional path to save annotated video output (e.g. output.mp4)"
    )
    parser.add_argument(
        "--model", "-m",
        default="",
        help="Path to pose_landmarker_lite.task model (defaults to pose_landmarker_lite.task in script dir)"
    )
    parser.add_argument(
        "--min-hold",
        type=int,
        default=6,
        help="Minimum consecutive frames required to confirm a step (default: 6 frames)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV GUI window (ideal for background services / headless Raspberry Pi)"
    )
    parser.add_argument(
        "--log-csv",
        default="crossing_results.csv",
        help="Path to CSV file where crossing results will be logged"
    )
    parser.add_argument(
        "--auto-reset-sec",
        type=float,
        default=3.5,
        help="Seconds to wait after a total pass or person leaves before resetting for next person (default: 3.5s)"
    )
    return parser.parse_args()

def log_result_to_csv(csv_path, report):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Step1 (Look & Point Left)",
                "Step2 (Look & Point Right)",
                "Step3 (Look & Point Forward)",
                "Total Result",
                "Standard Compliant"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report.get("step1_look_point_left", "NG"),
            report.get("step2_look_point_right", "NG"),
            report.get("step3_look_point_forward", "NG"),
            report.get("total_result", "TOTAL NG"),
            "YES" if report.get("is_standard_compliant") else "NO"
        ])

def main():
    args = parse_arguments()
    
    # Determine video source
    source_val = args.source
    if source_val.isdigit():
        source_val = int(source_val)
        is_live_stream = True
    elif source_val.startswith("rtsp://") or source_val.startswith("http://") or source_val.startswith("https://"):
        is_live_stream = True
    else:
        is_live_stream = False
        if not os.path.exists(source_val):
            print(f"[ERROR] Video source file not found: {source_val}")
            sys.exit(1)

    print(f"[*] Initializing Road Crossing Safety Checker...")
    print(f"[*] Source: {args.source}")
    print(f"[*] Headless Mode: {args.headless}")
    print(f"[*] Min Hold Frames: {args.min_hold}")

    model_path = args.model if args.model else os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
    checker = RoadCrossingSafetyChecker(model_path=model_path, min_hold_frames=args.min_hold)

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output Video Writer
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"[*] Recording output to: {args.output}")

    total_pass_timestamp = None
    frame_count = 0
    t0 = time.time()

    print("[*] System active. Press 'q' in window to exit, or 'r' to reset manually.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_live_stream:
                    print("[WARN] Video stream disconnected. Retrying in 1s...")
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(source_val)
                    continue
                else:
                    # Video file ended
                    break

            frame_count += 1
            cur_time = frame_count / fps if not is_live_stream else time.time()
            
            # Process Frame
            annotated_frame, info = checker.process_frame(frame, timestamp_sec=cur_time)
            
            # Auto-reset after successful crossing check
            if info["total_ok"]:
                if total_pass_timestamp is None:
                    total_pass_timestamp = time.time()
                    report = checker.generate_final_report()
                    log_result_to_csv(args.log_csv, report)
                    print(f"\n[EVENT] Standard COMPLETE! Result logged: {report}")
                elif time.time() - total_pass_timestamp > args.auto_reset_sec:
                    print("[INFO] Resetting checker for next pedestrian...")
                    checker.reset()
                    total_pass_timestamp = None
            
            # Write to output video file
            if out_writer is not None:
                out_writer.write(annotated_frame)

            # Display GUI window
            if not args.headless:
                cv2.imshow("Road Crossing Safety Standard Checker", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quit requested by user.")
                    break
                elif key == ord('r'):
                    print("[INFO] Manual reset triggered.")
                    checker.reset()
                    total_pass_timestamp = None

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = time.time() - t0
    final_report = checker.generate_final_report()
    log_result_to_csv(args.log_csv, final_report)

    print("\n=======================================================")
    print("FINAL SUMMARY REPORT")
    print("=======================================================")
    print(f"Frames Processed : {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.1f} FPS)")
    print(f"Step 1 (Look & Point Left)   : {final_report['step1_look_point_left']}")
    print(f"Step 2 (Look & Point Right)  : {final_report['step2_look_point_right']}")
    print(f"Step 3 (Look & Point Forward): {final_report['step3_look_point_forward']}")
    print(f"TOTAL RESULT                 : {final_report['total_result']}")
    print(f"Standard Compliant           : {final_report['is_standard_compliant']}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
