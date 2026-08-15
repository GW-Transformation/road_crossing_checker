#!/usr/bin/env python3
"""
Pedestrian Road Crossing Safety Standard Checker
================================================
Comprehensive Computer Vision & Safety Enforcement System:
  - Point of Interest (POI) crossing corridor detection
  - Active monitoring upon detecting face & person in frame / POI
  - Multi-person tracking with independent 3-Step Safety verification:
      * Step 1: Look Left AND Point Left simultaneously
      * Step 2: Look Right AND Point Right simultaneously
      * Step 3: Look Forward AND Point Forward simultaneously
  - Automated NG evaluation when safety steps are incomplete
  - Real-time Voice Alarm: "กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง"
  - Face snapping, evidence video clipping, and SQLite DB logging
  - External Face Recognition API client integration
  - Built-in Web Incident Management Dashboard & REST API
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
import threading
from datetime import datetime

from safety_checker import RoadCrossingSafetyChecker
from web_dashboard import app as flask_app

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
        help="Path to pose_landmarker_lite.task model"
    )
    parser.add_argument(
        "--poi",
        default="poi_config.json",
        help="Path to POI polygon config JSON"
    )
    parser.add_argument(
        "--min-hold",
        type=int,
        default=6,
        help="Minimum consecutive frames required to confirm a step (default: 6 frames)"
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=4.0,
        help="Max seconds allowed in POI to complete safety checklist before NG trigger (default: 4.0s)"
    )
    parser.add_argument(
        "--no-alarm",
        action="store_true",
        help="Disable voice alarm playback"
    )
    parser.add_argument(
        "--face-api",
        default="",
        help="URL to external Face Recognition API (e.g. http://localhost:5000/api/mock_face_recognition)"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start background Web Incident Dashboard (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for Web Incident Dashboard (default: 8080)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV GUI window (ideal for headless embedded systems)"
    )
    parser.add_argument(
        "--db-path",
        default="data/crossing_records.db",
        help="Path to SQLite database file"
    )
    return parser.parse_args()

def run_web_dashboard_async(port=8080):
    def _run():
        for p in [port, 8080, 8081, 5001, 5000]:
            try:
                print(f"[*] Starting Web Incident Dashboard at: http://localhost:{p}")
                flask_app.run(host="0.0.0.0", port=p, debug=False, use_reloader=False)
                break
            except Exception:
                continue
    t = threading.Thread(target=_run, daemon=True)
    t.start()

def main():
    args = parse_arguments()
    
    # Start web dashboard if requested
    if args.web:
        run_web_dashboard_async(port=args.port)

    # Determine video source
    source_val = args.source
    if source_val.isdigit():
        source_val = int(source_val)
        is_live_stream = True
    elif source_val.startswith("rtsp://") or source_val.startswith("http://") or source_val.startswith("https://"):
        is_live_stream = True
    else:
        is_live_stream = False
        source_val = os.path.expanduser(source_val)
        if not os.path.exists(source_val):
            print(f"[ERROR] Video source file not found: {source_val}")
            sys.exit(1)

    print(f"[*] Initializing Road Crossing Safety Checker...")
    print(f"[*] Source: {args.source}")
    print(f"[*] POI Config: {args.poi}")
    print(f"[*] Voice Alarm Enabled: {not args.no_alarm}")
    print(f"[*] Headless Mode: {args.headless}")

    model_path = args.model if args.model else os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
    checker = RoadCrossingSafetyChecker(
        model_path=model_path,
        min_hold_frames=args.min_hold,
        poi_config_path=args.poi,
        db_path=args.db_path,
        face_api_url=args.face_api if args.face_api else None,
        enable_alarm=not args.no_alarm,
        max_crossing_wait_sec=args.max_wait
    )

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or math.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"[*] Recording output to: {args.output}")

    frame_count = 0
    t0 = time.time()
    print("[*] System active. Monitoring Point of Interest (POI). Press 'q' in window to exit.")

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
                    # End of video file
                    break

            frame_count += 1
            cur_time = frame_count / fps if not is_live_stream else time.time()
            
            # Process Frame
            annotated_frame, info = checker.process_frame(frame, timestamp_sec=cur_time)
            
            # Write output
            if out_writer is not None:
                out_writer.write(annotated_frame)

            # Display GUI window
            if not args.headless:
                cv2.imshow("Road Crossing Safety Standard Monitor", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quit requested by user.")
                    break

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

    print("\n=======================================================")
    print("SESSION SUMMARY REPORT")
    print("=======================================================")
    print(f"Frames Processed : {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.1f} FPS)")
    print(f"Total Evaluated  : {len(final_report.get('persons', []))} pedestrian(s)")
    print(f"Final Result     : {final_report['total_result']}")
    print(f"All Compliant    : {final_report['is_standard_compliant']}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
