# 🚸 Pedestrian Road Crossing Safety Standard Checker — Master Prompts & Specification

This document contains the complete **System Prompt, Technical Specifications, and Architectural Blueprint** used to design and build the Pedestrian Road Crossing Safety Standard Checker.

---

## 📋 Master System Prompt (for AI Agents & Developers)

```text
You are an expert Computer Vision, Embedded AI, and Edge Systems Architect.

Build a real-time Embedded-Ready AI Safety Standard Checker system in Python using MediaPipe Pose, OpenCV, SQLite, and Flask to monitor and enforce the Japanese/Thai 3-Step Pedestrian Road Crossing Standard:

### 🎯 Safety Standard Protocol:
1. Step 1 (Look & Point Side 1): Pedestrian looks Left AND points hand/finger Left (or Right first).
2. Step 2 (Look & Point Side 2): Pedestrian looks the opposite direction AND points hand/finger in that direction.
3. Step 3 (Look & Point Forward): Pedestrian looks Forward AND points hand/finger straight Forward.
4. Evaluation:
   - TOTAL OK (PASS): All 3 steps completed before crossing.
   - SAFETY VIOLATION (NG): Pedestrian leaves the Point of Interest (POI) crossing corridor without completing all 3 safety steps.

### 🚀 Core Engineering Requirements:
1. POI Zone Corridor Management:
   - Configured via `poi_config.json` with normalized (0.0 - 1.0) polygon coordinates.
   - On-Screen Interactive GUI Editor: Users can press [E] to toggle Edit Mode, drag vertex handles (P1-P4) with the mouse to reshape the corridor in real time, and press [S] to save.
   - Web Dashboard SVG Editor: Users can drag SVG handles and click Save via REST API (/api/poi).
   - Only activate tracking, evaluation, and alarms for pedestrians inside the POI.

2. Robust Multi-Person Spatial Tracking:
   - SimpleMultiPersonTracker with 1-to-1 spatial centroid/IOU Hungarian-style matching to eliminate cross-track ID swapping.
   - Temporal qualification: Suppress spurious single-frame ghost blips (< 4 frames) before qualifying as a verified pedestrian.

3. Computer Vision & Gesture Recognition:
   - Head Yaw Classification: Ratio of nose to left/right ears ((nose.x - ear_mid_x) / ear_w) with calibrated angle thresholds (-0.16 for Left, +0.14 for Right, [-0.35, 0.35] for Front).
   - Arm Pointing Direction: Vector calculation of shoulder -> elbow -> wrist angles. Instant recognition for Forward pointing; minimum hold frames for Left/Right.
   - Order Flexibility: Support checking Left first OR Right first, followed by Step 3 Forward.

4. Voice Alarm & Audio Feedback:
   - Crisp confirmation chime on Step 1, Step 2, and Step 3 completion.
   - Non-blocking audio queue playback (using macOS `afplay` or Linux `aplay`) playing Thai voice alert: "กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง" when an unqualified pedestrian leaves the POI.

5. Full Head, Hair & Neck Face Portrait Crop:
   - High-resolution face evidence portrait extraction dynamically including top of hairstyle/hair, forehead, ears, chin, and neck down to the clavicle / shoulder line.

6. Evidence Video Recording:
   - Circular frame buffer (10 FPS subsampled) recording the entire crossing event.
   - Live rendered skeleton overlay and HUD status card embedded into H.264 mp4 browser-playable evidence video.

7. Database & Web Incident Dashboard:
   - SQLite (`data/test_records.db`) logging event UUID, track ID, status (TOTAL_OK / NG), timestamps, face image path, video path, and external recognition ID.
   - Flask Web Incident Dashboard (`http://localhost:5000`) with video player modal, face portrait zoom, batch delete, clear all, and external face recognition webhook integration.
```

---

## 🛠️ Module Architecture Breakdown

| Module | Purpose | Key Classes / Functions |
| :--- | :--- | :--- |
| [`safety_checker.py`](file:///Users/m4ck/Projects/road_crossing_checker/safety_checker.py) | Main CV Pipeline & Gesture Logic | `RoadCrossingSafetyChecker`, `process_frame`, `_extract_face_crop` |
| [`person_tracker.py`](file:///Users/m4ck/Projects/road_crossing_checker/person_tracker.py) | Multi-person state & spatial tracker | `SimpleMultiPersonTracker`, `TrackedPerson` |
| [`poi_manager.py`](file:///Users/m4ck/Projects/road_crossing_checker/poi_manager.py) | POI polygon & on-screen handle editor | `POIManager`, `find_nearest_vertex`, `update_vertex`, `draw_poi_overlay` |
| [`alarm_player.py`](file:///Users/m4ck/Projects/road_crossing_checker/alarm_player.py) | Non-blocking audio warning engine | `AlarmWarningPlayer`, `play_step_ok`, `play_warning_alarm` |
| [`db_manager.py`](file:///Users/m4ck/Projects/road_crossing_checker/db_manager.py) | SQLite incident & audit log storage | `CrossingDatabase`, `log_event`, `delete_records` |
| [`web_dashboard.py`](file:///Users/m4ck/Projects/road_crossing_checker/web_dashboard.py) | Flask REST API & Web Dashboard | `/`, `/api/records`, `/api/poi`, `/data/faces/`, `/data/evidence/` |
| [`app.py`](file:///Users/m4ck/Projects/road_crossing_checker/app.py) | Main CLI runner & OpenCV GUI | Mouse drag callback, keyboard shortcuts (`[E]`, `[S]`, `[R]`, `[Q]`) |

---

## 🎮 Shortcut Guide (OpenCV Live Window)

- **`[E]` Key:** Toggle **POI Edit Mode** on/off.
- **Mouse Left-Click & Drag:** Drag any vertex handle (`P1`, `P2`, `P3`, `P4`) to reshape the crossing corridor on-screen.
- **`[S]` Key:** Save the active POI coordinates directly to `poi_config.json`.
- **`[R]` Key:** Reset the POI corridor to default rectangular bounds.
- **`[Q]` / `[ESC]` Key:** Exit the application.

---

## 🧪 Verification & Test Commands

```bash
# 1. Run all advanced feature tests (Pass, NG trigger, DB logging)
./venv/bin/python test_advanced_features.py

# 2. Run POI exit logic tests
./venv/bin/python test_poi_exit_logic.py

# 3. Run edge-case verification suite
./venv/bin/python test_edge_cases.py

# 4. Run test videos benchmark
./venv/bin/python test_videos.py
```
