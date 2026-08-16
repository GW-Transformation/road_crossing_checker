# Pedestrian Road Crossing Safety Standard Checker 🚸

An embedded-ready AI computer vision and safety enforcement system designed to verify if pedestrians follow the 3-step safety standard before crossing the road.

---

## 🎯 Safety Standard Protocol
1. **Step 1:** Look Left **AND** Point finger/hand to Left simultaneously.
2. **Step 2:** Look Right **AND** Point finger/hand to Right simultaneously.
3. **Step 3:** Look Forward **AND** Point finger/hand Forward simultaneously.
4. **Outcome Evaluation:**
   - **TOTAL OK (PASS):** All 3 steps executed in order ➔ Logged as compliant.
   - **SAFETY VIOLATION (NG):** Incomplete or incorrect steps ➔ Triggers voice alarm warning, snaps face image, clips evidence video, and queries External Face Recognition API.

---

## 🚀 Key Features
- **Interactive On-Screen POI Editor:** Edit safety crossing zones directly in the live OpenCV window by dragging vertex handles (`P1-P4`), or via the Web Dashboard interactive SVG editor (`poi_config.json`).
- **Full Head & Face Portrait Capture:** Captures high-res face portrait evidence showing full head, hair, ears, chin, and neck down to shoulders.
- **Active Face & Person Monitoring:** Activates tracking and monitoring when a person/face enters the POI.
- **Multi-Person Tracking:** Tracks individual pedestrians with separate safety checklists.
- **🔊 Voice Alarm Warning:** Plays *"กรุณาหยุด ชี้นิ้วตามทางแยกให้ถูกต้อง"* in real-time upon violation.
- **📸 Face Snapping & Evidence Video:** Captures high-res face crops and clips evidence video of the entire crossing event.
- **🌐 External Face Recognition API Integration:** Webhook client to match and identify pedestrians against external HR/security systems.
- **💾 SQLite Database Logging:** Full audit log of all events, timestamps, face paths, and video recordings.
- **🖥️ Web Incident Dashboard & REST API:** Built-in web dashboard (`http://localhost:5000`) with video playback and record retrieval.
- **⚡ Embedded Ready:** Optimized for Raspberry Pi 4 / 5 and low-spec hardware using lightweight MediaPipe Pose.

---

## 🎮 GUI & POI Editor Controls (On-Screen Window)
- **`[E]` Key:** Toggle **POI Edit Mode** on/off.
- **Left-Click & Drag:** Click and drag any POI vertex handle (`P1`, `P2`, `P3`, `P4`) to reshape the safety zone in real-time.
- **`[S]` Key:** **Save POI configuration** immediately to disk (`poi_config.json`) with on-screen visual confirmation.
- **`[R]` Key:** **Reset POI** to default rectangular safety crossing corridor.
- **`[Q]` / `[ESC]` Key:** Exit / Quit.

---

## 💻 How to Run

### 1. Run Live Webcam
```bash
/Users/m4ck/projects/road_crossing_checker/venv/bin/python app.py --source 0 --web
```
*Access the Web Incident Dashboard at `http://localhost:5000` while camera runs.*

### 2. Run CCTV / IP RTSP Stream
```bash
/Users/m4ck/projects/road_crossing_checker/venv/bin/python app.py --source "rtsp://username:password@camera_ip:554/stream" --web
```

### 3. Run on Video Files (Test with 1.mov / 2.mov)
```bash
/Users/m4ck/projects/road_crossing_checker/venv/bin/python app.py --source ~/Desktop/1.mov --output output_1.mp4
```

### 4. Start Standalone Web Incident Dashboard
```bash
/Users/m4ck/projects/road_crossing_checker/venv/bin/python web_dashboard.py
```

### 5. Run Automated Verification Tests
```bash
/Users/m4ck/projects/road_crossing_checker/venv/bin/python test_advanced_features.py
```
