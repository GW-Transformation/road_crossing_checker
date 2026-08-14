# Pedestrian Road Crossing Safety Standard Checker 🚸

An embedded-ready AI computer vision application for detecting and verifying if a pedestrian follows the safety crossing standard before crossing the road.

## 🎯 Safety Standard Rules
1. **Step 1:** Look Left **AND** Point finger/hand to Left simultaneously.
2. **Step 2:** Look Right **AND** Point finger/hand to Right simultaneously.
3. **Step 3:** Look Forward **AND** Point finger/hand Forward simultaneously.
4. **Total Evaluation:** If all 3 steps pass sequentially ➔ **TOTAL OK (SAFE TO CROSS)**. If any step is missed or failed ➔ Displays **Step NG** and **TOTAL NG**.

---

## ⚡ Key Features & Embedded Optimization
- **Ultra-lightweight:** Powered by MediaPipe Pose Lite (TFLite / XNNPACK).
- **Embedded Hardware Compatible:** Runs smoothly on Raspberry Pi 4 / 5, low-spec PCs, Intel NUC, and Jetson Nano.
- **Universal Input:** Webcams (`0`, `1`), CCTV / IP Cameras (`rtsp://...`, `http://...`), or Video files (`.mov`, `.mp4`).
- **Real-Time On-Screen HUD:** Visual step checklist with live indicators and skeleton overlay.
- **CSV Event Logging:** Automatically logs all pass/fail events with timestamps.
- **Headless Mode:** Can run without a GUI display as a background daemon on embedded Linux/Raspberry Pi.

---

## 🚀 Quick Start Guide

### 1. Run on Webcam (Live Stream)
```bash
./venv/bin/python app.py --source 0
```

### 2. Run on CCTV / RTSP Stream
```bash
./venv/bin/python app.py --source rtsp://admin:password@192.168.1.100:554/stream1
```

### 3. Run on Test Videos (e.g. 1.mov and 2.mov)
```bash
# Test 1.mov and record annotated output
./venv/bin/python app.py --source ~/Desktop/1.mov --output output_1.mp4

# Test 2.mov
./venv/bin/python app.py --source ~/Desktop/2.mov --output output_2.mp4
```

### 4. Run Automated Test Suite
```bash
./venv/bin/python test_edge_cases.py
```
