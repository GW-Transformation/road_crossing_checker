import requests
import cv2
import base64
import os
import threading
import json
import time

class ExternalFaceRecognitionClient:
    """
    Client for calling external Face Recognition System API to identify NG pedestrians.
    """
    def __init__(self, api_url=None, timeout_sec=3.0, on_match_callback=None):
        self.api_url = api_url # e.g. "http://external-hr-system.internal/api/face_match"
        self.timeout_sec = timeout_sec
        self.on_match_callback = on_match_callback

    def recognize_face_async(self, face_image, record_id, metadata=None):
        """Dispatches recognition in background thread to avoid dropping camera FPS"""
        thread = threading.Thread(
            target=self._recognize_worker,
            args=(face_image, record_id, metadata),
            daemon=True
        )
        thread.start()

    def _recognize_worker(self, face_image, record_id, metadata):
        if face_image is None or face_image.size == 0:
            return
            
        # Encode image to JPEG
        ret, buf = cv2.imencode(".jpg", face_image)
        if not ret:
            return
            
        img_bytes = buf.tobytes()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        result = {
            "matched": False,
            "person_id": None,
            "name": "Unknown",
            "confidence": 0.0
        }
        
        if self.api_url:
            try:
                payload = {
                    "image_base64": img_b64,
                    "record_id": record_id,
                    "timestamp": time.time(),
                    "metadata": metadata or {}
                }
                resp = requests.post(self.api_url, json=payload, timeout=self.timeout_sec)
                if resp.status_code == 200:
                    data = resp.json()
                    result["matched"] = data.get("matched", True)
                    result["person_id"] = data.get("person_id", f"EXT-{record_id:04d}")
                    result["name"] = data.get("name", "Identified Subject")
                    result["confidence"] = float(data.get("confidence", 0.92))
                    print(f"[FACE RECOGNITION API] Match found: {result['name']} (ID: {result['person_id']})")
                else:
                    print(f"[FACE RECOGNITION API] Server responded with code {resp.status_code}")
            except Exception as e:
                print(f"[FACE RECOGNITION API] Connection failed: {e}")
        else:
            # Standalone / Simulation mode
            # Generate simulated ID based on track record
            result["matched"] = True
            result["person_id"] = f"EMP-{int(record_id) % 9000 + 1000}"
            result["name"] = f"Pedestrian #{record_id}"
            result["confidence"] = 0.88
            
        if self.on_match_callback:
            self.on_match_callback(record_id, result)
        return result
