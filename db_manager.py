import sqlite3
import os
import json
from datetime import datetime

class CrossingDatabase:
    """
    Manages SQLite database for road crossing safety compliance records.
    Stores:
      - record_id, track_id, timestamp, status (TOTAL_OK, NG)
      - step1_status, step2_status, step3_status
      - face_image_path, evidence_video_path
      - external_person_id, external_name, external_confidence
      - notes / metadata
    """
    def __init__(self, db_path="data/crossing_records.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crossing_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_uuid TEXT UNIQUE,
                    track_id INTEGER,
                    timestamp TEXT,
                    status TEXT, -- 'TOTAL_OK' or 'NG'
                    step1_status TEXT, -- 'OK' or 'NG'
                    step2_status TEXT, -- 'OK' or 'NG'
                    step3_status TEXT, -- 'OK' or 'NG'
                    duration_sec REAL,
                    face_image_path TEXT,
                    evidence_video_path TEXT,
                    external_person_id TEXT,
                    external_name TEXT,
                    external_confidence REAL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON crossing_records(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON crossing_records(timestamp)")
            conn.commit()

    def insert_record(self, record_data):
        """
        record_data dict:
          event_uuid, track_id, timestamp, status, step1_status, step2_status, step3_status,
          duration_sec, face_image_path, evidence_video_path, external_person_id,
          external_name, external_confidence, metadata_json
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO crossing_records (
                    event_uuid, track_id, timestamp, status,
                    step1_status, step2_status, step3_status,
                    duration_sec, face_image_path, evidence_video_path,
                    external_person_id, external_name, external_confidence,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_data.get("event_uuid"),
                record_data.get("track_id", 0),
                record_data.get("timestamp", datetime.now().isoformat()),
                record_data.get("status", "NG"),
                record_data.get("step1_status", "NG"),
                record_data.get("step2_status", "NG"),
                record_data.get("step3_status", "NG"),
                record_data.get("duration_sec", 0.0),
                record_data.get("face_image_path", ""),
                record_data.get("evidence_video_path", ""),
                record_data.get("external_person_id", None),
                record_data.get("external_name", None),
                record_data.get("external_confidence", 0.0),
                json.dumps(record_data.get("metadata", {}))
            ))
            conn.commit()
            return cursor.lastrowid

    def update_external_recognition(self, record_id, person_id, name, confidence):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE crossing_records
                SET external_person_id = ?, external_name = ?, external_confidence = ?
                WHERE id = ? OR event_uuid = ?
            """, (person_id, name, confidence, record_id, str(record_id)))
            conn.commit()

    def get_records(self, limit=50, offset=0, status_filter=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM crossing_records"
            params = []
            if status_filter:
                query += " WHERE status = ?"
                params.append(status_filter)
            query += " ORDER BY id DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_record_by_id(self, record_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM crossing_records WHERE id = ? OR event_uuid = ?", (record_id, str(record_id)))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_stats(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM crossing_records")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM crossing_records WHERE status = 'TOTAL_OK'")
            total_ok = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM crossing_records WHERE status = 'NG'")
            total_ng = cursor.fetchone()[0]
            
            return {
                "total_events": total,
                "total_ok": total_ok,
                "total_ng": total_ng,
                "compliance_rate": (total_ok / total * 100) if total > 0 else 0.0
            }
