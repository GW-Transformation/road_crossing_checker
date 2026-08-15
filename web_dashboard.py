from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
from db_manager import CrossingDatabase
from poi_manager import POIManager

app = Flask(__name__, static_folder="static", template_folder="templates")
db = CrossingDatabase()
poi = POIManager()

@app.route("/")
def index():
    stats = db.get_stats()
    records = db.get_records(limit=100)
    return render_template("index.html", stats=stats, records=records, poi_poly=poi.polygon)

@app.route("/api/records", methods=["GET"])
def api_records():
    status = request.args.get("status")
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    records = db.get_records(limit=limit, offset=offset, status_filter=status)
    return jsonify({"success": True, "records": records, "stats": db.get_stats()})

@app.route("/api/records/<int:record_id>", methods=["GET"])
def api_record_detail(record_id):
    rec = db.get_record_by_id(record_id)
    if rec:
        return jsonify({"success": True, "record": rec})
    return jsonify({"success": False, "error": "Not found"}), 404

@app.route("/api/poi", methods=["GET", "POST"])
def api_poi():
    if request.method == "POST":
        data = request.get_json()
        polygon = data.get("poi_polygon", [])
        poi.set_polygon_normalized(polygon)
        return jsonify({"success": True, "poi_polygon": poi.polygon})
    return jsonify({"success": True, "poi_polygon": poi.polygon})

@app.route("/api/records/delete", methods=["POST"])
def api_delete_records():
    data = request.get_json() or {}
    ids = data.get("ids", [])
    if isinstance(ids, int):
        ids = [ids]
    deleted = db.delete_records(ids)
    return jsonify({"success": True, "deleted_count": deleted, "stats": db.get_stats()})

@app.route("/api/records/clear_all", methods=["POST"])
def api_clear_all_records():
    db.clear_all_records()
    return jsonify({"success": True, "message": "All records cleared", "stats": db.get_stats()})

@app.route("/data/faces/<path:filename>")
@app.route("/faces/<path:filename>")
def serve_face(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(os.path.join(base_dir, "data", "faces"), filename)

@app.route("/data/evidence/<path:filename>")
@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ev_dir = os.path.join(base_dir, "data", "evidence")
    return send_from_directory(ev_dir, filename, mimetype="video/mp4")

# Mock External Face Recognition Server endpoint for instant testing
@app.route("/api/mock_face_recognition", methods=["POST"])
def mock_face_recognition():
    data = request.get_json() or {}
    record_id = data.get("record_id", 1)
    return jsonify({
        "matched": True,
        "person_id": f"EMP-{int(record_id) * 111 % 9000 + 1000}",
        "name": f"Employee John #{record_id}",
        "department": "Logistics & Operations",
        "confidence": 0.94
    })

def start_web_server(port=8080):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    start_web_server()
