"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Main Application: visionport.py
Model Name: traffic_detector
Project: VisionPort
Tasks: 1-20 (fully connected)
"""

import io
import os
import subprocess
import sys
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from vehicle_detector import TrafficDetector

app = FastAPI(title="VisionPort: Intelligent Vehicle Detection and Traffic Monitoring API")

# Allow the HTML page to talk to this server (fixes "Failed to fetch")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Counts", "X-Total", "X-Density", "X-Congestion",
        "X-Signal-Status", "X-Avg-Wait", "X-Max-Speed", "X-Task-Result"
    ],
)

# Initialize core model: traffic_detector
detector = TrafficDetector(model_path="yolov8n.pt")
model = detector.model
VEHICLE_CLASSES = detector.VEHICLE_CLASSES
LOW_MAX = detector.LOW_MAX
MEDIUM_MAX = detector.MEDIUM_MAX

# Map task IDs -> script files and primary output images
TASK_MAP = {
    1:  {"script": "task1_setup_and_verify.py",           "output": "task1_opencv_test.jpg"},
    2:  {"script": "task2_capture_traffic_signal.py",     "output": "captured_images/captured_traffic_signal.jpg"},
    3:  {"script": "task3_read_display_traffic_image.py", "output": "task3_display_output.jpg"},
    5:  {"script": "task5_extract_video_frames.py",       "output": "extracted_frames/frame_0001.jpg"},
    6:  {"script": "task6_detect_cars.py",                "output": "task6_car_detection_output.jpg"},
    7:  {"script": "task7_detect_multiple_vehicles.py",   "output": "task7_multiple_vehicles_output.jpg"},
    8:  {"script": "task8_display_labels_confidence.py",  "output": "task8_labels_confidence_output.jpg"},
    9:  {"script": "task9_count_vehicles.py",             "output": "task9_vehicle_count_output.jpg"},
    10: {"script": "task10_traffic_density.py",           "output": "task10_traffic_density_output.jpg"},
    11: {"script": "task11_traffic_congestion.py",        "output": "task11_congestion_alert_output.jpg"},
    12: {"script": "task12_track_vehicles.py",            "output": "task12_vehicle_tracking_output.jpg"},
    13: {"script": "task13_speed_estimation.py",          "output": "task13_speed_estimation_output.jpg"},
    14: {"script": "task14_traffic_signal_status.py",     "output": "task14_signal_status_output.jpg"},
    15: {"script": "task15_traffic_waiting_time.py",      "output": "task15_waiting_time_output.jpg"},
    16: {"script": "task16_junction_entry_exit.py",       "output": "task16_entry_exit_output.jpg"},
    17: {"script": "task17_wrong_turn_detection.py",      "output": "task17_wrong_turn_output.jpg"},
    18: {"script": "task18_wrong_way_detection.py",       "output": "task18_wrong_way_output.jpg"},
    19: {"script": "task19_illegal_parking.py",           "output": "task19_illegal_parking_output.jpg"},
    20: {"script": "task20_triple_riding_detection.py",   "output": "task20_triple_riding_output.jpg"},
}


# ──────────────────────────────────────────────
# ROOT: Serve the web portal
# ──────────────────────────────────────────────
@app.get("/")
def home():
    """Serves the VisionPort dashboard web portal."""
    if os.path.exists("visionport.html"):
        return FileResponse("visionport.html")
    elif os.path.exists("visionport (1).html"):
        return FileResponse("visionport (1).html")
    return JSONResponse({"status": "active", "project": "VisionPort", "model": detector.model_name})


# ──────────────────────────────────────────────
# STATUS: System health
# ──────────────────────────────────────────────
@app.get("/status")
def system_status():
    """System health check and module capability overview."""
    return {
        "project": "VisionPort",
        "model_name": detector.model_name,
        "status": "online",
        "supported_classes": list(VEHICLE_CLASSES.values()),
        "tasks_implemented": list(range(1, 21)),
        "density_thresholds": {"low_max": LOW_MAX, "medium_max": MEDIUM_MAX}
    }


# ──────────────────────────────────────────────
# TASKS LIST: Return all tasks metadata
# ──────────────────────────────────────────────
@app.get("/tasks")
def list_tasks():
    """Return all available task IDs and their scripts."""
    return {
        str(tid): {"script": info["script"], "output": info["output"]}
        for tid, info in TASK_MAP.items()
    }


# ──────────────────────────────────────────────
# RUN TASK: Execute any task by ID (Tasks 1–20)
# ──────────────────────────────────────────────
@app.post("/run/{task_id}")
async def run_task(task_id: int, file: UploadFile = File(None)):
    """
    Run a specific task (1–20) by ID.
    Optionally accepts an uploaded image that is saved as sample_traffic.jpg
    before the task script runs. Returns the task output image.
    """
    if task_id not in TASK_MAP:
        return JSONResponse({"error": f"Task {task_id} is not defined."}, status_code=404)

    task_info = TASK_MAP[task_id]
    script = task_info["script"]
    output_img = task_info["output"]

    # If a file was uploaded, save it as sample_traffic.jpg so task scripts pick it up
    if file and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite("sample_traffic.jpg", img)

    # Run the task script
    python_exe = sys.executable
    try:
        result = subprocess.run(
            [python_exe, script],
            capture_output=True, text=True, timeout=90
        )
        success = result.returncode == 0
        stdout = result.stdout[-2000:] if result.stdout else ""
        stderr = result.stderr[-1000:] if result.stderr else ""
    except subprocess.TimeoutExpired:
        success = False
        stdout = ""
        stderr = f"Task {task_id} timed out after 90 seconds."

    # Return the output image if it exists, else a JSON status report
    if os.path.exists(output_img):
        ext = os.path.splitext(output_img)[1].lower()
        media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        headers = {"X-Task-Result": "SUCCESS" if success else "WARNING"}
        return FileResponse(output_img, media_type=media_type, headers=headers)

    return JSONResponse({
        "task_id": task_id,
        "script": script,
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "note": "No output image was generated by this task."
    })


# ──────────────────────────────────────────────
# DETECT: Real-time vehicle detection (Tasks 6–10 pipeline)
# ──────────────────────────────────────────────
@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Main image detection endpoint (Tasks 6-11 pipeline).
    Returns annotated image stream with X-Headers for the UI dashboard.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image payload"}, status_code=400)

    # Multi-vehicle detection (Tasks 6-9)
    detections = detector.detect_multiple_vehicles(img)
    counts, total, deduped = detector.count_vehicles(detections)
    density, density_color = detector.get_density(total)

    # Congestion & Signal detection (Tasks 11, 14)
    is_congested, persistence_ratio = detector.check_congestion(density)
    signal_status, sig_color, _ = detector.detect_traffic_signal(img)

    # Draw bounding boxes with labels and confidence (Task 8)
    annotated_img = detector.draw_detections(img, deduped, show_label=True, show_conf=True)

    # Draw HUD overlay panel
    lines = [
        f"VisionPort: {detector.model_name}",
        f"Cars: {counts['car']}",
        f"Motorcycles: {counts['motorcycle']}",
        f"Buses: {counts['bus']}",
        f"Trucks: {counts['truck']}",
        f"Total Vehicles: {total}",
        f"Traffic Density: {density}",
        f"Signal Status: {signal_status}",
        f"Congestion Alert: {'YES' if is_congested else 'NO'}"
    ]
    cv2.rectangle(annotated_img, (5, 5), (360, 15 + 28 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        if text.startswith("Traffic"):
            line_color = density_color
        elif text.startswith("Signal"):
            line_color = sig_color
        elif "Congestion Alert: YES" in text:
            line_color = (0, 0, 255)
        elif i == 0:
            line_color = (0, 255, 255)
        else:
            line_color = (255, 255, 255)
        cv2.putText(annotated_img, text, (15, 30 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 2)

    _, encoded_img = cv2.imencode(".jpg", annotated_img)
    headers = {
        "X-Counts": ", ".join(f"{k}: {v}" for k, v in counts.items()),
        "X-Total": str(total),
        "X-Density": density,
        "X-Congestion": "ALERT" if is_congested else "NORMAL",
        "X-Signal-Status": signal_status
    }
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()),
                             media_type="image/jpeg", headers=headers)


# ──────────────────────────────────────────────
# ANALYZE: Detailed JSON analysis (all tasks)
# ──────────────────────────────────────────────
@app.post("/analyze/")
async def analyze_traffic(file: UploadFile = File(...)):
    """
    Detailed JSON analysis endpoint.
    Returns vehicle counts, density, signal status, congestion, tracking and waiting time.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image payload"}, status_code=400)

    detections = detector.detect_multiple_vehicles(img)
    counts, total, deduped = detector.count_vehicles(detections)
    density, _ = detector.get_density(total)
    is_congested, persistence_ratio = detector.check_congestion(density)
    signal_status, _, signal_details = detector.detect_traffic_signal(img)
    tracks = detector.track_vehicles(deduped)
    waiting_report = detector.estimate_waiting_time(tracks, signal_status)

    return {
        "project": "VisionPort",
        "model": detector.model_name,
        "vehicle_counts": counts,
        "total_vehicles": total,
        "traffic_density": density,
        "congestion_alert": is_congested,
        "congestion_persistence_ratio": round(persistence_ratio, 2),
        "signal_status": signal_status,
        "signal_pixel_metrics": signal_details,
        "waiting_time_report": waiting_report
    }


# ──────────────────────────────────────────────
# STATIC FILES: Serve images/videos for the UI
# ──────────────────────────────────────────────
@app.get("/{filename}")
def serve_static_file(filename: str):
    """Serve images and video media files for the dashboard (fixes 404 errors)."""
    # Security: prevent path traversal
    safe_path = os.path.normpath(filename)
    if ".." in safe_path or os.path.isabs(safe_path):
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    if os.path.exists(safe_path) and os.path.isfile(safe_path):
        ext = os.path.splitext(safe_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            return FileResponse(safe_path, media_type="image/jpeg")
        elif ext == ".png":
            return FileResponse(safe_path, media_type="image/png")
        elif ext == ".mp4":
            return FileResponse(safe_path, media_type="video/mp4")
    return JSONResponse({"error": "File not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    print("[VisionPort] Starting API server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
