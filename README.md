# VisionPort – Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning

**Project Name:** VisionPort  
**Main Model Name:** `traffic_detector`  
**Architecture:** YOLOv8 Deep Learning Pipeline integrated with OpenCV Computer Vision & FastAPI

---

## 🚦 Overview

VisionPort is an end-to-end intelligent traffic monitoring and vehicle detection system built using deep learning and computer vision. It is capable of detecting multiple vehicle categories, performing real-time counting with spatial deduplication, classifying traffic density levels, detecting persistent congestion at junctions, tracking individual vehicles with persistent IDs, estimating approximate vehicle speeds, classifying traffic signal status via HSV analysis, and estimating traffic waiting times.

---

## 📁 Updated Project Folder Structure

```
VisionPort/
├── .vscode/
│   ├── launch.json                          # VS Code run & debug configurations
│   └── settings.json                        # Interpreter settings (Python 3.12)
├── captured_images/
│   └── captured_traffic_signal.jpg          # Output from Task 2 (Traffic Image Capture)
├── extracted_frames/
│   ├── frame_0001.jpg                       # Output from Task 5 (Video Frame Extraction)
│   ├── frame_0002.jpg
│   └── ...
├── sample_traffic.jpg                       # Reference traffic junction image
├── traffic_sample.mp4                       # Reference traffic video stream
├── yolov8n.pt                               # YOLOv8 neural network weights (traffic_detector)
├── requirements.txt                         # Project dependencies
├── run_visionport.bat                       # One-click Windows runner batch script
├── run_all_tasks.py                         # Master 15-task automated verification suite
├── vehicle_detector.py                      # Reusable core engine & TrafficDetector class
├── visionport.py                            # FastAPI main server & REST endpoints
├── visionport.html                          # Live interactive web portal dashboard
├── task1_setup_and_verify.py                # Task 1: Environment & library verification
├── task1_opencv_test.jpg                    # Task 1 generated test image
├── task2_capture_traffic_signal.py          # Task 2: Webcam & snapshot traffic capture
├── task3_read_display_traffic_image.py      # Task 3: Read & display image with path handling
├── task3_display_output.jpg                 # Task 3 display render
├── task4_load_play_traffic_video.py         # Task 4: Video playback (pause, resume, exit)
├── task5_extract_video_frames.py            # Task 5: Frame extraction to directory
├── task6_detect_cars.py                     # Task 6: Car detection with YOLO bounding boxes
├── task6_car_detection_output.jpg           # Task 6 detection output image
├── task7_detect_multiple_vehicles.py        # Task 7: Cars, motorcycles, buses, trucks
├── task7_multiple_vehicles_output.jpg       # Task 7 multi-class output image
├── task8_display_labels_confidence.py       # Task 8: Class names & confidence badges
├── task8_labels_confidence_output.jpg       # Task 8 output image
├── task9_count_vehicles.py                  # Task 9: Vehicle counting with deduplication
├── task9_vehicle_count_output.jpg           # Task 9 vehicle count visualization
├── task10_traffic_density.py                # Task 10: LOW, MEDIUM, HIGH density classification
├── task10_traffic_density_output.jpg        # Task 10 density classification visualization
├── task11_traffic_congestion.py             # Task 11: Multi-frame congestion alerts
├── task11_congestion_alert_output.jpg       # Task 11 congestion alert visualization
├── task12_track_vehicles.py                 # Task 12: Unique vehicle tracking IDs & trails
├── task12_vehicle_tracking_output.jpg       # Task 12 tracking visualization
├── task13_speed_estimation.py               # Task 13: Approximate vehicle speed estimation
├── task13_speed_estimation_output.jpg       # Task 13 speed estimation visualization
├── task14_traffic_signal_status.py          # Task 14: Traffic signal status (RED/YELLOW/GREEN)
├── task14_signal_status_output.jpg          # Task 14 signal lamp visualization
├── task15_traffic_waiting_time.py           # Task 15: Traffic waiting-time estimation
├── task15_waiting_time_output.jpg           # Task 15 waiting-time visualization
├── task15_waiting_time_report.json          # Task 15 structured JSON waiting report
└── README.md                                # Project documentation
```

---

## 📋 Completed Tasks (1 – 20)

| Task # | Task Description | Expected Output | Implementation File | Status |
|:---:|:---|:---|:---|:---:|
| **1** | Install Python, OpenCV and required libraries | Working Python environment | `task1_setup_and_verify.py` | ✅ PASSED |
| **2** | Capture an image of a traffic signal using camera | Traffic image capture | `task2_capture_traffic_signal.py` | ✅ PASSED |
| **3** | Read and display traffic images using OpenCV | Image display window | `task3_read_display_traffic_image.py` | ✅ PASSED |
| **4** | Load and play a traffic video using OpenCV | Video playback | `task4_load_play_traffic_video.py` | ✅ PASSED |
| **5** | Extract individual frames from traffic videos | Frames saved as images | `task5_extract_video_frames.py` | ✅ PASSED |
| **6** | Detect cars in a traffic video using YOLO | Car detection with bounding boxes | `task6_detect_cars.py` | ✅ PASSED |
| **7** | Detect bikes, buses and trucks | Multiple vehicle detection | `task7_detect_multiple_vehicles.py` | ✅ PASSED |
| **8** | Display vehicle names and confidence scores | Labelled detection output | `task8_display_labels_confidence.py` | ✅ PASSED |
| **9** | Count the number of vehicles in a video frame | Vehicle count (deduplicated) | `task9_count_vehicles.py` | ✅ PASSED |
| **10** | Calculate traffic density based on vehicle count | Low, medium, high traffic | `task10_traffic_density.py` | ✅ PASSED |
| **11** | Detect traffic congestion at signal junctions | Congestion alerts (temporal filter) | `task11_traffic_congestion.py` | ✅ PASSED |
| **12** | Track individual vehicles across video frames | Vehicle tracking IDs & trails | `task12_track_vehicles.py` | ✅ PASSED |
| **13** | Calculate approximate vehicle speed | Speed estimation (with calibration note) | `task13_speed_estimation.py` | ✅ PASSED |
| **14** | Identify traffic signal status (red, yellow, green)| Signal status detection | `task14_traffic_signal_status.py` | ✅ PASSED |
| **15** | Estimate traffic waiting time | Waiting-time report (JSON/HUD) | `task15_traffic_waiting_time.py` | ✅ PASSED |
| **16** | Count vehicles entering and leaving a junction | Entry and exit vehicle count | `task16_junction_entry_exit.py` | ✅ PASSED |
| **17** | Detect vehicles making wrong turns at intersections | Wrong-turn detection alerts | `task17_wrong_turn_detection.py` | ✅ PASSED |
| **18** | Identify vehicles moving in the wrong direction | Wrong-way vehicle alerts | `task18_wrong_way_detection.py` | ✅ PASSED |
| **19** | Detect vehicles parked in restricted traffic areas | Illegal parking detection | `task19_illegal_parking.py` | ✅ PASSED |
| **20** | Detect two-wheelers carrying more than two persons | Triple-riding detection | `task20_triple_riding_detection.py` | ✅ PASSED |

---

## 🛠️ Required Dependencies

All packages are listed in `requirements.txt`:
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `pillow>=10.0.0`
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `ultralytics>=8.0.0`
- `fastapi>=0.100.0`
- `uvicorn>=0.20.0`
- `python-multipart>=0.0.6`
- `matplotlib>=3.7.0`

---

## 🚀 Exact Commands to Run

### 1. Run Any Individual Task
```powershell
python task1_setup_and_verify.py
python task2_capture_traffic_signal.py
python task3_read_display_traffic_image.py
python task4_load_play_traffic_video.py
python task5_extract_video_frames.py
python task6_detect_cars.py
python task7_detect_multiple_vehicles.py
python task8_display_labels_confidence.py
python task9_count_vehicles.py
python task10_traffic_density.py
python task11_traffic_congestion.py
python task12_track_vehicles.py
python task13_speed_estimation.py
python task14_traffic_signal_status.py
python task15_traffic_waiting_time.py
python task16_junction_entry_exit.py
python task17_wrong_turn_detection.py
python task18_wrong_way_detection.py
python task19_illegal_parking.py
python task20_triple_riding_detection.py
```

### 2. Run All Tasks in One Command
```powershell
python run_all_tasks.py
```

### 3. Launch VisionPort Live Web Dashboard
```powershell
python visionport.py
```
Then navigate to: **`http://localhost:8000`** in any web browser.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Serves the interactive web portal |
| GET  | `/status` | System health & capability overview |
| GET  | `/tasks` | List all 20 task IDs, scripts, and outputs |
| POST | `/run/{task_id}` | Run any task (1–20) by ID; returns output image |
| POST | `/detect/` | Real-time vehicle detection (returns annotated image + X-Headers) |
| POST | `/analyze/` | Full JSON analysis (counts, density, signal, tracking, waiting time) |
| GET  | `/{filename}` | Serve any output image or video from the project directory |

---

## 🚀 Deployment Notes

- **Static files**: Any `.jpg`, `.jpeg`, `.png`, or `.mp4` output file in the project root is served directly via `GET /{filename}`.
- **Task runner**: `POST /run/{task_id}` executes the task script using the same Python interpreter and returns the output image. Optionally upload an image with the request to override `sample_traffic.jpg`.
- **Git**: To push to GitHub, run:
  ```powershell
  git init
  git remote add origin https://github.com/ruhitha007/2520080053-Deep_learning.git
  git add .
  git commit -m "VisionPort: All 20 tasks implemented and connected"
  git push -u origin main
  ```
