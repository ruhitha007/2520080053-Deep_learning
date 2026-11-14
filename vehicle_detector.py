"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Core Module: vehicle_detector.py
Model Name: traffic_detector
Project: VisionPort
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

class TrafficDetector:
    """
    VisionPort Core Traffic Detection and Analysis Engine
    Model Name: traffic_detector
    """
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.25):
        self.model_name = "traffic_detector"
        self.model_path = model_path
        self.conf_thresh = conf_thresh

        print(f"[VisionPort] Initializing {self.model_name} with weights '{model_path}'...")
        self.model = YOLO(model_path)
        
        # Target COCO vehicle classes
        self.VEHICLE_CLASSES = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

        # Class color palette for visual distinction (BGR format)
        self.CLASS_COLORS = {
            "car": (255, 140, 0),        # Deep sky blue / orange-blue
            "motorcycle": (200, 50, 255), # Vivid magenta
            "bus": (50, 205, 50),         # Lime green
            "truck": (0, 140, 255),       # Orange
            "vehicle": (0, 255, 255)      # Yellow default
        }

        # Traffic density thresholds
        self.LOW_MAX = 5
        self.MEDIUM_MAX = 10

        # Tracking state
        self.next_track_id = 1
        self.active_tracks = {}   # track_id -> dict(box, centroid, history, class_name, speed, wait_time, stopped_frames)
        self.max_disappeared = 15

        # Congestion monitoring buffer
        self.density_history = []

    # ---------------------------------------------------------
    # Task 6: Car Detection
    # ---------------------------------------------------------
    def detect_cars(self, frame, conf=None):
        """Detect ONLY cars (COCO class ID 2)."""
        conf_val = conf if conf is not None else self.conf_thresh
        results = self.model(frame, classes=[2], conf=conf_val, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "class_id": 2,
                    "class_name": "car",
                    "confidence": confidence
                })
        return detections

    # ---------------------------------------------------------
    # Task 7: Multiple Vehicle Detection (cars, bikes, buses, trucks)
    # ---------------------------------------------------------
    def detect_multiple_vehicles(self, frame, conf=None):
        """Detect cars, motorcycles, buses, and trucks."""
        conf_val = conf if conf is not None else self.conf_thresh
        target_classes = list(self.VEHICLE_CLASSES.keys())
        results = self.model(frame, classes=target_classes, conf=conf_val, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0])
                cls_name = self.VEHICLE_CLASSES.get(cls_id, "vehicle")
                confidence = float(box.conf[0])
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": confidence
                })
        return detections

    # ---------------------------------------------------------
    # Task 8: Display Labels and Confidence Scores
    # ---------------------------------------------------------
    def draw_detections(self, frame, detections, show_label=True, show_conf=True):
        """Draw high-contrast bounding boxes with class names and confidence scores."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = self.CLASS_COLORS.get(cls_name, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label string
            if show_label and show_conf:
                label = f"{cls_name} {conf:.2f}"
            elif show_label:
                label = f"{cls_name}"
            elif show_conf:
                label = f"{conf:.2f}"
            else:
                label = ""

            if label:
                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                # Label badge background
                tag_y1 = max(0, y1 - lh - baseline - 4)
                tag_y2 = y1
                tag_x2 = min(frame.shape[1], x1 + lw + 8)
                cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
                # Text in black or white depending on background
                cv2.putText(annotated, label, (x1 + 4, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return annotated

    # ---------------------------------------------------------
    # Task 9: Vehicle Counting with Deduplication
    # ---------------------------------------------------------
    def count_vehicles(self, detections, iou_threshold=0.65):
        """
        Count detected vehicles per class and total count.
        Avoids duplicate counts within the same frame via IoU deduplication.
        """
        # Deduplication via Non-Maximum Suppression / spatial IoU
        deduped = []
        for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            keep = True
            for kept_det in deduped:
                if self._calculate_iou(det["box"], kept_det["box"]) > iou_threshold:
                    keep = False
                    break
            if keep:
                deduped.append(det)

        counts = {name: 0 for name in self.VEHICLE_CLASSES.values()}
        for det in deduped:
            if det["class_name"] in counts:
                counts[det["class_name"]] += 1

        total = sum(counts.values())
        return counts, total, deduped

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    # ---------------------------------------------------------
    # Task 10: Traffic Density Classification
    # ---------------------------------------------------------
    def get_density(self, total_count, low_max=None, med_max=None):
        """Classify traffic density into LOW, MEDIUM, HIGH."""
        l_max = low_max if low_max is not None else self.LOW_MAX
        m_max = med_max if med_max is not None else self.MEDIUM_MAX

        if total_count <= l_max:
            return "LOW", (0, 200, 0)       # Green (BGR)
        elif total_count <= m_max:
            return "MEDIUM", (0, 165, 255)  # Orange
        else:
            return "HIGH", (0, 0, 255)      # Red

    # ---------------------------------------------------------
    # Task 11: Traffic Congestion Detection
    # ---------------------------------------------------------
    def check_congestion(self, current_density, window_size=10, min_high_ratio=0.7):
        """
        Detect persistent traffic congestion across multiple frames.
        Avoids false alerts from temporary single-frame spikes.
        """
        self.density_history.append(current_density)
        if len(self.density_history) > window_size:
            self.density_history.pop(0)

        high_frames = self.density_history.count("HIGH")
        persistence_ratio = high_frames / len(self.density_history)

        is_congested = (len(self.density_history) >= 5 and persistence_ratio >= min_high_ratio)
        return is_congested, persistence_ratio

    # ---------------------------------------------------------
    # Task 12: Individual Vehicle Tracking
    # ---------------------------------------------------------
    def track_vehicles(self, detections, max_distance=60):
        """
        Vehicle tracker assigning persistent IDs across video frames.
        Computes centroids and updates trajectory history.
        """
        current_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            current_centroids.append((cx, cy, det))

        updated_active = {}
        used_current = set()

        # Match existing tracks to nearest current detections
        for track_id, track_data in self.active_tracks.items():
            prev_cx, prev_cy = track_data["centroid"]
            best_match_idx = None
            best_dist = float("inf")

            for idx, (cx, cy, det) in enumerate(current_centroids):
                if idx in used_current:
                    continue
                dist = np.hypot(cx - prev_cx, cy - prev_cy)
                if dist < best_dist and dist <= max_distance:
                    best_dist = dist
                    best_match_idx = idx

            if best_match_idx is not None:
                cx, cy, det = current_centroids[best_match_idx]
                used_current.add(best_match_idx)
                
                history = track_data["history"]
                history.append((cx, cy))
                if len(history) > 30:
                    history.pop(0)

                updated_active[track_id] = {
                    "box": det["box"],
                    "centroid": (cx, cy),
                    "history": history,
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "disappeared": 0,
                    "speed_kmh": track_data.get("speed_kmh", 0.0),
                    "wait_time_sec": track_data.get("wait_time_sec", 0.0),
                    "stopped_frames": track_data.get("stopped_frames", 0)
                }
            else:
                # Mark track as disappeared
                disappeared = track_data.get("disappeared", 0) + 1
                if disappeared <= self.max_disappeared:
                    track_data["disappeared"] = disappeared
                    updated_active[track_id] = track_data

        # Register new detections as new tracks
        for idx, (cx, cy, det) in enumerate(current_centroids):
            if idx not in used_current:
                new_id = self.next_track_id
                self.next_track_id += 1
                updated_active[new_id] = {
                    "box": det["box"],
                    "centroid": (cx, cy),
                    "history": [(cx, cy)],
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "disappeared": 0,
                    "speed_kmh": 0.0,
                    "wait_time_sec": 0.0,
                    "stopped_frames": 0
                }

        self.active_tracks = updated_active
        return self.active_tracks

    # ---------------------------------------------------------
    # Task 13: Approximate Vehicle Speed Estimation
    # ---------------------------------------------------------
    def estimate_speed(self, tracks, fps=20.0, pixels_per_meter=15.0):
        """
        Estimate approximate speed (km/h) using centroid movement across frames.
        Speed = (pixel_displacement / pixels_per_meter) / elapsed_time * 3.6
        Camera calibration note: Exact speed requires real-world measurement calibration.
        """
        dt = 1.0 / max(1.0, fps)
        for track_id, data in tracks.items():
            history = data["history"]
            if len(history) >= 2:
                # Average displacement over last 3 frames for stability
                pts = history[-min(len(history), 4):]
                p_start = pts[0]
                p_end = pts[-1]
                steps = len(pts) - 1
                
                disp_px = np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
                disp_meters = disp_px / pixels_per_meter
                speed_mps = disp_meters / (steps * dt)
                speed_kmh = speed_mps * 3.6
                data["speed_kmh"] = round(float(speed_kmh), 1)
            else:
                data["speed_kmh"] = 0.0
        return tracks

    # ---------------------------------------------------------
    # Task 14: Traffic Signal Status Detection
    # ---------------------------------------------------------
    def detect_traffic_signal(self, frame, roi=None):
        """
        Identify traffic signal status (RED, YELLOW, GREEN, UNCERTAIN)
        using HSV color segmentation and circle/light detection.
        """
        if roi is not None:
            x, y, w, h = roi
            signal_crop = frame[y:y+h, x:x+w]
        else:
            # Automatic ROI search: look for traffic light class or upper quadrant of junction
            signal_crop = frame[0:int(frame.shape[0] * 0.45), :]

        if signal_crop.size == 0:
            return "UNCERTAIN", (128, 128, 128), None

        hsv = cv2.cvtColor(signal_crop, cv2.COLOR_BGR2HSV)

        # HSV color ranges for traffic signals
        # Red has two wraps in Hue (0-10 and 160-180)
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        yellow_mask = cv2.inRange(hsv, np.array([18, 100, 100]), np.array([35, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([90, 255, 255]))

        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)

        pixel_counts = {
            "RED": (red_pixels, (0, 0, 255)),
            "YELLOW": (yellow_pixels, (0, 255, 255)),
            "GREEN": (green_pixels, (0, 255, 0))
        }

        # Threshold for minimum active pixels to avoid random noise
        min_pixels = 30
        best_state = "UNCERTAIN"
        best_color = (128, 128, 128)
        max_count = 0

        for state, (cnt, col) in pixel_counts.items():
            if cnt > max_count and cnt >= min_pixels:
                max_count = cnt
                best_state = state
                best_color = col

        return best_state, best_color, {"red": red_pixels, "yellow": yellow_pixels, "green": green_pixels}

    # ---------------------------------------------------------
    # Task 15: Traffic Waiting-Time Estimation
    # ---------------------------------------------------------
    def estimate_waiting_time(self, tracks, signal_status, fps=20.0, stop_speed_thresh=5.0):
        """
        Estimate traffic waiting time for stationary vehicles near junction during RED/stop.
        Returns detailed report distinguishing estimated wait from actual measured wait.
        """
        dt = 1.0 / max(1.0, fps)
        is_stop_phase = (signal_status == "RED" or signal_status == "UNCERTAIN")

        for track_id, data in tracks.items():
            speed = data.get("speed_kmh", 0.0)
            if speed < stop_speed_thresh and is_stop_phase:
                data["stopped_frames"] = data.get("stopped_frames", 0) + 1
                data["wait_time_sec"] = round(data["stopped_frames"] * dt, 1)
            else:
                # Vehicle moving or green signal
                if not is_stop_phase and speed >= stop_speed_thresh:
                    data["stopped_frames"] = max(0, data.get("stopped_frames", 0) - 1)
                    data["wait_time_sec"] = round(data["stopped_frames"] * dt, 1)

        # Generate aggregated report
        waiting_vehicles = [
            {"id": tid, "class": d["class_name"], "wait_time_sec": d["wait_time_sec"], "speed": d["speed_kmh"]}
            for tid, d in tracks.items() if d.get("wait_time_sec", 0.0) > 0.0
        ]
        
        avg_wait = np.mean([v["wait_time_sec"] for v in waiting_vehicles]) if waiting_vehicles else 0.0
        max_wait = max([v["wait_time_sec"] for v in waiting_vehicles]) if waiting_vehicles else 0.0

        report = {
            "signal_status": signal_status,
            "total_tracked_vehicles": len(tracks),
            "waiting_vehicle_count": len(waiting_vehicles),
            "average_estimated_wait_sec": round(float(avg_wait), 1),
            "maximum_estimated_wait_sec": round(float(max_wait), 1),
            "waiting_details": waiting_vehicles,
            "estimation_note": "ESTIMATED metric based on stationary frame duration and signal phase. Actual measured wait requires loop detectors or GPS ground truth."
        }
        return report
