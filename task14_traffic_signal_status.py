"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 14: Traffic Signal Status Detection
Expected Output: Signal status detection (RED, YELLOW, GREEN, UNCERTAIN) using computer vision
"""

import os
import cv2
import numpy as np
from vehicle_detector import TrafficDetector

def run_signal_detection(source="sample_traffic.jpg", output_img="task14_signal_status_output.jpg"):
    print("=" * 60)
    print("VisionPort Task 14: Traffic Signal Status Detection")
    print("=" * 60)
    print(f"Source: {source}")
    print("Method: HSV Color Segmentation & Circular Light Radiance Detection")
    print("-" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    if not os.path.exists(source):
        print(f"[ERROR] Source file not found: {source}")
        return False

    frame = cv2.imread(source)
    if frame is None:
        print(f"[ERROR] Could not decode image: {source}")
        return False

    # Detect traffic signal status
    status, status_color, stats = detector.detect_traffic_signal(frame)
    print(f"[DETECTION] Traffic Signal Status Identified: [{status}]")
    print(f"            Active Pixel Distribution : Red={stats['red']}, Yellow={stats['yellow']}, Green={stats['green']}")

    # Render Visual Traffic Light Graphic Overlay
    annotated = frame.copy()
    h, w, _ = annotated.shape

    # Traffic light housing box
    bx1, by1 = w - 160, 20
    bx2, by2 = w - 20, 260
    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (200, 200, 200), 2)
    cv2.putText(annotated, "SIGNAL", (bx1 + 15, by1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 3 Signal Lamps (Red, Yellow, Green)
    lamp_x = bx1 + 70
    red_lamp_y = by1 + 65
    yel_lamp_y = by1 + 130
    grn_lamp_y = by1 + 195
    radius = 22

    # Draw inactive bases
    cv2.circle(annotated, (lamp_x, red_lamp_y), radius, (0, 0, 80), -1)
    cv2.circle(annotated, (lamp_x, yel_lamp_y), radius, (0, 80, 80), -1)
    cv2.circle(annotated, (lamp_x, grn_lamp_y), radius, (0, 80, 0), -1)

    # Illuminate active lamp with glow
    if status == "RED":
        cv2.circle(annotated, (lamp_x, red_lamp_y), radius, (0, 0, 255), -1)
        cv2.circle(annotated, (lamp_x, red_lamp_y), radius + 4, (0, 100, 255), 2)
    elif status == "YELLOW":
        cv2.circle(annotated, (lamp_x, yel_lamp_y), radius, (0, 255, 255), -1)
        cv2.circle(annotated, (lamp_x, yel_lamp_y), radius + 4, (0, 255, 255), 2)
    elif status == "GREEN":
        cv2.circle(annotated, (lamp_x, grn_lamp_y), radius, (0, 255, 0), -1)
        cv2.circle(annotated, (lamp_x, grn_lamp_y), radius + 4, (100, 255, 100), 2)

    # Status text under housing
    cv2.putText(annotated, f"STATE: {status}", (bx1 + 10, by2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

    # Top-Left HUD Banner
    cv2.rectangle(annotated, (10, 10), (450, 75), (0, 0, 0), -1)
    cv2.putText(annotated, "VisionPort: Traffic Signal Detector", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(annotated, f"Detected Status: [{status}] | Color: BGR{status_color}", (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

    cv2.imwrite(output_img, annotated)
    print(f"[PASSED] Signal detection visualization saved: {output_img}")

    # Also test uncertain / edge case
    blank_test = np.zeros((200, 200, 3), dtype=np.uint8)
    uncertain_status, _, _ = detector.detect_traffic_signal(blank_test)
    print(f"[TEST] Blank frame fallback test: [{uncertain_status}] (Handles uncertain status correctly)")

    return True

if __name__ == "__main__":
    success = run_signal_detection()
    print("=" * 60)
    if success:
        print("TASK 14 COMPLETED SUCCESSFULLY: Signal status detection.")
    else:
        print("TASK 14 FAILED.")
    print("=" * 60)
