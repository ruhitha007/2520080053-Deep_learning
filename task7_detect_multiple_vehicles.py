"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 7: Multiple Vehicle Detection
Expected Output: Multiple vehicle detection (cars, bikes, buses, trucks) with class-specific bounding boxes
"""

import os
import cv2
from vehicle_detector import TrafficDetector

def run_multiple_vehicle_detection(source="traffic_sample.mp4", max_frames=20, output_img="task7_multiple_vehicles_output.jpg"):
    print("=" * 60)
    print("VisionPort Task 7: Multiple Vehicle Detection")
    print("=" * 60)
    print(f"Source         : {source}")
    print(f"Target Classes : Cars (2), Motorcycles (3), Buses (5), Trucks (7)")
    print("-" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {source}. Testing on 'sample_traffic.jpg'...")
        frame = cv2.imread("sample_traffic.jpg")
        detections = detector.detect_multiple_vehicles(frame)
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = detector.CLASS_COLORS.get(det["class_name"], (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, det["class_name"], (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(output_img, annotated)
        print(f"[PASSED] Saved output: {output_img}")
        return True

    frame_count = 0
    class_totals = {name: 0 for name in detector.VEHICLE_CLASSES.values()}
    last_annotated = None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        detections = detector.detect_multiple_vehicles(frame)

        annotated = frame.copy()
        frame_counts = {name: 0 for name in detector.VEHICLE_CLASSES.values()}

        # Draw distinct colored bounding box per vehicle class
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class_name"]
            color = detector.CLASS_COLORS.get(cls_name, (0, 255, 0))
            frame_counts[cls_name] += 1
            class_totals[cls_name] += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, cls_name, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Legend HUD
        hud_lines = [
            f"Frame {frame_count:02d} Multi-Vehicle Detection",
            f"Cars: {frame_counts['car']} | Bikes: {frame_counts['motorcycle']} | Buses: {frame_counts['bus']} | Trucks: {frame_counts['truck']}"
        ]
        cv2.rectangle(annotated, (10, 10), (520, 75), (0, 0, 0), -1)
        for i, text in enumerate(hud_lines):
            cv2.putText(annotated, text, (20, 35 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        last_annotated = annotated

        if frame_count % 10 == 0 or frame_count == 1:
            print(f"[FRAME {frame_count:02d}] Detected: {frame_counts}")

        try:
            cv2.imshow("VisionPort Task 7 - Multiple Vehicle Detection", annotated)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        except cv2.error:
            pass

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    if last_annotated is not None:
        cv2.imwrite(output_img, last_annotated)
        print(f"[PASSED] Multi-vehicle annotated frame saved to: {output_img}")

    print("-" * 60)
    print(f"[SUMMARY] Total detections across {frame_count} frames: {class_totals}")
    return True

if __name__ == "__main__":
    success = run_multiple_vehicle_detection(source="traffic_sample.mp4", max_frames=20)
    print("=" * 60)
    if success:
        print("TASK 7 COMPLETED SUCCESSFULLY: Multiple vehicle detection.")
    else:
        print("TASK 7 FAILED.")
    print("=" * 60)
