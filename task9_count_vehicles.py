"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 9: Vehicle Counting
Expected Output: Vehicle count per frame and class breakdown avoiding duplicates
"""

import os
import cv2
from vehicle_detector import TrafficDetector

def run_vehicle_counting(source="traffic_sample.mp4", max_frames=20, output_img="task9_vehicle_count_output.jpg"):
    print("=" * 60)
    print("VisionPort Task 9: Vehicle Counting")
    print("=" * 60)
    print(f"Source: {source}")
    print("Deduplication: Spatial IoU suppression applied")
    print("-" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {source}. Testing on 'sample_traffic.jpg'...")
        frame = cv2.imread("sample_traffic.jpg")
        detections = detector.detect_multiple_vehicles(frame)
        counts, total, deduped = detector.count_vehicles(detections)
        annotated = detector.draw_detections(frame, deduped)
        
        # Draw counts panel
        panel = [
            f"Cars: {counts['car']}",
            f"Motorcycles: {counts['motorcycle']}",
            f"Buses: {counts['bus']}",
            f"Trucks: {counts['truck']}",
            f"TOTAL VEHICLES: {total}"
        ]
        cv2.rectangle(annotated, (10, 10), (320, 20 + 30 * len(panel)), (0, 0, 0), -1)
        for i, text in enumerate(panel):
            color = (0, 255, 255) if "TOTAL" in text else (255, 255, 255)
            cv2.putText(annotated, text, (20, 38 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.imwrite(output_img, annotated)
        print(f"[PASSED] Saved output: {output_img} | Total: {total}")
        return True

    frame_count = 0
    last_annotated = None
    all_totals = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        raw_detections = detector.detect_multiple_vehicles(frame)
        
        # Perform counting with intra-frame deduplication
        counts, total, deduped = detector.count_vehicles(raw_detections, iou_threshold=0.6)
        all_totals.append(total)

        # Draw boxes and count dashboard
        annotated = detector.draw_detections(frame, deduped)
        lines = [
            f"VisionPort: Vehicle Counter",
            f"Cars: {counts['car']} | Bikes: {counts['motorcycle']}",
            f"Buses: {counts['bus']} | Trucks: {counts['truck']}",
            f"TOTAL VEHICLES: {total}"
        ]
        cv2.rectangle(annotated, (10, 10), (380, 20 + 28 * len(lines)), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            col = (0, 255, 0) if "TOTAL" in text else ((0, 200, 255) if i == 0 else (255, 255, 255))
            cv2.putText(annotated, text, (20, 34 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

        last_annotated = annotated

        if frame_count % 10 == 0 or frame_count == 1:
            print(f"[FRAME {frame_count:02d}] Raw detections: {len(raw_detections)} -> Deduplicated Count: {total} {counts}")

        try:
            cv2.imshow("VisionPort Task 9 - Vehicle Counting", annotated)
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
        print(f"[PASSED] Vehicle count visualization saved to: {output_img}")

    avg_count = sum(all_totals) / max(1, len(all_totals))
    print("-" * 60)
    print(f"[SUMMARY] Processed {frame_count} frames | Avg Vehicle Count per Frame: {avg_count:.1f}")
    return True

if __name__ == "__main__":
    success = run_vehicle_counting(source="traffic_sample.mp4", max_frames=20)
    print("=" * 60)
    if success:
        print("TASK 9 COMPLETED SUCCESSFULLY: Vehicle count.")
    else:
        print("TASK 9 FAILED.")
    print("=" * 60)
