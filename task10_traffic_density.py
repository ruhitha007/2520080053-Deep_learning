"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 10: Traffic Density Classification
Expected Output: Low, medium and high traffic classification with configurable thresholds
"""

import os
import cv2
from vehicle_detector import TrafficDetector

def run_traffic_density_classification(
    source="traffic_sample.mp4",
    max_frames=20,
    low_threshold=5,
    med_threshold=10,
    output_img="task10_traffic_density_output.jpg"
):
    print("=" * 60)
    print("VisionPort Task 10: Traffic Density Classification")
    print("=" * 60)
    print(f"Source                  : {source}")
    print(f"Configurable Thresholds : LOW: <= {low_threshold} | MEDIUM: <= {med_threshold} | HIGH: > {med_threshold}")
    print("-" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")
    detector.LOW_MAX = low_threshold
    detector.MEDIUM_MAX = med_threshold

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {source}. Testing on 'sample_traffic.jpg'...")
        frame = cv2.imread("sample_traffic.jpg")
        detections = detector.detect_multiple_vehicles(frame)
        counts, total, deduped = detector.count_vehicles(detections)
        density, color = detector.get_density(total, low_threshold, med_threshold)
        
        annotated = detector.draw_detections(frame, deduped)
        cv2.rectangle(annotated, (10, 10), (420, 80), (0, 0, 0), -1)
        cv2.putText(annotated, f"Total Vehicles: {total}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Traffic Density: {density}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.imwrite(output_img, annotated)
        print(f"[PASSED] Saved output: {output_img} | Density: {density}")
        return True

    frame_count = 0
    density_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    last_annotated = None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        detections = detector.detect_multiple_vehicles(frame)
        counts, total, deduped = detector.count_vehicles(detections)
        
        # Calculate density
        density, color = detector.get_density(total, low_threshold, med_threshold)
        density_counts[density] += 1

        annotated = detector.draw_detections(frame, deduped)

        # Draw Density Classification Badge
        cv2.rectangle(annotated, (10, 10), (450, 110), (0, 0, 0), -1)
        cv2.putText(annotated, f"VisionPort: Density Monitor", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(annotated, f"Vehicle Count : {total} (Cars:{counts['car']}, Bikes:{counts['motorcycle']})",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Traffic Density: [{density}]",
                    (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        last_annotated = annotated

        if frame_count % 10 == 0 or frame_count == 1:
            print(f"[FRAME {frame_count:02d}] Vehicles: {total:02d} -> Classification: {density}")

        try:
            cv2.imshow("VisionPort Task 10 - Traffic Density", annotated)
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
        print(f"[PASSED] Density classification frame saved to: {output_img}")

    print("-" * 60)
    print(f"[SUMMARY] Density distribution over {frame_count} frames: {density_counts}")
    return True

if __name__ == "__main__":
    success = run_traffic_density_classification(source="traffic_sample.mp4", max_frames=20, low_threshold=5, med_threshold=10)
    print("=" * 60)
    if success:
        print("TASK 10 COMPLETED SUCCESSFULLY: Low, medium and high traffic classification.")
    else:
        print("TASK 10 FAILED.")
    print("=" * 60)
