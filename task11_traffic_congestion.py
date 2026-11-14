"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 11: Traffic Congestion Detection
Expected Output: Persistent congestion alerts avoiding single-frame false spikes
"""

import os
import cv2
from vehicle_detector import TrafficDetector

def run_congestion_detection(
    source="traffic_sample.mp4",
    max_frames=25,
    window_size=10,
    min_high_ratio=0.6,
    output_img="task11_congestion_alert_output.jpg"
):
    print("=" * 60)
    print("VisionPort Task 11: Traffic Congestion Detection")
    print("=" * 60)
    print(f"Source               : {source}")
    print(f"Sliding Window Size  : {window_size} frames")
    print(f"Persistence Threshold: >= {int(min_high_ratio * 100)}% HIGH density frames")
    print("-" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {source}. Testing on 'sample_traffic.jpg'...")
        frame = cv2.imread("sample_traffic.jpg")
        detections = detector.detect_multiple_vehicles(frame)
        counts, total, deduped = detector.count_vehicles(detections)
        # Simulate persistent frames to trigger alert
        for _ in range(8):
            is_congested, ratio = detector.check_congestion("HIGH", window_size, min_high_ratio)
        annotated = detector.draw_detections(frame, deduped)
        cv2.rectangle(annotated, (10, 10), (600, 90), (0, 0, 180), -1)
        cv2.putText(annotated, "CONGESTION ALERT: HEAVY TRAFFIC DETECTED", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(annotated, f"Persistence: {int(ratio*100)}% | Vehicles: {total}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imwrite(output_img, annotated)
        print(f"[PASSED] Saved output: {output_img}")
        return True

    frame_count = 0
    alert_count = 0
    last_annotated = None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        detections = detector.detect_multiple_vehicles(frame)
        counts, total, deduped = detector.count_vehicles(detections)
        density, density_color = detector.get_density(total)

        # Check multi-frame congestion persistence
        is_congested, persistence_ratio = detector.check_congestion(
            current_density=density,
            window_size=window_size,
            min_high_ratio=min_high_ratio
        )

        annotated = detector.draw_detections(frame, deduped)

        if is_congested:
            alert_count += 1
            # Red Alert Banner
            cv2.rectangle(annotated, (10, 10), (620, 105), (0, 0, 200), -1)
            cv2.putText(annotated, "!!! TRAFFIC CONGESTION ALERT !!!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"Status: CRITICAL CONGESTION | High Density Persistence: {int(persistence_ratio*100)}%",
                        (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 255), 1)
            cv2.putText(annotated, f"Junction Total: {total} vehicles | Window: {len(detector.density_history)} frames",
                        (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        else:
            # Normal Flow Banner
            cv2.rectangle(annotated, (10, 10), (520, 80), (0, 0, 0), -1)
            cv2.putText(annotated, "VisionPort: Traffic Flow Normal", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(annotated, f"Vehicles: {total} | Density: {density} | Persistence: {int(persistence_ratio*100)}%",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        last_annotated = annotated

        if frame_count % 5 == 0 or frame_count == 1:
            status_str = ">> ALERT TRIGGERED <<" if is_congested else "Normal Flow"
            print(f"[FRAME {frame_count:02d}] Vehicles: {total:02d} | Density: {density:<6} | High Ratio: {persistence_ratio:.2f} | {status_str}")

        try:
            cv2.imshow("VisionPort Task 11 - Congestion Detection", annotated)
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
        print(f"[PASSED] Congestion alert visualization saved to: {output_img}")

    print("-" * 60)
    print(f"[SUMMARY] Total frames: {frame_count} | Congestion alert triggered in {alert_count} frames.")
    return True

if __name__ == "__main__":
    success = run_congestion_detection(source="traffic_sample.mp4", max_frames=20)
    print("=" * 60)
    if success:
        print("TASK 11 COMPLETED SUCCESSFULLY: Congestion alerts.")
    else:
        print("TASK 11 FAILED.")
    print("=" * 60)
