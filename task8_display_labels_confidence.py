"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 8: Vehicle Names and Confidence Scores
Expected Output: Labelled detection output with readable class names and confidence scores
"""

import os
import cv2
from vehicle_detector import TrafficDetector

def run_label_confidence_display(source="traffic_sample.mp4", max_frames=20, output_img="task8_labels_confidence_output.jpg"):
    print("=" * 60)
    print("VisionPort Task 8: Vehicle Names and Confidence Scores")
    print("=" * 60)
    print(f"Source: {source}")
    print("-" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {source}. Testing on 'sample_traffic.jpg'...")
        frame = cv2.imread("sample_traffic.jpg")
        detections = detector.detect_multiple_vehicles(frame)
        annotated = detector.draw_detections(frame, detections, show_label=True, show_conf=True)
        cv2.imwrite(output_img, annotated)
        print(f"[PASSED] Saved output: {output_img}")
        return True

    frame_count = 0
    last_annotated = None
    all_confs = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        detections = detector.detect_multiple_vehicles(frame)
        
        # Draw high-contrast labels and confidence scores
        annotated = detector.draw_detections(frame, detections, show_label=True, show_conf=True)

        for det in detections:
            all_confs.append(det["confidence"])

        # HUD banner
        cv2.rectangle(annotated, (10, 10), (480, 50), (0, 0, 0), -1)
        cv2.putText(annotated, f"VisionPort Task 8: Labels & Conf Scores | Frame {frame_count:02d}",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        last_annotated = annotated

        if frame_count % 10 == 0 or frame_count == 1:
            sample_labels = [f"{d['class_name']}:{d['confidence']:.2f}" for d in detections[:4]]
            print(f"[FRAME {frame_count:02d}] Sample labels: {', '.join(sample_labels)}")

        try:
            cv2.imshow("VisionPort Task 8 - Labels & Confidence", annotated)
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
        print(f"[PASSED] Labelled detection frame saved to: {output_img}")

    avg_conf = sum(all_confs) / max(1, len(all_confs))
    print("-" * 60)
    print(f"[SUMMARY] Total labelled vehicles: {len(all_confs)} | Avg Confidence: {avg_conf:.2f}")
    return True

if __name__ == "__main__":
    success = run_label_confidence_display(source="traffic_sample.mp4", max_frames=20)
    print("=" * 60)
    if success:
        print("TASK 8 COMPLETED SUCCESSFULLY: Labelled detection output.")
    else:
        print("TASK 8 FAILED.")
    print("=" * 60)
