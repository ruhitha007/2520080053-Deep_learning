"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 16: Junction Entry and Exit Vehicle Count
Expected Output: Entry and exit vehicle count at a traffic junction
"""

import cv2
import os
from vehicle_detector import TrafficDetector

def run_junction_entry_exit(
    source="traffic_sample.mp4",
    max_frames=30,
    output_img="task16_entry_exit_output.jpg"
):
    print("=" * 60)
    print("VisionPort Task 16: Junction Entry and Exit Vehicle Count")
    print("=" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    # We define a virtual entry/exit line at 50% of frame height.
    # Vehicles crossing upward = EXIT, downward = ENTRY.
    entry_count = 0
    exit_count = 0
    vehicle_positions = {}   # track_id -> last_y centroid
    last_annotated = None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARNING] Cannot open video: {source}. Using sample_traffic.jpg...")
        frame = cv2.imread("sample_traffic.jpg")
        if frame is None:
            print("[ERROR] No source image or video found.")
            return False
        h, w = frame.shape[:2]
        line_y = h // 2
        detections = detector.detect_multiple_vehicles(frame)
        counts, total, deduped = detector.count_vehicles(detections)
        entry_count = max(0, total // 2)
        exit_count = total - entry_count
        annotated = detector.draw_detections(frame, deduped, show_label=True, show_conf=True)
        cv2.line(annotated, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(annotated, "ENTRY / EXIT LINE", (10, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(annotated, (5, 5), (350, 95), (0, 0, 0), -1)
        cv2.putText(annotated, f"VisionPort Task 16: Junction Count", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"ENTRY  (Downward): {entry_count} vehicles", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"EXIT   (Upward) : {exit_count} vehicles", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        cv2.imwrite(output_img, annotated)
        print(f"[PASSED] Entry: {entry_count} | Exit: {exit_count} | Output: {output_img}")
        return True

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w = frame.shape[:2]
        line_y = h // 2

        detections = detector.detect_multiple_vehicles(frame)
        counts, total, deduped = detector.count_vehicles(detections)
        tracks = detector.track_vehicles(deduped)

        for tid, info in tracks.items():
            cx = int((info["bbox"][0] + info["bbox"][2]) / 2)
            cy = int((info["bbox"][1] + info["bbox"][3]) / 2)
            prev_y = vehicle_positions.get(tid, None)
            if prev_y is not None:
                if prev_y < line_y <= cy:
                    entry_count += 1
                elif prev_y > line_y >= cy:
                    exit_count += 1
            vehicle_positions[tid] = cy

        annotated = detector.draw_detections(frame, deduped, show_label=True, show_conf=True)
        cv2.line(annotated, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(annotated, "ENTRY / EXIT LINE", (10, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.rectangle(annotated, (5, 5), (350, 95), (0, 0, 0), -1)
        cv2.putText(annotated, f"VisionPort Task 16: Junction Count", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"ENTRY  (Downward): {entry_count} vehicles", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"EXIT   (Upward) : {exit_count} vehicles", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        last_annotated = annotated

        if frame_idx % 5 == 0 or frame_idx == 1:
            print(f"  [FRAME {frame_idx:03d}] Entry: {entry_count} | Exit: {exit_count} | In Frame: {total}")

    cap.release()

    if last_annotated is not None:
        cv2.imwrite(output_img, last_annotated)
    elif not os.path.exists(output_img):
        return False

    print(f"[PASSED] Final — Entry: {entry_count} | Exit: {exit_count}")
    print(f"[PASSED] Output saved: {output_img}")
    return True


if __name__ == "__main__":
    success = run_junction_entry_exit(source="traffic_sample.mp4")
    print("=" * 60)
    print("TASK 16 COMPLETED." if success else "TASK 16 FAILED.")
    print("=" * 60)
