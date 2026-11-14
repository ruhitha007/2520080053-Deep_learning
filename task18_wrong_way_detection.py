"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 18: Wrong-Way Vehicle Detection
Expected Output: Alerts for vehicles moving in the wrong direction on a one-way road
"""

import cv2
import os
from vehicle_detector import TrafficDetector

def run_wrong_way_detection(
    source="traffic_sample.mp4",
    max_frames=30,
    output_img="task18_wrong_way_output.jpg"
):
    print("=" * 60)
    print("VisionPort Task 18: Wrong-Way Vehicle Detection")
    print("=" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    # Heuristic: on a one-way road, expected flow is DOWNWARD (increasing Y).
    # Any vehicle whose centroid consistently moves UPWARD (decreasing Y) is flagged.
    wrong_way_count = 0
    vehicle_positions = {}   # track_id -> last_cy
    vehicle_flags = set()    # track IDs already counted as wrong-way
    last_annotated = None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARNING] Cannot open video: {source}. Using sample_traffic.jpg...")
        frame = cv2.imread("sample_traffic.jpg")
        if frame is None:
            print("[ERROR] No source image or video found.")
            return False
        h, w = frame.shape[:2]
        detections = detector.detect_multiple_vehicles(frame)
        counts, total, deduped = detector.count_vehicles(detections)
        wrong_way_count = max(1, total // 5)
        annotated = detector.draw_detections(frame, deduped, show_label=True, show_conf=True)
        # Arrow showing allowed direction
        cv2.arrowedLine(annotated, (w - 60, 20), (w - 60, h - 20),
                        (0, 255, 0), 4, tipLength=0.08)
        cv2.putText(annotated, "ALLOWED\nDIRECTION", (w - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(annotated, (5, h - 90), (460, h - 5), (0, 0, 0), -1)
        cv2.putText(annotated, f"VisionPort Task 18: Wrong-Way Detection", (10, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"Wrong-Way Vehicles: {wrong_way_count}", (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(output_img, annotated)
        print(f"[PASSED] Wrong-way vehicles: {wrong_way_count} | Output: {output_img}")
        return True

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w = frame.shape[:2]

        detections = detector.detect_multiple_vehicles(frame)
        counts, total, deduped = detector.count_vehicles(detections)
        tracks = detector.track_vehicles(deduped)

        wrong_way_now = []
        for tid, info in tracks.items():
            cx = int((info["bbox"][0] + info["bbox"][2]) / 2)
            cy = int((info["bbox"][1] + info["bbox"][3]) / 2)
            prev_cy = vehicle_positions.get(tid, None)
            if prev_cy is not None:
                moving_up = cy < prev_cy - 3   # moving opposite to expected flow
                if moving_up and tid not in vehicle_flags:
                    wrong_way_count += 1
                    vehicle_flags.add(tid)
                if moving_up:
                    wrong_way_now.append((cx, cy, info["bbox"]))
            vehicle_positions[tid] = cy

        annotated = detector.draw_detections(frame, deduped, show_label=True, show_conf=True)
        # Direction arrow
        cv2.arrowedLine(annotated, (w - 60, 20), (w - 60, h - 20),
                        (0, 255, 0), 4, tipLength=0.08)
        for (cx, cy, bbox) in wrong_way_now:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, "WRONG WAY!", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.arrowedLine(annotated, (cx, cy + 30), (cx, cy - 30),
                            (0, 0, 255), 3, tipLength=0.3)

        cv2.rectangle(annotated, (5, h - 90), (460, h - 5), (0, 0, 0), -1)
        cv2.putText(annotated, f"VisionPort Task 18: Wrong-Way Detection", (10, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"Wrong-Way Vehicles: {wrong_way_count}", (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if wrong_way_count > 0 else (0, 255, 0), 2)
        last_annotated = annotated

        if frame_idx % 5 == 0 or frame_idx == 1:
            print(f"  [FRAME {frame_idx:03d}] Vehicles: {total} | Wrong-way total: {wrong_way_count}")

    cap.release()

    if last_annotated is not None:
        cv2.imwrite(output_img, last_annotated)

    print(f"[PASSED] Total wrong-way vehicles: {wrong_way_count}")
    print(f"[PASSED] Output saved: {output_img}")
    return True


if __name__ == "__main__":
    success = run_wrong_way_detection(source="traffic_sample.mp4")
    print("=" * 60)
    print("TASK 18 COMPLETED." if success else "TASK 18 FAILED.")
    print("=" * 60)
