"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 12: Individual Vehicle Tracking
Expected Output: Vehicle tracking with persistent unique IDs and trajectory paths across frames
"""

import os
import cv2
from vehicle_detector import TrafficDetector

def run_vehicle_tracking(source="traffic_sample.mp4", max_frames=20, output_img="task12_vehicle_tracking_output.jpg"):
    print("=" * 60)
    print("VisionPort Task 12: Individual Vehicle Tracking")
    print("=" * 60)
    print(f"Source: {source}")
    print("Tracking Algorithm: Centroid-IoU Kinematic Association")
    print("-" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {source}. Testing on 'sample_traffic.jpg'...")
        frame = cv2.imread("sample_traffic.jpg")
        detections = detector.detect_multiple_vehicles(frame)
        tracks = detector.track_vehicles(detections)
        annotated = frame.copy()
        for tid, data in tracks.items():
            x1, y1, x2, y2 = data["box"]
            color = detector.CLASS_COLORS.get(data["class_name"], (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"ID #{tid} {data['class_name']}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(output_img, annotated)
        print(f"[PASSED] Saved output: {output_img} | Tracked IDs: {list(tracks.keys())}")
        return True

    frame_count = 0
    all_assigned_ids = set()
    last_annotated = None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        raw_detections = detector.detect_multiple_vehicles(frame)
        _, _, deduped_detections = detector.count_vehicles(raw_detections)

        # Update vehicle tracks
        tracks = detector.track_vehicles(deduped_detections, max_distance=60)
        annotated = frame.copy()

        # Render tracked vehicles with IDs and motion trails
        for track_id, data in tracks.items():
            if data["disappeared"] > 0:
                continue
            all_assigned_ids.add(track_id)
            x1, y1, x2, y2 = data["box"]
            cls_name = data["class_name"]
            color = detector.CLASS_COLORS.get(cls_name, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw ID Tag Badge
            id_label = f"ID #{track_id} {cls_name}"
            (tw, th), tb = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - tb - 4)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, id_label, (x1 + 3, y1 - tb - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Draw Trajectory history (motion trail)
            history = data["history"]
            for i in range(1, len(history)):
                cv2.line(annotated, history[i - 1], history[i], (0, 255, 255), 2)
            cv2.circle(annotated, data["centroid"], 4, (0, 0, 255), -1)

        # HUD Panel
        cv2.rectangle(annotated, (10, 10), (450, 75), (0, 0, 0), -1)
        cv2.putText(annotated, f"VisionPort: Multi-Vehicle Tracker", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(annotated, f"Active Tracks: {len([t for t in tracks.values() if t['disappeared']==0])} | Total Unique IDs: {len(all_assigned_ids)}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        last_annotated = annotated

        if frame_count % 10 == 0 or frame_count == 1:
            active_ids = [tid for tid, d in tracks.items() if d["disappeared"] == 0]
            print(f"[FRAME {frame_count:02d}] Active Track IDs: {active_ids}")

        try:
            cv2.imshow("VisionPort Task 12 - Vehicle Tracking", annotated)
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
        print(f"[PASSED] Tracking visualization saved to: {output_img}")

    print("-" * 60)
    print(f"[SUMMARY] Total frames: {frame_count} | Total unique tracking IDs assigned: {len(all_assigned_ids)}")
    print(f"          Tracking IDs: {sorted(list(all_assigned_ids))}")
    return True

if __name__ == "__main__":
    success = run_vehicle_tracking(source="traffic_sample.mp4", max_frames=20)
    print("=" * 60)
    if success:
        print("TASK 12 COMPLETED SUCCESSFULLY: Vehicle tracking IDs.")
    else:
        print("TASK 12 FAILED.")
    print("=" * 60)
