"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 13: Approximate Vehicle Speed Estimation
Expected Output: Vehicle speed estimation with camera calibration disclaimer and metrics
"""

import os
import cv2
from vehicle_detector import TrafficDetector

def run_speed_estimation(
    source="traffic_sample.mp4",
    max_frames=20,
    pixels_per_meter=15.0,  # Camera calibration scale (15 pixels = 1 meter)
    output_img="task13_speed_estimation_output.jpg"
):
    print("=" * 60)
    print("VisionPort Task 13: Approximate Vehicle Speed Estimation")
    print("=" * 60)
    print(f"Source                  : {source}")
    print(f"Calibration Scale (PPM) : {pixels_per_meter:.1f} pixels/meter")
    print("[DISCLAIMER] Speeds are APPROXIMATE optical estimates.")
    print("             Accurate speed requires survey-grade camera calibration.")
    print("-" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {source}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = 0
    last_annotated = None
    observed_speeds = {}

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        detections = detector.detect_multiple_vehicles(frame)
        _, _, deduped = detector.count_vehicles(detections)
        tracks = detector.track_vehicles(deduped)

        # Estimate speed using tracked positions over time
        tracks = detector.estimate_speed(tracks, fps=fps, pixels_per_meter=pixels_per_meter)

        annotated = frame.copy()

        for track_id, data in tracks.items():
            if data["disappeared"] > 0:
                continue
            x1, y1, x2, y2 = data["box"]
            cls_name = data["class_name"]
            speed = data.get("speed_kmh", 0.0)
            color = detector.CLASS_COLORS.get(cls_name, (0, 255, 0))

            if speed > 0:
                observed_speeds[track_id] = (cls_name, speed)

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Speed and ID badge
            label = f"ID #{track_id} {cls_name} | {speed:.1f} km/h"
            (tw, th), tb = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - tb - 4)), (x1 + tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - tb - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Calibration Disclaimer HUD
        cv2.rectangle(annotated, (10, 10), (550, 85), (0, 0, 0), -1)
        cv2.putText(annotated, "VisionPort: Vehicle Speed Estimator", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(annotated, f"FPS: {fps:.1f} | Calibration: {pixels_per_meter:.1f} px/m (Approximate)",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated, "* Ground truth requires 3D roadway perspective calibration",
                    (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

        last_annotated = annotated

        if frame_count % 10 == 0 or frame_count == 1:
            sample_speeds = [f"ID#{tid}:{d['speed_kmh']:.1f}km/h" for tid, d in list(tracks.items())[:3] if d['disappeared']==0]
            print(f"[FRAME {frame_count:02d}] Active Speeds: {', '.join(sample_speeds)}")

        try:
            cv2.imshow("VisionPort Task 13 - Speed Estimation", annotated)
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
        print(f"[PASSED] Speed estimation frame saved to: {output_img}")

    print("-" * 60)
    print(f"[SUMMARY] Tracked vehicle speed estimates:")
    for tid, (cname, spd) in sorted(observed_speeds.items()):
        print(f"          Vehicle ID #{tid:<3} ({cname:<10}) : Estimated Speed = {spd:.1f} km/h")
    return True

if __name__ == "__main__":
    success = run_speed_estimation(source="traffic_sample.mp4", max_frames=20)
    print("=" * 60)
    if success:
        print("TASK 13 COMPLETED SUCCESSFULLY: Speed estimation.")
    else:
        print("TASK 13 FAILED.")
    print("=" * 60)
