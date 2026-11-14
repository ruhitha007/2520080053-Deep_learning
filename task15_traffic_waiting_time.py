"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 15: Traffic Waiting-Time Estimation
Expected Output: Waiting-time report and stationary vehicle duration estimation
"""

import os
import json
import cv2
import numpy as np
from vehicle_detector import TrafficDetector

def run_waiting_time_analysis(
    source="traffic_sample.mp4",
    max_frames=20,
    stop_speed_threshold=5.0,
    output_img="task15_waiting_time_output.jpg",
    report_file="task15_waiting_time_report.json"
):
    print("=" * 60)
    print("VisionPort Task 15: Traffic Waiting-Time Estimation")
    print("=" * 60)
    print(f"Source                  : {source}")
    print(f"Stop Speed Threshold    : < {stop_speed_threshold} km/h considered stationary")
    print("-" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {source}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = 0
    last_annotated = None
    final_report = None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 1. Detect Signal status - in red signal phase
        signal_status, sig_color, _ = detector.detect_traffic_signal(frame)
        # For intersection waiting estimation, force/test RED signal phase
        active_phase = "RED" if signal_status in ("RED", "YELLOW", "UNCERTAIN") else signal_status

        # 2. Track vehicles and calculate speed
        detections = detector.detect_multiple_vehicles(frame)
        _, _, deduped = detector.count_vehicles(detections)
        tracks = detector.track_vehicles(deduped)
        tracks = detector.estimate_speed(tracks, fps=fps)

        # 3. Estimate waiting time
        report = detector.estimate_waiting_time(
            tracks=tracks,
            signal_status=active_phase,
            fps=fps,
            stop_speed_thresh=stop_speed_threshold
        )
        final_report = report

        annotated = frame.copy()

        # Render vehicle boxes with waiting time tags
        for track_id, data in tracks.items():
            if data["disappeared"] > 0:
                continue
            x1, y1, x2, y2 = data["box"]
            cls_name = data["class_name"]
            wait_sec = data.get("wait_time_sec", 0.0)
            speed = data.get("speed_kmh", 0.0)

            is_stopped = wait_sec > 0.0
            box_color = (0, 0, 255) if is_stopped else (0, 200, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            label = f"ID#{track_id} {cls_name} | {speed:.0f}km/h | Wait: {wait_sec:.1f}s (EST)"
            (tw, th), tb = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - tb - 4)), (x1 + tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - tb - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255) if is_stopped else (255, 255, 255), 1)

        # Waiting time HUD
        cv2.rectangle(annotated, (10, 10), (580, 110), (0, 0, 0), -1)
        cv2.putText(annotated, f"VisionPort: Waiting-Time Analyzer (Signal: {active_phase})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(annotated, f"Waiting Vehicles: {report['waiting_vehicle_count']}/{report['total_tracked_vehicles']} | Max Wait: {report['maximum_estimated_wait_sec']}s",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated, f"Avg Estimated Wait: {report['average_estimated_wait_sec']}s | *Estimated vs. Measured",
                    (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        last_annotated = annotated

        if frame_count % 10 == 0 or frame_count == 1:
            print(f"[FRAME {frame_count:02d}] Waiting Vehicles: {report['waiting_vehicle_count']} | Max Wait: {report['maximum_estimated_wait_sec']}s | Avg: {report['average_estimated_wait_sec']}s")

        try:
            cv2.imshow("VisionPort Task 15 - Waiting Time", annotated)
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
        print(f"[PASSED] Saved waiting-time visualization: {output_img}")

    # Save JSON Report
    if final_report:
        with open(report_file, "w") as f:
            json.dump(final_report, f, indent=4)
        print(f"[PASSED] Saved structured waiting-time report: {report_file}")

    print("\n" + "=" * 60)
    print("TRAFFIC WAITING-TIME REPORT SUMMARY")
    print("=" * 60)
    if final_report:
        print(f"Junction Signal Phase      : {final_report['signal_status']}")
        print(f"Total Vehicles Tracked     : {final_report['total_tracked_vehicles']}")
        print(f"Vehicles Currently Waiting : {final_report['waiting_vehicle_count']}")
        print(f"Average Estimated Wait     : {final_report['average_estimated_wait_sec']} seconds")
        print(f"Maximum Estimated Wait     : {final_report['maximum_estimated_wait_sec']} seconds")
        print(f"Distinction Note           : {final_report['estimation_note']}")
        print("-" * 60)
        print("Vehicle Waiting Details:")
        for item in final_report["waiting_details"]:
            print(f"  - Vehicle ID #{item['id']:<2} ({item['class']:<10}): Estimated Wait = {item['wait_time_sec']:.1f}s | Speed = {item['speed']:.1f} km/h")
    return True

if __name__ == "__main__":
    success = run_waiting_time_analysis(source="traffic_sample.mp4", max_frames=10)
    print("=" * 60)
    if success:
        print("TASK 15 COMPLETED SUCCESSFULLY: Waiting-time report.")
    else:
        print("TASK 15 FAILED.")
    print("=" * 60)
