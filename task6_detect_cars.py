"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 6: Car Detection Using YOLO
Expected Output: Car detection with bounding boxes
"""

import os
import cv2
from vehicle_detector import TrafficDetector

def run_car_detection(source="traffic_sample.mp4", max_frames=30, output_img="task6_car_detection_output.jpg"):
    print("=" * 60)
    print("VisionPort Task 6: Car Detection Using YOLO")
    print("=" * 60)
    print(f"Source       : {source}")
    print(f"Target Class : 'car' (COCO class ID: 2)")
    print("-" * 60)

    detector = TrafficDetector(model_path="yolov8n.pt")

    # If source is an image
    if source.lower().endswith(('.jpg', '.jpeg', '.png')):
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Could not read image: {source}")
            return False
        
        cars = detector.detect_cars(frame)
        print(f"[INFO] Total cars detected: {len(cars)}")
        for idx, car in enumerate(cars, 1):
            print(f"       Car #{idx}: Box {car['box']} | Confidence: {car['confidence']:.2f}")

        # Draw bounding boxes (only boxes as required by Task 6)
        annotated = frame.copy()
        for car in cars:
            x1, y1, x2, y2 = car["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 140, 0), 2)
            cv2.putText(annotated, "car", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)

        cv2.imwrite(output_img, annotated)
        print(f"[PASSED] Car detection output saved: {output_img}")
        return True

    # If source is a video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {source}. Falling back to 'sample_traffic.jpg'...")
        return run_car_detection(source="sample_traffic.jpg", output_img=output_img)

    frame_count = 0
    total_cars_seen = 0
    last_annotated = None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        cars = detector.detect_cars(frame)
        total_cars_seen += len(cars)

        # Draw bounding boxes
        annotated = frame.copy()
        for car in cars:
            x1, y1, x2, y2 = car["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 140, 0), 2)
            cv2.putText(annotated, "car", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)

        # Add HUD
        cv2.putText(annotated, f"VisionPort Task 6: Car Detection | Frame {frame_count} | Cars: {len(cars)}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        last_annotated = annotated

        if frame_count % 10 == 0 or frame_count == 1:
            print(f"[FRAME {frame_count:02d}] Cars detected in frame: {len(cars)}")

        try:
            cv2.imshow("VisionPort Task 6 - Car Detection", annotated)
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
        print(f"[PASSED] Saved sample annotated frame to: {output_img}")

    print("-" * 60)
    print(f"[SUMMARY] Processed {frame_count} frames. Average cars/frame: {total_cars_seen / max(1, frame_count):.1f}")
    return True

if __name__ == "__main__":
    success = run_car_detection(source="traffic_sample.mp4", max_frames=20)
    print("=" * 60)
    if success:
        print("TASK 6 COMPLETED SUCCESSFULLY: Car detection with bounding boxes.")
    else:
        print("TASK 6 FAILED.")
    print("=" * 60)
