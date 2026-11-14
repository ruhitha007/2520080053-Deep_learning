"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 5: Video Frame Extraction
Expected Output: Frames extracted from video and saved as organized image files
"""

import os
import cv2

def extract_frames(video_path="traffic_sample.mp4", output_dir="extracted_frames", frame_stride=2, max_frames=30):
    print("=" * 60)
    print("VisionPort Task 5: Video Frame Extraction")
    print("=" * 60)
    print(f"Source Video   : {video_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Frame Stride    : Every {frame_stride} frame(s)")
    print(f"Max Extraction  : {max_frames} frames")
    print("-" * 60)

    if not os.path.exists(video_path):
        print(f"[ERROR] Source video '{video_path}' not found.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return []

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_files = []
    current_frame = 0
    saved_count = 0

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        if current_frame % frame_stride == 0:
            saved_count += 1
            filename = f"frame_{saved_count:04d}.jpg"
            file_path = os.path.join(output_dir, filename)
            
            # Save extracted frame
            cv2.imwrite(file_path, frame)
            saved_files.append(file_path)

            if saved_count <= 5 or saved_count % 10 == 0:
                print(f"[EXTRACTED] Saved {filename} (video frame {current_frame}/{total_video_frames})")

    cap.release()

    print("-" * 60)
    print(f"[SUMMARY] Successfully extracted and saved {len(saved_files)} frames into '{output_dir}'.")
    return saved_files

if __name__ == "__main__":
    saved = extract_frames(video_path="traffic_sample.mp4", output_dir="extracted_frames", frame_stride=2, max_frames=20)
    print("=" * 60)
    if len(saved) > 0:
        print(f"TASK 5 COMPLETED SUCCESSFULLY: {len(saved)} frames saved as images.")
    else:
        print("TASK 5 FAILED: No frames were extracted.")
    print("=" * 60)
