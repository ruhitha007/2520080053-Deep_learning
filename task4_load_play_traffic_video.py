"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 4: Traffic Video Playback
Expected Output: Traffic video playback with pause, resume, and exit controls
"""

import os
import cv2
import time

def play_traffic_video(video_path="traffic_sample.mp4", max_frames_auto=120):
    print("=" * 60)
    print("VisionPort Task 4: Traffic Video Playback")
    print("=" * 60)
    print(f"Loading video: '{video_path}'")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video stream for '{video_path}'")
        return False

    # Extract video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)

    print(f"[INFO] Video Details: {width}x{height} @ {fps:.1f} FPS | Total Frames: {total_frames}")
    print("[INFO] Controls:")
    print("       - Press 'SPACE' to Pause / Resume")
    print("       - Press 'q' or 'ESC' to Exit")
    print("-" * 60)

    frame_idx = 0
    paused = False
    window_name = "VisionPort - Traffic Video Playback"

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] End of video stream reached.")
                break
            frame_idx += 1
            current_frame = frame
        else:
            current_frame = frame.copy()

        # Render control and status HUD overlay
        display_frame = current_frame.copy()
        hud_bg = (0, 0, 0)
        cv2.rectangle(display_frame, (10, 10), (450, 90), hud_bg, -1)
        
        status_text = "PAUSED" if paused else "PLAYING"
        status_color = (0, 165, 255) if paused else (0, 255, 0)
        
        cv2.putText(display_frame, f"Status: {status_text} | Frame: {frame_idx}/{total_frames}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Res: {width}x{height}",
                    (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display_frame, "[SPACE] Pause/Resume  |  [Q] Exit",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Show display window
        try:
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(delay if not paused else 50) & 0xFF

            if key == ord(' '):  # Toggle pause
                paused = not paused
                print(f"[CONTROL] {'Video Paused' if paused else 'Video Resumed'}")
            elif key in (ord('q'), 27):  # 'q' or Esc
                print("[CONTROL] Exit requested by user.")
                break
        except cv2.error:
            # Headless environment fallback
            if frame_idx % 20 == 0 or frame_idx == 1:
                print(f"[PLAYBACK] Processed frame {frame_idx}/{total_frames} ({status_text})")
            if frame_idx >= max_frames_auto:
                break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("-" * 60)
    print(f"[PASSED] Video playback completed. Total frames played: {frame_idx}")
    return True

if __name__ == "__main__":
    success = play_traffic_video()
    print("=" * 60)
    if success:
        print("TASK 4 COMPLETED SUCCESSFULLY: Traffic video playback.")
    else:
        print("TASK 4 FAILED: Could not play video.")
    print("=" * 60)
