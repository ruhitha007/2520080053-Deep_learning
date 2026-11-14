"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Master Task Runner: run_all_tasks.py
Runs and verifies all 15 tasks consecutively
"""

import subprocess
import sys
import time

TASKS = [
    (1, "task1_setup_and_verify.py", "Environment and Library Verification"),
    (2, "task2_capture_traffic_signal.py", "Traffic Image Capture"),
    (3, "task3_read_display_traffic_image.py", "Read and Display Traffic Images"),
    (4, "task4_load_play_traffic_video.py", "Traffic Video Playback"),
    (5, "task5_extract_video_frames.py", "Video Frame Extraction"),
    (6, "task6_detect_cars.py", "Car Detection Using YOLO"),
    (7, "task7_detect_multiple_vehicles.py", "Multiple Vehicle Detection"),
    (8, "task8_display_labels_confidence.py", "Vehicle Names and Confidence Scores"),
    (9, "task9_count_vehicles.py", "Vehicle Counting & Deduplication"),
    (10, "task10_traffic_density.py", "Traffic Density Classification"),
    (11, "task11_traffic_congestion.py", "Traffic Congestion Detection"),
    (12, "task12_track_vehicles.py", "Individual Vehicle Tracking"),
    (13, "task13_speed_estimation.py", "Approximate Vehicle Speed Estimation"),
    (14, "task14_traffic_signal_status.py", "Traffic Signal Status Detection"),
    (15, "task15_traffic_waiting_time.py", "Traffic Waiting-Time Estimation")
]

def main():
    print("=" * 70)
    print("VISIONPORT: MASTER 15-TASK VERIFICATION SUITE")
    print("Model: traffic_detector | Project: VisionPort")
    print("=" * 70)

    results = []
    py_exec = sys.executable

    for num, script, desc in TASKS:
        print(f"\n>>> Running Task {num:02d}: {desc} ({script})...")
        t0 = time.time()
        try:
            res = subprocess.run([py_exec, script], capture_output=True, text=True, timeout=120)
            elapsed = time.time() - t0
            passed = (res.returncode == 0)
            status = "PASSED" if passed else "FAILED"
            results.append((num, desc, script, status, f"{elapsed:.1f}s"))
            print(f"    Status: [{status}] in {elapsed:.1f}s")
            if not passed:
                print(f"    Error output:\n{res.stderr[:200]}")
        except Exception as e:
            results.append((num, desc, script, "ERROR", f"{time.time()-t0:.1f}s"))
            print(f"    Status: [ERROR] ({e})")

    print("\n" + "=" * 70)
    print(f"{'Task #':<8} {'Description':<38} {'Status':<10} {'Runtime'}")
    print("-" * 70)
    all_ok = True
    for num, desc, script, status, runtime in results:
        print(f"Task {num:<3} {desc:<38} {status:<10} {runtime}")
        if status != "PASSED":
            all_ok = False
    print("=" * 70)
    if all_ok:
        print("ALL 15 VISIONPORT TASKS COMPLETED & VERIFIED SUCCESSFULLY!")
    else:
        print("SOME TASKS ENCOUNTERED ISSUES. CHECK LOGS ABOVE.")
    print("=" * 70)

if __name__ == "__main__":
    main()
