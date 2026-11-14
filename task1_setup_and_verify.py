"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 1: Environment and Library Setup Verification
Expected Output: Working Python Environment Verified
"""

import sys
import importlib
import numpy as np
import cv2

print("*" * 60)
print("VisionPort Task 1: Environment & Library Verification")
print("=" * 60)

# Check Python version
print(f"Python Version: {sys.version}")
if sys.version_info < (3, 8):
    print("WARNING: Python 3.8 or higher is recommended for VisionPort.")
else:
    print("[OK] Python version is compatible.\n")

# Required libraries for the VisionPort project
required_libraries = [
    ("cv2", "OpenCV (opencv-python)"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("ultralytics", "Ultralytics YOLOv8"),
    ("torch", "PyTorch"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn")
]

all_passed = True

print("Checking installed packages:")
print("-" * 60)
for module_name, display_name in required_libraries:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "Installed")
        print(f"[PASSED] {display_name:<28} : v{version}")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[FAILED] {display_name:<28} : NOT INSTALLED ({e})")
        all_passed = False

print("-" * 60)

# OpenCV Verification & Test Image Generation
test_img_path = "task1_opencv_test.jpg"
try:
    test_img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Background color
    test_img[:] = (35, 30, 30)
    # Draw test shapes and banner
    cv2.rectangle(test_img, (50, 50), (550, 350), (0, 200, 100), 2)
    cv2.putText(test_img, "VisionPort: Environment Verified", (70, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(test_img, f"OpenCV v{cv2.__version__}", (70, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(test_img, "Status: Ready for Traffic Detection", (70, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Traffic signal circles
    cv2.circle(test_img, (150, 300), 20, (0, 0, 255), -1)   # Red
    cv2.circle(test_img, (300, 300), 20, (0, 255, 255), -1) # Yellow
    cv2.circle(test_img, (450, 300), 20, (0, 255, 0), -1)   # Green
    
    cv2.imwrite(test_img_path, test_img)
    print(f"[PASSED] OpenCV Image Processing Check : Created {test_img_path}")
except Exception as e:
    print(f"[FAILED] OpenCV Image Processing Check : {e}")
    all_passed = False

print("=" * 60)
if all_passed:
    print("TASK 1 COMPLETED SUCCESSFULLY: Working Python environment.")
else:
    print("TASK 1 FAILED: Please install the missing packages.")
print("=" * 60)
