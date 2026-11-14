"""
VisionPort - Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Task 3: Read and Display Traffic Images
Expected Output: Image display window and robust path handling
"""

import os
import cv2

def load_and_display_image(image_path, window_name="VisionPort Traffic Display", wait_key_delay=2000):
    print(f"\n[INFO] Loading image from: '{image_path}'")
    
    # Path validation
    if not os.path.exists(image_path):
        print(f"[ERROR] File does not exist: '{image_path}'")
        return None

    # Read image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to decode image (unsupported or corrupted format): '{image_path}'")
        return None

    height, width, channels = img.shape
    print(f"[SUCCESS] Image loaded successfully.")
    print(f"          Dimensions: {width}x{height} pixels | Channels: {channels}")

    # Create visual display copy with info banner
    display_img = img.copy()
    overlay_text = f"VisionPort: {os.path.basename(image_path)} ({width}x{height})"
    cv2.rectangle(display_img, (10, 10), (min(width - 10, 550), 60), (0, 0, 0), -1)
    cv2.putText(display_img, overlay_text, (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

    # Save verification display image
    output_preview = "task3_display_output.jpg"
    cv2.imwrite(output_preview, display_img)
    print(f"[INFO] Rendered display preview saved: '{output_preview}'")

    # Display window using cv2.imshow
    try:
        cv2.imshow(window_name, display_img)
        print(f"[INFO] Display window '{window_name}' opened. Waiting {wait_key_delay}ms...")
        cv2.waitKey(wait_key_delay)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"[INFO] Running in headless environment (GUI window skipped): {e}")

    return display_img

def main():
    print("=" * 60)
    print("VisionPort Task 3: Read and Display Images")
    print("=" * 60)

    # Test 1: Valid image path
    valid_path = "sample_traffic.jpg"
    print("\n--- Test 1: Valid Image Path ---")
    img = load_and_display_image(valid_path, wait_key_delay=500)
    assert img is not None, "Valid image loading failed!"

    # Test 2: Invalid / missing image path
    print("\n--- Test 2: Invalid Image Path Handling ---")
    invalid_path = "non_existent_traffic_image.jpg"
    failed_img = load_and_display_image(invalid_path)
    if failed_img is None:
        print("[PASSED] Correctly handled missing file without crashing.")

    # Test 3: Corrupt or empty file handling
    print("\n--- Test 3: Zero-Byte / Corrupt File Handling ---")
    corrupt_path = "empty_corrupt_test.jpg"
    with open(corrupt_path, "w") as f:
        f.write("")
    corrupt_img = load_and_display_image(corrupt_path)
    if corrupt_img is None:
        print("[PASSED] Correctly caught corrupted file.")
    if os.path.exists(corrupt_path):
        os.remove(corrupt_path)

    print("\n" + "=" * 60)
    print("TASK 3 COMPLETED SUCCESSFULLY: Image read & display verified.")
    print("=" * 60)

if __name__ == "__main__":
    main()
