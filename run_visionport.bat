@echo off
echo ============================================================
echo VisionPort: Intelligent Vehicle Detection and Traffic Monitoring
echo Main Model: traffic_detector
echo ============================================================
echo.
echo Select an option:
echo 1. Start VisionPort FastAPI Web Portal (http://localhost:8000)
echo 2. Run All 15 Tasks Verification Suite
echo 3. Run Specific Task (1 - 15)
echo 4. Exit
echo.
set /p opt="Enter choice (1-4): "

set PYTHON_EXE=C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe

if "%opt%"=="1" (
    echo Starting VisionPort Server...
    "%PYTHON_EXE%" visionport.py
) else if "%opt%"=="2" (
    echo Running All Tasks...
    "%PYTHON_EXE%" run_all_tasks.py
) else if "%opt%"=="3" (
    set /p tasknum="Enter task number (1-15): "
    if "%tasknum%"=="1" "%PYTHON_EXE%" task1_setup_and_verify.py
    if "%tasknum%"=="2" "%PYTHON_EXE%" task2_capture_traffic_signal.py
    if "%tasknum%"=="3" "%PYTHON_EXE%" task3_read_display_traffic_image.py
    if "%tasknum%"=="4" "%PYTHON_EXE%" task4_load_play_traffic_video.py
    if "%tasknum%"=="5" "%PYTHON_EXE%" task5_extract_video_frames.py
    if "%tasknum%"=="6" "%PYTHON_EXE%" task6_detect_cars.py
    if "%tasknum%"=="7" "%PYTHON_EXE%" task7_detect_multiple_vehicles.py
    if "%tasknum%"=="8" "%PYTHON_EXE%" task8_display_labels_confidence.py
    if "%tasknum%"=="9" "%PYTHON_EXE%" task9_count_vehicles.py
    if "%tasknum%"=="10" "%PYTHON_EXE%" task10_traffic_density.py
    if "%tasknum%"=="11" "%PYTHON_EXE%" task11_traffic_congestion.py
    if "%tasknum%"=="12" "%PYTHON_EXE%" task12_track_vehicles.py
    if "%tasknum%"=="13" "%PYTHON_EXE%" task13_speed_estimation.py
    if "%tasknum%"=="14" "%PYTHON_EXE%" task14_traffic_signal_status.py
    if "%tasknum%"=="15" "%PYTHON_EXE%" task15_traffic_waiting_time.py
) else (
    exit /b
)
pause
