# CodeAlpha_Object-Detection-and-Tracking-System

Advanced Object Detection and Tracking System
Overview
This Python application implements a real-time object detection and tracking system using YOLOv4-tiny and OpenCV. The system detects objects in a video stream, tracks them across frames, and provides analytics about detected objects.

Features
Real-time Object Detection: Uses YOLOv4-tiny for fast object detection

Multi-Object Tracking: Tracks objects across frames with unique IDs

Performance Analytics: Displays FPS and most frequently detected objects

Visualization: Shows bounding boxes, object paths, and labels

GPU Acceleration: Uses CUDA if available, falls back to CPU

Prerequisites
Python 3.6+

OpenCV with DNN and CUDA support

YOLOv4-tiny weights and configuration files

COCO dataset class names

Installation
Install required packages:

bash
pip install opencv-python numpy
Download the necessary YOLO files:

Download yolov4-tiny.weights from the official YOLO repository

Download yolov4-tiny.cfg from the official YOLO repository

Download coco.names which contains the 80 COCO class labels

Place all files in the same directory as the script

Usage
Run the script directly:

bash
python object_detector.py
The application will:

Initialize the webcam

Start detecting and tracking objects in real-time

Display the video stream with bounding boxes and analytics

Exit when the ESC key is pressed

Key Components
AdvancedObjectDetector Class
Handles network initialization and configuration

Manages object detection, tracking, and visualization

Maintains tracking history and object counters

Main Functions
detect_objects(): Processes frames through YOLO network

update_tracking(): Maintains object tracks across frames

draw_analytics(): Visualizes results and analytics on the frame

Performance Notes
With CUDA enabled: Expected 20-30 FPS on mid-range GPUs

CPU-only: Expected 5-15 FPS depending on hardware

Lower confidence thresholds increase detection but reduce precision

Customization
Adjust Detection Sensitivity
Modify the confidence threshold in detect_objects() method:

python
if confidence > 0.5:  # Change this value
Change Tracking Parameters
Adjust the tracking distance threshold in update_tracking():

python
if dist < min_dist and dist < 50:  # Change distance threshold
Add New Classes
Edit the coco.names file to add or modify class labels (requires retraining YOLO for custom classes)

Troubleshooting
Common Issues
"File not found" errors: Ensure YOLO files are in the correct directory

Low FPS: Try using the full YOLOv4 model for better accuracy or adjust frame size

CUDA not available: Check OpenCV CUDA support with cv2.cuda.getCudaEnabledDeviceCount()

Webcam Issues
Change video capture index if webcam isn't detected: cv2.VideoCapture(1)

Verify webcam permissions on your system

Extending the Application
Add New Analytics
Extend the draw_analytics() method to display additional information like:

Object count over time

Movement patterns

Zone intrusion detection

Export Results
Add functionality to save:

Detection logs to file

Screenshots of detected objects

Video recordings with annotations

License
This project uses OpenCV and YOLO, which have their respective licenses. Please ensure you comply with these licenses when using this code.

References
YOLO: https://github.com/AlexeyAB/darknet

OpenCV: https://opencv.org/

COCO Dataset: https://cocodataset.org/
