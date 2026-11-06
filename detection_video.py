#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# EXPERIMENT 10: VIDEO OBJECT DETECTION AND CLASSIFICATION
# Algorithm: YOLOv8 Nano
# ============================================================

# Step 1: Install required libraries
get_ipython().run_line_magic('pip', 'install ultralytics opencv-python matplotlib requests')

# Step 2: Import libraries
from ultralytics import YOLO
import cv2
import requests
from collections import Counter
from IPython.display import Video, display

# Step 3: Download sample video
video_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"

response = requests.get(video_url)
response.raise_for_status()

with open("sample_video.mp4", "wb") as file:
    file.write(response.content)

print("Sample video downloaded successfully!")

# Step 4: Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

print("YOLOv8 model loaded successfully!")

# Step 5: Open the input video
cap = cv2.VideoCapture("sample_video.mp4")

if not cap.isOpened():
    raise FileNotFoundError("Unable to open sample video.")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video
output = cv2.VideoWriter(
    "detected_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Step 6: Initialize object counter
object_counts = Counter()
frame_number = 0

print("Processing video...")

# Step 7: Process video frame by frame
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Perform YOLO object detection
    results = model.predict(
        source=frame,
        conf=0.25,
        imgsz=416,
        verbose=False
    )

    result = results[0]

    # Count detected object classes in this frame
    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        object_counts[class_name] += 1

    # Draw bounding boxes and labels
    annotated_frame = result.plot()

    # Save processed frame
    output.write(annotated_frame)

# Step 8: Release video resources
cap.release()
output.release()

print("Video processing completed!")
print("Total frames processed:", frame_number)

# Step 9: Display object classification results
print("\nOBJECT DETECTION RESULTS")
print("------------------------------")

for object_name, count in object_counts.items():
    print(object_name, ":", count)

# Step 10: Display processed video
print("\nProcessed video:")

display(Video("detected_video.mp4", embed=True))

print("\nExperiment completed successfully!")

import os
os.startfile("detected_video.mp4")


# In[ ]:




