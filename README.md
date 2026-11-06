# Experiment 10: Video Object Detection and Classification Using YOLOv8

## 1. Project Overview

This project demonstrates video-based object detection and classification using the YOLOv8 (You Only Look Once) deep learning algorithm.

The program processes a video frame by frame, identifies objects, classifies them into different categories and displays bounding boxes with confidence scores.

The detected frames are combined and saved as a new annotated video.

## 2. Objectives

- Understand video-based object detection using YOLOv8.
- Process video frames using OpenCV.
- Detect multiple objects in video frames.
- Classify detected objects into different categories.
- Display bounding boxes and confidence scores.
- Generate and save an output video containing detection results.

## 3. Algorithm Used

**YOLOv8 – You Only Look Once**

YOLOv8 is a single-stage, deep-learning-based object detection model.

It processes individual video frames and predicts:
- Object bounding box coordinates.
- Object class labels.
- Confidence scores.

The model performs detection independently on each frame.

## 4. Technologies and Libraries

- Programming Language: Python
- Deep Learning Model: YOLOv8 Nano
- Framework: Ultralytics
- Video Processing: OpenCV
- Environment: Jupyter Notebook
- Output Format: MP4

## 5. Sample Video

The experiment downloads a sample video directly from the Ultralytics public assets repository.

Sample video URL:

https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4

The video is automatically downloaded when the notebook is executed.

## 6. Installation

Install the required libraries:

```bash
pip install ultralytics opencv-python matplotlib requests
```

## 7. Execution Procedure

1. Clone or download this repository.
2. Open `Experiment_10_Video_Object_Detection.ipynb` in Jupyter Notebook.
3. Install the required Python libraries.
4. Execute the notebook cells sequentially.
5. The sample video and YOLOv8 Nano model are downloaded if required.
6. The program processes the video frame by frame.
7. View the detected objects and classification results.
8. Open the generated `detected_video.mp4` file to view the output.

## 8. Expected Output

The program generates:

- Sample input video.
- Object detection results for individual frames.
- Bounding boxes around detected objects.
- Object class labels.
- Confidence scores.
- Class-wise detection counts across processed frames.
- Annotated output video (`detected_video.mp4`).

**Note:** The reported counts represent detections across video frames, not unique physical objects. The same object may be detected repeatedly in multiple frames.

## 9. Project Files

- `Experiment_10_Video_Object_Detection.ipynb` – Complete Jupyter Notebook program.
- `video_object_detection.py` – Python source code.
- `requirements.txt` – Required Python libraries.

The input sample video is downloaded automatically during execution.

## 10. Applications

- Autonomous mobile robots
- Traffic surveillance
- Industrial monitoring
- Smart security systems
- Video analytics
- Robotics vision applications

## 11. Author

**Arun Kumar K**  
Department of Robotics and Automation

## 12. References

- Ultralytics Documentation: https://docs.ultralytics.com/
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics
- OpenCV Documentation: https://docs.opencv.org/
