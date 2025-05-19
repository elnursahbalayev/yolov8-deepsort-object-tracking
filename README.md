# YOLOv8 DeepSORT Object Tracking

## Overview
This project implements an object tracking system using YOLOv8 for object detection and DeepSORT for object tracking. It processes video files, detects objects in each frame, tracks them across frames, and outputs a video with bounding boxes around the tracked objects.

## Features
- Object detection using YOLOv8
- Object tracking using DeepSORT
- Visualization of tracked objects with unique colored bounding boxes
- Video processing with OpenCV

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/yolov8-deepsort-object-tracking.git
   cd yolov8-deepsort-object-tracking
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model:
   ```
   # The model will be downloaded automatically when running the script
   # or you can manually download it from https://github.com/ultralytics/ultralytics
   ```

## Usage
1. Place your input video in the `data` directory or update the `video_path` variable in `main.py`.

2. Run the main script:
   ```
   python main.py
   ```

3. The output video with tracked objects will be saved as `out.mp4` in the project directory.

## Configuration
You can adjust the following parameters in `main.py`:
- `detection_threshold`: Confidence threshold for object detection (default: 0.5)
- `video_path`: Path to the input video
- `video_out_path`: Path for the output video

## Project Structure
- `main.py`: Main script for video processing and visualization
- `tracker.py`: Implementation of the Tracker class using DeepSORT
- `model_data/`: Directory containing the DeepSORT model
- `data/`: Directory for input videos

## Dependencies
- ultralytics (YOLOv8)
- deep_sort
- OpenCV
- TensorFlow
- NumPy
- FilterPy

## License
[Include license information here]

## Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/nwojke/deep_sort)
