# V7 Labs - Auto and Manual Annotation Tool

A two-part annotation tool. A FastAPI service runs YOLOv8 for automatic object detection on images and videos, and can propagate manually drawn boxes through a video with an OpenCV CSRT tracker. A Streamlit front end uploads files, shows auto-annotations, lets you rename labels, and lets you draw extra boxes on a canvas.

## Features

- Automatic detection on a single image, or on every frame of a video, using a YOLOv8 model (`yolov8n.pt`).
- Label renaming: map detected class names to your own labels in the UI.
- Manual boxes drawn on the image or video canvas, rendered in blue; automatic detections are drawn in red.
- Video mode: a manual box on one frame is tracked forward and backward with CSRT and kept only while YOLO still detects the same label with IoU above 0.3.
- Download of the annotated image or video from the UI.

## Project structure

- `inference_service.py` - FastAPI backend. Endpoints: `/get_initial_labels`, `/predict_image`, `/predict_image_manual`, `/predict_video`, `/predict_video_manual`. Loads `yolov8n.pt` at startup.
- `annotation_ui.py` - Streamlit front end. Calls the backend at `http://localhost:8000`.
- `run.txt` - commands to start both parts.
- `annotated_image.png` - sample annotated output.
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt` - YOLOv8 weights. Only `yolov8n.pt` is loaded by the included code.
- `yolov10s.pt`, `yolov10m.pt` - YOLOv10 weights. The included code does not load them.

## Requirements

Imports seen in the code: `fastapi`, `uvicorn`, `cv2` (CSRT tracker needs opencv-contrib), `numpy`, `PIL`, `ultralytics`, `torch`, `streamlit`, `streamlit_drawable_canvas`, `requests`. FastAPI form and file uploads also need `python-multipart`.

```bash
pip install fastapi uvicorn python-multipart ultralytics opencv-contrib-python pillow numpy torch streamlit streamlit-drawable-canvas requests
```

## Installation

```bash
git clone <this-repo-url>
cd V7-Labs
pip install fastapi uvicorn python-multipart ultralytics opencv-contrib-python pillow numpy torch streamlit streamlit-drawable-canvas requests
```

## Usage

Open two terminals in this folder.

```bash
# Terminal 1: start the inference API on port 8000 (expects yolov8n.pt in this folder)
uvicorn inference_service:app --host 0.0.0.0 --port 8000

# Terminal 2: start the UI
streamlit run annotation_ui.py
```

In the UI, choose **Image** or **Video**, upload a `jpg`, `jpeg`, `png`, `mp4`, `avi` or `mov` file, and wait for auto-detection. Rename labels as needed, draw manual boxes, and download the result.

## Notes

- The backend URL is fixed to `http://localhost:8000` in `annotation_ui.py`.
- `inference_service.py` loads `yolov8n.pt` from the current working directory, so start `uvicorn` from this folder.
- The `__pycache__` folder from the original working copy is not included; Python regenerates it.
- Video processing is done in memory: all frames are read into a list, so long videos use a lot of RAM.
- Auto-detection runs on every frame in `/predict_video` and `/predict_video_manual`, so processing can be slow on CPU.
- `ImageFont.truetype("arial.ttf")` is tried first and falls back to PIL's default font if Arial is not found.
- The `seconds` form field of `/get_initial_labels` is accepted but unused.

## Pretrained YOLO weights (not included)

The following off-the-shelf Ultralytics checkpoints were removed from this repository because they are publicly downloadable. The code loads them by name, and Ultralytics downloads them automatically on first use:

- `yolov10m.pt`
- `yolov10s.pt`
- `yolov8l.pt`
- `yolov8m.pt`
- `yolov8n.pt`
- `yolov8s.pt`

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # downloaded automatically if not present
```

Or download them manually from https://github.com/ultralytics/assets/releases
