# Reachy Mini Object Detector

Real-time object detection for [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) using TensorFlow Hub's SSD MobileNet V2 model. The robot sees objects through its camera, tracks them with its head, and wiggles its antennas when it spots something new.

## Project Structure

```
.
├── object_detection_standalone.ipynb       # Stage 1 — standalone proof of concept
└── reachy_mini_object_detector/            # Stage 2 — Reachy Mini app
    ├── pyproject.toml
    ├── index.html / style.css              # Hugging Face Spaces landing page
    └── reachy_mini_object_detector/
        ├── detector.py                     # TF Hub model wrapper
        ├── main.py                         # App: head tracking + web dashboard
        └── static/                         # Web UI (served at :8042)
```

## Stage 1: Standalone Notebook

**`object_detection_standalone.ipynb`** runs entirely on your laptop — no robot needed. It uses your webcam and the same SSD MobileNet V2 model to demonstrate the detection pipeline:

- Load the TF Hub model (~30 MB, downloaded once)
- Run inference on a single frame
- Live detection loop with bounding boxes, labels, and FPS overlay

### Requirements

- Python 3.12+
- `tensorflow`, `tensorflow-hub`, `opencv-python`, `numpy`, `Pillow`

All dependencies are installed in the first notebook cell.

## Stage 2: Reachy Mini App

**`reachy_mini_object_detector/`** is a full [Reachy Mini app](https://github.com/pollen-robotics/reachy_mini) that integrates the same detection model with the robot:

- **Head tracking** — smoothly follows the largest detected object
- **Antenna wiggle** — reacts when a new object class is first spotted
- **Web dashboard** at `http://0.0.0.0:8042` — live annotated video feed, detection list, FPS counter, and a toggle to enable/disable tracking

### Install on the robot

From the Reachy Mini dashboard (Apps > Install), or manually:

```bash
pip install git+https://huggingface.co/spaces/backtoengineering/reachy_mini_object_detector
```

### Dependencies

- `reachy-mini`
- `tensorflow >= 2.15`
- `tensorflow-hub >= 0.16`
- `opencv-python >= 4.8`
- `numpy`

## Model

Both stages use [SSD MobileNet V2](https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1) trained on Open Images V4. It runs at ~10 FPS on CPU, prioritizing real-time performance for responsive robot behavior.