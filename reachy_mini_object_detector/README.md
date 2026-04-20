---
title: Reachy Mini Object Detector
emoji: 👁️
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Real-time object detection with head tracking
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini Object Detector

Real-time object detection app for [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) using TensorFlow Hub's SSD MobileNet V2 model.

## Features

- **Live object detection** via the robot's camera using SSD MobileNet V2 (~10 FPS on CPU)
- **Head tracking** — the robot follows the largest detected object with smooth interpolation
- **Antenna wiggle** — antennas wiggle when a new object class is first spotted
- **Web UI** — live annotated video feed with detection list, FPS counter, and tracking toggle

## How it works

1. The camera captures frames and feeds them to the TF Hub object detector
2. Detected objects are drawn as bounding boxes on the video stream
3. The robot's head smoothly tracks the largest object in frame
4. A web dashboard at `http://0.0.0.0:8042` shows the live feed and controls

## Installation

Install from the Reachy Mini dashboard or manually:

```bash
pip install git+https://huggingface.co/spaces/backtoengineering/reachy_mini_object_detector
```

## Dependencies

- `reachy-mini`
- `tensorflow >= 2.15`
- `tensorflow-hub >= 0.16`
- `opencv-python >= 4.8`
- `numpy`