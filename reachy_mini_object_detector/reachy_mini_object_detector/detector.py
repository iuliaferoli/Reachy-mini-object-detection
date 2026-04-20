import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import logging

logger = logging.getLogger(__name__)

# SSD MobileNet V2 — fast, good for real-time (~10 FPS on CPU)
MODEL_HANDLE = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"


class Detection:
    """A single detected object."""

    def __init__(self, label: str, score: float, box: tuple[float, float, float, float]):
        self.label = label
        self.score = score
        # box is (ymin, xmin, ymax, xmax) in normalized [0, 1] coordinates
        self.ymin, self.xmin, self.ymax, self.xmax = box

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    @property
    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(self.score, 3),
            "box": [self.ymin, self.xmin, self.ymax, self.xmax],
            "center": [self.center_x, self.center_y],
        }


class ObjectDetector:
    """TensorFlow Hub SSD MobileNet V2 object detector."""

    def __init__(self, min_score: float = 0.3, max_detections: int = 10):
        self.min_score = min_score
        self.max_detections = max_detections
        self._detector = None
        self._loaded = False

    def load(self):
        """Load the TF Hub model. Call once before detect()."""
        logger.info("Loading TF Hub model: %s", MODEL_HANDLE)
        t0 = time.monotonic()
        self._detector = hub.load(MODEL_HANDLE)
        self._loaded = True
        elapsed = time.monotonic() - t0
        logger.info("Model loaded in %.1fs", elapsed)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR numpy frame (from OpenCV/camera).

        Args:
            frame: numpy array (H, W, 3) uint8, BGR format

        Returns:
            List of Detection objects, filtered by min_score, sorted by area (largest first).
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert BGR to RGB and normalize to float32 [0, 1]
        rgb = frame[:, :, ::-1]
        img_tensor = tf.image.convert_image_dtype(rgb, tf.float32)[tf.newaxis, ...]

        # Use the model's signature for inference
        results = self._detector.signatures['default'](img_tensor)

        # Convert tensors to numpy and flatten batch dimension
        boxes = np.array(results["detection_boxes"])
        scores = np.array(results["detection_scores"])
        labels = np.array(results["detection_class_entities"])

        if boxes.ndim > 2:
            boxes = boxes[0]
        if scores.ndim > 1:
            scores = scores[0]
        if labels.ndim > 1:
            labels = labels[0]

        scores = np.atleast_1d(scores)

        detections = []
        for i in range(min(len(scores), self.max_detections)):
            if scores[i] < self.min_score:
                continue
            label = labels[i].decode("utf-8") if isinstance(labels[i], bytes) else str(labels[i])
            detections.append(Detection(
                label=label,
                score=float(scores[i]),
                box=tuple(float(v) for v in boxes[i]),
            ))

        # Sort by area descending (largest object first)
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections