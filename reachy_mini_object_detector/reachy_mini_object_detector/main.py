
import threading
import time
import logging

import cv2
import numpy as np
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from .detector import ObjectDetector, Detection

logger = logging.getLogger(__name__)

# How far the camera FOV maps to head rotation (degrees).
# Tweak these if the robot over- or under-shoots when tracking.
CAMERA_FOV_H_DEG = 60.0  # horizontal field of view
CAMERA_FOV_V_DEG = 45.0  # vertical field of view

# Smoothing factor for head tracking (0 = no move, 1 = instant snap)
TRACKING_ALPHA = 0.15

# How long antennas wiggle when a new object class is first detected
ANTENNA_WIGGLE_DURATION = 1.5  # seconds


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on a frame (BGR)."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    for det in detections:
        x1, y1 = int(det.xmin * w), int(det.ymin * h)
        x2, y2 = int(det.xmax * w), int(det.ymax * h)

        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{det.label} {det.score:.0%}"
        font_scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return annotated


class ReachyMiniObjectDetector(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = "default"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = ObjectDetector(min_score=0.3, max_detections=10)

        # Shared state (written by inference thread, read by control loop)
        self._lock = threading.Lock()
        self._detections: list[Detection] = []
        self._annotated_frame: np.ndarray | None = None
        self._inference_fps: float = 0.0
        self._seen_classes: set[str] = set()
        self._wiggle_until: float = 0.0

        # Tracking state
        self._current_yaw: float = 0.0
        self._current_pitch: float = 0.0
        self._tracking_enabled: bool = True

        # Register FastAPI routes now (server starts before run() is called)
        self._register_routes()

    def _register_routes(self):
        if self.settings_app is None:
            return

        class TrackingState(BaseModel):
            enabled: bool

        @self.settings_app.post("/tracking")
        def update_tracking(state: TrackingState):
            self._tracking_enabled = state.enabled
            return {"tracking_enabled": self._tracking_enabled}

        @self.settings_app.get("/detections")
        def get_detections():
            with self._lock:
                return JSONResponse({
                    "detections": [d.to_dict() for d in self._detections],
                    "fps": round(self._inference_fps, 1),
                    "tracking_enabled": self._tracking_enabled,
                })

        @self.settings_app.get("/video_feed")
        def video_feed():
            return StreamingResponse(
                self._mjpeg_generator(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        # --- Load model ---
        logger.info("Loading object detection model...")
        self.detector.load()
        logger.info("Model ready!")

        # --- Start inference thread ---
        inference_thread = threading.Thread(
            target=self._inference_loop,
            args=(reachy_mini, stop_event),
            daemon=True,
        )
        inference_thread.start()

        # --- Main control loop (~50Hz) ---
        t0 = time.monotonic()
        while not stop_event.is_set():
            t = time.monotonic()

            with self._lock:
                detections = list(self._detections)

            if self._tracking_enabled and detections:
                largest = detections[0]
                # Map normalized center to yaw/pitch offset
                # center_x=0.5 means centered, 0=left edge, 1=right edge
                target_yaw = -(largest.center_x - 0.5) * CAMERA_FOV_H_DEG
                target_pitch = (largest.center_y - 0.5) * CAMERA_FOV_V_DEG

                # Smooth tracking
                self._current_yaw += TRACKING_ALPHA * (target_yaw - self._current_yaw)
                self._current_pitch += TRACKING_ALPHA * (target_pitch - self._current_pitch)
            elif not detections:
                # Slowly return to center when nothing is detected
                self._current_yaw *= 0.98
                self._current_pitch *= 0.98

            head_pose = create_head_pose(
                yaw=self._current_yaw,
                pitch=self._current_pitch,
                degrees=True,
            )

            # Antenna wiggle on new detection
            if t < self._wiggle_until:
                phase = (t - t0) * 8.0  # fast wiggle
                antenna_val = np.deg2rad(20.0 * np.sin(phase))
                antennas = np.array([antenna_val, -antenna_val])
            else:
                antennas = np.array([0.0, 0.0])

            reachy_mini.set_target(head=head_pose, antennas=antennas)
            time.sleep(0.02)  # ~50Hz

    def _inference_loop(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Runs in a separate thread: grabs frames, runs detection."""
        # Camera initialization retry loop (USB takes time to start)
        logger.info("Waiting for camera...")
        t_wait_start = time.monotonic()
        while not stop_event.is_set():
            frame = reachy_mini.media.get_frame()
            if frame is not None:
                logger.info("Camera ready! Frame shape: %s", frame.shape)
                break
            elapsed = time.monotonic() - t_wait_start
            if elapsed > 30:
                logger.error("Timeout: camera not available after 30s")
                return
            logger.info("Camera not ready yet (%.0fs)... retrying", elapsed)
            time.sleep(1.0)

        while not stop_event.is_set():
            t_start = time.monotonic()

            frame = reachy_mini.media.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            detections = self.detector.detect(frame)
            annotated = draw_detections(frame, detections)

            # Check for newly seen classes
            new_labels = {d.label for d in detections}
            with self._lock:
                unseen = new_labels - self._seen_classes
                if unseen:
                    self._seen_classes.update(unseen)
                    self._wiggle_until = time.monotonic() + ANTENNA_WIGGLE_DURATION
                    logger.info("New objects detected: %s", unseen)

                self._detections = detections
                self._annotated_frame = annotated

            elapsed = time.monotonic() - t_start
            self._inference_fps = 1.0 / max(elapsed, 0.001)

    def _mjpeg_generator(self):
        """Yields MJPEG frames for the /video_feed endpoint."""
        while not self.stop_event.is_set():
            with self._lock:
                frame = self._annotated_frame

            if frame is None:
                time.sleep(0.1)
                continue

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
            time.sleep(0.066)  # ~15 FPS to browser


if __name__ == "__main__":
    app = ReachyMiniObjectDetector()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()