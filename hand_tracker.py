"""
hand_tracker.py
----------------
Wraps MediaPipe's HandLandmarker (the current "Tasks" API) to detect a
single hand in a webcam frame and report which fingers are extended,
plus key landmark coordinates.

NOTE: Modern mediapipe (0.10.x+) removed the old `mp.solutions.hands`
API that most older tutorials use. This module uses the current
HandLandmarker Task API instead, which requires a downloaded model
file: hand_landmarker.task (downloaded automatically on first run).

Landmark reference (MediaPipe Hands, 21 points per hand):
    0  = wrist
    4  = thumb tip
    8  = index fingertip
    12 = middle fingertip
    16 = ring fingertip
    20 = pinky fingertip
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model_downloaded(model_path=MODEL_FILENAME):
    """Download the HandLandmarker model file if it isn't already present."""
    if os.path.exists(model_path):
        return model_path
    print(f"Downloading hand landmark model to {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Model download complete.")
    return model_path


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.6, tracking_confidence=0.5,
                 model_path=MODEL_FILENAME):
        model_path = ensure_model_downloaded(model_path)

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,  # synchronous, one frame at a time
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self.results = None
        self._start_time = time.time()

    def find_hands(self, frame, draw=True):
        """Run detection on a BGR frame; optionally draw landmarks. Returns the frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if draw and self.results and self.results.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in self.results.hand_landmarks:
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        return frame

    def get_landmark_positions(self, frame, hand_index=0):
        """Return a list of (id, x_px, y_px) for the requested hand, or []."""
        landmark_list = []
        if self.results and self.results.hand_landmarks:
            if hand_index < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[hand_index]
                h, w, _ = frame.shape
                for lm_id, lm in enumerate(hand):
                    x_px, y_px = int(lm.x * w), int(lm.y * h)
                    landmark_list.append((lm_id, x_px, y_px))
        return landmark_list

    def fingers_up(self, landmark_list):
        """
        Return a list of 5 booleans [thumb, index, middle, ring, pinky]
        indicating which fingers are extended, based on landmark geometry.
        """
        if not landmark_list or len(landmark_list) < 21:
            return [False, False, False, False, False]

        fingers = []
        lm = {lm_id: (x, y) for lm_id, x, y in landmark_list}

        # Thumb: compare x-coordinates (works for a hand facing the camera, mirrored view)
        fingers.append(lm[4][0] > lm[3][0])

        # Other four fingers: tip above (smaller y) than pip joint = extended
        for tip_id in [8, 12, 16, 20]:
            pip_id = tip_id - 2
            fingers.append(lm[tip_id][1] < lm[pip_id][1])

        return fingers

    def get_gesture(self, landmark_list):
        """
        Classify the current hand pose into a simple gesture string used
        by the app: 'draw' (index only), 'select' (index+middle), or 'idle'.
        """
        fingers = self.fingers_up(landmark_list)
        if not landmark_list:
            return "none", fingers

        index_up = fingers[1]
        middle_up = fingers[2]
        ring_up = fingers[3]
        pinky_up = fingers[4]

        if index_up and middle_up and not ring_up and not pinky_up:
            return "select", fingers
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "draw", fingers
        return "idle", fingers

    def close(self):
        """Release the landmarker's resources."""
        self.landmarker.close()