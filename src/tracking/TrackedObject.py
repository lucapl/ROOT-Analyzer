import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod

from src.Event import Event
from src.viz.images import draw_bbox
from src.utils import create_tracker


class TrackedObject(ABC):

    def __init__(self, name: str, tracker_type: str, velocity_sensivity=10, redetect_timer=48):
        self.name = name
        self.tracker = create_tracker(tracker_type)
        self.event = Event(redetect_timer)

        self.vel_sens = velocity_sensivity
        self.velocity = np.array([0, 0])
        self.last_bbox = None
        self.contour = None

        self.timer = 0

        self.redetect_timer = redetect_timer

    def init_tracker(self, frame, contour):
        bd = cv.boundingRect(contour)
        self.tracker.init(frame, cv.boundingRect(contour))

    def _update_velocity(self, bbox):
        x, y, _, _ = self.last_bbox
        x_p, y_p, _, _ = bbox

        self.velocity = np.array([x_p - x, y_p - y])

    @abstractmethod
    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        return frame

    @abstractmethod
    def redetect(self, frame):
        return None

    def is_moving(self):
        return np.linalg.norm(self.velocity, 2) > self.vel_sens

    def update(self, raw_frame):
        ok, bbox = self.tracker.update(raw_frame)
        if ok:
            self.check_if_lost(bbox)
            if self.last_bbox is not None:
                self._update_velocity(bbox)

            self.last_bbox = bbox
            return bbox
        else:
            if self.timer > self.redetect_timer:
                self.redetect(raw_frame)
            self.timer += 1
            return None

    def check_if_lost(self, bbox):
        self.timer = 0

    def detection_fail_msg(self, frame):
        return self.draw_bbox(frame, f"{self.name} Lost", (0, 0, 255))

    def draw_bbox(self, frame, msg=None, color=(0, 255, 0)):
        if msg is None:
            msg = self.name
        x, y, w, h = self.last_bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        draw_bbox(frame, self.last_bbox, color)

        if self.is_moving():
            frame = cv.putText(frame, "Moving", (x, y + h - 2), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
                               cv.LINE_AA)
        return cv.putText(frame, msg, (x, y - 2), cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv.LINE_AA)
