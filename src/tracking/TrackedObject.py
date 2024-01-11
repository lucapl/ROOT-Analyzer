import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod

from src.viz.images import draw_bbox
from src.utils import create_tracker


class TrackedObject(ABC):

    def __init__(self, name: str, tracker_type: str, starting_contour, first_frame, velocity_sensivity=10,redetect_timer=48):
        self.name = name
        self.tracker = create_tracker(tracker_type)

        self.init_tracker(first_frame, starting_contour)

        self.ini_cont = starting_contour
        self.vel_sens = velocity_sensivity
        self.velocity = np.array([0, 0])
        self.last_bbox = None

        self.timer = 0

        self.redetect_timer = redetect_timer

        self.events = []

    def init_tracker(self, frame, contour):
        self.tracker.init(frame, cv.boundingRect(contour))


    def _update_velocity(self, bbox):
        x, y, _, _ = self.last_bbox
        x_p, y_p, _, _ = bbox

        self.velocity = np.array([x_p - x, y_p - y])

    @abstractmethod
    def redetect(self,frame):
        return None

    def is_moving(self):
        return np.linalg.norm(self.velocity, 2) > self.vel_sens

    def update(self, raw_frame):
        ok, bbox = self.tracker.update(raw_frame)
        if ok:
            if self.last_bbox is not None:
                self._update_velocity(bbox)

            self.last_bbox = bbox
            return bbox
        else:
            return None

    def detection_fail_msg(self, frame):
        return self.draw_bbox(frame, f"{self.name} Lost", (0, 0, 255))

    def draw_bbox(self, frame, msg=None, color=(0, 255, 0)):
        if msg is None:
            msg = self.name
        x, y, w, h = self.last_bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        draw_bbox(frame, self.last_bbox, color)

        if self.is_moving():
            frame = cv.putText(frame, "Moving", (x, y + h - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                               cv.LINE_AA)
        return cv.putText(frame, msg, (x, y - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

    @abstractmethod
    def detect_events(self, frame_num: int):
        return None

    # def redetection(self,)
