import numpy as np
import cv2 as cv

from src.tracking.StaticObject import StaticObject
from src.tracking.TrackedObject import TrackedObject
from src.detection.elements import descriptor_detect
from src.viz.images import draw_bbox


class Card(TrackedObject):

    def __init__(self, name, tracker_type, ref, distance=.5, velocity_sensivity=10):
        super().__init__(name, tracker_type, velocity_sensivity)
        self.ref = ref
        self.distance = distance

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        self.event.update()

        if self.is_moving():
            x, y = self.velocity
            player = "Orange" if y > 0 else "Blue"
            self.event.msg = f"Card drawn by {player}"
            self.event.reset()

        frame = cv.putText(frame, self.event.get(), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
        return cv.putText(frame, self.event.get(), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv.LINE_AA)

    def redetect(self, frame):
        output = descriptor_detect(frame, self.ref, distance=self.distance, draw_matches=False)
        if output is None:
            return
        _, self.contour = output
        self.init_tracker(frame, self.contour)

    def check_if_lost(self, bbox):
        x, y, w, h = cv.boundingRect(self.contour)
        px, py, pw, ph = bbox
        if np.linalg.norm(np.subtract([x, y], [px, py])) > w:
            self.timer += 1
        else:
            self.timer = 0


class CardPile(StaticObject):
    def __init__(self, name, ref, distance=0.5):
        super().__init__(name)
        self.ref = ref
        self.distance = distance
        self.M = None
        self.contour = None

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        return frame

    def redetect(self, frame, M_board):
        output = descriptor_detect(frame, self.ref, distance=self.distance, draw_matches=False)
        if output is None:
            return None
        _, self.contour = output

    def draw(self, frame, msg=None, color=(0, 255, 0)):
        if msg is None:
            msg = self.name
        x, y, w, h = cv.boundingRect(self.contour)
        draw_bbox(frame, (x, y, w, h), color)
        return cv.putText(frame, msg, (x, y - 2), cv.FONT_HERSHEY_SIMPLEX, 1.5, color,
                          3, cv.LINE_AA)
