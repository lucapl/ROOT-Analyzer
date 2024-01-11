import numpy as np
import cv2 as cv

from src.tracking.StaticObject import StaticObject
from src.tracking.TrackedObject import TrackedObject
from src.detection.elements import descriptor_detect
from src.viz.images import draw_bbox


class Card(TrackedObject):

    def __init__(self, name, tracker_type, starting_contour, first_frame, velocity_sensivity=10):
        super().__init__(name, tracker_type, starting_contour, first_frame, velocity_sensivity)

    def detect_events(self, frame_num: int):
        if self.is_moving():
            x, y = self.velocity
            player = "Orange" if y > 0 else "Blue"
            event_msg = f"Card drawn by {player}"
            self.events.append((frame_num, event_msg))
            return event_msg

    def redetect(self, frame):
        self.init_tracker(frame, self.contour)

    def check_if_lost(self, bbox):
        x, y, w, h = cv.boundingRect(self.contour)
        px, py, pw, ph = bbox
        if np.linalg.norm(np.subtract([x, y], [px, py])) > w:
            self.timer += 1
        else:
            self.timer = 0


class CardPile(StaticObject):
    def __init__(self, name, contour, ref, distance=0.5):
        super().__init__(name)
        self.ref = ref
        self.distance = distance
        self.M = None
        self.contour = contour

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        return frame

    def redetect(self, frame, M_board):
        output = descriptor_detect(frame, self.ref, distance=self.distance, draw_matches=False)
        if output is None:
            return None
        M, cont = output
        self.contour = cont
        self.M = M

    def draw(self, frame, msg=None, color=(0, 255, 0)):
        if msg is None:
            msg = self.name
        x, y, w, h = cv.boundingRect(self.contour)
        draw_bbox(frame, (x, y, w, h), color)
        return cv.putText(frame, msg, (x, y - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, color,
                          1, cv.LINE_AA)
