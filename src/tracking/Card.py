import numpy as np
import cv2 as cv

from src.tracking.StaticObject import StaticObject
from src.tracking.TrackedObject import TrackedObject
from src.detection.elements import detect_from_reference
from src.viz.images import draw_bbox


class Card(TrackedObject):

    def __init__(self, name, tracker_type, card_pile, distance=.5, velocity_sensitivity=10):
        super().__init__(name, tracker_type, velocity_sensitivity)
        self.distance = distance
        self.card_pile = card_pile

    def detect_events(self, frame: np.ndarray):
        self.event.update()

        if self.is_moving():
            x, y = self.velocity
            player = "Orange" if y > 0 else "Blue"
            self.event.msg = f"Card drawn by {player}"
            self.event.reset()

    def re_detect(self, frame):
        self.init_tracker(frame, self.card_pile.contour)
        self.last_bbox = cv.boundingRect(self.card_pile.contour)

    def update_timer(self, bbox):
        x, y, w, h = cv.boundingRect(self.card_pile.contour)
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
        self.contour = None

    def re_detect(self, frame):
        _, contour, _ = detect_from_reference(frame, self.ref, distance=self.distance, draw_matches=False)

        if contour is None:
            return

        self.contour = contour

    def draw(self, frame, msg=None, color=(0, 255, 0)):
        return cv.drawContours(frame, [self.contour], -1, color, 2)
