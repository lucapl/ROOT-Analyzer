import numpy as np

from src.tracking.StaticObject import StaticObject
from src.tracking.TrackedObject import TrackedObject


class Card(TrackedObject):

    def __init__(self, name, tracker_type, starting_contour, first_frame, velocity_sensivity=10):
        super().__init__(name, tracker_type, starting_contour, first_frame, velocity_sensivity)

    def detect_events(self, frame_num: int):
        if self.is_moving():
            self.events.append((frame_num, "Card Moved from the pile"))
            return "Card Moved from the pile"


class CardPile(StaticObject):
    def __init__(self, name, contour):
        super().__init__(name, contour)

    def detect_events(self, frame_num: int, frame: np.ndarray) -> str | None:
        return None
