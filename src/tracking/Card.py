import numpy as np

from src.tracking.StaticObject import StaticObject
from src.tracking.TrackedObject import TrackedObject
from src.detection.elements import descriptor_detect


class Card(TrackedObject):

    def __init__(self, name, tracker_type, starting_contour, first_frame, velocity_sensivity=10):
        super().__init__(name, tracker_type, starting_contour, first_frame, velocity_sensivity)


    def detect_events(self, frame_num: int):
        if self.is_moving():
            x,y = self.velocity
            player = "Orange" if y > 0 else "Blue"
            event_msg = f"Card drawn by {player}"
            self.events.append((frame_num, event_msg))
            return event_msg
    def redetect(self, frame):
        return None


class CardPile(StaticObject):
    def __init__(self, name, contour,ref,distance = 0.5):
        super().__init__(name, contour)
        self.ref = ref
        self.distance = distance

    def detect_events(self, frame_num: int, frame: np.ndarray) -> None:
        return None
    
    def redetect(self, frame):
        output = descriptor_detect(frame,self.ref,distance=self.distance,draw_matches=False)
        if output is None:
            return None
        M,cont = output
        super().contour = cont
        self.M = M
