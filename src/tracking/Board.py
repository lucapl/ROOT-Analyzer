import numpy as np

from src.tracking.StaticObject import StaticObject
from src.detection.elements import descriptor_detect


class Board(StaticObject):

    def __init__(self, name, board_ref, distance=0.25):
        super().__init__(name)
        self.ref = board_ref
        self.distance = 0.25
        self.M = None

    def redetect(self, frame,):
        output = descriptor_detect(frame, self.ref, distance=self.distance, draw_matches=False)
        if output is None:
            return None
        M, cont = output
        self.M = M

    def detect_events(self, frame_num: int, frame) -> np.ndarray:
        return frame

    def draw(self, frame, msg=None, color=(0, 255, 0)):
        return frame
