import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod
from src.Event import Event

from src.viz.images import draw_bbox


class StaticObject(ABC):

    BLUE_COLOR = (255,0,0)
    ORANGE_COLOR = (0,125,255)

    def __init__(self, name: str, contour: np.ndarray,event_timer_limit=48) -> None:
        self.name = name
        self.contour = contour
        self.last_bbox = cv.boundingRect(contour)
        print(self.last_bbox)
        self.events = []
        self.event = Event(event_timer_limit)

    @abstractmethod
    def detect_events(self, frame_num: int, frame: np.ndarray) -> str:
        return None
    
    @abstractmethod
    def redetect(self,frame):
        return None

    def draw_bbox(self, frame, msg=None, color=(255, 0, 0)):
        if msg is None:
            msg = self.name
        x, y, w, h = self.last_bbox
        draw_bbox(frame, self.last_bbox, color)

        return cv.putText(frame, msg, (x, y - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, color,
                          1, cv.LINE_AA)

    def set_contour(self,contour):
        self.contour = contour