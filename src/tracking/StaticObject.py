import numpy as np
from abc import ABC, abstractmethod
from src.Event import Event


class StaticObject(ABC):
    BLUE_COLOR = (255, 0, 0)
    ORANGE_COLOR = (0, 125, 255)

    def __init__(self, name: str, event_timer_limit=48) -> None:
        self.name = name
        self.events = []
        self.event = Event(event_timer_limit)
        self.M = None

    @abstractmethod
    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        return frame

    @abstractmethod
    def redetect(self, frame, M_board) -> None:
        return

    @abstractmethod
    def draw(self, frame, msg=None, color=(255, 0, 0)) -> np.ndarray:
        return frame
