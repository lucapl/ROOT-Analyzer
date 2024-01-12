import numpy as np
from abc import ABC, abstractmethod
from src.Event import Event


class StaticObject(ABC):
    BLUE_COLOR = (255, 0, 0)
    ORANGE_COLOR = (0, 125, 255)

    LOWER_ORANGE = np.array([0, 100, 100])
    UPPER_ORANGE = np.array([20, 255, 255])
    LOWER_DARK_BLUE = np.array([100, 50, 50])
    UPPER_DARK_BLUE = np.array([140, 255, 255])

    def __init__(self, name: str, event_timer_limit=120) -> None:
        self.name = name
        self.event = Event(event_timer_limit)

    @abstractmethod
    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        return frame

    @abstractmethod
    def redetect(self, frame, M_board) -> None:
        return

    @abstractmethod
    def draw(self, frame, msg=None, color=(255, 0, 0)) -> np.ndarray:
        return frame
