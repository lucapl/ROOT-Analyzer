import numpy as np
from abc import ABC, abstractmethod
from src.tracking.Event import Event


class StaticObject(ABC):
    """
    Abstract class for static objects

    :var name: Name of the object
    :type name: str
    :var refresh_rate: Refresh rate of the tracker
    :type refresh_rate: int
    :var event: Event object
    :type event: Event
    :var BLUE_COLOR: Blue color
    :type BLUE_COLOR: tuple[int, int, int]
    :var ORANGE_COLOR: Orange color
    :type ORANGE_COLOR: tuple[int, int, int]
    :var LOWER_ORANGE: Lower orange RGB threshold
    :type LOWER_ORANGE: np.ndarray
    :var UPPER_ORANGE: Upper orange RGB threshold
    :type UPPER_ORANGE: np.ndarray
    :var LOWER_DARK_BLUE: Lower dark blue RGB threshold
    :type LOWER_DARK_BLUE: np.ndarray
    :var UPPER_DARK_BLUE: Upper dark blue RGB threshold
    :type UPPER_DARK_BLUE: np.ndarray
    """
    BLUE_COLOR = (255, 0, 0)
    ORANGE_COLOR = (0, 125, 255)

    LOWER_ORANGE = np.array([0, 100, 100])
    UPPER_ORANGE = np.array([20, 255, 255])
    LOWER_DARK_BLUE = np.array([100, 50, 50])
    UPPER_DARK_BLUE = np.array([140, 255, 255])

    def __init__(self, name: str, refresh_rate=900, event_timer_limit=120) -> None:
        """
        Initializes the static object

        :param name: Name of the object
        :type name: str
        :param refresh_rate: Refresh rate of the tracker, defaults to 900
        :type refresh_rate: int, optional
        :param event_timer_limit: Event timer limit, defaults to 120
        :type event_timer_limit: int, optional
        """
        self.name = name
        self.refresh_rate = refresh_rate
        self.event = Event(event_timer_limit)

    @abstractmethod
    def re_detect(self, frame: np.ndarray) -> None:
        """
        Re-detects the object in the frame

        :param frame: Frame to detect the object in
        :type frame: np.ndarray
        """
        return

    @abstractmethod
    def draw(self, frame: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
        """
        Draws the object on the frame

        :param frame: Frame to draw the object on
        :type frame: np.ndarray
        :param color: Color of the object, defaults to (255, 0, 0)
        :type color: tuple[int, int, int], optional
        :return: Frame with the object drawn on it
        """
        return frame

    def detect_events(self, frame: np.ndarray) -> None:
        """
        Detects events in the frame

        :param frame: Frame to detect events in
        :type frame: np.ndarray
        """
        self.event.update()
