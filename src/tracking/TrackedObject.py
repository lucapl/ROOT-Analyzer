import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod

from src.tracking.Event import Event
from src.viz.images import draw_bbox
from src.utils.helpers import create_tracker


class TrackedObject(ABC):
    """
    Abstract class for tracked objects

    :var name: Name of the object
    :type name: str
    :var tracker: Tracker object
    :type tracker: cv.Tracker
    :var is_init: Whether the tracker is initialized
    :type is_init: bool
    :var event: Event object
    :type event: Event
    :var vel_sens: Velocity sensitivity threshold
    :type vel_sens: int
    :var velocity: Velocity of the object
    :type velocity: np.ndarray
    :var last_bbox: Last bounding box of the object
    :type last_bbox: tuple[int, int, int, int] | None
    :var timer: Timer for refreshing the tracker
    :type timer: int
    :var refresh_rate: Refresh rate of the tracker
    :type refresh_rate: int
    """
    def __init__(self, name: str, tracker_type: str, velocity_sensitivity=10, refresh_rate=48, event_timer_limit=120):
        """
        Initializes the tracked object

        :param name: Name of the object
        :type name: str
        :param tracker_type: Type of the tracker
        :type tracker_type: str
        :param velocity_sensitivity: Velocity sensitivity threshold, defaults to 10
        :type velocity_sensitivity: int, optional
        :param refresh_rate: Refresh rate of the tracker, defaults to 48
        :type refresh_rate: int, optional
        :param event_timer_limit: Event timer limit, defaults to 120
        :type event_timer_limit: int, optional
        """
        self.name = name
        self.tracker = create_tracker(tracker_type)
        self.is_init = False
        self.event = Event(event_timer_limit)

        self.vel_sens = velocity_sensitivity
        self.velocity = np.array([0, 0])
        self.last_bbox = None

        self.timer = 0
        self.refresh_rate = refresh_rate

    @abstractmethod
    def re_detect(self, frame):
        """
        Re-detects the object in the frame

        :param frame: Frame to detect the object in
        :type frame: np.ndarray
        """
        return

    def detect_events(self, frame: np.ndarray):
        """
        Detects events in the frame

        :param frame: Frame to detect events in
        :type frame: np.ndarray
        """
        self.event.update()

    def init_tracker(self, frame, contour):
        """
        Initializes the tracker with the given frame and contour

        :param frame: Frame to initialize the tracker with
        :type frame: np.ndarray
        :param contour: Contour to initialize the tracker with
        :type contour: np.ndarray
        """
        self.tracker.init(frame, cv.boundingRect(contour))
        self.is_init = True

    def is_moving(self):
        """
        Checks if the object is moving

        :return: Whether the object is moving
        :rtype: bool
        """
        return np.linalg.norm(self.velocity, 2) > self.vel_sens

    def update(self, frame) -> bool:
        """
        Updates the tracker with the given frame

        :param frame: Frame to update the tracker with
        :type frame: np.ndarray
        :return: Whether the update was successful
        :rtype: bool
        """
        if not self.is_init:
            return False

        if self.timer > self.refresh_rate:
            self.re_detect(frame)
            self.timer = 0
            return True

        ok, bbox = self.tracker.update(frame)

        if ok:
            bbox = tuple(bbox)

            if self.last_bbox is not None:
                self._update_velocity(bbox)

            self.last_bbox = bbox

            self.update_timer(bbox)

            return True
        else:
            self.timer += 1
            return False

    def update_timer(self, bbox):
        """
        Updates the timer

        :param bbox: Bounding box of the object
        :type bbox: tuple[int, int, int, int]
        """
        self.timer = 0

    def detection_fail_msg(self, frame):
        """
        Returns the message to display when the detection fails

        :param frame: Frame to display the message in
        :type frame: np.ndarray
        :return: Message to display
        :rtype: str
        """
        return self.draw_bbox(frame, f"{self.name} Lost", (0, 0, 255))

    def draw_bbox(self, frame, msg=None, color=(0, 0, 255)):
        """
        Draws the bounding box of the object in the frame

        :param frame: Frame to draw the bounding box in
        :type frame: np.ndarray
        :param msg: Message to display, defaults to None
        :type msg: str, optional
        :param color: Color of the bounding box, defaults to (0, 255, 0)
        :type color: tuple[int, int, int], optional
        :return: Frame with the bounding box drawn
        :rtype: np.ndarray
        """
        if self.last_bbox is None:
            return frame

        if msg is None:
            msg = self.name

        frame = draw_bbox(frame, self.last_bbox, color)

        x, y, w, h = self.last_bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        return cv.putText(frame, msg, (x + w//2 - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)

    def _update_velocity(self, bbox):
        """
        Updates the velocity of the object

        :param bbox: Bounding box of the object
        :type bbox: tuple[int, int, int, int]
        """
        x, y, _, _ = self.last_bbox
        x_p, y_p, _, _ = bbox

        self.velocity = np.array([x_p - x, y_p - y])
