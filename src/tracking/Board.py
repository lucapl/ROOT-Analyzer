import numpy as np
import cv2 as cv

from src.tracking.StaticObject import StaticObject
from src.detection.elements import detect_from_reference


class Board(StaticObject):
    """
    Represents the board of the game

    :var name: Name of the object
    :type name: str
    :var ref: Reference image of the board
    :type ref: np.ndarray
    :var distance: Distance threshold for the matching algorithm, defaults to 0.25
    :type distance: float, optional
    :var contour: Contour of the board
    :type contour: np.ndarray | None
    """
    def __init__(self, name, ref, distance=0.25):
        """
        Initializes the board object
        """
        super().__init__(name)
        self.ref = ref
        self.distance = distance
        self.m = None
        self.contour = None

    def re_detect(self, frame):
        m, contour, _ = detect_from_reference(frame, self.ref, distance=self.distance)

        if m is None or contour is None:
            return

        self.m = m
        self.contour = contour

    def draw(self, frame, color=(0, 122, 0)) -> np.ndarray:
        if self.contour is None:
            return frame

        return cv.drawContours(frame, [self.contour], -1, color, 2)
