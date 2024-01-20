from typing import Sequence

import cv2 as cv
import PIL.Image
import numpy as np
from IPython.display import display

from src.tracking.Event import Event


def imshow(img: np.ndarray):
    """
    Displays the image, courtesy of the lab notebooks

    :param img: Image to display
    """
    img = img.clip(0, 255).astype("uint8")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(img))


def scaled_imshow(img: np.ndarray, fx=0.3, fy=0.3):
    """
    Displays the image scaled by the given factor

    :param img: Image to display
    :type img: np.ndarray
    :param fx: Factor to scale the image by in the x-axis, defaults to 0.3
    :type fx: float, optional
    :param fy: Factor to scale the image by in the y-axis, defaults to 0.3
    :type fy: float, optional
    """
    scaled_img = cv.resize(img, None, fx=fx, fy=fy)
    imshow(scaled_img)


def draw_bbox(image: np.ndarray, bbox: Sequence[int], color=(255, 255, 255), thickness=2) -> np.ndarray:
    """
    Draws a bounding box in the given image

    :param image: Image to draw the bounding box in
    :type image: np.ndarray
    :param bbox: Bounding box to draw
    :type bbox: Sequence[int]
    :param color: Color of the bounding box, defaults to (255, 255, 255)
    :type color: tuple[int, int, int], optional
    :param thickness: Thickness of the bounding box, defaults to 2
    :type thickness: int, optional
    :return: Image with the bounding box drawn
    :rtype: np.ndarray
    """
    x, y, w, h = bbox
    p1 = (int(x), int(y))
    p2 = (int(x + w), int(y + h))

    return cv.rectangle(image, p1, p2, color, thickness, 1)


def display_events(image: np.ndarray, events: list[Event]) -> np.ndarray:
    """
    Displays the events in the image

    :param image: Image to display the events in
    :type image: np.ndarray
    :param events: List of events to display
    :type events: list[Event]
    :return: Image with the events displayed
    """
    sorted_events = sorted([event for event in events if event.get() != ""], key=lambda e: e.timer)

    for idx, event in enumerate(sorted_events):
        image = cv.putText(image, f"{event.msg}", (10, 50 + 75 * idx), cv.FONT_HERSHEY_SIMPLEX, 2,
                           (0, 0, 0), 10, cv.LINE_AA)
        image = cv.putText(image, f"{event.msg}", (10, 50 + 75 * idx), cv.FONT_HERSHEY_SIMPLEX, 2,
                           (255, 255, 255), 3, cv.LINE_AA)
    return image
