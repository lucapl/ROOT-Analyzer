import cv2 as cv
import numpy as np


def create_tracker(tracker_type: str) -> cv.Tracker:
    """
    Creates a tracker of the specified type

    :param tracker_type: Type of the tracker
    :type tracker_type: str
    :return: Tracker of the specified type
    :rtype: cv.Tracker
    """
    trackers = {
        "BOOSTING": cv.legacy.TrackerBoosting_create,
        "MIL": cv.legacy.TrackerMIL_create,
        "KCF": cv.legacy.TrackerKCF_create,
        "TLD": cv.legacy.TrackerTLD_create,
        "MEDIANFLOW": cv.legacy.TrackerMedianFlow_create,
        # "GOTURN": cv.legacy.TrackerGOTURN_create,
        "MOSSE": cv.legacy.TrackerMOSSE_create,
        "CSRT": cv.legacy.TrackerCSRT_create,
    }

    return trackers[tracker_type]() if tracker_type in trackers else None


def safe_division(a: float, b: float) -> float:
    """
    Safely divides a by b, returns 0 if b is 0

    :param a: Dividend
    :type a: float
    :param b: Divisor
    :type b: float
    :return: Result of a / b
    :rtype: float
    """
    return a / b if b != 0 else 0


def get_highest_hierarchy(hierarchy: np.ndarray) -> list[int]:
    """
    Returns the indices of the highest hierarchy contours (parentless)

    :param hierarchy: Hierarchical representation of contours
    :type hierarchy: np.ndarray
    :return: Indices of the highest hierarchy contours
    :rtype: list[int]
    """
    return [i for i, node in enumerate(hierarchy[0]) if node[3] == -1]


def get_lowest_hierarchy(hierarchy: np.ndarray) -> list[int]:
    """
    Returns the indices and contours of the lowest hierarchy contours (childless)

    :param hierarchy: Hierarchical representation of contours
    :type hierarchy: np.ndarray
    :return: Indices and contours of the lowest hierarchy contours
    :rtype: list[int]
    """
    return [i for i, node in enumerate(hierarchy[0]) if node[2] == -1]


def get_children(i: int, hierarchy: np.ndarray) -> list[int]:
    """
    Returns the indices of the children of the contour with index i

    :param i: Index of the contour
    :type i: int
    :param hierarchy: Hierarchical representation of contours
    :type hierarchy: np.ndarray
    :return: Indices of the children of the contour with index i
    :rtype: list[int]
    """
    return [j for j, node in enumerate(hierarchy[0]) if node[3] == i]
