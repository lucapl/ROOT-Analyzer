import cv2 as cv
import numpy as np
from copy import deepcopy


def crop_image(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Crops the image to the bounding rectangle of the contour

    :param img: Image to be cropped
    :type img: np.ndarray
    :param contour: Contour to crop the image to
    :type contour: np.ndarray
    :return: Cropped image
    :rtype: np.ndarray  
    """
    x, y, w, h = cv.boundingRect(contour)
    cropped = img[y:y + h, x:x + w]
    return cropped


def saturate_image(img: np.ndarray, factor=2.0) -> np.ndarray:
    """
    Increases the saturation of the image by the specified factor

    :param img: Image to be saturated
    :type img: np.ndarray
    :param factor: Factor to increase the saturation by
    :type factor: float
    :return: Saturated image
    :rtype: np.ndarray
    """
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv.split(img_hsv)
    s = s * factor
    s = np.clip(s, 0, 255)
    img_hsv = cv.merge([h, s, v])

    return cv.cvtColor(img_hsv.astype("uint8"), cv.COLOR_HSV2BGR)


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates the image by the specified angle

    :param img: Image to be rotated
    :type img: np.ndarray
    :param angle: Angle to rotate the image by
    :type angle: float
    :return: Rotated image
    :rtype: np.ndarray
    """
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(img_center, angle, 1.0)
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


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
        "GOTURN": cv.legacy.TrackerGOTURN_create,
        "MOSSE": cv.legacy.TrackerMOSSE_create,
        "CSRT": cv.legacy.TrackerCSRT_create,
    }

    return trackers[tracker_type]() if tracker_type in trackers else None


def calculate_color_percentage(img: np.ndarray, lower_color: np.ndarray, upper_color: np.ndarray) -> float:
    """
    Calculates the percentage of pixels in the image that are in the specified color range

    :param img: The image to be analyzed
    :type img: np.ndarray
    :param lower_color: Lower bound of the color range
    :type lower_color: np.ndarray
    :param upper_color: Upper bound of the color range
    :type upper_color: np.ndarray
    :return: Percentage of pixels in the image that are in the specified color range
    :rtype: float
    """
    # Convert the image to the HSV color space (Hue, Saturation, Value)
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Create a binary mask for the specified color range
    color_mask = cv.inRange(hsv_image, lower_color, upper_color)

    # Calculate the percentage of non-zero pixels in the mask
    total_pixels = np.prod(color_mask.shape)
    colored_pixels = np.count_nonzero(color_mask)
    percentage = (colored_pixels / total_pixels) * 100

    return percentage


def warp_contour(contour: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Warps the contour with the transformation matrix m

    :param contour: Contour to be warped
    :type contour: np.ndarray
    :param m: Transformation matrix
    :type m: np.ndarray
    :return: Warped contour
    :rtype: np.ndarray
    """
    return cv.perspectiveTransform(contour.astype(np.float64), m).astype(np.int32)


def resize_contour(contour: np.ndarray, factor=0.25):
    """
    Resizes the contour by a factor
    :param contour:
    :param factor:
    :return:
    """
    resized_contour = np.copy(contour)
    resized_contour[:, :, 0] = contour[:, :, 0] * factor
    resized_contour[:, :, 1] = contour[:, :, 1] * factor
    return resized_contour


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


def reorganize_contours(contours: list[tuple[int, np.ndarray]], swap_list: list[tuple[int, int]]) \
        -> list[tuple[int, np.ndarray]]:
    """
    Reorganizes the contours according to the swap list

    :param contours: Contours to be reorganized
    :type contours: list[tuple[int, np.ndarray]]
    :param swap_list: List of tuples of indices to be swapped
    :type swap_list: list[tuple[int, int]]
    :return: Reorganized contours
    :rtype: list[tuple[int, np.ndarray]]
    """
    c = deepcopy(contours)
    for i, j in swap_list:
        c[i], c[j] = c[j], c[i]
    c.reverse()
    c[-4:] = c[-4:][::-1]
    return c


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


def get_lowest_children(i: int, hierarchy: np.ndarray) -> list[int]:
    """
    Returns the indices of the children of the contour with index i that have no children themselves

    :param i: Index of the contour
    :type i: int
    :param hierarchy: Hierarchical representation of contours
    :type hierarchy: np.ndarray
    :return: Indices of the children of the contour with index i
    :rtype: list[int]
    """
    return [j for j, node in enumerate(hierarchy[0]) if node[3] == i and node[2] == -1]
