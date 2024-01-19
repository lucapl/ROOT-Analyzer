import cv2 as cv
import numpy as np
from copy import deepcopy


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


def reorder_contours(contours: list[tuple[int, np.ndarray]], swap_list: list[tuple[int, int]]) \
        -> list[tuple[int, np.ndarray]]:
    """
    Reorders the contours according to the swap list

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
