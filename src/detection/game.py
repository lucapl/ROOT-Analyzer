import cv2 as cv
import numpy as np

from src.utils.images import calculate_color_coverage, crop_image


def calculate_current_score(img: np.ndarray, cell_contours: list[np.ndarray],
                            orange: tuple[np.ndarray, np.ndarray],
                            blue: tuple[np.ndarray, np.ndarray]) \
        -> tuple[int, int]:
    """
    Calculates the current score of the game

    :param img: Image of the game
    :type img: np.ndarray
    :param cell_contours: Contours of the cells
    :type cell_contours: list[np.ndarray]
    :param orange: Color range of the orange team
    :type orange: tuple[np.ndarray, np.ndarray]
    :param blue: Color range of the blue team
    :type blue: tuple[np.ndarray, np.ndarray]
    :return: Current score of the game (Orange score, Blue score)
    :rtype: tuple[int, int]
    """
    orange_coverage = list(map(lambda cont: get_contour_coverage(img, cont, orange[0], orange[1]), cell_contours))
    blue_coverage = list(map(lambda cont: get_contour_coverage(img, cont, blue[0], blue[1]), cell_contours))

    orange_score = orange_coverage.index(max(orange_coverage))
    blue_score = blue_coverage.index(max(blue_coverage))

    return orange_score, blue_score


def calculate_current_buildings_control(img: np.ndarray, cell_contours: list[np.ndarray],
                                        orange: tuple[np.ndarray, np.ndarray],
                                        blue: tuple[np.ndarray, np.ndarray], color_sensitivity=0.33) \
        -> tuple[list[bool], list[bool]]:
    """
    Calculates the current buildings of the game by color

    :param img: Image of the game
    :type img: np.ndarray
    :param cell_contours: Contours of the cells
    :type cell_contours: list[np.ndarray]
    :param orange: Color range of the orange team
    :param orange: tuple[np.ndarray, np.ndarray]
    :param blue: Color range of the blue team
    :type blue: tuple[np.ndarray, np.ndarray]
    :param color_sensitivity: Sensitivity of the color detection, defaults to 0.33
    :type color_sensitivity: float, optional
    :return: Current buildings of the game (Orange buildings, Blue buildings)
    :rtype: tuple[list[bool], list[bool]]
    """
    orange_coverage = list(map(lambda cont: get_contour_coverage(img, cont, orange[0], orange[1]), cell_contours))
    blue_coverage = list(map(lambda cont: get_contour_coverage(img, cont, blue[0], blue[1]), cell_contours))

    orange_buildings = [i > color_sensitivity for i in orange_coverage]
    blue_buildings = [i > color_sensitivity for i in blue_coverage]

    return orange_buildings, blue_buildings


def calculate_current_clearing_control(orange_pawns: dict[int, list[np.ndarray]],
                                       blue_pawns: dict[int, list[np.ndarray]]) \
        -> tuple[list[bool], list[bool]]:
    """
    Calculates the current clearing control of the game by pawns

    :param orange_pawns: Dictionary of the orange pawns by clearing
    :type orange_pawns: dict[int, list[np.ndarray]]
    :param blue_pawns:  Dictionary of the blue pawns by clearing
    :type blue_pawns: dict[int, list[np.ndarray]]
    :return: Current clearing control of the game (Orange control, Blue control)
    :rtype: tuple[list[bool], list[bool]]
    """
    orange_clearings = [False] * 12
    blue_clearings = [False] * 12

    for i in range(12):
        orange_clearings[i] = len(orange_pawns[i]) > 0 and len(orange_pawns[i]) > len(blue_pawns[i])
        blue_clearings[i] = len(blue_pawns[i]) > 0 and len(blue_pawns[i]) >= len(orange_pawns[i])

    return orange_clearings, blue_clearings


def get_contour_coverage(img: np.ndarray, contour: np.ndarray, lower_color: np.ndarray, upper_color: np.ndarray) \
        -> float:
    """
    Calculates the specified color coverage of the contour

    :param img: Image of the game
    :type img: np.ndarray
    :param contour: Contour of the cell
    :type contour: np.ndarray
    :param lower_color: Lower bound of the color range
    :type lower_color: np.ndarray
    :param upper_color: Upper bound of the color range
    :type upper_color: np.ndarray
    :return: Color coverage of the contour
    :rtype: float
    """
    cell = crop_image(img, contour)
    return calculate_color_coverage(cell, lower_color, upper_color)
