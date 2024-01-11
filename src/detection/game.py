import cv2 as cv
import numpy as np

from src.utils import calculate_color_percentage, crop_contour
from src.viz.images import imshow


def calculate_current_score(img: np.ndarray, cell_contours: list[np.ndarray],
                            orange: tuple[np.ndarray, np.ndarray],
                            blue: tuple[np.ndarray, np.ndarray]) -> tuple[int, int]:
    """ calculates the current score of the game """
    def get_score(cell_contour: np.ndarray, lower_color: np.ndarray, upper_color: np.ndarray) -> int:
        cell = crop_contour(img, cell_contour)
        return calculate_color_percentage(cell, lower_color, upper_color)

    orange_percenteges = list(map(lambda cont: get_score(cont, orange[0], orange[1]), cell_contours))
    blue_percenteges = list(map(lambda cont: get_score(cont, blue[0], blue[1]), cell_contours))

    orange_score = orange_percenteges.index(max(orange_percenteges))
    blue_score = blue_percenteges.index(max(blue_percenteges))

    return orange_score, blue_score


def calculate_current_buildings(img: np.ndarray, cell_contours: list[np.ndarray], orange: tuple[np.ndarray, np.ndarray],
                                blue: tuple[np.ndarray, np.ndarray]) -> tuple[int, int]:
    """ calculates the current score of the game """
    def get_score(cell_contour: np.ndarray, lower_color: np.ndarray, upper_color: np.ndarray) -> int:
        cell = crop_contour(img, cell_contour)
        return calculate_color_percentage(cell, lower_color, upper_color)

    orange_percenteges = list(map(lambda cont: get_score(cont, orange[0], orange[1]), cell_contours))
    blue_percenteges = list(map(lambda cont: get_score(cont, blue[0], blue[1]), cell_contours))

    orange_score = sum([1 for i in orange_percenteges if i > 33])
    blue_score = sum([1 for i in blue_percenteges if i > 33])

    return orange_score, blue_score
