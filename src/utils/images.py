import cv2 as cv
import numpy as np


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


def calculate_color_coverage(img: np.ndarray, lower_color: np.ndarray, upper_color: np.ndarray) -> float:
    """
    Calculates the specified color coverage of the image

    :param img: The image to be analyzed
    :type img: np.ndarray
    :param lower_color: Lower bound of the color range
    :type lower_color: np.ndarray
    :param upper_color: Upper bound of the color range
    :type upper_color: np.ndarray
    :return: Coverage of the specified color range
    :rtype: float
    """
    # Convert the image to the HSV color space (Hue, Saturation, Value)
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Create a binary mask for the specified color range
    color_mask = cv.inRange(hsv_image, lower_color, upper_color)

    # Calculate the percentage of non-zero pixels in the mask
    total_pixels = np.prod(color_mask.shape)
    colored_pixels = np.count_nonzero(color_mask)
    return colored_pixels / total_pixels
