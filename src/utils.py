import cv2
import numpy as np


def crop_contour(img: np.ndarray, contour: cv2.Mat) -> np.ndarray:
    """ crops the contour out of the image """
    x, y, w, h = cv2.boundingRect(contour)
    cropped = img[y:y + h, x:x + w]
    return cropped


def saturate_image(img: np.ndarray, saturation=2) -> np.ndarray:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")

    (h, s, v) = cv2.split(img_hsv)
    s = s * saturation
    s = np.clip(s, 0, 255)
    img_hsv = cv2.merge([h, s, v])

    return cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
