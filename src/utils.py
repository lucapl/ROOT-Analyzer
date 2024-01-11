import cv2 as cv
import numpy as np

TRACKER_TYPES = {
    # "BOOSTING": cv2.TrackerBoosting_create(),
    "MIL": cv2.TrackerMIL_create(),
    "KCF": cv2.TrackerKCF_create(),
    # "TLD": cv2.TrackerTLD_create(),
    # "MEDIANFLOW": cv2.TrackerMedianFlow_create(),
    # "GOTURN": cv2.TrackerGOTURN_create(),
    # "MOSSE": cv2.TrackerMOSSE_create(),
    "CSRT": cv2.TrackerCSRT_create(),
}


def crop_contour(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
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


def create_tracker(tracker_type) -> cv.Tracker:
    ''' 
    This function needs to create new trackers each time, so it can't be the previous solution
    Tracker types:

        "BOOSTING",
        "MIL",
        "KCF",
        "TLD",
        "MEDIANFLOW",
        "GOTURN",
        "MOSSE",
        "CSRT",
    '''

    if tracker_type == "BOOSTING":
        return cv.TrackerBoosting_create()
    if tracker_type == "MIL":
        return cv.TrackerMIL_create()
    if tracker_type == "KCF":
        return cv.TrackerKCF_create()
    if tracker_type == "TLD":
        return cv.TrackerTLD_create()
    if tracker_type == "MEDIANFLOW":
        return cv.TrackerMedianFlow_create()
    if tracker_type == "GOTURN":
        return cv.TrackerGOTURN_create()
    if tracker_type == "MOSSE":
        return cv.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        return cv.TrackerCSRT_create()

def dominant_colors(image: np.ndarray, n_clusters=3):
    data = cv2.resize(image, (100, 100)).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)

    cluster_sizes = np.bincount(labels.flatten())

    return centers, cluster_sizes


def create_tracker(tracker_type):
    return TRACKER_TYPES.get(tracker_type, None)


def warp_contours(contours, M):
    return [cv2.perspectiveTransform(cont.astype(np.float64), M).astype(np.int32) for cont in contours]


def calculate_color_percentage(img, lower_color, upper_color):
    # Convert the image to the HSV color space (Hue, Saturation, Value)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a binary mask for the specified color range
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Calculate the percentage of non-zero pixels in the mask
    total_pixels = np.prod(color_mask.shape)
    colored_pixels = np.count_nonzero(color_mask)
    percentage = (colored_pixels / total_pixels) * 100

    return percentage
