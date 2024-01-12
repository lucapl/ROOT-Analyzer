import cv2 as cv
import numpy as np


def crop_contour(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """ crops the contour out of the image """
    x, y, w, h = cv.boundingRect(contour)
    cropped = img[y:y + h, x:x + w]
    return cropped


def saturate_image(img: np.ndarray, saturation=2) -> np.ndarray:
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv.split(img_hsv)
    s = s * saturation
    s = np.clip(s, 0, 255)
    img_hsv = cv.merge([h, s, v])

    return cv.cvtColor(img_hsv.astype("uint8"), cv.COLOR_HSV2BGR)


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(img_center, angle, 1.0)
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def create_tracker(tracker_type) -> cv.Tracker:
    trackers = {
        "BOOSTING": cv.legacy.TrackerBoosting_create(),
        "MIL": cv.legacy.TrackerMIL_create(),
        "KCF": cv.legacy.TrackerKCF_create(),
        "TLD": cv.legacy.TrackerTLD_create(),
        "MEDIANFLOW": cv.legacy.TrackerMedianFlow_create(),
        # "GOTURN": cv.legacy.TrackerGOTURN_create(),
        "MOSSE": cv.legacy.TrackerMOSSE_create(),
        "CSRT": cv.legacy.TrackerCSRT_create(),
    }

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
    data = cv.resize(image, (100, 100)).reshape(-1, 3)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)

    cluster_sizes = np.bincount(labels.flatten())

    return centers, cluster_sizes


def warp_contour(contour, M):
    return cv.perspectiveTransform(contour.astype(np.float64), M).astype(np.int32)


def calculate_color_percentage(img, lower_color, upper_color):
    # Convert the image to the HSV color space (Hue, Saturation, Value)
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Create a binary mask for the specified color range
    color_mask = cv.inRange(hsv_image, lower_color, upper_color)

    # Calculate the percentage of non-zero pixels in the mask
    total_pixels = np.prod(color_mask.shape)
    colored_pixels = np.count_nonzero(color_mask)
    percentage = (colored_pixels / total_pixels) * 100

    return percentage


def resize_contour(contour, factor=0.25):
    resized_contour = np.copy(contour)
    resized_contour[:, :, 0] = contour[:, :, 0] * factor
    resized_contour[:, :, 1] = contour[:, :, 1] * factor
    return resized_contour

def get_clearings_and_buildings(clearing_mask):

    def contours_reorg(contours):
        swaps = [(0,3),
                (0,2),
                (1,3),
                (5,6)]
        for i,j in swaps:
            contours[i],contours[j] = contours[j],contours[i]
        contours.reverse()
        contours[-4:] = contours[-4:][::-1]
        return contours
    
    def get_children(i,contours,hierarchy):

        _,_,child,_ = hierarchy[0][i]
        children = []
        while True:
            if child == -1:
                break
            children.append(contours[child])
            child, _, _, _ = hierarchy[0][child]
        return children
    
    contours, hierarchy = cv.findContours(clearing_mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    filtered = [(i,cont) for i,cont in enumerate(contours) if hierarchy[0][i][3] == -1]
    reorged = contours_reorg(filtered)
    reorged_conts = []

    buildings = {}
    for j,(i,cont) in enumerate(reorged):
        buildings[j] = get_children(i,contours,hierarchy)
        reorged_conts.append(cont)

    return reorged_conts,buildings