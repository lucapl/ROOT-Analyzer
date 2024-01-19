import cv2 as cv
import numpy as np

from src.utils.images import crop_image
from src.utils.helpers import get_highest_hierarchy, get_children, safe_division
from src.utils.contours import reorder_contours


def detect_dice_tray(img: np.ndarray, thresh=50, draw_contours=False) \
        -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Detects the dice tray and the dice inside it
    This function utilizes the fact that dice tray is all black and dice are white

    :param img: Image to detect the dice tray and dice in
    :type img: np.ndarray
    :param thresh: Threshold for the dice tray detection
    :type thresh: int
    :param draw_contours: Whether to draw the contours over the image, defaults to False
    :return: Dice tray contour, dice 1 contour, dice 2 contour, image with contours drawn (if draw_contours is True).
        Dice contours are None if not found
    :rtype: tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]
    """
    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    filtered = cv.bilateralFilter(gray, 9, 250, 250)

    # define thresholds
    _, edges = cv.threshold(filtered, thresh, 255, cv.THRESH_BINARY)
    edges = 255 - edges

    # find contours
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, 2)

    # detect dice tray, take the largest contour
    tray_idx, tray = max(enumerate(contours), key=lambda i_c: cv.contourArea(i_c[1]))

    # detect dice, take the two smallest contours
    tray_children = get_children(tray_idx, hierarchy)
    tray_children_mapped = map(lambda i: cv.boundingRect(contours[i]), tray_children)
    tray_children_sorted = sorted(zip(tray_children, tray_children_mapped),
                                  key=lambda bound: -bound[1][2] * bound[1][3])

    dice1, dice2 = None, None

    if len(tray_children_sorted) >= 2:
        dice1, dice2 = contours[tray_children_sorted[0][0]], contours[tray_children_sorted[1][0]]

    if draw_contours:
        img_contours = np.copy(img)
        if dice1 is not None and dice2 is not None:
            cv.drawContours(img_contours, [dice1, dice2], -1, (0, 255, 0), 2)
        cv.drawContours(img_contours, [tray], -1, (0, 0, 255), 2)

        return tray, dice1, dice2, img_contours
    else:
        return tray, dice1, dice2, None


def detect_from_reference(img: np.ndarray, ref: np.ndarray, distance=0.25, draw_matches=False) \
        -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Detects the reference object using descriptors

    :param img: Image to detect the reference object in
    :type img: np.ndarray
    :param ref: Reference object image
    :type ref: np.ndarray
    :param distance: Distance threshold for the matches
    :type distance: float
    :param draw_matches: Whether to draw the matches over the image, defaults to False
    :type draw_matches: bool
    :return: Homography matrix, contour of the reference object, image with matches drawn (if draw_matches is True).
        Matrix and contour are None if not found
    :rtype: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
    """
    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ref_gray = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)

    # detect key points and descriptors
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)
    kp2, desc2 = sift.detectAndCompute(ref_gray, None)

    # match descriptors and filter them by distance
    matcher = cv.FlannBasedMatcher()
    matches = matcher.match(desc2, desc)
    max_distance = max(matches, key=lambda x: x.distance).distance
    matches = list(filter(lambda x: x.distance < distance * max_distance, matches))

    # if no matches found, return
    if len(matches) == 0:
        return None, None, None

    # find homography matrix
    src_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    # find contour of the reference object
    ref_gray = cv.warpPerspective(ref_gray, m, (img.shape[1], img.shape[0]))
    _, ref_gray = cv.threshold(ref_gray, 50, 255, cv.THRESH_BINARY)
    ref_gray = cv.morphologyEx(ref_gray, cv.MORPH_CLOSE, kernel=np.ones((7, 7)))
    contours, _ = cv.findContours(ref_gray, cv.RETR_TREE, 2)
    obj_contour = contours[0]

    if draw_matches:
        draw_params = dict(matchColor=(255, 0, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=mask.ravel().tolist(),  # draw only inliers
                           flags=2 | 4,
                           )
        img_matches = cv.drawMatches(ref, kp2, img, kp, matches, None, **draw_params)
        return m, obj_contour, img_matches
    else:
        return m, obj_contour, None


def detect_score_board(img: np.ndarray, mask: np.ndarray, thresh_args=(55, 15), hor_ker=np.ones((1, 40)),
                       ver_ker=np.ones((30, 1)), hor_ver_ker=np.ones((7, 7))) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Detects the score board and the cells inside it using a mask

    :param img: Image to detect the score board and cells in
    :type img: np.ndarray
    :param mask: Mask of the score board
    :type mask: np.ndarray
    :param thresh_args: Arguments for the adaptive thresholding (block size, constant), defaults to (55, 15)
    :type thresh_args: tuple[int, int]
    :param hor_ker: Horizontal kernel for the morphological operations, defaults to np.ones((1, 40))
    :type hor_ker: np.ndarray
    :param ver_ker: Vertical kernel for the morphological operations, defaults to np.ones((30, 1))
    :type ver_ker: np.ndarray
    :param hor_ver_ker: Horizontal and vertical kernel for the morphological operations, defaults to np.ones((7, 7))
    :type hor_ver_ker: np.ndarray
    :return: List of cell contours, score board contour
    :rtype: tuple[list[np.ndarray], np.ndarray]
    """
    # Find the score board contour and crop the image to it
    mask_cont = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0][0]
    cropped = crop_image(img, mask_cont)
    img_gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

    # Adaptive thresholding
    threshold = cv.adaptiveThreshold(
        img_gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        *thresh_args
    )

    # Morphological operations
    hor = 255 - cv.erode(cv.dilate(threshold, hor_ker), hor_ker)
    ver = 255 - cv.erode(cv.dilate(threshold, ver_ker), ver_ker)
    hor_ver = cv.morphologyEx(hor + ver, cv.MORPH_CLOSE, hor_ver_ker)

    # Find contours of the cells and the score board
    contours, hierarchy = cv.findContours(hor_ver, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    score_board_idx, _ = max(enumerate(contours), key=lambda i_c: cv.contourArea(i_c[1]))
    cells = get_children(score_board_idx, hierarchy)

    return [contours[i] for i in cells[::-1]], mask_cont


def detect_clearings_and_buildings(mask: np.ndarray) \
        -> tuple[list[np.ndarray], dict[int, list[np.ndarray]]]:
    """
    Detects the clearings and buildings from the clearing mask 
    
    :param mask: Mask of the clearings
    :type mask: np.ndarray 
    :return: List of clearing contours, dictionary of building contours for each clearing
    :rtype: tuple[list[np.ndarray], dict[int, list[np.ndarray]]]
    """
    # Find contours of the clearings and buildings
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Get the highest hierarchy contours (parentless) as clearings and reorganize them
    clearings = [(cont_idx, contours[cont_idx]) for cont_idx in get_highest_hierarchy(hierarchy)]
    clearings = reorder_contours(clearings, [(0, 3), (0, 2), (1, 3), (5, 6)])

    # Get the lowest hierarchy contours (childless) for each clearing as buildings
    buildings = {i: [contours[j] for j in get_children(c_idx, hierarchy)]
                 for i, (c_idx, _) in enumerate(clearings)}
    clearings = [cont for _, cont in clearings]

    return clearings, buildings


def detect_pawns(img: np.ndarray, mask: np.ndarray, clearings: list[np.ndarray],
                 orange: tuple[np.ndarray, np.ndarray], blue: tuple[np.ndarray, np.ndarray],
                 diff_sensitivity=0.5, area_sensitivity=0.3) \
        -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    """
    Detects the pawns from the clearing mask

    :param img: Image to detect the pawns in
    :type img: np.ndarray
    :param mask: Mask of the clearings
    :type mask: np.ndarray
    :param clearings: Clearing contours
    :type clearings: list[np.ndarray]
    :param orange: Color range of the orange team
    :type orange: tuple[np.ndarray, np.ndarray]
    :param blue: Color range of the blue team
    :type blue: tuple[np.ndarray, np.ndarray]
    :param diff_sensitivity: Sensitivity of the difference between areas of the next sorted contours, defaults to 0.5
    :type diff_sensitivity: float, optional
    :param area_sensitivity: Sensitivity of the area of the contour to the biggest contour, defaults to 0.3
    :type area_sensitivity: float, optional
    :return: Dictionary of orange pawns for each clearing, dictionary of blue pawns for each clearing
    :rtype: tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = cv.bitwise_and(hsv, hsv, mask=mask)
    pawns: list[dict[int, list[np.ndarray]]] = []

    for color_range in (orange, blue):
        pawns.append({})
        color_mask = cv.inRange(hsv, *color_range)
        color_mask = cv.erode(color_mask, np.ones((5, 5)))
        contours, _ = cv.findContours(color_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        biggest_area = np.max(tuple(map(lambda c: cv.contourArea(c), contours)))

        for c_idx, c_cont in enumerate(clearings):
            pawns[-1][c_idx] = []
            clearing = crop_image(color_mask, c_cont)
            contours, _ = cv.findContours(clearing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            areas = list(sorted(map(lambda c: (c, cv.contourArea(c)), contours), reverse=True, key=lambda x: x[1]))
            diffs = [safe_division((areas[i][1] - areas[i + 1][1]), areas[i][1]) for i in range(len(areas) - 1)]
            if len(diffs) == 0 and len(areas) > 0:
                diffs.append(1)
            for i in range(len(diffs)):
                if safe_division(areas[i][1], biggest_area) < area_sensitivity:
                    break

                pawns[-1][c_idx].append(areas[i][0])

                if diffs[i] > diff_sensitivity:
                    break

    return pawns[0], pawns[1]
