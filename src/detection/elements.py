import cv2 as cv
import numpy as np

from src.utils import crop_contour

from typing import Dict


def detect_dice_tray(img: np.ndarray, thresh=50) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ this function uses the fact that the dice tray is all black """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    filtered = cv.bilateralFilter(gray, 9, 250, 250)

    # define thesholds
    # edges = cv.Canny(filtered, 50, 60)
    _, edges = cv.threshold(filtered, thresh, 255, cv.THRESH_BINARY)
    edges = 255 - edges
    # edges = cv.morphologyEx(edges, cv.MORPH_DILATE, np.ones((2,2)))

    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, 2)
    # detect dice tray
    i, largest_contour = max(enumerate(contours), key=lambda i_c: cv.contourArea(i_c[1]))

    # draw contours over image
    img_contours = np.copy(img)
    cv.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    cv.drawContours(img_contours, [largest_contour], -1, (0, 0, 255), 2)

    _, _, i_d1, _ = hierarchy[0, i]
    i_d2, _, _, _ = hierarchy[0, i_d1]

    return largest_contour, contours[i_d1], contours[i_d2], img_contours


def descriptor_detect(img: np.ndarray, board_ref: np.ndarray, distance=0.25):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    board_gray = cv.cvtColor(board_ref, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    kp, desc = sift.detectAndCompute(gray, None)
    kp2, desc2 = sift.detectAndCompute(board_gray, None)

    matcher = cv.FlannBasedMatcher()
    matches = matcher.match(desc2, desc)

    good_matches = sorted(matches, key=lambda x: x.distance)
    _max = max(matches, key=lambda x: x.distance).distance
    good_matches = [m for m in matches if m.distance < distance * _max]

    if len(good_matches)==0:
        return None

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    draw_params = dict(matchColor=(255, 0, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=mask.ravel().tolist(),  # draw only inliers
                       flags=2 | 4
                       )

    drawn_matches = cv.drawMatches(board_ref, kp2, img, kp, good_matches, None, **draw_params)

    warped_board = cv.warpPerspective(board_gray, M, (img.shape[1], img.shape[0]))

    _, edges2 = cv.threshold(warped_board, 50, 255, cv.THRESH_BINARY)
    edges2 = cv.morphologyEx(edges2, cv.MORPH_CLOSE, kernel=np.ones((7, 7)))

    contours, _ = cv.findContours(edges2, cv.RETR_TREE, 2)
    # i,largest_contour = max(enumerate(contours), key=lambda i_c:cv.contourArea(i_c[1]))

    return M, drawn_matches, contours[0]


def detect_score_board(img,
                       mask,
                       thresh_arg=(55, 15),
                       hor_ker=np.ones((1, 40)),
                       ver_ker=np.ones((30, 1)),
                       hor_ver_ker=np.ones((7, 7))):
    mask_cont = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0][0]
    cropped = crop_contour(img, mask_cont)
    img_gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

    threshold = cv.adaptiveThreshold(
        img_gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        *thresh_arg
    )

    hor = 255 - cv.erode(cv.dilate(threshold, hor_ker), hor_ker)
    ver = 255 - cv.erode(cv.dilate(threshold, ver_ker), ver_ker)
    hor_ver = cv.morphologyEx(hor + ver, cv.MORPH_CLOSE, hor_ver_ker)

    contours, hierarchy = cv.findContours(hor_ver, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    i, _ = max(enumerate(contours), key=lambda i_c: cv.contourArea(i_c[1]))

    cells = []
    _, _, child, _ = hierarchy[0][i]
    j = child

    while True:
        _next, _, _, _ = hierarchy[0][j]
        cells.append(j)
        j = _next
        if _next == -1:
            break

    return cells, [contours[i] for i in cells[::-1]], mask_cont


def detect_buildings(mask):
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    buildings = [cont for cont_idx, cont in enumerate(contours) if hierarchy[0][cont_idx][2] == -1] #childless contours
    return buildings

def detect_clearing(mask):
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    clearings = [cont for cont_idx,cont in enumerate(contours) if hierarchy[0][cont_idx][3] == -1] #parentless contours
    return clearings

def detect_pawns(frame,clearing_mask,pawn_colors:Dict[str,tuple]):
    clearings = detect_clearing(clearing_mask)
    pawns = {}
    hsv_frame = cv.cvtColor(cv.COLOR_BGR2HSV)
    for cont in clearings:
        img = crop_contour(frame,cont)
        for player,color_range in pawn_colors.items():
            color_mask = cv.inRange(hsv_frame,*color_range)
            contours,hierarchy = cv.findContours(color_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            pawns[player] = len(contours)

    return pawns


