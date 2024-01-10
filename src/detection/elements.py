import cv2 as cv
import numpy as np

def detect_dice_tray(img):
    ''' this function uses the fact that the dice tray is all black '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    filtered = cv.bilateralFilter(gray, 9, 250, 250) 

    # define thesholds
    #edges = cv.Canny(filtered, 50, 60)
    _,edges = cv.threshold(filtered,50,255,cv.THRESH_BINARY)
    edges = 255-edges
    #edges = cv.morphologyEx(edges, cv.MORPH_DILATE, np.ones((2,2)))

    contours,hierarchy = cv.findContours(edges, cv.RETR_TREE, 2)
    # detect dice tray
    i,largest_contour = max(enumerate(contours), key=lambda i_c:cv.contourArea(i_c[1]))

    # draw contours over image
    img_contours = np.copy(img)
    cv.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    cv.drawContours(img_contours, [largest_contour], -1, (0, 0, 255), 2)

    _,_,i_d1,_ = hierarchy[0,i]
    i_d2,_,_,_ = hierarchy[0,i_d1]

    return largest_contour,contours[i_d1],contours[i_d2],img_contours

def detect_board(img,board_ref,distance=0.25):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    board_gray = cv.cvtColor(board_ref,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    kp, desc = sift.detectAndCompute(gray, None)
    kp2, desc2 = sift.detectAndCompute(board_gray, None)

    matcher = cv.FlannBasedMatcher()
    matches = matcher.match(desc2, desc)

    good_matches = sorted(matches, key=lambda x: x.distance)
    _max =  max(matches, key=lambda x: x.distance).distance
    good_matches = [m for m in matches if m.distance < distance * _max]

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = mask.ravel().tolist(), # draw only inliers
                    flags = 2|4)
    drawn_matches = cv.drawMatches(board_ref,kp2,img,kp,good_matches,None,**draw_params)

    warped_board = cv.warpPerspective(board_gray, M, (img.shape[1], img.shape[0]))

    _,edges2 = cv.threshold(warped_board,50,255,cv.THRESH_BINARY)
    edges2 = cv.morphologyEx(edges2,cv.MORPH_CLOSE,kernel=np.ones((7,7)))

    contours,_ = cv.findContours(edges2, cv.RETR_TREE, 2)
    #i,largest_contour = max(enumerate(contours), key=lambda i_c:cv.contourArea(i_c[1]))

    return M,drawn_matches,contours[0]