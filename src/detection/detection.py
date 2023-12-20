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