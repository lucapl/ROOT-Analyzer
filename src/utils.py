import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd
from IPython.display import display, HTML

def crop_contour(img,contour):
    ''' crops the countour out of the image '''
    x,y,w,h = cv2.boundingRect(contour)
    cropped = img[y:y+h,x:x+w]
    return cropped

def saturate_image(img,saturation=2):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    
    (h, s, v) = cv2.split(img_hsv)
    s = s*saturation
    s = np.clip(s,0,255)
    img_hsv = cv2.merge([h,s,v])
    
    return cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

def dominant_colors(image,n_clusters=3):
    data = cv2.resize(image, (100, 100)).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)

    cluster_sizes = np.bincount(labels.flatten())

    # palette = []
    # for cluster_idx in np.argsort(-cluster_sizes):
    #     palette.append(np.full((image.shape[0], image.shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))
    return centers,cluster_sizes

def create_tracker(tracker_type):
    '''
    Available trackers:
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
        return cv2.TrackerBoosting_create()
    if tracker_type == "MIL":
        return cv2.TrackerMIL_create()
    if tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    if tracker_type == "TLD":
        return cv2.TrackerTLD_create()
    if tracker_type == "MEDIANFLOW":
        return cv2.TrackerMedianFlow_create()
    if tracker_type == "GOTURN":
        return cv2.TrackerGOTURN_create()
    if tracker_type == "MOSSE":
        return cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
