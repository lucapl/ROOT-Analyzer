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