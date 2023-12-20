import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd
from IPython.display import display, HTML


def imshow(img):
    ''' imshow function courtesy of our lab notebooks '''

    img = img.clip(0, 255).astype("uint8")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(img))

def scaled_imshow(img,fx=0.3,fy=0.3):
    ''' scaling the image so the notebook doesn't take too much space '''
    scaled_img = cv2.resize(img,None,fx=fx,fy=fy)
    imshow(scaled_img)

def crop_contour(img,contour):
    ''' crops the countour out of the image '''
    x,y,w,h = cv2.boundingRect(contour)
    cropped = img[y:y+h,x:x+w]
    return cropped