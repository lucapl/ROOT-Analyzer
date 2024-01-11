import cv2 as cv
import numpy as np

class Detected_Object:

    def __init__(self,name:str,contours,first_frame,redetect_timer=48):
        self.name = name

        self.contours = contours

        self.redetect_timer = redetect_timer

        self.state = []

        self.data = []

    def _redetect(self,frame):
        return None
    
    def update(self,raw_frame):
        read_state(raw_frame)

    def _get_state(self,frame):
        return None

    def read_state(self,frame):
        self.state.append(self._get_state(frame))