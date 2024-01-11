import cv2 as cv

from src.tracking.Tracked_Object import Tracked_Object

class Card(Tracked_Object):

    def __init__(self,name,tracker:cv.Tracker,starting_contour,first_frame,velocity_sensivity=10):
        super().__init__(name,tracker,starting_contour,first_frame,velocity_sensivity)