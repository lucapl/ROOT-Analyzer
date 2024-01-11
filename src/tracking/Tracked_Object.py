import cv2 as cv
import numpy as np

from src.viz.images import draw_bbox
from src.utils import create_tracker

class Tracked_Object:

    def __init__(self,name:str,tracker_type:str,starting_contour,first_frame,velocity_sensivity=10,redetect_timer=48):
        self.name = name

        self.tracker_type = tracker_type
        self.init_tracker(first_frame,starting_contour)
        self.ini_cont = starting_contour

        self.vel_sens = velocity_sensivity
        self.velocity = np.array([0,0])
        self.last_bbox = None

        self.redetect_timer = redetect_timer

        self.events = []
    
    def init_tracker(self,frame,contour):
        self.tracker = create_tracker(self.tracker_type)
        self.tracker.init(frame,cv.boundingRect(contour))

    def _update_velocity(self,bbox):
        x,y,_,_ = self.last_bbox
        x_p,y_p,_,_ = bbox

        self.velocity = np.array([x_p-x,y_p-y])

    def is_moving(self):
        return np.linalg.norm(self.velocity,2) > self.vel_sens

    def update(self,raw_frame):
        ok, bbox = self.tracker.update(raw_frame)
        if ok:
            if self.last_bbox != None:
                self._update_velocity(bbox)

            self.last_bbox = bbox
            return bbox
        else:
            return None
        
    def detection_fail_msg(self,frame):
        return self.draw_bbox(frame,f"{self.name} Lost",(0,0,255))
    
    def draw_bbox(self,frame,msg=None,color=(0,255,0)):
        if msg == None:
            msg = self.name
        x,y,w,h = self.last_bbox
        draw_bbox(frame,self.last_bbox,color)
        
        if self.is_moving():
            frame = cv.putText(frame, "Moving",(x,y+h-2),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv.LINE_AA)
        return cv.putText(frame, msg,(x,y-2),cv.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv.LINE_AA)
    
    def detect_events(self):
        return None
    
    #def redetection(self,)