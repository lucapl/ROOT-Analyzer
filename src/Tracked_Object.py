import cv2 as cv
import numpy as np

from src.viz.images import draw_bbox

class Tracked_Object:

    def __init__(self,name:str,tracker:cv.Tracker,starting_contour,first_frame,velocity_sensivity=10):
        self.name = name

        self.tracker = tracker
        self.ini_cont = starting_contour
        self.tracker.init(first_frame,cv.boundingRect(self.ini_cont))

        self.vel_sens = velocity_sensivity
        self.velocity = np.array([0,0])
        self.last_bbox = None
    
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