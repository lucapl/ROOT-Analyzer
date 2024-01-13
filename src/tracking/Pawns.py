import cv2 as cv
import numpy as np

from collections import Counter
from src.tracking.StaticObject import StaticObject
from src.detection.elements import detect_pawns
from src.utils import get_clearings_and_buildings,warp_contour

class Pawns(StaticObject):
    def __init__(self,name,board,clearing_mask,diff_sensivity=0.4,area_sensivity=0.3):
        super().__init__(name)
        self.mask = clearing_mask
        self.warp_mask = None
        self.board = board
        self.clearing_contours,_ = get_clearings_and_buildings(clearing_mask)
        self.warped_contours = None

        self.pawn_colors = {"orange":(StaticObject.LOWER_ORANGE,StaticObject.UPPER_ORANGE),
                            "blue":(StaticObject.LOWER_DARK_BLUE,StaticObject.UPPER_DARK_BLUE)}
        self.diff_sensivity = diff_sensivity
        self.area_sensivity = area_sensivity
        self.counted_pawns = None
        self.counted_pawns_over_time = []
        self.total_pawns = None
        self.total_pawns_over_time = []
        self.clearing_control = None
        self.clearing_control_over_time = []

    def redetect(self,frame):
        self.warped_contours = [warp_contour(cont,self.board.M) for cont in self.clearing_contours]
        self.warp_mask =  cv.warpPerspective(self.mask, self.board.M, (frame.shape[1], frame.shape[0]))

    def _determine_control(self,orange_pawns,blue_pawns):
        if blue_pawns == 0 and orange_pawns == 0:
            return None
        # in case of a tie the blue faction controls the clearings
        return "blue" if blue_pawns >= orange_pawns else "orange"

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        self.event.update()

        self.counted_pawns,self.total_pawns = detect_pawns(frame,self.warped_contours,self.warp_mask,self.pawn_colors,self.diff_sensivity,self.area_sensivity,with_ids=False)

        self.clearing_control = []

        for i in range(len(self.clearing_contours)):
            orange_pawns = self.counted_pawns["orange"][i]
            blue_pawns = self.counted_pawns["blue"][i]
            self.clearing_control.append(self._determine_control(orange_pawns,blue_pawns))
        
        self.clearing_control_over_time.append(self.clearing_control)
        if len(self.clearing_control) > 30:
            self.clearing_control_over_time.pop(0)

        return frame

    def draw(self,frame):
        #c = np.repeat(np.copy(board_mask[:,:,0])[:,:,np.newaxis],3,axis=2)

        clearing_control = []
        transposed = tuple(zip(*self.clearing_control_over_time))
        for i in range(len(self.warped_contours)):
            clearing_control.append(Counter(transposed[i]).most_common(1)[0][0])

        blue_clearings = [cont for i,cont in enumerate(self.warped_contours) if clearing_control[i] == "blue"]
        orange_clearings = [cont for i,cont in enumerate(self.warped_contours) if clearing_control[i] == "orange"]
        not_controlled = [cont for i,cont in enumerate(self.warped_contours) if clearing_control[i] == None]

        for i,cont in enumerate(self.warped_contours):
            orange_pawns = self.counted_pawns["orange"][i]
            blue_pawns = self.counted_pawns["blue"][i]
            x,y,w,h = cv.boundingRect(cont)
            frame = cv.putText(frame,str(orange_pawns),(x,y),cv.FONT_HERSHEY_COMPLEX,2,(0,122,255),2)

            frame = cv.putText(frame,str(blue_pawns),(x+w,y),cv.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)


        cv.drawContours(frame,blue_clearings,-1,(255,0,0),3)
        cv.drawContours(frame,orange_clearings,-1,(0,122,255),3)
        cv.drawContours(frame,not_controlled,-1,(0,122,0),3)

        return frame
