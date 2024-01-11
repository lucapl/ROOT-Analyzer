import cv2 as cv
import numpy as np
from src.detection.game import calculate_current_buildings
from src.detection.elements import detect_buildings
from src.tracking.StaticObject import StaticObject
from src.utils import warp_contours


class Buildings(StaticObject):
    def __init__(self, name, mask, board, orange, blue):
        super().__init__(name, board.contour)
        self.board = board
        self.mask = mask
        self.orange = orange
        self.blue = blue
        self.building_contours = None
        self.orange_buildings,self.blue_buildings= None,None
        self.current_score = None

    def redetect(self, frame):
        building_contours = detect_buildings(self.mask)
        building_contours = warp_contours([cont for cont in building_contours], self.board.M)
        self.building_contours = building_contours
        self.orange_buildings,self.blue_buildings,orange_score,blue_score = calculate_current_buildings(frame,building_contours , self.orange, self.blue)
        self.current_score = (orange_score,blue_score)

    def detect_events(self, frame_num: int, frame: np.ndarray) -> str:
        ob,bb,orange_score,blue_score = calculate_current_buildings(frame, self.building_contours, self.orange, self.blue)
        cur_orange_score,cur_blue_score = self.current_score

        self.event.update()

        if (orange_score, blue_score) != self.current_score:
            self.event.reset()
            orange_diff = orange_score-cur_orange_score
            blue_diff = blue_score-cur_blue_score
            
            state = lambda diff: "built" if diff >0 else "lost"

            event_str = f"Orange {state(orange_diff)}: {orange_diff} Blue {state(blue_diff)}: {blue_diff}"

            self.event.msg = event_str
            self.events.append((frame_num, event_str))

            self.current_score = (orange_score,blue_score)
            self.orange_buildings,self.blue_buildings=ob,bb

        return self.event

    def draw_bbox(self, frame, msg=None, color=(0, 122, 0)):
        orange_buildings = [self.building_contours[i] for i in range(len(self.building_contours)) if self.orange_buildings[i]]
        blue_buildings = [self.building_contours[i] for i in range(len(self.building_contours)) if self.blue_buildings[i]]
        frame = cv.drawContours(frame, self.building_contours, -1, color, 3)
        frame = cv.drawContours(frame, orange_buildings, -1, StaticObject.ORANGE_COLOR, 3)
        frame = cv.drawContours(frame, blue_buildings,-1,StaticObject.BLUE_COLOR, 3)
        return frame
