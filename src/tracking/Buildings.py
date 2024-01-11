import cv2 as cv
import numpy as np
from src.detection.game import calculate_current_buildings
from src.tracking.StaticObject import StaticObject


class Buildings(StaticObject):
    def __init__(self, name, contour, frame, buildings_contours, orange, blue):
        super().__init__(name, contour)
        self.buildings_contours = buildings_contours
        self.orange = orange
        self.blue = blue
        self.orange_buildings,self.blue_buildings,orange_score,blue_score = calculate_current_buildings(frame, self.buildings_contours, self.orange, self.blue)
        self.current_score = (orange_score,blue_score)

    def redetect(self, frame):
        return None

    def detect_events(self, frame_num: int, frame: np.ndarray) -> str:
        ob,bb,orange_score,blue_score = calculate_current_buildings(frame, self.buildings_contours, self.orange, self.blue)
        cur_orange_score,cur_blue_score = self.current_score

        self.event.update()

        if (orange_score, blue_score) != self.current_score:
            self.event.reset()
            event_str = f"Building Constructed Orange: {orange_score-cur_orange_score} Blue: {blue_score-cur_blue_score}"

            self.event.msg = event_str
            self.events.append((frame_num, event_str))

            self.current_score = (orange_score,blue_score)
            self.orange_buildings,self.blue_buildings=ob,bb

            return self.event

    def draw_bbox(self, frame, msg=None, color=(0, 122, 0)):
        orange_buildings = [self.buildings_contours[i] for i in range(len(self.buildings_contours)) if self.orange_buildings[i]]
        blue_buildings = [self.buildings_contours[i] for i in range(len(self.buildings_contours)) if self.blue_buildings[i]]
        frame = cv.drawContours(frame, self.buildings_contours, -1, color, 3)
        frame = cv.drawContours(frame, orange_buildings, -1, StaticObject.ORANGE_COLOR, 3)
        frame = cv.drawContours(frame, blue_buildings,-1,StaticObject.BLUE_COLOR, 3)
        return frame
