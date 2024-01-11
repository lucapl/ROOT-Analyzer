import numpy as np
import cv2 as cv
from src.tracking.StaticObject import StaticObject
from src.detection.game import calculate_current_score


class ScoreBoard(StaticObject):
    def __init__(self, name, contour, first_frame, cell_contours, orange, blue):
        super().__init__(name, contour)
        self.cell_contours = cell_contours
        self.orange = orange
        self.blue = blue
        self.current_score = calculate_current_score(first_frame, self.cell_contours, self.orange, self.blue)

    def redetect(self, frame):
        return None

    def draw_bbox(self, frame, msg=None, color=(0, 122, 0)):
        orange_score,blue_score = self.current_score
        frame = cv.drawContours(frame, self.cell_contours, -1, color, 3)
        frame = cv.drawContours(frame, [self.cell_contours[blue_score]], -1, StaticObject.BLUE_COLOR, 3)
        frame = cv.drawContours(frame, [self.cell_contours[orange_score]],-1,StaticObject.ORANGE_COLOR, 3)
        return frame

    def detect_events(self, frame_num: int, frame: np.ndarray) -> str:
        orange_score, blue_score = calculate_current_score(frame, self.cell_contours, self.orange, self.blue)
        cur_orange_score,cur_blue_score = self.current_score
        if (orange_score, blue_score) != self.current_score:
            event_str = f"Score change Orange: {orange_score-cur_orange_score} Blue: {blue_score-cur_blue_score}"
            self.events.append((frame_num, event_str))
            self.current_score = (orange_score,blue_score)
            return event_str

