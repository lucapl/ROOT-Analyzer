import cv2 as cv
import numpy as np

from src.detection.game import calculate_current_buildings,calculate_building_score
from src.detection.elements import detect_buildings
from src.tracking.StaticObject import StaticObject
from src.utils import warp_contour


class Buildings(StaticObject):
    def __init__(self, name,board, mask):
        super().__init__(name)
        self.mask = mask
        self.building_contours = None
        self.board = board
        self.orange_buildings, self.blue_buildings = [], []
        self.current_score = None
        self.current_scores = []

    def redetect(self, frame,):
        building_contours = detect_buildings(self.mask)
        self.building_contours = list(map(lambda c: warp_contour(c, self.board.M), building_contours))

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        self.event.update()

        ob, bb = calculate_current_buildings(frame, self.building_contours,
                                             (StaticObject.LOWER_ORANGE, StaticObject.UPPER_ORANGE),
                                             (StaticObject.LOWER_DARK_BLUE, StaticObject.UPPER_DARK_BLUE))
        new_score = calculate_building_score(ob, bb)

        self.current_scores.append(new_score)

        if len(self.current_scores) > 30:
            self.current_scores.pop(0)

        average_score = self.get_average_score()

        if self.current_score != average_score:
            self.event.msg = (f"Building Constructed Orange: {average_score[0]} "
                              f"Blue: {average_score[1]}")
            self.orange_buildings, self.blue_buildings = ob, bb
            self.current_score = average_score
            self.event.reset()

        frame = cv.putText(frame, self.event.get(), (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
        return cv.putText(frame, self.event.get(), (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv.LINE_AA)

    def draw(self, frame, msg=None, color=(0, 122, 0)):
        orange_buildings = [self.building_contours[i] for i in range(len(self.building_contours)) if
                            self.orange_buildings[i]]
        blue_buildings = [self.building_contours[i] for i in range(len(self.building_contours)) if
                          self.blue_buildings[i]]
        frame = cv.drawContours(frame, self.building_contours, -1, color, 2)
        frame = cv.drawContours(frame, orange_buildings, -1, StaticObject.ORANGE_COLOR, 3)
        frame = cv.drawContours(frame, blue_buildings, -1, StaticObject.BLUE_COLOR, 3)
        return frame

    def get_average_score(self):
        return (np.mean([score[0] for score in self.current_scores], axis=0, dtype=int),
                np.mean([score[1] for score in self.current_scores], axis=0, dtype=int))
