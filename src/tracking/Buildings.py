import cv2 as cv
import numpy as np

from src.detection.elements import detect_buildings
from src.detection.game import calculate_current_buildings, calculate_building_score
from src.tracking.StaticObject import StaticObject
from src.utils import warp_contour, resize_contour


class Buildings(StaticObject):
    def __init__(self, name, buildings_contours, mask, first_frame, orange, blue):
        super().__init__(name)
        self.building_contours = buildings_contours
        self.mask = mask
        self.orange = orange
        self.blue = blue
        self.orange_buildings, self.blue_buildings = calculate_current_buildings(first_frame, self.building_contours,
                                                                                 self.orange,
                                                                                 self.blue)
        self.current_score = calculate_building_score(self.orange_buildings, self.blue_buildings)
        self.current_scores = [self.current_score]
        self.msg = ""

    def redetect(self, frame, M_board):
        building_contours = detect_buildings(self.mask)
        self.building_contours = list(map(lambda cont: warp_contour(cont, M_board), building_contours))

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        ob, bb = calculate_current_buildings(frame, self.building_contours, self.orange,
                                             self.blue)

        new_score = calculate_building_score(ob, bb)

        self.current_scores.append(new_score)

        if len(self.current_scores) > 30:
            self.current_scores.pop(0)

        if frame_num % 60 == 59:
            self.msg = ""

        average_score = self.get_average_score()

        if self.current_score != average_score and frame_num % 60 == 0:
            self.msg = (f"Building Constructed Orange: {average_score[0] - self.current_score[1]} "
                        f"Blue: {average_score[1] - self.current_score[1]}")
            self.orange_buildings, self.blue_buildings = ob, bb
            self.current_score = average_score

        frame = cv.putText(frame, self.msg, (0, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)
        return cv.putText(frame, self.msg, (0, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    def draw(self, frame, msg=None, color=(0, 122, 0)):
        orange_buildings = [self.building_contours[i] for i in range(len(self.building_contours)) if
                            self.orange_buildings[i]]
        blue_buildings = [self.building_contours[i] for i in range(len(self.building_contours)) if
                          self.blue_buildings[i]]
        frame = cv.drawContours(frame, self.building_contours, -1, color, 1)
        frame = cv.drawContours(frame, orange_buildings, -1, StaticObject.ORANGE_COLOR, 1)
        frame = cv.drawContours(frame, blue_buildings, -1, StaticObject.BLUE_COLOR, 1)
        return frame

    def get_average_score(self):
        return (np.mean([score[0] for score in self.current_scores], axis=0, dtype=int),
                np.mean([score[1] for score in self.current_scores], axis=0, dtype=int))