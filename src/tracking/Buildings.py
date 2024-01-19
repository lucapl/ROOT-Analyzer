import cv2 as cv
import numpy as np

from src.detection.game import calculate_current_buildings_control
from src.detection.elements import detect_clearings_and_buildings
from src.tracking.Board import Board
from src.tracking.StaticObject import StaticObject
from src.utils.contours import warp_contour


class Buildings(StaticObject):
    def __init__(self, name, board: Board, mask: np.ndarray):
        super().__init__(name)
        self.board = board

        _, buildings_by_clearing = detect_clearings_and_buildings(mask)
        self.static_contours = [building for clearing in buildings_by_clearing.values() for building in clearing]
        self.building_contours = None

        self.orange_buildings, self.blue_buildings = [], []
        self.current_score = None
        self.scores = []

    def re_detect(self, frame):
        self.building_contours = list(map(lambda c: warp_contour(c, self.board.m), self.static_contours))

    def detect_events(self, frame: np.ndarray):
        self.event.update()

        ob, bb = calculate_current_buildings_control(frame, self.building_contours,
                                                     (StaticObject.LOWER_ORANGE, StaticObject.UPPER_ORANGE),
                                                     (StaticObject.LOWER_DARK_BLUE, StaticObject.UPPER_DARK_BLUE))
        new_score = self._calculate_score(ob, bb)

        self.scores.append(new_score)

        if len(self.scores) > 30:
            self.scores.pop(0)

        average_score = self._get_average_score()

        if self.current_score != average_score:
            self.orange_buildings, self.blue_buildings = ob, bb
            self.current_score = average_score
            self.event.msg = (f"Buildings Constructed Orange: {average_score[0]} "
                              f"Blue: {average_score[1]}")
            self.event.reset()

    def draw(self, frame, msg=None, color=(0, 122, 0)):
        orange_buildings = [self.building_contours[i] for i in range(len(self.building_contours)) if
                            self.orange_buildings[i]]
        blue_buildings = [self.building_contours[i] for i in range(len(self.building_contours)) if
                          self.blue_buildings[i]]
        frame = cv.drawContours(frame, self.building_contours, -1, color, 2)
        frame = cv.drawContours(frame, orange_buildings, -1, StaticObject.ORANGE_COLOR, 3)
        frame = cv.drawContours(frame, blue_buildings, -1, StaticObject.BLUE_COLOR, 3)
        return frame

    def _get_average_score(self) -> tuple[int, int]:
        return (np.mean([score[0] for score in self.scores], axis=0, dtype=int),
                np.mean([score[1] for score in self.scores], axis=0, dtype=int))

    @staticmethod
    def _calculate_score(orange_buildings: list[bool], blue_buildings: list[bool]) -> tuple[int, int]:
        return sum(orange_buildings), sum(blue_buildings)
