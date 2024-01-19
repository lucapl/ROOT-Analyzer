import numpy as np
import cv2 as cv

from src.tracking.StaticObject import StaticObject
from src.tracking.Board import Board
from src.detection.game import calculate_current_score
from src.detection.elements import detect_score_board
from src.utils.contours import warp_contour


class ScoreBoard(StaticObject):
    def __init__(self, name, board: Board, mask: np.ndarray):
        super().__init__(name)
        self.mask = mask
        self.board = board
        self.cell_contours = None
        self.static_contours, score_ref = detect_score_board(self.board.ref, self.mask)
        score_x, score_y, _, _ = cv.boundingRect(score_ref)
        self.score_offset = [score_x, score_y]

        self.current_score = None
        self.scores = []

    def re_detect(self, frame):
        self.cell_contours = list(map(lambda c: warp_contour(c, self.board.m),
                                      [c + self.score_offset for c in self.static_contours]))

    def draw(self, frame, color=(0, 122, 0)):
        if self.cell_contours is None:
            return frame

        frame = cv.drawContours(frame, self.cell_contours, -1, color, 2)
        frame = cv.drawContours(frame, [self.cell_contours[self.current_score[1]]], -1, StaticObject.BLUE_COLOR, 3)
        frame = cv.drawContours(frame, [self.cell_contours[self.current_score[0]]], -1, StaticObject.ORANGE_COLOR, 3)
        return frame

    def detect_events(self, frame: np.ndarray):
        self.event.update()

        new_score = calculate_current_score(frame, self.cell_contours,
                                            (StaticObject.LOWER_ORANGE, StaticObject.UPPER_ORANGE),
                                            (StaticObject.LOWER_DARK_BLUE, StaticObject.UPPER_DARK_BLUE))

        self.scores.append(new_score)

        if len(self.scores) > 30:
            self.scores.pop(0)

        average_score = self._get_average_score()

        if self.current_score != average_score:
            self.current_score = average_score
            self.event.msg = (f"Score - Orange: {average_score[0]} "
                              f"Blue: {average_score[1]}")
            self.event.reset()

    def _get_average_score(self):
        return (np.mean([score[0] for score in self.scores], axis=0, dtype=int),
                np.mean([score[1] for score in self.scores], axis=0, dtype=int))
