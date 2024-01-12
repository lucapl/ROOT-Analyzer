import numpy as np
import cv2 as cv
from src.tracking.StaticObject import StaticObject
from src.detection.game import calculate_current_score
from src.detection.elements import detect_score_board
from src.utils import warp_contour


class ScoreBoard(StaticObject):
    def __init__(self, name, board_ref, mask):
        super().__init__(name)
        self.board_ref = board_ref
        self.mask = mask
        self.cell_contours = None
        self.current_score = None
        self.current_scores = []

    def redetect(self, frame, M):
        cell_contours, score_ref = detect_score_board(self.board_ref, self.mask)
        score_x, score_y, _, _ = cv.boundingRect(score_ref)
        self.cell_contours = list(map(lambda c: warp_contour(c, M), [c + [score_x, score_y] for c in cell_contours]))

    def draw(self, frame, msg=None, color=(0, 122, 0)):
        if self.cell_contours is None:
            return frame

        orange_score, blue_score = self.get_average_score()
        frame = cv.drawContours(frame, self.cell_contours, -1, color, 2)
        frame = cv.drawContours(frame, [self.cell_contours[blue_score]], -1, StaticObject.BLUE_COLOR, 3)
        frame = cv.drawContours(frame, [self.cell_contours[orange_score]], -1, StaticObject.ORANGE_COLOR, 3)
        return frame

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        self.event.update()

        new_score = calculate_current_score(frame, self.cell_contours,
                                            (StaticObject.LOWER_ORANGE, StaticObject.UPPER_ORANGE),
                                            (StaticObject.LOWER_DARK_BLUE, StaticObject.UPPER_DARK_BLUE))

        self.current_scores.append(new_score)

        if len(self.current_scores) > 30:
            self.current_scores.pop(0)

        average_score = self.get_average_score()

        if self.current_score != average_score:
            self.event.reset()
            self.event.msg = (f"Score change Orange: {average_score[0]} "
                              f"Blue: {average_score[1]}")
            self.current_score = average_score

        msg = self.event.get()

        frame = cv.putText(frame, self.event.get(), (0, 250), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
        return cv.putText(frame, self.event.get(), (0, 250), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv.LINE_AA)

    def get_average_score(self):
        return (np.mean([score[0] for score in self.current_scores], axis=0, dtype=int),
                np.mean([score[1] for score in self.current_scores], axis=0, dtype=int))
