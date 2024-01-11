import numpy as np
import cv2 as cv
from src.tracking.StaticObject import StaticObject
from src.detection.game import calculate_current_score
from src.detection.elements import detect_score_board
from src.utils import warp_contour, resize_contour


class ScoreBoard(StaticObject):
    def __init__(self, name, cell_contours, board_ref, mask, first_frame, orange, blue):
        super().__init__(name)
        self.board_ref = board_ref
        self.mask = mask
        self.cell_contours = cell_contours
        self.orange = orange
        self.blue = blue
        self.current_score = calculate_current_score(first_frame, self.cell_contours, self.orange, self.blue)
        self.current_scores = [self.current_score]
        self.msg = ""

    def redetect(self, frame, M):
        cell_contours, score_ref = detect_score_board(self.board_ref, self.mask)
        score_x, score_y, _, _ = cv.boundingRect(score_ref)
        self.cell_contours = list(map(lambda c: warp_contour(c, M), [c + [score_x, score_y] for c in cell_contours]))
        # self.cell_contours = list(map(resize_contour, cell_contours))

    def draw(self, frame, msg=None, color=(0, 122, 0)):
        orange_score, blue_score = self.get_average_score()
        frame = cv.drawContours(frame, self.cell_contours, -1, color, 1)
        frame = cv.drawContours(frame, [self.cell_contours[blue_score]], -1, StaticObject.BLUE_COLOR, 2)
        frame = cv.drawContours(frame, [self.cell_contours[orange_score]], -1, StaticObject.ORANGE_COLOR, 2)
        return frame

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        new_score = calculate_current_score(frame, self.cell_contours, self.orange, self.blue)

        self.current_scores.append(new_score)

        if len(self.current_scores) > 30:
            self.current_scores.pop(0)

        if frame_num % 60 == 0:
            self.msg = ""

        average_score = self.get_average_score()

        if self.current_score != average_score:
            self.msg = (f"Score change Orange: {average_score[0] - self.current_score[0]} "
                        f"Blue: {average_score[1] - self.current_score[1]}")
            self.current_score = average_score

        frame = cv.putText(frame, self.msg, (0, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)
        return cv.putText(frame, self.msg, (0, 50, ), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    def get_average_score(self):
        return (np.mean([score[0] for score in self.current_scores], axis=0, dtype=int),
                np.mean([score[1] for score in self.current_scores], axis=0, dtype=int))
