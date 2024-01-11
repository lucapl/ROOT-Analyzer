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
        self.start_score = calculate_current_score(first_frame, self.cell_contours, self.orange, self.blue)

    def draw_bbox(self, frame, msg=None, color=(0, 255, 0)):
        return cv.drawContours(frame, self.cell_contours, -1, (255, 0, 0), 3)

    def detect_events(self, frame_num: int, frame: np.ndarray) -> str | None:
        orange_score, blue_score = calculate_current_score(frame, self.cell_contours, self.orange, self.blue)
        if (orange_score, blue_score) != self.start_score:
            self.events.append((frame_num, f"Orange: {orange_score} Blue: {blue_score}"))
            return f"Orange: {orange_score} Blue: {blue_score}"
