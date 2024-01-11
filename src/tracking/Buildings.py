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
        self.start_score = calculate_current_buildings(frame, self.buildings_contours, self.orange, self.blue)

    def detect_events(self, frame_num: int, frame: np.ndarray) -> str | None:
        orange_score, blue_score = calculate_current_buildings(frame, self.buildings_contours, self.orange, self.blue)
        if (orange_score, blue_score) != self.start_score:
            self.events.append((frame_num, f"Orange: {orange_score} Blue: {blue_score}"))
            return f"Orange: {orange_score} Blue: {blue_score}"

    def draw_bbox(self, frame, msg=None, color=(0, 255, 0)):
        return cv.drawContours(frame, self.buildings_contours, -1, (255, 0, 0), 3)
