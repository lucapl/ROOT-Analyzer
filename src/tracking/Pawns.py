import cv2 as cv
import numpy as np

from src.tracking.StaticObject import StaticObject
from src.tracking.Board import Board
from src.detection.elements import detect_pawns, detect_clearings_and_buildings
from src.detection.game import calculate_current_clearing_control
from src.utils.contours import warp_contour
from src.viz.images import draw_bbox


class Pawns(StaticObject):
    def __init__(self, name, board: Board, mask: np.ndarray, diff_sensitivity=0.4, area_sensitivity=0.3):
        super().__init__(name)
        self.board = board
        self.static_mask = mask
        self.static_contours, _ = detect_clearings_and_buildings(mask)

        self.mask = None
        self.contours = None

        self.diff_sensitivity = diff_sensitivity
        self.area_sensitivity = area_sensitivity

        self.orange_pawns, self.blue_pawns = {}, {}
        self.orange_clearings, self.blue_clearings = [], []
        self.current_count = None
        self.counts = []

    def re_detect(self, frame):
        self.contours = [warp_contour(cont, self.board.m) for cont in self.static_contours]
        self.mask = cv.warpPerspective(self.static_mask, self.board.m, (frame.shape[1], frame.shape[0]))

    def detect_events(self, frame: np.ndarray):
        self.event.update()

        op, bp = detect_pawns(frame, self.mask, self.contours, (StaticObject.LOWER_ORANGE, StaticObject.UPPER_ORANGE),
                              (StaticObject.LOWER_DARK_BLUE, StaticObject.UPPER_DARK_BLUE), self.diff_sensitivity,
                              self.area_sensitivity)

        count = sum([self._count_pawns(clearing) for clearing in op.values()]), \
            sum([self._count_pawns(clearing) for clearing in bp.values()])

        self.counts.append(count)

        if len(self.counts) > 30:
            self.counts.pop(0)

        average_count = self._get_average_count()

        if self.current_count != average_count:
            self.orange_pawns, self.blue_pawns = op, bp
            self.orange_clearings, self.blue_clearings = calculate_current_clearing_control(op, bp)
            self.current_count = average_count

            self.event.msg = f"Pawn Placed - Orange: {average_count[0]} Blue: {average_count[1]}"
            self.event.reset()

    def draw(self, frame, color=(0, 122, 0)):
        orange_clearings = [cont for i, cont in enumerate(self.contours) if self.orange_clearings[i]]
        blue_clearings = [cont for i, cont in enumerate(self.contours) if self.blue_clearings[i]]
        not_controlled = [cont for i, cont in enumerate(self.contours) if
                          not self.orange_clearings[i] and not self.blue_clearings[i]]

        rects = [cv.boundingRect(cont) for cont in self.contours]
        orange_pawns = [self._count_pawns(clearing) for clearing in self.orange_pawns.values()]
        blue_pawns = [self._count_pawns(clearing) for clearing in self.blue_pawns.values()]

        for i, cont in enumerate(self.contours):
            x, y, w, h = rects[i]

            frame = cv.putText(frame, str(orange_pawns[i]), (x + w//2 - 30, y - 10), cv.FONT_HERSHEY_COMPLEX, 1,
                               StaticObject.ORANGE_COLOR, 2)
            frame = cv.putText(frame, ":", (x + w//2 - 7, y - 10), cv.FONT_HERSHEY_COMPLEX, 1,
                               (0, 0, 0), 5)
            frame = cv.putText(frame, ":", (x + w//2 - 7, y - 10), cv.FONT_HERSHEY_COMPLEX, 1,
                               (255, 255, 255), 2)
            frame = cv.putText(frame, str(blue_pawns[i]), (x + w//2 + 10, y - 10), cv.FONT_HERSHEY_COMPLEX, 1,
                               StaticObject.BLUE_COLOR, 2)

        frame = cv.drawContours(frame, blue_clearings, -1, (255, 0, 0), 3)
        frame = cv.drawContours(frame, orange_clearings, -1, (0, 122, 255), 3)
        frame = cv.drawContours(frame, not_controlled, -1, (0, 122, 0), 3)

        op = [cv.boundingRect(pawn + [rects[c_idx][0], rects[c_idx][1]])
              for c_idx, clearing in self.orange_pawns.items()
              for pawn in clearing]
        bp = [cv.boundingRect(pawn + [rects[c_idx][0], rects[c_idx][1]])
              for c_idx, clearing in self.blue_pawns.items()
              for pawn in clearing]

        for pawns, color in ((op, StaticObject.ORANGE_COLOR), (bp, StaticObject.BLUE_COLOR)):
            for rect in pawns:
                frame = draw_bbox(frame, rect, color)

        return frame

    def _get_average_count(self) -> tuple[int, int]:
        return (np.mean([count[0] for count in self.counts], axis=0, dtype=int),
                np.mean([count[1] for count in self.counts], axis=0, dtype=int))

    @staticmethod
    def _count_pawns(pawns: list[np.ndarray]) -> int:
        return len(pawns)
