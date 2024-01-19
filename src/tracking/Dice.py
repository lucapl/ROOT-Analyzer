import cv2 as cv
import numpy as np

from src.detection.elements import detect_dice_tray
from src.tracking.StaticObject import StaticObject
from src.tracking.TrackedObject import TrackedObject


class Dice(TrackedObject):
    def __init__(self, name, tracker_type, dice_tray, dice_num, velocity_sensitivity=10, threshold=30):
        super().__init__(name, tracker_type, velocity_sensitivity)
        self.dice_num = dice_num
        self.threshold = threshold
        self.dice_tray = dice_tray

    def re_detect(self, frame):
        # the dice should be tracked by the same object because calculating the dice tray each time is quite expensive
        # _, dice_1, dice_2, _ = detect_dice_tray(frame, self.threshold)
        dice = self.dice_tray.dice_1 if self.dice_num == 1 else self.dice_tray.dice_2

        if dice is not None:
            self.is_init = False
            self.init_tracker(frame, dice)

    def detect_events(self, frame: np.ndarray):
        self.event.update()

        if self.is_moving():
            self.event.msg = f"{self.name} is rolling"
            self.event.reset()

    def update_timer(self, bbox):
        x, y, w, h = cv.boundingRect(self.dice_tray.tray)
        px, py, pw, ph = bbox

        if np.linalg.norm(np.subtract([x, y], [px, py])) > pw * 2:
            self.timer += 1
        else:
            self.timer = 0


class DiceTray(StaticObject):
    def __init__(self, name, threshold=30):
        super().__init__(name, refresh_rate=120)  # match tracked object refresh rate
        self.dice_1 = None
        self.dice_2 = None
        self.tray = None
        self.threshold = threshold

    def re_detect(self, frame):
        self.tray, self.dice_1, self.dice_2, _ = detect_dice_tray(frame, self.threshold, False)

    def draw(self, frame, msg=None, color=(0, 122, 0)):
        return cv.drawContours(frame, [self.tray], -1, color, 2)
