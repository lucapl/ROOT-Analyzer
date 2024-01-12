import cv2 as cv
import numpy as np
from src.tracking.TrackedObject import TrackedObject
from src.detection.elements import detect_dice_tray


class Dice(TrackedObject):
    def __init__(self, name, tracker_type, dice_num, velocity_sensivity=10, threshold=30):
        super().__init__(name, tracker_type, velocity_sensivity)
        self.dice_num = dice_num
        self.threshold = threshold

    def redetect(self, frame):
        # the dice should be tracked by the same object because calculating the dice tray each time is quite expensive
        _, dice_1, dice_2, _ = detect_dice_tray(frame, self.threshold)
        dice = dice_1 if self.dice_num == 1 else dice_2
        if dice != None:
            self.contour = dice
            self.init_tracker(frame, self.contour)

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        self.event.update()

        if self.is_moving():
            self.event.reset()
            self.event.msg = f"{self.name} is rolling"

        frame = cv.putText(frame, self.event.get(), (100 * self.dice_num, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
        return cv.putText(frame, self.event.get(), (100 * self.dice_num, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv.LINE_AA)