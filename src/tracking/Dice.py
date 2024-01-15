import cv2 as cv
import numpy as np
from src.tracking.TrackedObject import TrackedObject
from src.tracking.StaticObject import StaticObject
from src.detection.elements import detect_dice_tray


class Dice(TrackedObject):
    def __init__(self, name, tracker_type,diceTray, dice_num, velocity_sensivity=10, threshold=30):
        super().__init__(name, tracker_type, velocity_sensivity)
        self.dice_num = dice_num
        self.threshold = threshold
        self.diceTray = diceTray

    def redetect(self, frame):
        # the dice should be tracked by the same object because calculating the dice tray each time is quite expensive
        #_, dice_1, dice_2, _ = detect_dice_tray(frame, self.threshold)
        dice_1,dice_2 = self.diceTray.dice1,self.diceTray.dice2
        dice = dice_1 if self.dice_num == 1 else dice_2
        if type(dice) != type(None):
            self.isInit = False
            self.contour = dice
            self.init_tracker(frame, self.contour)

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        self.event.update()

        if self.is_moving():
            self.event.reset()
            self.event.msg = f"{self.name} is rolling"

        frame = cv.putText(frame, self.event.get(), (100 * self.dice_num, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
        return cv.putText(frame, self.event.get(), (100 * self.dice_num, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv.LINE_AA)
    


class DiceTray(StaticObject):
    def __init__(self,name,threshold=30):
        super().__init__(name)
        self.dice1 = None
        self.dice2 = None
        self.tray = None
        self.threshold=30

    def redetect(self, frame,):
        self.tray, self.dice1, self.dice2 = detect_dice_tray(frame, self.threshold,False)

    def detect_events(self, frame_num: int, frame: np.ndarray) -> np.ndarray:
        return frame

    def draw(self, frame, msg=None, color=(0, 122, 0)):
        return cv.drawContours(frame, [self.tray], -1, color, 2)