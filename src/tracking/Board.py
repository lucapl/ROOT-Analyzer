
from src.tracking.StaticObject import StaticObject
from src.detection.elements import descriptor_detect

class Board(StaticObject):

    def __init__(self, name, contour, board_ref,distance=0.25):
        super().__init__(name, contour)
        self.board_ref = board_ref
        self.distance = 0.25
        self.M = None

    def redetect(self, frame):
        output = descriptor_detect(frame,self.board_ref,distance=self.distance,draw_matches=False)
        if output is None:
            return None
        M,cont = output
        super().contour = cont 
        self.M = M