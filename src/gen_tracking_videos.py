import cv2 as cv

from tracking.algorithms import track_video
from tracking.Card import Card, CardPile
from tracking.ScoreBoard import ScoreBoard
from tracking.Buildings import Buildings
from tracking.Board import Board
from tracking.Dice import Dice
from file_reading import read_pdf


DATA_DIR = "../data"
DIFFICULTIES = ["easy", "medium", "hard"]
CLIP_DIRS = dict([(diff, f"{DATA_DIR}/{diff}") for diff in DIFFICULTIES])

clip = lambda idx: f"clip_{idx}"
clip_mp4 = lambda idx: f"{clip(idx)}.mp4"

board_mask = cv.imread(f"{DATA_DIR}/game_data/board_mask.png")
board_ref = read_pdf(f"{DATA_DIR}/game_data/board.pdf")
card_ref = read_pdf(f"{DATA_DIR}/game_data/card_reverse.pdf")


def upload_clip(diff: str, idx: int) -> None:
    clip_name = f"{diff}_{clip(idx)}"
    
    print(f"Processing {clip_name}")

    video = cv.VideoCapture(f"{CLIP_DIRS[diff]}/{clip_mp4(idx)}")

    if video.isOpened():
        print("Loaded")

    card = Card("card", "CSRT", card_ref)
    dice_1 = Dice("dice 1", "CSRT", 1, threshold=30)
    dice_2 = Dice("dice 2", "CSRT", 2, threshold=30)
    card_pile = CardPile("pile of cards", card_ref)
    score_board = ScoreBoard("score", board_ref, board_mask[:, :, 2])
    buildings = Buildings("buildings", board_mask[:, :, 0])
    board = Board("board", board_ref)

    track_video(video, f"../ignore/{clip_name}", board, [card, dice_1, dice_2], [card_pile, score_board, buildings])


if __name__ == "__main__":
    for difficulty in DIFFICULTIES:
        for i in range(3):
            upload_clip(difficulty, i)
