import cv2 as cv
import numpy as np
from tqdm import tqdm
import subprocess

from src.utils.data import get_pdf_page
from src.tracking import TrackedObject, StaticObject, Board, Buildings, Card, CardPile, Dice, DiceTray, ScoreBoard, \
    Pawns
from src.viz.images import display_events

DATA_DIR = "../data"
UPLOAD_DIR = "../results"
DIFFICULTIES = ["easy", "medium", "hard"]
CLIP_DIRS = dict([(diff, f"{DATA_DIR}/{diff}") for diff in DIFFICULTIES])

BOARD_MASK = cv.imread(f"{DATA_DIR}/game_data/board_mask.png")
BOARD_REF = get_pdf_page(f"{DATA_DIR}/game_data/board.pdf")
CARD_REF = get_pdf_page(f"{DATA_DIR}/game_data/card_reverse.pdf")


def get_clip_name(video_idx: int) -> str:
    return f"clip_{video_idx}"


def record(reader: cv.VideoCapture, writer: cv.VideoWriter, tracked: list[TrackedObject], statics: list[StaticObject],
           start=0, sec=None):
    fps = reader.get(cv.CAP_PROP_FPS)

    if sec is None:
        sec = int(reader.get(cv.CAP_PROP_FRAME_COUNT)) / fps

    reader.set(cv.CAP_PROP_POS_FRAMES, start)

    print(f"Recording {sec} seconds of video from {start//fps} at {fps} fps")
    print("Recording...")

    for frame_id in tqdm(range(start, start + int(sec * fps))):
        ret, frame = reader.read()

        if not ret:
            print("Failed to read frame")
            break

        raw_frame = np.copy(frame)

        for obj in statics:
            if frame_id % obj.refresh_rate == 0:
                obj.re_detect(raw_frame)
            obj.detect_events(raw_frame)
            frame = obj.draw(frame)

        for obj in tracked:
            if frame_id % obj.refresh_rate == 0:
                obj.re_detect(raw_frame)

            found = obj.update(raw_frame)

            if not found:
                frame = obj.detection_fail_msg(frame)
                continue

            obj.detect_events(raw_frame)
            frame = obj.draw_bbox(frame)

        events = [obj.event for obj in tracked + statics]
        display_events(frame, events)

        writer.write(frame)


def make_clip(diff: str, idx: int) -> None:
    clip_name = f"{diff}_{get_clip_name(idx)}"

    print(f"Processing... Difficulty: {diff} | File: {get_clip_name(idx)}.mp4")

    reader = cv.VideoCapture(f"{CLIP_DIRS[diff]}/{get_clip_name(idx)}.mp4")
    writer = cv.VideoWriter(
        f"{UPLOAD_DIR}/{clip_name}.avi",
        cv.VideoWriter_fourcc(*"DIVX"),
        reader.get(cv.CAP_PROP_FPS),
        (int(reader.get(cv.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv.CAP_PROP_FRAME_HEIGHT))),
    )

    if reader.isOpened() and writer.isOpened():
        print("Loaded")
    else:
        print("Failed to load")
        return

    board = Board("board", BOARD_REF)
    card_pile = CardPile("pile of cards", CARD_REF)
    card = Card("card", "CSRT", card_pile)
    dice_tray = DiceTray("dice tray")
    dice_1 = Dice("dice 1", "CSRT", dice_tray, 1)
    dice_2 = Dice("dice 2", "CSRT", dice_tray, 2)
    score_board = ScoreBoard("score", board, BOARD_MASK[:, :, 2])
    buildings = Buildings("buildings", board, BOARD_MASK[:, :, 0])
    pawns = Pawns("pawns", board, BOARD_MASK[:, :, 0])

    record(reader, writer, [card, dice_1, dice_2], [board, dice_tray, card_pile, score_board, buildings, pawns])

    print("Done")

    reader.release()
    writer.release()


def convert_to_mp4(diff: str, idx: int) -> None:
    clip_name = f"{diff}_{get_clip_name(idx)}"
    ffmpeg_command = (f"ffmpeg -hide_banner -loglevel error -i {UPLOAD_DIR}/{clip_name}.avi "
                      f"-y {UPLOAD_DIR}/{clip_name}.mp4")
    print(f"Running: {ffmpeg_command}")
    subprocess.run(ffmpeg_command.split())


if __name__ == "__main__":
    for difficulty in DIFFICULTIES:
        for i in range(3):
            make_clip(difficulty, i)
            convert_to_mp4(difficulty, i)
