import cv2 as cv
import numpy as np
from tqdm import tqdm
import subprocess

from src.tracking.TrackedObject import TrackedObject
from src.tracking.StaticObject import StaticObject
from src.tracking.Board import Board


def track_video(video, out_path, tracked: list[TrackedObject], statics: list[StaticObject],
                start=0, sec=None):
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)

    if sec is None:
        sec = int(video.get(cv.CAP_PROP_FRAME_COUNT)) / fps

    track = cv.VideoWriter(
        out_path + '.avi',
        cv.VideoWriter_fourcc(*"DIVX"),
        fps,
        (width, height),
    )

    video.set(cv.CAP_PROP_POS_FRAMES, start)

    for i in tqdm(range(int(sec * fps))):
        if not video.isOpened():
            break
        if i > sec * fps:
            break

        ret, frame = video.read()
        raw_frame = np.copy(frame)

        if i % 900 == 0:
            #board.redetect(raw_frame, board.M)
            for static in statics:
                static.redetect(raw_frame)

        if i % 120 == 0:
            for obj in tracked:
                obj.redetect(raw_frame)

        if not ret:
            break

        for obj in tracked:
            bbox = obj.update(raw_frame)
            if bbox is None:
                frame = obj.detection_fail_msg(frame)
                continue
            frame = obj.detect_events(i, frame)
            frame = obj.draw_bbox(frame)

        for obj in statics:
            frame = obj.detect_events(i, frame)
            frame = obj.draw(frame)

        track.write(frame)

    track.release()

    ffmpeg_command = f"ffmpeg -hide_banner -loglevel error -i {out_path}.avi -y {out_path}.mp4"
    print(f"Running: {ffmpeg_command}")
    subprocess.run(ffmpeg_command.split())
