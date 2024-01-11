import cv2 as cv
import numpy as np
from tqdm import tqdm
import subprocess


from src.Event import Event
from src.tracking.TrackedObject import TrackedObject
from src.tracking.StaticObject import StaticObject


def track_video(video, out_path, tracked: list[TrackedObject], statics: list[StaticObject], start=0, sec=None):
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

        if i % (10 *fps) == 0 :
            for static in statics:
                static.redetect(raw_frame)
            for tracked_obj in tracked:
                tracked_obj.redetect(raw_frame)
            

        if not ret:
            break

        for idx, obj in enumerate(statics):
            frame = obj.draw_bbox(frame)
            event = obj.detect_events(i, raw_frame)
            if type(event) is Event:
                event = event.get()
            if event is not None:
                frame = cv.putText(frame, event, (0, 50 + idx * 25), cv.FONT_HERSHEY_SIMPLEX,
                                   1, (0, 0, 255), 2, cv.LINE_AA)

        for obj in tracked:
            bbox = obj.update(raw_frame)
            if bbox is None:
                frame = obj.detection_fail_msg(frame)
                continue
            frame = obj.draw_bbox(frame)
            event = obj.detect_events(i)
            if type(event) is Event:
                event = event.get()
            if event is not None:
                frame = cv.putText(frame, event, (0, 35), cv.FONT_HERSHEY_SIMPLEX,
                                   1, (0, 0, 255), 2, cv.LINE_AA)

        # for cont in static:
        #     draw_bbox(frame, cv.boundingRect(cont), (255, 0, 0))

        # for tracker in trackers:
        #     ok, bbox = tracker.update(raw_frame)
        #     if ok:
        #         draw_bbox(frame, bbox, (0, 255, 0))
        #     else:
        #         frame = cv.putText(frame, 'FAILED', (0,0), cv.FONT_HERSHEY_SIMPLEX,
        #         1, (0,0,255), 2, cv.LINE_AA)

        # hsv = cv.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # ret, track_window = cv.CamShift(dst, track_window, term_crit)
        # pts = np.int0(cv.boxPoints(ret))
        track.write(frame)

    track.release()

    ffmpeg_command = f"ffmpeg -hide_banner -loglevel error -i {out_path}.avi -y {out_path}.mp4"
    print(f"Running: {ffmpeg_command}")
    subprocess.run(ffmpeg_command.split())
