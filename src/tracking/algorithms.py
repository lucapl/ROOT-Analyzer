import cv2 as cv
import numpy as np
import tqdm

def track_video(video,out_path,tracked,start=0,sec=None):
    track = cv.VideoWriter(
        out_path,
        cv.VideoWriter_fourcc(*"DIVX"),
        fps,
        (width, height),
    )

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)

    if sec==None:
        sec = int(video.get(cv.CAP_PROP_FRAME_COUNT))/fps

    video.set(cv.CAP_PROP_POS_FRAMES,start)
    for i in tqdm(range(int(sec*fps))):
        if not video.isOpened():
            break
        if i > sec*fps:
            break

        ret, frame = video.read()
        raw_frame = np.copy(frame)

        if ret:
            
            for obj in tracked:
                bbox = obj.update(raw_frame)
                if bbox == None:
                    frame = obj.detection_fail_msg(frame)
                    continue
                frame = obj.draw_bbox(frame)
                

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
        else:
            break

    track.release()