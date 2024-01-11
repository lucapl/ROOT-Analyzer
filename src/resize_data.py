import moviepy.editor as mpy
import os

'''
Script courtesy of: https://www.python-engineer.com/posts/video-editing-with-python/
'''

VCODEC = "libx264"
QUALITY = "23"
COMPRESSION = "fast"


def resize_vid(path: str, out_path: str, factor=0.25, threads=16, fps=24) -> None:
    # load file
    video = mpy.VideoFileClip(path)
    video = video.resize(factor)
    video.write_videofile(out_path, threads=threads, fps=fps,
                          codec=VCODEC,
                          preset=COMPRESSION,
                          audio=False,
                          ffmpeg_params=["-crf", QUALITY])

    video.close()


if __name__ == "__main__":
    data_dir = "./data"

    types = {
        "easy": 3, 
        "medium": 3, 
        "hard": 3
    }

    for _type, n in types.items():
        if not os.path.exists(f"{data_dir}/{_type}/resized"):
            os.makedirs(f"{data_dir}/{_type}/resized")

        for i in range(n):
            if os.path.exists(f"{data_dir}/{_type}/resized/clip_{i}.mp4"):
                continue

            resize_vid(f"{data_dir}/{_type}/clip_{i}.mp4", f"{data_dir}/{_type}/resized/clip_{i}.mp4")
