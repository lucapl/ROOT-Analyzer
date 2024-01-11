import moviepy.editor as mpy
import os


CODEC = "libx264"
QUALITY = "23"
COMPRESSION = "fast"


def rotate_vid(path, out_path, angle=180, threads=16, fps=24):
    # load file
    video = mpy.VideoFileClip(path)
    video = video.rotate(angle)
    video.write_videofile(out_path, threads=threads, fps=fps,
                          codec=CODEC,
                          preset=COMPRESSION,
                          audio=False,
                          ffmpeg_params=["-crf", QUALITY])

    video.close()


if __name__ == "__main__":
    data_dir = "./data/hard"
    
    if not os.path.exists(f"{data_dir}/rotated"):
        os.makedirs(f"{data_dir}/rotated")

    for i in range(3):
        if os.path.exists(f"{data_dir}/rotated/clip_{i}.mp4"):
            continue

        rotate_vid(f"{data_dir}/clip_{i}.mp4", f"{data_dir}/rotated/clip_{i}.mp4")
