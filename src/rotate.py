import moviepy.editor as mpy

vcodec =   "libx264"

videoquality = "23"

compression = "fast"

def rotate_vid(path,out_path,angle=180,threads=16,fps=24):
    # load file
    video = mpy.VideoFileClip(path)
    video = video.rotate(angle)
    video.write_videofile(out_path, threads=threads, fps=fps,
                        codec=vcodec,
                        preset=compression,
                        audio = False,
                        ffmpeg_params=["-crf",videoquality])

    video.close()

if __name__ == "__main__":
    data_dir = ".\\data\\hard\\"
    for i in range(3):
        rotate_vid(data_dir+f"clip_{i}.mp4",data_dir+f"rotated\\clip_{i}.mp4")