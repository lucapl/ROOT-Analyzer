import moviepy.editor as mpy
'''
Script courtesy of: https://www.python-engineer.com/posts/video-editing-with-python/
'''

vcodec =   "libx264"

videoquality = "23"

compression = "fast"

def resize_vid(path,out_path,factor=0.25,threads=16,fps=24):
    # load file
    video = mpy.VideoFileClip(path)
    video = video.resize(factor)
    video.write_videofile(out_path, threads=threads, fps=fps,
                        codec=vcodec,
                        preset=compression,
                        audio = False,
                        ffmpeg_params=["-crf",videoquality])

    video.close()

if __name__ == "__main__":
    data_dir = ".\\data\\"

    types = ["easy","medium","hard"]
    clips = [3,3,3]

    for _type,n in zip(types,clips):
        for i in range(n):
            resize_vid(data_dir+_type+f"\\clip_{i}.mp4",data_dir+_type+f"\\resized\\clip_{i}.mp4")

