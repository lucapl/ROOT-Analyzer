import moviepy.editor as mpy
'''
Script courtesy of: https://www.python-engineer.com/posts/video-editing-with-python/
'''

vcodec =   "libx264"

videoquality = "23"

compression = "fast"

def split_vid(path,_dir, cuts,threads=16,fps=24):
    # load file
    video = mpy.VideoFileClip(path)
    for i,cut in enumerate(cuts):
        clip = video.subclip(cut[0], cut[1])
        out_path = f"{_dir}\\clip_{i}.mp4"
        clip.write_videofile(out_path, threads=threads, fps=fps,
                            codec=vcodec,
                            preset=compression,
                            audio = False,
                            ffmpeg_params=["-crf",videoquality])

    video.close()

if __name__ == "__main__":
    data_dir = ".\\data"

    hard_cuts = [('00:00:00.000','00:01:55.000'),
                ('00:01:55.000','00:04:04.000'),
                ('00:07:39.000','00:08:51.000')]

    medium_cuts = [('00:00:00.000','00:01:55.000'),
                ('00:01:55.000','00:04:04.000'),
                ('00:07:39.000','00:08:51.000')]

    easy_cuts = [('00:00:00.000','00:02:00.000'),
                ('00:02:00.000','00:04:30.000'),
                ('00:04:30.000','00:07:26.000')]

    data = {"easy":easy_cuts,
            "medium":medium_cuts,
            "hard":hard_cuts}
    
    for _type,cuts in data.items():
        split_vid(data_dir+f'\\{_type}\\1.mp4',data_dir+f'\\{_type}\\',cuts)


