import cv2
from sys import argv

def getFirstFrame(_input,_output="first_frame.png"):
    print(_input,_output)
    vidcap = cv2.VideoCapture(_input)
    if vidcap.isOpened():
        print("Video loaded")
    success, image = vidcap.read()
    if success:
        print(_output)
        cv2.imwrite(_output, image)

if __name__ == '__main__':
    getFirstFrame(*argv[1:])