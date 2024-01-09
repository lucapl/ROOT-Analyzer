import cv2
from sys import argv

def getFirstFrame(_input,verbose=False):
    if verbose: print(_input,_output)
    vidcap = cv2.VideoCapture(_input)
    if vidcap.isOpened():
        if verbose: print("Video loaded")
    success, image = vidcap.read()
    if success:
        return image

if __name__ == '__main__':
    image = getFirstFrame(*argv[1:])
    if image != None:
        cv2.imwrite("first_frame.png",image)