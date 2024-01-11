import cv2 as cv
from sys import argv


def get_first_frame(_input: str, verbose=False) -> cv.Mat | None:
    if verbose:
        print(_input)

    vidcap = cv.VideoCapture(_input)

    if not vidcap.isOpened():
        if verbose:
            print("Failed to open video")
        return None

    if verbose:
        print("Video loaded")

    success, img = vidcap.read()

    if success:
        return img


if __name__ == '__main__':
    image = get_first_frame(*argv[1:])

    if image is not None:
        cv.imwrite("first_frame.png", image)
