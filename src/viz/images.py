import cv2 as cv
import PIL.Image
from IPython.display import display


def imshow(img):
    """ imshow function courtesy of our lab notebooks """

    img = img.clip(0, 255).astype("uint8")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(img))


def scaled_imshow(img, fx=0.3, fy=0.3):
    """ scaling the image so the notebook doesn't take too much space """
    scaled_img = cv.resize(img, None, fx=fx, fy=fy)
    imshow(scaled_img)


def draw_bbox(frame, bbox, color=(255, 255, 255)):
    x, y, w, h = bbox
    p1 = (int(x), int(y))
    p2 = (int(x + w), int(y + h))
    cv.rectangle(frame, p1, p2, color, 2, 1)
