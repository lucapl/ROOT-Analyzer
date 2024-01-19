import cv2 as cv
import numpy as np
import fitz


def get_frame(video_path: str, seconds: int = 0) -> np.ndarray | None:
    """
    Gets the frame at a certain time in the video, returns None if the video is not opened or the frame is not found

    :param video_path: Path to the video
    :type video_path: str
    :param seconds: Time in seconds, defaults to 0
    :type seconds: int, optional
    :return: Frame at the specified time
    :rtype: np.ndarray | None
    """
    video = cv.VideoCapture(video_path)

    if not video.isOpened():
        return None

    fps = video.get(cv.CAP_PROP_FPS)
    frame_idx = int(seconds * fps)
    video.set(cv.CAP_PROP_POS_FRAMES, frame_idx)

    success, img = video.read()

    video.release()

    if not success:
        return None

    return img


def get_pdf_page(pdf_path: str, page_num: int = 0) -> np.ndarray | None:
    """
    Gets the page of the pdf at the specified index

    :param pdf_path: Path to the pdf
    :type pdf_path: str
    :param page_num: Index of the page, defaults to 0
    :type page_num: int, optional
    :return: Page at the specified index
    :rtype: np.ndarray | None
    """
    pdf = fitz.open(pdf_path)

    if pdf.is_closed or page_num >= len(pdf):
        return None

    page = pdf[page_num]

    page_width, page_height = int(page.rect.width) + 1, int(page.rect.height) + 1

    img = cv.cvtColor(
        np.frombuffer(
            page.get_pixmap().samples,
            dtype=np.uint8).
        reshape((page_height, page_width, 3)),
        cv.COLOR_RGB2BGR)

    pdf.close()

    return img
