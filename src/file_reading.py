import fitz_new as fitz
import numpy as np
import cv2 as cv


def read_pdf(path: str, i=0) -> cv.Mat:
    pdf_document = fitz.open(path)

    page = pdf_document[i]

    page_width, page_height = int(page.rect.width) + 1, int(page.rect.height) + 1

    img = cv.cvtColor(
        np.frombuffer(
            page.get_pixmap().samples,
            dtype=np.uint8).
        reshape((page_height, page_width, 3)),
        cv.COLOR_RGB2BGR)

    pdf_document.close()

    return img
