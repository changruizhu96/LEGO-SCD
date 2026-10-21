from __future__ import annotations
import cv2
import numpy as np

def crop_with_context(image, mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return image, mask
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    pad_x = max(12, round(width * 0.22))
    pad_y = max(12, round(height * 0.22))
    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(image.shape[1], int(xs.max()) + pad_x + 1)
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(image.shape[0], int(ys.max()) + pad_y + 1)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def fit_panel(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel

