# 4109061012 B.S.Chen
import cv2
import numpy as np

IMAGE_PATH = "homework_5/src/Fig0628(b)(jupiter-Io-closeup).tif"
TARGET_FOLDER = "homework_5/result"
RANGE_X, RANGE_Y = [115, 135], [385, 405]

def image_segement(img: np.ndarray, sel_x: tuple, sel_y: tuple) -> np.ndarray:
    new_img = np.zeros(img.shape[0 : 2])
    sel_region = img[sel_y[0]: sel_y[1], sel_x[0]: sel_x[1], :]

    mean = np.mean(sel_region, axis=(0, 1))
    dev = np.std(sel_region, axis=(0, 1))

    th_low = mean - 1.25 * dev
    th_high = mean + 1.25 * dev
    seg = (img[:, :, 0] >= th_low[0]) & (img[:, :, 0] < th_high[0]) \
        & (img[:, :, 1] >= th_low[1]) & (img[:, :, 1] < th_high[1]) \
        & (img[:, :, 2] >= th_low[2]) & (img[:, :, 2] < th_high[2])

    new_img[seg] = 255
    return new_img

if __name__ == "__main__":
    img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR), dtype=np.uint8)
    img = image_segement(img, RANGE_X, RANGE_Y)
    cv2.imwrite(f"{TARGET_FOLDER}/P10_Result.jpg", img.astype(np.uint8))
