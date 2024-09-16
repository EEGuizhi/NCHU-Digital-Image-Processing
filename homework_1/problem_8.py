# 4109061012 B.S.Chen
import cv2
import numpy as np


FILE_PATH = "homework_1/src/Fig0221(a)(ctskull-256).tif"
NEW_LEVELS = 2
I_MAX = 2**8 - 1  # original max intensity value


def reduce_intensity_levels(img: np.ndarray, levels: int) -> np.ndarray:
    """ Reduce the number of intensity levels in an image. """
    if levels <= 0: raise ValueError("the new number of intensity levels must > 0")
    elif levels == 1: return np.zeros_like(img)

    interval = I_MAX / levels
    intensities = (np.arange(levels) * I_MAX / (levels - 1)).astype(int)
    print(f">> new intensities ({levels} levels): \n{intensities}\n")

    for i in range(levels):
        img[(img >= interval * i) & (img < interval * (i+1))] = intensities[i]
    return img


if __name__ == "__main__":
    img = np.asarray(cv2.imread(FILE_PATH, cv2.IMREAD_GRAYSCALE))
    img = reduce_intensity_levels(img, NEW_LEVELS)
    cv2.imwrite("problem_8(b)_output.jpg", img)

    print(">> done.")
