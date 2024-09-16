# 4109061012 B.S.Chen
import cv2
import numpy as np

IMAGE_PATH = "homework_5/src/Fig0110(4)(WashingtonDC Band4).TIF"
TARGET_FOLDER = "homework_5/result"
TH = 40

def pseudo_color_process(img: np.ndarray, threshold: int) -> np.ndarray:
    """ Problem(a): the range of gray levels under the `threshold` will be yellow after processing,
        the other range of gray levels will be remained. """
    new_img = np.empty((img.shape[0], img.shape[1], 3))
    new_img = np.stack([img, img, img], axis=-1)
    new_img[img < threshold, :] = [0, 255, 255]  # yellow
    return new_img

if __name__ == "__main__":
    # Problem (b): using Fig. 1.10(4)
    img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
    img = pseudo_color_process(img, TH)
    cv2.imwrite(f"{TARGET_FOLDER}/P8(b)_Result.jpg", img.astype(np.uint8))
