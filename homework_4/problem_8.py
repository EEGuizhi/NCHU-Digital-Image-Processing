# 4109061012 B.S.Chen
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_PATH = "homework_4/src/Fig0507(a)(ckt-board-orig).tif"
TARGET_FOLDER = "homework_4/result"
PA, PB = 0.2, 0.2

def median_filtering(img: np.ndarray, size: int) -> np.ndarray:
    """ Problem (a): Median Filtering """
    s, pad = size, size // 2
    new_img = np.empty_like(img, dtype=np.float32)
    img = np.pad(img, ((pad, pad), (pad, pad)), "symmetric")

    for r in tqdm(range(new_img.shape[0])):
        for c in range(new_img.shape[1]):
            new_img[r, c] = np.median(img[r : r+s, c : c+s])
    return new_img

def salt_and_pepper_noise(img: np.ndarray, Pa: float, Pb: float) -> np.ndarray:
    """ Problem (b): Adding salt and pepper noise on image """
    noise = np.random.choice(3, img.shape, p=[1 - Pa - Pb, Pa, Pb])
    img[noise == 1] = 0
    img[noise == 2] = 255
    return img

if __name__ == "__main__":
    # Problem (c)
    img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
    img = salt_and_pepper_noise(img, PA, PB)
    cv2.imwrite(f"{TARGET_FOLDER}/P8_Image_with_noise.jpg", img.clip(0, 255).astype(np.uint8))
    img = median_filtering(img, 3)
    cv2.imwrite(f"{TARGET_FOLDER}/P8_Result.jpg", img.clip(0, 255).astype(np.uint8))
