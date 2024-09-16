# 4109061012 B.S.Chen
import cv2
import numpy as np

IMAGE_PATH = "homework_5/src/Fig0635(bottom_left_stream).tif"
TARGET_FOLDER = "homework_5/result"

def histogram(img: np.ndarray, value_range: tuple, normalize: bool) -> np.ndarray:
    """ Compute histogram of input image (from homework 2) """
    img = img.ravel()
    hist = np.zeros(value_range[1] - value_range[0])
    for i in range(img.shape[0]):
        hist[img[i]] += 1
    if normalize:
        hist = hist / img.shape[0]
    return hist

def hist_equalize(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Histogram equalization (from homework 2) """
    # Cumulated PDF of original image intensities
    hist = histogram(img, (0, 256), normalize=True)
    cumsum_hist = np.cumsum(hist)

    # Transform
    new_img = np.empty_like(img)
    for i in range(256):
        intensity = round(255 * cumsum_hist[i])
        new_img[img == i] = intensity
    return new_img

def sep_hist_equal(img: np.ndarray) -> np.ndarray:
    """ Problem (a): Histogram-equalize the R, G, and B images separately. """
    for ch in range(3):
        img[:, :, ch] = hist_equalize(img[:, :, ch])
    return img

def ave_hist_equal(img: np.ndarray) -> np.ndarray:
    """ Problem (b): Form an average histogram from the three histograms first. """
    ave_hist = histogram(img[:, :, 0], (0, 256), True) \
             + histogram(img[:, :, 1], (0, 256), True) \
             + histogram(img[:, :, 2], (0, 256), True)
    ave_hist = ave_hist / 3
    cumsum_hist = np.cumsum(ave_hist)

    trans_func = np.empty(256)
    for i in range(256):
        intensity = round(255 * cumsum_hist[i])
        trans_func[i] = intensity

    new_img = np.empty_like(img)
    for i in range(256):
        new_img[img == i] = trans_func[i]
    return new_img


if __name__ == "__main__":
    orig_img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR), dtype=np.uint8)

    img = sep_hist_equal(orig_img.copy())
    cv2.imwrite(f"{TARGET_FOLDER}/P9(a)_Result.jpg", img.astype(np.uint8))
    cv2.imwrite(f"{TARGET_FOLDER}/P9(a)_Result.tif", img.astype(np.uint8))

    img = ave_hist_equal(orig_img)
    cv2.imwrite(f"{TARGET_FOLDER}/P9(b)_Result.jpg", img.astype(np.uint8))
    cv2.imwrite(f"{TARGET_FOLDER}/P9(b)_Result.tif", img.astype(np.uint8))
