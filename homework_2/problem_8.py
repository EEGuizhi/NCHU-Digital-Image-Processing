# 4109061012 B.S.Chen
import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_PATH = "homework_2/src/Fig0308(a)(fractured_spine).tif"
TARGET_PATH = "homework_2/result"
I_MAX = 2**8


def histogram(img: np.ndarray, value_range: tuple, normalize: bool) -> np.ndarray:
    """ Problem 8(a): Compute histogram of input image """
    img = img.ravel()
    hist = np.zeros(value_range[1] - value_range[0])
    for i in range(img.shape[0]):
        hist[img[i]] += 1
    if normalize:
        hist = hist / img.shape[0]
    return hist


def hist_equalize(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Histogram equalization """
    # Cumulated PDF of original image intensities
    hist = histogram(img, (0, I_MAX), normalize=True)
    cumsum_hist = np.cumsum(hist)

    # Transform
    new_img = np.empty_like(img)
    trans_func = np.empty(I_MAX)
    for i in range(I_MAX):
        intensity = round((I_MAX - 1) * cumsum_hist[i])
        new_img[img == i] = intensity
        trans_func[i] = intensity
    return new_img, trans_func


if __name__ == "__main__":
    orig_img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
    new_img , trans_func= hist_equalize(orig_img)

    cv2.imwrite(TARGET_PATH + "/prob8_result_img.jpg", new_img)
    plt.hist(orig_img.ravel(), I_MAX, (0, I_MAX)), plt.xlim(left=0, right=I_MAX)
    plt.savefig(TARGET_PATH + "/prob8_orig_hist.jpg"), plt.clf()
    plt.hist(new_img.ravel(), I_MAX, (0, I_MAX)), plt.xlim(left=0, right=I_MAX)
    plt.savefig(TARGET_PATH + "/prob8_result_hist.jpg"), plt.clf()
    plt.plot(trans_func), plt.xlim(left=0, right=I_MAX)
    plt.savefig(TARGET_PATH + "/prob8_transform_func.jpg")
