# 4109061012 B.S.Chen
import cv2
import numpy as np


IMAGE_PATH = "homework_2/src/Fig0340(a)(dipxe_text).tif"
TARGET_PATH = "homework_2/result"
COEFF = 4.5  # highboost ( > 1)


def filtering(img: np.ndarray, kernel: np.ndarray, coeff = 9.0) -> np.ndarray:
    """ Filtering an image with symmetric padding """
    s, pad = kernel.shape[0], kernel.shape[0] // 2
    new_img = np.empty_like(img)
    img = np.pad(img, ((pad, pad), (pad, pad)), "symmetric")

    for r in range(new_img.shape[0]):
        for c in range(new_img.shape[1]):
            new_img[r, c] = round(coeff * np.average(kernel * img[r : r+s, c : c+s]))
    return new_img


def Unsharp_masking(img: np.ndarray, coeff: float) -> np.ndarray:
    """ Apply unsharp masking on input image. """
    img = img.astype(dtype=np.int32)

    # Filter setting
    ave_filter = np.ones((3, 3), dtype=np.int32)

    # Enhancing
    mask = img - filtering(img, ave_filter, 1)
    img = img + coeff * mask
    img[img < 0], img[img > 255] = 0, 255
    return img.astype(dtype=np.uint8)


if __name__ == "__main__":
    orig_img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

    print(">> start masking..")
    new_img = Unsharp_masking(orig_img, COEFF)
    cv2.imwrite(TARGET_PATH + "/prob10_result_img.jpg", new_img)
    print(">> done")
