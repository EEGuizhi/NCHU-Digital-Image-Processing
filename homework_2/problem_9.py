# 4109061012 B.S.Chen
import cv2
import numpy as np


IMAGE_PATH = "homework_2/src/Fig0338(a)(blurry_moon).tif"
TARGET_PATH = "homework_2/result"
SIZE = 3


def setting_filter() -> np.ndarray:
    i = 0
    filter = np.empty((SIZE, SIZE), dtype=np.int32)
    print(">> Please input values of the filter (use ',' to separate):")
    while i < SIZE**2:
        data = input().split(',')
        for value in data:
            filter[i // SIZE, i % SIZE] = int(value.replace(' ', ''))
            i += 1
    return filter


def filtering(img: np.ndarray, kernel: np.ndarray, coeff = 9.0) -> np.ndarray:
    """ Filtering an image with symmetric padding """
    s, pad = kernel.shape[0], kernel.shape[0] // 2
    new_img = np.empty_like(img)
    img = np.pad(img, ((pad, pad), (pad, pad)), "symmetric")

    for r in range(new_img.shape[0]):
        for c in range(new_img.shape[1]):
            new_img[r, c] = round(coeff * np.average(kernel * img[r : r+s, c : c+s]))
    return new_img


def Laplacian_filtering(img: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Enhancement using laplacian with user inputted filter """
    img = img.astype(dtype=np.int32)

    # Enhancing
    mask = filtering(img, kernel)
    scale_mask = (mask - mask.min()) / mask.max() * 255  # Eqs. (2.6-10), (2.6-11)
    img = img + mask

    img[img < 0], img[img > 255] = 0, 255
    mask[mask < 0], mask[mask > 255] = 0, 255
    scale_mask[scale_mask < 0], scale_mask[scale_mask > 255] = 0, 255
    return img.astype(np.uint8), mask.astype(np.uint8), scale_mask.astype(np.uint8)


if __name__ == "__main__":
    orig_img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
    kernel = setting_filter()

    print(">> start enhancing..")
    new_img, mask, scale_mask = Laplacian_filtering(orig_img, kernel)
    cv2.imwrite(TARGET_PATH + "/prob9_result_img.jpg", new_img)
    cv2.imwrite(TARGET_PATH + "/prob9_mask.jpg", mask)
    cv2.imwrite(TARGET_PATH + "/prob9_scale_mask.jpg", scale_mask)
    print(">> done")
