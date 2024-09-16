# 4109061012 B.S.Chen
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

IMAGE_PATH = "homework_3/src/Fig0235(c)(kidney_original).tif"
TARGET_FOLDER = "homework_3/result"

T = 115

def filtering(img: np.ndarray, kernel: np.ndarray, coeff = 9) -> np.ndarray:
    """ Filtering an image with symmetric padding """
    s, pad = kernel.shape[0], kernel.shape[0] // 2
    new_img = np.empty_like(img, dtype=np.float32)
    img = np.pad(img, ((pad, pad), (pad, pad)), "symmetric")

    for r in tqdm(range(new_img.shape[0])):
        for c in range(new_img.shape[1]):
            new_img[r, c] = round(coeff * np.average(kernel * img[r : r+s, c : c+s]))
    return new_img

def Edge_detection(img: np.ndarray, threshold: int) -> np.ndarray:
    """ Problem (a): Edge detection using sobel filters """
    mask_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ])
    mask_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    Gx, Gy = filtering(img, mask_x), filtering(img, mask_y)
    grad_sum = abs(Gx) + abs(Gy)

    # Histogram
    plt.hist(grad_sum.ravel(), 256, (0, 256)), plt.xlim(left=0, right=256)
    plt.savefig(f"{TARGET_FOLDER}/P9_Grad_Hist.jpg"), plt.clf()

    # Binarization
    grad_sum[grad_sum < threshold] = 0
    grad_sum[grad_sum > 0] = 255
    return grad_sum.astype(np.uint8)

if __name__ == "__main__":
    # Problem (b): smoothing with 3x3 mask & use program from (a)
    img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

    print(f">> Smoothing..")
    img = filtering(img, np.ones((3, 3)), 1)

    print(f">> Computing Gradients (Gx, Gy)..")
    edge = Edge_detection(img, T)

    cv2.imwrite(f"{TARGET_FOLDER}/P9_Result_T{T}.jpg", edge)
    print(f">> Threshold T: {T}")
