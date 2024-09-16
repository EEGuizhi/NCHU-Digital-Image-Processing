# 4109061012 B.S.Chen
import cv2
import numpy as np

IMAGE_PATH = "homework_3/src/Fig0457(a)(thumb_print).tif"
TARGET_FOLDER = "homework_3/result"

D0 = 50
T = 10

def Gauss_Highpass_Filter(shape: tuple, cutoff: int, center: tuple) -> np.ndarray:
    """ Problem (a): Gaussian highpass filter """
    x = np.linspace(0, shape[1]-1, shape[1])
    y = np.linspace(0, shape[0]-1, shape[0])
    xv, yv = np.meshgrid(x, y)

    xv = xv - center[1]
    yv = yv - center[0]
    D2_uv = np.square(xv) + np.square(yv)

    H_uv = np.ones_like(D2_uv) - np.exp(-D2_uv / (2 * cutoff**2))
    return H_uv

if __name__ == "__main__":
    # Problem (b)
    img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
    m, n = img.shape[0], img.shape[1]
    p, q = 2*m - 1, 2*n - 1

    ft_img = np.zeros((p, q))
    ft_img[:m, :n] = img
    hp = Gauss_Highpass_Filter((p, q), D0, (m, n))
    cv2.imwrite(f"{TARGET_FOLDER}/P10_Highpass_Filter.jpg", hp * 255)

    ft_img = np.fft.fft2(ft_img)
    ft_img = np.fft.fftshift(ft_img)
    ft_img = ft_img * hp
    ft_img = np.fft.ifftshift(ft_img)
    img = np.fft.ifft2(ft_img)

    img = img.real[:m, :n]
    cv2.imwrite(f"{TARGET_FOLDER}/P10_Filtered_image.jpg", img)

    img[img < T] = 0
    img[img > 0] = 255
    cv2.imwrite(f"{TARGET_FOLDER}/P10_Result.jpg", img)
