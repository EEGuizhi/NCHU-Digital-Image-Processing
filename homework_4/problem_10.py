# 4109061012 B.S.Chen
import cv2
import numpy as np

IMAGE_PATH = "homework_4/src/Fig0526(a)(original_DIP).tif"
TARGET_FOLDER = "homework_4/result"
K = 0.0003

def blurring_filter(shape: tuple, a: float, b: float, T: float) -> np.ndarray:
    """ Problem (a): Implement blurring filter as in Eq.(5.6-11) """
    y, x = np.arange(shape[0]), np.arange(shape[1])
    xv, yv = np.meshgrid(x, y)
    tmp = np.pi * (a * xv + b * yv)
    tmp[tmp == 0] = 0.0001

    mask = T * np.sin(tmp) * np.exp(-1j * tmp) / tmp
    return mask

def gaussian_noise(img: np.ndarray, mean: int, var: int) -> np.ndarray:
    """ Problem (a): Adding gaussian noise on the image """
    z_len = 512 - 1  # posible values: pos = 1~255, neg = -1~-255, center = 0, total: 511
    coef = np.sqrt(2 * np.pi * var * np.ones(z_len))
    hist = coef * np.exp(-np.square(np.arange(z_len) - 255 - mean) / (2 * var))

    mask = np.random.choice(z_len, img.shape, p=hist/hist.sum()) - 255
    img = img + mask
    return img.clip(0, 255)

def Wiener_filtering(G: np.ndarray, H: np.ndarray, K: float) -> np.ndarray:
    H_square = H * H.conj()
    F_hat = (G / H) * (H_square / (H_square + K))
    return F_hat

def display_ft_image(ft_img: np.ndarray, name: str):
    """ Display (saving) the image which is in fourier spectrum """
    save_img = np.log(1 + abs(ft_img))
    save_img = save_img / save_img.max() * 255
    cv2.imwrite(f"{TARGET_FOLDER}/{name}", save_img)

if __name__ == "__main__":
    # Problem (b): Blurring
    img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
    blur_filter = blurring_filter(img.shape, -0.1, 0.1, 1)
    display_ft_image(blur_filter, "P10_Blurring_filter.jpg")

    ft_img = np.fft.fft2(img)
    ft_img = np.fft.fftshift(ft_img)
    ft_img = ft_img * blur_filter
    ft_img = np.fft.ifftshift(ft_img)
    img = np.fft.ifft2(ft_img).real

    img = img.clip(0, 255)
    cv2.imwrite(f"{TARGET_FOLDER}/P10(b)_Result.jpg", img.astype(np.uint8))


    # Problem (c): Adding noise
    img = gaussian_noise(img, 0, 10)
    cv2.imwrite(f"{TARGET_FOLDER}/P10(c)_Result.jpg", img.astype(np.uint8))

    # Problem (d): Restoring the image
    ft_img = np.fft.fft2(img)
    ft_img = np.fft.fftshift(ft_img)
    ft_img = Wiener_filtering(ft_img, blur_filter, K)
    ft_img = np.fft.ifftshift(ft_img)
    img = np.fft.ifft2(ft_img).real
    cv2.imwrite(f"{TARGET_FOLDER}/P10(d)_Result.jpg", img.clip(0, 255).astype(np.uint8))
