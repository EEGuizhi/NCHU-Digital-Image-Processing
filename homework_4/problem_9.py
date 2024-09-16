# 4109061012 B.S.Chen
import cv2
import numpy as np

IMAGE_PATH = "homework_4/src/Fig0526(a)(original_DIP).tif"
TARGET_FOLDER = "homework_4/result"


def sinusoidal_noise(img: np.ndarray, A: float, u0: float, v0: float) -> np.ndarray:
    """ Problem (a): Adding sinusoidal noise (from prob5.14) on image """
    y, x = np.arange(img.shape[0])/img.shape[0], np.arange(img.shape[1])/img.shape[1]
    xv, yv = np.meshgrid(x, y)
    noise = A * np.sin(u0 * xv + v0 * yv)
    return img + noise


def horizontal_notch_filter(shape: tuple, D0: int, mid: int) -> np.ndarray:
    """ Horizontal, ideal notch reject filter """
    notch_pass = np.zeros(shape)
    notch_pass[shape[0]//2 - D0 : shape[0]//2 + D0 + 1, :] = 1
    notch_pass[:, shape[1]//2 - mid//2 : shape[1]//2 + mid//2 + 1] = 0
    display_ft_image(notch_pass, "P9_Notch_pass_filter.jpg")
    return 1 - notch_pass  # retrun is notch "reject" filter


def display_ft_image(ft_img: np.ndarray, name: str):
    save_img = np.log(1 + abs(ft_img))
    save_img = save_img / save_img.max() * 255
    cv2.imshow("Display Image", save_img.astype(np.uint8)), cv2.waitKey(0), cv2.destroyAllWindows()
    saving_image(save_img, name)


def saving_image(img: np.ndarray, name: str):
    cv2.imwrite(f"{TARGET_FOLDER}/{name}", np.clip(img, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    # Problem (b)
    img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
    img = sinusoidal_noise(img, A=60, u0=img.shape[0]/2, v0=0)
    saving_image(img, "P9_Image_with_noise.jpg")

    # Problem (c)
    ft_img = np.fft.fft2(img)
    ft_img = np.fft.fftshift(ft_img)
    display_ft_image(ft_img, "P9_FT_spectrum.jpg")

    # Problem (d)
    notch_reject = horizontal_notch_filter(ft_img.shape, 0, ft_img.shape[0]//10)
    ft_img = ft_img * notch_reject
    ft_img = np.fft.ifftshift(ft_img)
    img = np.fft.ifft2(ft_img).real
    saving_image(img, "P9_Result.jpg")
