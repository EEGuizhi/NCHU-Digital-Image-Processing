# 4109061012 B.S.Chen
import cv2
import numpy as np

IMAGE_PATH = "homework_3/src/Fig0441(a)(characters_test_pattern).tif"
TARGET_FOLDER = "homework_3/result"

if __name__ == "__main__":
    img = np.asarray(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

    # Problem (a): Compute (centered) Fourier spectrum
    ft_img = np.fft.fft2(img)
    ave = abs(ft_img[0, 0]) / (img.shape[0] * img.shape[1])
    ft_img = np.fft.fftshift(ft_img)

    # Problem (c): Average value
    print(f">> Average value of image = {ave}")
    print(f">> Average value of image in fourier spectrum = {np.average(ft_img)}")

    # Problem (b): Display (a)
    ft_img = np.log(1 + abs(ft_img))
    ft_img = ft_img / ft_img.max() * 255
    cv2.imwrite(f"{TARGET_FOLDER}/P8_centered_FT_image.jpg", ft_img)
