# 4109061012 B.S.Chen
import cv2


FILE_PATH = "homework_1/src/Fig0220(a)(chronometer 3692x2812  2pt25 inch 1250 dpi).tif"
FACTOR = 12


if __name__ == "__main__":
    """ Shrink the image by a factor first, then zoom back to original size. """
    img = cv2.imread(FILE_PATH)
    orig_size = (img.shape[1], img.shape[0])
    shrink_size = (round(img.shape[1] / FACTOR), round(img.shape[0] / FACTOR))

    # Shrinking
    img = cv2.resize(img, shrink_size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("problem_9(b)_output.jpg", img)

    # Zoom back to original size
    img = cv2.resize(img, orig_size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("problem_9(c)_output.jpg", img)

    print(">> done")
