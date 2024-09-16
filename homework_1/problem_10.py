# 4109061012 B.S.Chen
import cv2


FILE_PATH = "homework_1/src/Fig0236(a)(letter_T).tif"
ROTATE = -23  # ccw
SCALE = 2 / 3
SHIFT = (18, 22)  # x, y


def image_RSTI(img: cv2.Mat, rotate = 0.0, scale = 1.0, translate = (0, 0), interpolation = "nearest"):
    """ Apply rotating, scaling, translating(shifting), and using different interpolation on an image. """
    if interpolation not in ("nearest", "bilinear", "bicubic"):
        raise ValueError(f"interpolation must be \"nearest\", \"bilinear\", \"bicubic\"")

    center = (img.shape[1] // 2 + translate[0], img.shape[0] // 2 + translate[1])
    transform = cv2.getRotationMatrix2D(center, rotate, scale)
    if interpolation == "nearest":
        return cv2.warpAffine(img, transform, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
    elif interpolation == "bilinear":
        return cv2.warpAffine(img, transform, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    else:
        return cv2.warpAffine(img, transform, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)


if __name__ == "__main__":
    img = cv2.imread(FILE_PATH)
    nearest_img = image_RSTI(img, rotate=ROTATE, scale=SCALE, translate=SHIFT, interpolation="nearest")
    bilinear_img = image_RSTI(img, rotate=ROTATE, scale=SCALE, translate=SHIFT, interpolation="bilinear")
    bicubic_img = image_RSTI(img, rotate=ROTATE, scale=SCALE, translate=SHIFT, interpolation="bicubic")

    cv2.imwrite("problem_10(b)_nearest.jpg", nearest_img)
    cv2.imwrite("problem_10(b)_bilinear.jpg", bilinear_img)
    cv2.imwrite("problem_10(b)_bicubic.jpg", bicubic_img)

    print(">> done")
