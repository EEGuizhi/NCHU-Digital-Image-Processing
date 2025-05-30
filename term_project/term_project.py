# 4109061012 B.S.Chen
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Python implementation code of:
Pouli, Tania, and Erik Reinhard. "Progressive color transfer for images of arbitrary dynamic range." Computers & Graphics 35.1 (2011): 67-80.
- By B.S.Chen National Chung Hsing University
"""

# Settings
NUM = "04"
PATH_SRC = f"term_project/source_{NUM}.jpg"
PATH_TAR = f"term_project/target_{NUM}.jpg"

LINEAR = False
SHOW_HIST, SHOW_TRANSFER_FUNC = False, False
CHANNEL = 1

STEP = 0.1
B_MIN = 70
PERC = [20, 100]
W_A = -1  # 0.15  # Mask (set to <= 0 to deactivate)
CH_RANGE = (  # https://stackoverflow.com/questions/11386556/converting-an-opencv-bgr-8-bit-image-to-cie-lab
    (0, 256),
    (0, 256),
    (0, 256)
)


def compute_Smax(Bins: np.ndarray, Bmin: int) -> np.ndarray:
    """ Compute Smax according to Eq.7 , Note: Smax has 3 variables for each channel """
    Smax = np.floor(np.log2(Bins / Bmin))
    return Smax


def histogram(img: np.ndarray, value_range: tuple, normalize: bool) -> np.ndarray:
    """ Compute histogram of input image """
    hist, _ = np.histogram(
        img, np.arange(value_range[0], value_range[1] + 1),
        (value_range[0], value_range[1]),
        density=normalize
    )
    return hist


def resample_histogram(Hist: np.ndarray, Bins: int, normalize: bool) -> np.ndarray:
    """ Resampling histogram, including down- & up- sampling """
    orig_bins = Hist.shape[0]
    resampHist = np.zeros(Bins, dtype=Hist.dtype)

    if orig_bins < Bins:
        for i in range(Bins):
            index = round(i/Bins * orig_bins)
            resampHist[i] = Hist[index if index < orig_bins else orig_bins - 1]
    elif orig_bins > Bins:
        for i in range(orig_bins):
            index = round(i/orig_bins * Bins)
            resampHist[index if index < Bins else Bins - 1] += Hist[i]
    else:
        resampHist = Hist

    if normalize:
        total_area = np.sum(resampHist)
        resampHist = resampHist / total_area

    return resampHist


def Reshape_Histogram(Is: np.ndarray, It: np.ndarray, perc: int, Wa: float=-1, simple_transfer=False, show_hist=False, show_trans=False) -> np.ndarray:
    """ Reshape the histogram of `Is` to be similar as `It` """
    if perc == 0: return Is

    # Convert images to CIELab D65 color space
    source_lab = cv2.cvtColor(Is, cv2.COLOR_BGR2Lab)
    target_lab = cv2.cvtColor(It, cv2.COLOR_BGR2Lab)
    del Is, It

    # Initalize
    Io = np.empty_like(source_lab)
    Bins = np.array([CH_RANGE[ic][1] - CH_RANGE[ic][0] for ic in range(3)])

    # Region selection
    mask = np.ones_like(source_lab[:, :, 0], dtype=int)
    th1 = Wa * (source_lab[:, :, 1].max() - source_lab[:, :, 1].min()) + np.min(source_lab[:, :, 1] - 128)
    th2 = Wa * (source_lab[:, :, 2].max() - source_lab[:, :, 2].min()) + np.min(source_lab[:, :, 2] - 128)
    mask[(abs(source_lab[:, :, 1] - 128) <= th1) | (abs(source_lab[:, :, 2] - 128) <= th2)] = 0

    # Compute Smax
    Smax = compute_Smax(Bins, B_MIN)

    # For each channel
    for ic in range(3):
        # Compute histograms
        Hs = histogram(source_lab[mask == 1, ic], CH_RANGE[ic], False)
        Ht = histogram(target_lab[:, :, ic], CH_RANGE[ic], False)

        levels = list(np.arange(STEP, perc/100 + STEP, STEP) * Smax[ic])
        for k in tqdm(levels):
            scale = round(k, 2)

            # Compute Bk
            Bk = int(Bins[ic] * pow(2, scale - Smax[ic]))

            # Down-sample and then up-sample
            Hs_k, Ht_k = resample_histogram(Hs, Bk, True), resample_histogram(Ht, Bk, True)
            Hs_k, Ht_k = resample_histogram(Hs_k, Bins[ic], True), resample_histogram(Ht_k, Bins[ic], True)

            # Region transfer 1
            Rmin_t = findpeaks(Ht_k)
            Hs_kp = np.zeros_like(Hs_k)
            for m in range(Rmin_t.shape[0] - 1):
                Hs_kp[Rmin_t[m]:Rmin_t[m+1]] = RegionTransfer(
                    Hs_k[Rmin_t[m]:Rmin_t[m+1]],
                    Ht_k[Rmin_t[m]:Rmin_t[m+1]],
                    scale / Smax[ic],
                    simple_transfer
                )

            # Region transfer 2
            Rmin_s = findpeaks(Hs_kp)
            Ho_k = np.zeros_like(Hs_k)
            for m in range(Rmin_s.shape[0] - 1):
                Ho_k[Rmin_s[m]:Rmin_s[m+1]] = RegionTransfer(
                    Hs_kp[Rmin_s[m]:Rmin_s[m+1]],
                    Ht_k[Rmin_s[m]:Rmin_s[m+1]],
                    scale / Smax[ic],
                    simple_transfer
                )

            # Output becomes next round's input
            Hs = Ho_k

        print(f">> (Channel = {ic}) histogram matching..")
        Hs = histogram(source_lab[mask == 1, ic], CH_RANGE[ic], False)
        Io[:, :, ic] = histogram_matching(
            source_lab[:, :, ic],
            Hs,
            Ho_k,
            CH_RANGE[ic],
            source_lab.shape[0] * source_lab.shape[1],
            mask,
            show_trans
        )

        # Plot histograms
        if show_hist and (CHANNEL not in [0, 1, 2] or CHANNEL == ic):
            plt.plot(Hs / Hs.sum(), label="source")
            plt.plot(Ht / Ht.sum(), label="target")
            plt.plot(Ho_k, label="output")
            plt.xlim(left=0, right=256), plt.ylim(top=0.3, bottom=0), plt.legend()
            plt.show(), cv2.waitKey(0)

    # Convert Io back to RGB color space
    output_img = cv2.cvtColor(Io, cv2.COLOR_Lab2BGR)
    return output_img


def findpeaks(Hist: np.ndarray) -> np.ndarray:
    """ Search Rmin using Eq.8 & Eq.9 """
    H_grad = grad(Hist)
    H_grad2 = grad(H_grad)

    Rmin = []
    for i in range(H_grad2.shape[0]):
        if H_grad[i]*H_grad[i+1] < 0 and H_grad2[i] > 0:
            Rmin.append(i+1)

    Rmin.insert(0, 0)
    Rmin.append(Hist.shape[0])
    return np.array(Rmin, dtype=int)


def grad(arr: np.ndarray) -> np.ndarray:
    """ Compute gradients (Eq.8) """
    grad = np.empty(arr.shape[0] - 1, dtype=arr.dtype)
    for i in range(arr.shape[0] - 1):
        grad[i] = arr[i] - arr[i+1]
    return grad


def RegionTransfer(Hs: np.ndarray, Ht: np.ndarray, wt: float, simple_transfer=False) -> np.ndarray:
    """ Region transfer function using adjusted Eqs.10~12 """
    ws = 1 - wt
    Hs_avg, Ht_avg = Hs.mean(), Ht.mean()
    Hs_std, Ht_std = Hs.std(), Ht.std()

    if simple_transfer:
        Ho = Hs * ws + Ht * wt
    else:
        Ho = Hs.copy()
        if round(Hs_std, 8) != 0:
            Ho = (Ho - Hs_avg) * (wt * Ht_std + ws * Hs_std) / Hs_std + wt * Ht_avg + ws * Hs_avg
        else:
            Ho = Ho - Hs_avg + wt * Ht_avg + ws * Hs_avg

    return Ho


def histogram_matching(Is: np.ndarray, Hs: np.ndarray, Ho: np.ndarray, value_range: tuple, pixel_count: int, Im: np.ndarray, show_trans=False) -> np.ndarray:
    """ Performing histogram matching """
    # Initialize
    Imin, Imax = value_range[0], value_range[1]
    Hs_sum, Ho_sum = Hs.sum(), Ho.sum()
    Hs = Hs * pixel_count / Hs_sum
    Ho = Ho * pixel_count / Ho_sum

    # Cumulative Histogram
    Ho, Hs = Ho.cumsum().round(), Hs.cumsum().round()

    # Below line: second & third params are "x-coord" and "y-coord corresponding to x-coord",
    #  the first param is sampling x-coords, which is used to get the corresponding y-coords.
    new_pix_value = np.interp(Hs, Ho, np.arange(Imin, Imax))

    Io, Im = Is.ravel(), Im.ravel()
    Io[Im == 1] = new_pix_value[Io[Im == 1]]
    Io = np.reshape(Io, Is.shape).round()

    if show_trans:  # Showing transfer function
        plt.plot(np.arange(Imin, Imax), new_pix_value, label="transfer function")
        plt.xlim(left=0, right=256), plt.ylim(top=256, bottom=0), plt.legend()
        plt.show(), cv2.waitKey(0)

    return Io.astype(dtype=np.uint8)


def contrast_modify(Is: np.ndarray, Io: np.ndarray, wc: float) -> np.ndarray:
    """ Smooth the image """
    Is, Io = Is.copy(), Io.copy()
    Is, Io = Is.astype(dtype=np.float32), Io.astype(dtype=np.float32)
    # d = Is.shape[0] // 4
    # Ires_s = cv2.bilateralFilter(Is, d, 75, 75)
    # Ires_o = cv2.bilateralFilter(Io, d, 75, 75)
    Ires_s = cv2.medianBlur(Is, 5)
    Ires_o = cv2.medianBlur(Io, 5)
    Io = Io + wc * (Ires_s - Ires_o)
    Io[Io < 0], Io[Io > 255] = 0, 255
    return Io.astype(dtype=np.uint8)


def full_reshaped_MAE(Is: np.ndarray, It: np.ndarray, simple_transfer=False) -> np.ndarray:
    """ Compute MAE of "100% reshaped histogram" & "matched histogram using histogram matching" """
    # Using paper's algorithm
    test = Reshape_Histogram(Is, It, 100, Wa=-1, simple_transfer=simple_transfer)
    
    # Normal histogram matching
    source_lab = cv2.cvtColor(Is, cv2.COLOR_BGR2Lab)
    target_lab = cv2.cvtColor(It, cv2.COLOR_BGR2Lab)
    Io = np.empty_like(source_lab)
    for ic in range(3):
        Hs = histogram(source_lab[:, :, ic], CH_RANGE[ic], False)
        Ht = histogram(target_lab[:, :, ic], CH_RANGE[ic], False)
        mask = np.ones_like(source_lab[:, :, ic])
        Io[:, :, ic] = histogram_matching(
            source_lab[:, :, ic], Hs, Ht, CH_RANGE[ic],
            source_lab.shape[0] * source_lab.shape[1], mask
        )
    truth = cv2.cvtColor(Io, cv2.COLOR_Lab2BGR)
    
    # Compute mean absolute error
    diff = test - truth
    mae = np.mean(abs(diff))
    return mae


if __name__ == "__main__":
    # Load source and target images
    source_img = np.asarray(cv2.imread(PATH_SRC, cv2.IMREAD_COLOR), dtype=np.uint8)
    target_img = np.asarray(cv2.imread(PATH_TAR, cv2.IMREAD_COLOR), dtype=np.uint8)

    for perc in PERC:
        print("\n=======================================================")
        print(f">> Start the case of perc = {perc}%")
        output_img = Reshape_Histogram(
            source_img,
            target_img,
            perc,
            W_A,
            simple_transfer=LINEAR,
            show_hist=SHOW_HIST,
            show_trans=SHOW_TRANSFER_FUNC
        )
        output_img = contrast_modify(source_img, output_img, 0.1)
        if W_A <= 0:
            cv2.imwrite(f"term_project/output/output_{NUM}_perc_{perc}.jpg", output_img)
        else:
            cv2.imwrite(f"term_project/output/output_{NUM}_perc_{perc}_wa_{W_A}.jpg", output_img)
        print(">> done")

    # MAE
    error = full_reshaped_MAE(source_img, target_img, LINEAR)
    print(f">> MAE of image with 100% reshaped histogram = {error}")
