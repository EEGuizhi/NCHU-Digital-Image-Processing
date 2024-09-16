# 4109061012 B.S.Chen

- 程式語言：
    Python

- 所需套件：
    opencv-python, numpy, matplotlib, tqdm

- 使用：
    開起"term_project.py"檔案即可，
    程式碼內容前方有settings可以調整：

    ```
    # Settings
    NUM = "01"
    PATH_SRC = f"term_project/source_{NUM}.jpg"
    PATH_TAR = f"term_project/target_{NUM}.jpg"

    LINEAR = True
    SHOW_HIST, SHOW_TRANSFER_FUNC = False, False
    CHANNEL = 1

    STEP = 0.1
    B_MIN = 70
    PERC = [10, 15, 20, 40, 60, 100]
    W_A = -1  # 0.15  # Mask (set to <= 0 to deactivate)
    ```

    - `NUM`: 方便選用已經編號的圖片來設定圖片路徑。
    - `PATH_SRC`, `PATH_TAR`: 圖片路徑，可以調整。
    - `LINEAR`: 是否使用專題報告中的式(4)做計算，設False將使用式(2)(3)做計算。
    - `SHOW_HIST`: 顯示每次轉移後的直方圖。
    - `CHANNEL`: 承上，顯示指定的channel。(0=L*, 1=a*, 2=b*)
    - `SHOW_TRANSFER_FUNC`: 顯示轉移的Transfer function
    - `STEP`: 每次 scale k 前進的大小，在報告中有說明，建議設0.1就好。
    - `B_MIN`: Bmin，建議設高於50或不要更動。
    - `PERC`: 希望輸出有哪些perc的成果，必須是list，最小0、最大100。
    - `W_A`: Region selection的變數，設-1表示沒有特別設mask，設0.15適用於白色老虎的圖像(NUM="03")。
