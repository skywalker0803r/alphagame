import os
import numpy as np
import cv2  # OpenCV 用於影像顯示
import time

# === 設定資料路徑 ===
DATA_DIR = "data_x3d"

# 取得所有序列檔案，排序方便對應
sequence_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("sequence_") and f.endswith(".npy")])

print(f"共找到 {len(sequence_files)} 筆資料，按任意鍵播放，Esc 結束。")

for sequence_file in sequence_files:
    # 對應的標籤檔名
    index = sequence_file.split("_")[1].split(".")[0]
    action_file = f"action_{index}.txt"

    sequence_path = os.path.join(DATA_DIR, sequence_file)
    action_path = os.path.join(DATA_DIR, action_file)

    # 讀取序列資料 (C, T, H, W)
    data = np.load(sequence_path)  # shape: (3, 4, 182, 182)
    C, T, H, W = data.shape

    # 轉為 (T, H, W, C)，OpenCV 讀得懂
    frames = data.transpose(1, 2, 3, 0)

    # 讀取標籤
    if os.path.exists(action_path):
        with open(action_path, "r") as f:
            action_label = f.read().strip()
    else:
        action_label = "N/A"

    # 播放序列影格
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # PIL/np是RGB, OpenCV是BGR

        # 在右上角加上標籤
        cv2.putText(
            frame_bgr,
            f"Action: {action_label}",
            (5, 15),                   # 往下移一點點，避免被切到
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,                      # 字變小
            (0, 255, 0),
            1,                        # 線條也變細
            cv2.LINE_AA,
        )

        cv2.imshow("Sequence Viewer", cv2.resize(frame_bgr, (frame_bgr.shape[1] * 5, frame_bgr.shape[0] * 5)))
        key = cv2.waitKey(200)  # 每張顯示 200 毫秒（= 5 FPS）
        if key == 27:  # 按 ESC 中止
            break

    # 等使用者按鍵後再繼續下一筆
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()
