import os
import time
import mss  # 用來擷取螢幕畫面的套件
import threading  # 用於處理鍵盤監聽時的同步問題
import re  # 正則表達式，用來找出檔案編號
import numpy as np  # 數值與陣列處理
from PIL import Image  # 圖片格式轉換與處理
from pynput import keyboard  # 鍵盤監聽

# === 設定參數 ===
SAVE_DIR = "data_x3d"  # 資料儲存資料夾
os.makedirs(SAVE_DIR, exist_ok=True)  # 若資料夾不存在就自動建立

FRAMES_IN_SEQUENCE = 4  # 每個序列要收集幾張影格 (T)
CHANNELS = 3            # RGB 三通道 (C)
HEIGHT = 182            # 影格高度 (H)
WIDTH = 182             # 影格寬度 (W)

# === 全域變數 ===
current_keys = set()  # 記錄目前被按住的按鍵集合
lock = threading.Lock()  # 鎖，用來同步鍵盤輸入與主迴圈
frame_buffer = []  # 暫存影格序列用的 buffer

# === 檢查資料夾中目前的最大檔案編號，避免覆蓋舊檔案 ===
def get_latest_sequence_index(save_dir):
    pattern = re.compile(r"sequence_(\d{5})\.npy")
    max_index = -1
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        return 0
    for filename in os.listdir(save_dir):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            max_index = max(max_index, idx)
    return max_index + 1

# 初始化檔案編號計數器
sequence_count = get_latest_sequence_index(SAVE_DIR)

# === 圖像前處理：調整影像大小與格式 ===
def preprocess_image(img_pil):
    img_resized = img_pil.resize((WIDTH, HEIGHT))  # 縮放到統一尺寸
    return np.array(img_resized)  # 轉為 NumPy 陣列

# === 鍵盤事件處理 ===
def on_press(key):
    with lock:
        try:
            current_keys.add(key.char)  # 加入按下的字元鍵
        except AttributeError:
            current_keys.add(str(key))  # 特殊鍵（如 shift, ctrl）

def on_release(key):
    with lock:
        try:
            current_keys.discard(key.char)  # 移除放開的字元鍵
        except AttributeError:
            current_keys.discard(str(key))  # 特殊鍵

# 啟動鍵盤監聽器（背景執行）
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# === 主程式開始 ===
print("開始收集資料...")
print(f"儲存路徑為：'{SAVE_DIR}'，按 Ctrl+C 可停止程式。")

# 使用 mss 擷取螢幕畫面
with mss.mss() as sct:
    monitor = sct.monitors[1]  # 選擇主螢幕（index = 1）

    try:
        while True:
            # 擷取一張螢幕畫面
            img = sct.grab(monitor)
            img_pil = Image.frombytes("RGB", img.size, img.rgb)  # 轉為 PIL 圖片
            processed_frame = preprocess_image(img_pil)  # 前處理（縮放 + 轉陣列）
            frame_buffer.append(processed_frame)  # 加入影格序列

            # 若已收集滿一個序列
            if len(frame_buffer) == FRAMES_IN_SEQUENCE:
                # 取得目前的鍵盤輸入作為標籤
                with lock:
                    action_label = ",".join(sorted(list(current_keys)))  # 按鍵標籤（如 'a,s'）

                # 組合影格成 (T, H, W, C) → (C, T, H, W)
                sequence_data = np.stack(frame_buffer, axis=0)
                sequence_data = sequence_data.transpose(3, 0, 1, 2)  # PyTorch 格式

                # 組出儲存檔名
                sequence_filename = f"sequence_{sequence_count:05d}.npy"
                action_filename = f"action_{sequence_count:05d}.txt"
                
                sequence_path = os.path.join(SAVE_DIR, sequence_filename)
                action_path = os.path.join(SAVE_DIR, action_filename)

                # 儲存影格與對應標籤
                np.save(sequence_path, sequence_data)
                with open(action_path, "w") as f:
                    f.write(action_label)

                print(f"已儲存 {sequence_filename} 和 {action_filename}，標籤為：'{action_label}'")

                # 重置 buffer 與編號
                frame_buffer = []
                sequence_count += 1

            # 控制收集頻率（5 FPS ≈ 0.2 秒一張）
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n使用者中止，結束收集。")
    finally:
        listener.stop()  # 停止鍵盤監聽器

#現在我執行完這個程式碼了 寫一個check_data.py讓我可以觀察每個sequence_filename和action_filename
#sequence_filename作成影片撥放 將對應的action標註在右上角