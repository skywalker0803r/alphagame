import time
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
import mss
from pynput.keyboard import Controller,Key
from sklearn.preprocessing import LabelEncoder
import os
from model_def import ActionCNN  # ⬅️ 從另一個檔案引入模型結構
from collections import deque, Counter
# ---------- 模型相關 ----------

# 控制字元對應表
CONTROL_CHAR_MAP = {
    '\x03': 'c',  # Ctrl+C
    '\x1a': 'z',  # Ctrl+Z
    '\x18': 'x',  # Ctrl+X
    '\x08': 'Backspace',
    # 如果你知道還有哪些組合你會按，可以加進來
}

# 清理動作字串
def clean_action(action: str) -> str:
    for ctrl_char, key_name in CONTROL_CHAR_MAP.items():
        action = action.replace(ctrl_char, key_name)
    return action.strip()

# 檢查動作是否合法
def is_valid_action(action: str) -> bool:
    keywords = [
        "Key.up", "Key.down", "Key.left", "Key.right",  # 上下左右
        "Key.shift", "Key.ctrl_l", "Key.alt_l", "z"     # 喝水/技能/跳
    ]
    return any(kw in action for kw in keywords)

# 載入類別
def load_classes_from_txt(data_dir="merged_data"):
    labels = set()
    for f in os.listdir(data_dir):
        if f.endswith(".txt"):
            with open(os.path.join(data_dir, f), "r") as file:
                raw = file.read().strip()
                action = clean_action(raw)
                if is_valid_action(action):
                    labels.add(action)
    return sorted(list(labels))

# 載入類別並編碼
class_names = load_classes_from_txt()
encoder = LabelEncoder()
encoder.fit(class_names)
num_classes = len(class_names)

# 建立並載入模型
model = ActionCNN(num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location="cuda:0"))
model.eval()

# ---------- 預處理函數 ----------
def preprocess_image(img_pil):
    img_gray = img_pil.convert("L")
    img_resized = img_gray.resize((128, 128))
    return img_resized

# ---------- 螢幕截圖 + 模型推論 ----------
keyboard_ctrl = Controller()

# 根據按鍵組合決定持續時間
def get_duration_for_action(action_str):
    if ',' in action_str:
        print("⚠️ 組合鍵:", action_str)
        return 0.25  # 組合鍵稍長一點
    elif "Key.left" in action_str or "Key.right" in action_str or "Key.up" in action_str or "Key.down" in action_str:
        print("⚠️ 移動鍵:", action_str)
        return 0.4  # 移動鍵持續稍久
    else:
        print("⚠️ 單擊鍵:", action_str)
        return 0.12  # 單擊鍵

# 模擬按鍵組合
def press_combination(action_str):
    duration = get_duration_for_action(action_str)
    keys = []
    for key in action_str.split(","):
        key = key.strip()
        if not key:
            continue
        if key.startswith("Key."):
            key_attr = key.split(".", 1)[1]
            real_key = getattr(Key, key_attr)
        else:
            real_key = key
        keys.append(real_key)

    try:
        # 同時按下（使用 pressed context manager）
        with keyboard_ctrl.pressed(*keys):
            time.sleep(duration)  # 按著持續一小段時間
    except Exception as e:
        print(f"⚠️ 無法模擬按鍵組合 {action_str}: {e}")

# ---------- 重複動作懲罰 ----------
action_history = deque(maxlen=10)  # 最近 10 次動作
def apply_history_penalty(probabilities, encoder, action_history, penalty_strength=1.0):
    probabilities = probabilities.clone()
    
    if not action_history:
        return probabilities

    counts = Counter(action_history)
    total = sum(counts.values())

    for action_str, count in counts.items():
        index = encoder.transform([action_str])[0]
        freq = count / total  # 出現頻率
        penalty = 1.0 - penalty_strength * freq
        penalty = max(penalty, 0.0)  # 確保懲罰不會小於 0
        probabilities[0, index] *= penalty

    probabilities /= probabilities.sum()
    return probabilities

# ---------- 主迴圈 ----------
with mss.mss() as sct:
    monitor = sct.monitors[1]  # 主螢幕
    try:
        while True:
            # 擷取螢幕
            img = sct.grab(monitor)
            img_pil = Image.frombytes("RGB", img.size, img.rgb)
            # 預處理圖片
            processed = preprocess_image(img_pil)
            x = np.array(processed) / 255.0
            x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 128, 128)

            temperature = 1  # 越小越貪婪，越大越隨機
            
            with torch.no_grad():
                # 模型推論
                output = model(x_tensor)
                probabilities = torch.softmax(output / temperature, dim=1)
                # 加入根據歷史的懲罰
                #probabilities = apply_history_penalty(probabilities, encoder, action_history)
                # 取樣
                # 使用 multinomial 來隨機選擇一個動作
                pred_idx = torch.multinomial(probabilities, num_samples=1).item()
                action = encoder.inverse_transform([pred_idx])[0]
                action_history.append(action)
                # 輸出結果
                press_combination(action)
    except KeyboardInterrupt:
        print("停止推論")
