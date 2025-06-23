import time
import torch
import numpy as np
from PIL import Image
import mss
from pynput.keyboard import Controller,Key
from sklearn.preprocessing import LabelEncoder
import os
from model_def import ActionCNN  # ⬅️ 從另一個檔案引入模型結構
# ---------- 模型相關 ----------

CONTROL_CHAR_MAP = {
    '\x03': 'c',  # Ctrl+C
    '\x1a': 'z',  # Ctrl+Z
    '\x18': 'x',  # Ctrl+X
    '\x08': 'Backspace',
    # 如果你知道還有哪些組合你會按，可以加進來
}

def clean_action(action: str) -> str:
    for ctrl_char, key_name in CONTROL_CHAR_MAP.items():
        action = action.replace(ctrl_char, key_name)
    return action.strip()

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

def press_combination(action_str):
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
            time.sleep(1)  # 按著持續一小段時間
    except Exception as e:
        print(f"⚠️ 無法模擬按鍵組合 {action_str}: {e}")

with mss.mss() as sct:
    monitor = sct.monitors[1]  # 主螢幕
    try:
        while True:
            img = sct.grab(monitor)
            img_pil = Image.frombytes("RGB", img.size, img.rgb)

            processed = preprocess_image(img_pil)
            x = np.array(processed) / 255.0
            x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 128, 128)

            temperature = 0.8  # 越小越貪婪，越大越隨機
            
            with torch.no_grad():
                output = model(x_tensor)
                probabilities = torch.softmax(output / temperature, dim=1)
                pred_idx = torch.multinomial(probabilities, num_samples=1).item()
                action = encoder.inverse_transform([pred_idx])[0]
                print("動作預測：", action)
                press_combination(action)
    
    except KeyboardInterrupt:
        print("停止推論")
