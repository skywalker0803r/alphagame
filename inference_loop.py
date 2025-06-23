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
# 載入類別
def load_classes_from_txt(data_dir="data"):
    labels = set()
    for f in os.listdir(data_dir):
        if f.endswith(".txt"):
            with open(os.path.join(data_dir, f), "r") as file:
                action = file.read().strip()
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
with mss.mss() as sct:
    monitor = sct.monitors[1]  # 主螢幕
    try:
        while True:
            img = sct.grab(monitor)
            img_pil = Image.frombytes("RGB", img.size, img.rgb)

            processed = preprocess_image(img_pil)
            x = np.array(processed) / 255.0
            x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 128, 128)

            with torch.no_grad():
                output = model(x_tensor)
                pred_idx = output.argmax(dim=1).item()
                action = encoder.inverse_transform([pred_idx])[0]
                print("動作預測：", action)

                # 模擬按鍵
                for key in action.split(","):
                    key = key.strip()
                    if key == "":
                        continue
                    try:
                        # 特殊鍵處理，如 Key.space、Key.left
                        if key.startswith("Key."):
                            key_attr = key.split(".", 1)[1]
                            real_key = getattr(Key, key_attr)
                        else:
                            real_key = key  # 普通字元
                        keyboard_ctrl.press(real_key)
                        time.sleep(0.2)
                        keyboard_ctrl.release(real_key)
                    except Exception as e:
                        print(f"⚠️ 無法按鍵 {key}: {e}")
    except KeyboardInterrupt:
        print("停止推論")
