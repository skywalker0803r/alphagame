import os
import time
import mss
import threading
from PIL import Image
from pynput import keyboard

SAVE_DIR = "data4"
os.makedirs(SAVE_DIR, exist_ok=True)

frame_count = 0
current_keys = set()
lock = threading.Lock()

# 預處理函數：轉灰階 + 調整大小
def preprocess_image(img_pil):
    img_gray = img_pil.convert("L")              # 單通道灰階
    img_resized = img_gray.resize((128, 128))    # 調整大小
    return img_resized

# 鍵盤事件監聽
def on_press(key):
    with lock:
        try:
            current_keys.add(key.char)
        except AttributeError:
            current_keys.add(str(key))

def on_release(key):
    with lock:
        try:
            current_keys.discard(key.char)
        except AttributeError:
            current_keys.discard(str(key))

# 開始鍵盤監聽
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# 擷取螢幕並處理
with mss.mss() as sct:
    monitor = sct.monitors[1]  # 主螢幕

    try:
        while True:
            frame_count += 1
            img = sct.grab(monitor)
            img_pil = Image.frombytes("RGB", img.size, img.rgb)

            # 進行預處理（灰階 + 縮放）
            processed_img = preprocess_image(img_pil)

            filename = f"frame_{frame_count:05d}"
            img_path = os.path.join(SAVE_DIR, f"{filename}.png")
            action_path = os.path.join(SAVE_DIR, f"{filename}.txt")

            processed_img.save(img_path)

            with lock:
                with open(action_path, "w") as f:
                    f.write(",".join(current_keys))

            print(f"Saved {filename} with keys: {current_keys}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("停止擷取")
