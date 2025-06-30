import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1. 資料集載入
DATA_DIR = "data_x3d"
sequences = []
labels = []

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".npy"):
        npy_path = os.path.join(DATA_DIR, fname)
        base_name = os.path.splitext(fname)[0].replace("sequence", "action")
        txt_path = os.path.join(DATA_DIR, base_name + ".txt")
        
        if not os.path.exists(txt_path):
            print(f"跳過：找不到對應標籤 {txt_path}")
            continue

        data = np.load(npy_path)
        with open(txt_path, "r") as f:
            label_text = f.read().strip()
        label_tokens = label_text.split(",") if label_text else []

        sequences.append(data)
        labels.append(label_tokens)

# 2. 多標籤 binarizer
mlb = MultiLabelBinarizer()
label_array = mlb.fit_transform(labels)
print("所有類別:", mlb.classes_)

# 3. 訓練/測試集拆分
X_train, X_test, y_train, y_test = train_test_split(
    sequences, label_array, test_size=0.2, random_state=42
)

# 4. Dataset & DataLoader
class X3DDataset(Dataset):
    def __init__(self, X, y):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.y = [torch.tensor(l, dtype=torch.float32) for l in y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(X3DDataset(X_train, y_train), batch_size=8, shuffle=True)
test_loader = DataLoader(X3DDataset(X_test, y_test), batch_size=8, shuffle=False)

# 5. X3D 簡易版模型 (你可替換為更完整 X3D)
class SimpleX3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
            nn.Sigmoid(),  # 適用於 multi-label
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleX3D(num_classes=len(mlb.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# 6. 訓練模型
for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f}")

# 7. 評估
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        preds = preds.cpu().numpy()
        y_batch = y_batch.numpy()

        y_true.extend(y_batch)
        y_pred.extend((preds > 0.5).astype(int))

# 8. 分類報告
print("\n=== 分類報告 ===")
print(classification_report(y_true, y_pred, target_names=mlb.classes_))

# 9. 每一類別的混淆矩陣
mcm = multilabel_confusion_matrix(y_true, y_pred)
for i, label in enumerate(mlb.classes_):
    tn, fp, fn, tp = mcm[i].ravel()
    print(f"\n【{label}】")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")