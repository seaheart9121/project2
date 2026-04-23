import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys
import argparse
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LPRNet, CHARS
from dataset import LPRDataset, collate_fn


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose and self.counter > 0:
                print(f"⚠️ 早停计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("🛑 触发早停，停止训练")
        return self.early_stop


def get_prev_exp_dir(base_dir='lpr_runs'):
    os.makedirs(base_dir, exist_ok=True)
    exp_folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('exp'):
            try:
                exp_num = int(item.replace('exp', ''))
                exp_folders.append(exp_num)
            except ValueError:
                continue
    if not exp_folders:
        return None
    else:
        return max(exp_folders)


def get_current_exp_dir(prev_num, base_dir='lpr_runs'):
    if prev_num is None:
        new_num = 1
    else:
        new_num = prev_num + 1
    new_exp_dir = os.path.join(base_dir, f'exp{new_num}')
    os.makedirs(new_exp_dir, exist_ok=True)
    print(f"📁 本次训练保存到: {new_exp_dir}")
    return new_exp_dir


def train_lpr(resume=False):
    BATCH_SIZE = 8
    EPOCHS = 30
    INIT_LR = 0.0001
    PATIENCE = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_DIR = 'lpr_runs'
    DATASET_ROOT = 'lpr_dataset5'
    train_dir = os.path.join(DATASET_ROOT, 'train')
    val_dir = os.path.join(DATASET_ROOT, 'val')

    prev_exp_num = get_prev_exp_dir(BASE_DIR)
    exp_dir = get_current_exp_dir(prev_exp_num, BASE_DIR)

    checkpoint_path = os.path.join(exp_dir, 'lprnet_last.pth')
    best_model_path = os.path.join(exp_dir, 'lprnet_best.pth')
    log_path = os.path.join(exp_dir, f'{os.path.basename(exp_dir)}.txt')

    train_transform = A.Compose([
        A.Resize(height=24, width=94),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=24, width=94),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2()
    ])

    train_dataset = LPRDataset(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dataset = LPRDataset(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"✅ 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    model = LPRNet().to(DEVICE)
    criterion = nn.CTCLoss(blank=len(CHARS) - 1, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    start_epoch = 0
    best_val_loss = float('inf')
    weights_to_load = None

    if prev_exp_num is not None:
        prev_best_weights = os.path.join(BASE_DIR, f'exp{prev_exp_num}', 'lprnet_best.pth')
        if os.path.exists(prev_best_weights):
            weights_to_load = prev_best_weights

    # ==============================================
    # 🔥🔥🔥 这里是你唯一的致命错误！我已经改好了！
    # ==============================================
    if weights_to_load:
        print(f"📌 加载上一级权重: {weights_to_load}")
        try:
            state_dict = torch.load(weights_to_load, map_location=DEVICE)

            # 原来写的是 backbone → 错！
            # 现在改成 feature_extractor → 对！（和model.py完全一致）
            backbone_dict = {k: v for k, v in state_dict.items() if 'feature_extractor' in k}

            model.load_state_dict(backbone_dict, strict=False)
            print(f"✅ 权重加载成功！")
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")

    print(f"🚀 开始训练")
    log_file = open(log_path, 'w', encoding='utf-8')
    log_file.write(f"训练开始\n")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_train_loss = 0.0

        for i, (imgs, labels, lengths) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs).permute(1, 0, 2)
            input_lengths = torch.full(
                size=(imgs.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=DEVICE
            )
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels, lengths in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(imgs).permute(1, 0, 2)
                input_lengths = torch.full((imgs.size(0),), outputs.size(0), dtype=torch.long, device=DEVICE)
                loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if early_stopping(avg_val_loss):
            break

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🎉 保存最优模型!")

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train_lpr(resume=False)