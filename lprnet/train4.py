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

# 添加当前目录到path，以便直接运行
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LPRNet, CHARS
from dataset import LPRDataset, collate_fn


# 早停类（优化版，更严格）
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


# 1. 获取上一级EXP目录
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
        print(f"⚠️ 在 {base_dir} 中未找到任何exp文件夹，将从exp1开始。")
        return None
    else:
        return max(exp_folders)


# 2. 获取当前要创建的EXP目录
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
    # ===== 核心配置（优化版，解决级联过拟合）=====
    BATCH_SIZE = 8  # 小数据集用小batch，梯度更稳定
    EPOCHS = 30  # 减少轮数，防止过拟合
    INIT_LR = 0.0001  # 微调用低LR，不冲垮原有特征
    PATIENCE = 5  # 严格早停，快速终止过拟合
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_DIR = 'lpr_runs'

    # ===== 计算EXP序号 =====
    prev_exp_num = get_prev_exp_dir(BASE_DIR)
    exp_dir = get_current_exp_dir(prev_exp_num, BASE_DIR)

    # 路径设置
    checkpoint_path = os.path.join(exp_dir, 'lprnet_last.pth')
    best_model_path = os.path.join(exp_dir, 'lprnet_best.pth')
    log_path = os.path.join(exp_dir, f'{os.path.basename(exp_dir)}.txt')  # 自动保存训练日志

    # ===== 数据增强（新增，提升泛化能力）=====
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

    # ===== 数据集加载 =====
    train_dir = 'lpr_dataset/train'
    val_dir = 'lpr_dataset/val'
    for dir_path, dir_name in [(train_dir, "训练集"), (val_dir, "验证集")]:
        if not os.path.exists(dir_path):
            print(f"错误: 找不到{dir_name}目录 {dir_path}")
            return

    train_dataset = LPRDataset(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dataset = LPRDataset(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"✅ 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    # ===== 模型/损失/优化器/调度器（优化版）=====
    model = LPRNet().to(DEVICE)
    criterion = nn.CTCLoss(blank=len(CHARS) - 1, zero_infinity=True)
    # 微调用AdamW，权重衰减更强，防过拟合
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    start_epoch = 0
    best_val_loss = float('inf')
    weights_to_load = None

    # ===== 关键修复：自动加载上一级最优权重，仅加载主干 =====
    if prev_exp_num is not None:
        prev_best_weights = os.path.join(BASE_DIR, f'exp{prev_exp_num}', 'lprnet_best.pth')
        if os.path.exists(prev_best_weights):
            weights_to_load = prev_best_weights
        else:
            print(f"⚠️ 上级exp{prev_exp_num}未找到最优模型，使用随机初始化")
    else:
        # exp1用随机初始化
        print("📌 exp1 首次训练，使用随机初始化权重")

    # 加载权重（仅加载主干，不加载全模型，避免过拟合）
    if weights_to_load:
        print(f"📌 加载上一级权重: {weights_to_load}（仅加载主干特征，微调训练）")
        try:
            state_dict = torch.load(weights_to_load, map_location=DEVICE)
            # 仅加载主干网络权重，跳过分类头
            backbone_dict = {k: v for k, v in state_dict.items() if 'backbone' in k}
            model.load_state_dict(backbone_dict, strict=False)
            print(f"✅ 权重加载成功！仅微调分类头，防止过拟合")
        except Exception as e:
            print(f"❌ 权重加载失败: {e}, 使用随机初始化")

    print(f"🚀 开始训练 | 目标目录: {exp_dir} | 初始LR: {INIT_LR}")

    # ===== 训练日志保存 =====
    log_file = open(log_path, 'w', encoding='utf-8')
    log_file.write(f"训练配置: BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, INIT_LR={INIT_LR}\n")
    log_file.write(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}\n")
    log_file.write("=" * 50 + "\n")

    # ===== 训练+验证主循环 =====
    for epoch in range(start_epoch, EPOCHS):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        for i, (imgs, labels, lengths) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs).permute(1, 0, 2)  # CTC要求 [T, B, C]
            input_lengths = torch.full((imgs.size(0),), outputs.size(0), dtype=torch.long, device=DEVICE)

            loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            if i % 5 == 0:
                print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")

        # 验证阶段
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

        # 调度与早停
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"🔧 LR: {current_lr:.8f}")

        # 保存日志
        log_file.write(
            f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.8f}\n")

        if early_stopping(avg_val_loss):
            log_file.write(f"触发早停，停止训练\n")
            break

        # 保存模型
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': avg_val_loss
        }
        torch.save(checkpoint, checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🎉 保存最优模型! Loss: {best_val_loss:.4f}")

    # 训练结束
    log_file.write(f"\n训练完成! 最优验证Loss: {best_val_loss:.4f}\n")
    log_file.close()
    print(f"\n🏁 {exp_dir} 训练完成! 最优Loss: {best_val_loss:.4f}")
    print(f"📁 日志已保存到: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='从上次中断的检查点恢复训练')
    args = parser.parse_args()
    train_lpr(resume=args.resume)