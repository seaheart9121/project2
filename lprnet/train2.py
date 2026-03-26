# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import os
# import sys
# import argparse
#
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# from model import LPRNet, CHARS
# from dataset import LPRDataset, collate_fn
#
#
# def train_lpr(resume=False, weights=None):
#     # 配置
#     BATCH_SIZE = 32
#     EPOCHS = 50
#     LR = 0.001
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 数据集 - 新增验证集路径
#     train_dir = 'lpr_dataset/train'
#     val_dir = 'lpr_dataset/val'  # 新增：验证集目录
#
#     if not os.path.exists(train_dir):
#         print(f"错误: 找不到数据集 {train_dir}")
#         return
#
#     # 加载训练集
#     train_dataset = LPRDataset(train_dir)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
#
#     # 加载验证集（如果存在）
#     val_loader = None
#     if os.path.exists(val_dir):
#         val_dataset = LPRDataset(val_dir)
#         val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
#         print(f"✅ 加载验证集: {val_dir}, 共 {len(val_dataset)} 张图片")
#     else:
#         print(f"⚠️ 未找到验证集 {val_dir}, 将仅用训练loss保存模型")
#
#     model = LPRNet().to(DEVICE)
#     criterion = nn.CTCLoss(blank=len(CHARS) - 1, zero_infinity=True)
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#
#     start_epoch = 0
#     checkpoint_path = 'lpr_runs/lprnet_last.pth'
#
#     # 新增：初始化最优loss（无穷大）
#     best_loss = float('inf')
#
#     if resume:
#         if os.path.exists(checkpoint_path):
#             print(f"正在从 {checkpoint_path} 恢复训练...")
#             try:
#                 checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
#                 model.load_state_dict(checkpoint['state_dict'])
#                 optimizer.load_state_dict(checkpoint['optimizer'])
#                 start_epoch = checkpoint['epoch']
#                 # 恢复训练时，同步恢复best_loss（如果之前保存过）
#                 if 'best_loss' in checkpoint:
#                     best_loss = checkpoint['best_loss']
#                     print(f"成功恢复! 历史最优loss: {best_loss:.4f}, 将从第 {start_epoch + 1} 轮继续训练")
#                 else:
#                     print(f"成功恢复! 将从第 {start_epoch + 1} 轮继续训练（无历史最优loss）")
#             except Exception as e:
#                 print(f"恢复失败: {e}, 将重新开始")
#         else:
#             print(f"未找到检查点 {checkpoint_path}, 将重新开始")
#
#     if weights and os.path.exists(weights):
#         print(f"使用初始权重: {weights}")
#         model.load_state_dict(torch.load(weights, map_location=DEVICE))
#     elif weights:
#         print(f"⚠️ 指定的权重文件 {weights} 不存在，使用随机初始化权重")
#
#     print(f"开始训练LPRNet, 设备: {DEVICE}")
#     os.makedirs('lpr_runs', exist_ok=True)
#
#     for epoch in range(start_epoch, EPOCHS):
#         # ========== 训练阶段 ==========
#         model.train()
#         total_train_loss = 0
#
#         for i, (imgs, labels, lengths) in enumerate(train_loader):
#             imgs = imgs.to(DEVICE)
#             labels = labels.to(DEVICE)
#
#             optimizer.zero_grad()
#             outputs = model(imgs)  # [B, W, C]
#             outputs = outputs.permute(1, 0, 2)  # 适配CTC Loss [T, B, C]
#             input_lengths = torch.full(size=(imgs.size(0),), fill_value=outputs.size(0), dtype=torch.long)
#
#             loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
#             loss.backward()
#             optimizer.step()
#
#             total_train_loss += loss.item()
#
#             if i % 10 == 0:
#                 print(f"Epoch {epoch + 1}/{EPOCHS}, Step {i}, Loss: {loss.item():.4f}")
#
#         avg_train_loss = total_train_loss / len(train_loader)
#         print(f"Epoch {epoch + 1} | 训练平均Loss: {avg_train_loss:.4f}")
#
#         # ========== 验证阶段 ==========
#         avg_val_loss = None
#         if val_loader is not None:
#             model.eval()  # 切换到推理模式（关闭Dropout/BatchNorm）
#             total_val_loss = 0
#             with torch.no_grad():  # 禁用梯度计算，节省显存
#                 for imgs, labels, lengths in val_loader:
#                     imgs = imgs.to(DEVICE)
#                     labels = labels.to(DEVICE)
#
#                     outputs = model(imgs).permute(1, 0, 2)
#                     input_lengths = torch.full(size=(imgs.size(0),), fill_value=outputs.size(0), dtype=torch.long)
#                     loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
#                     total_val_loss += loss.item()
#
#             avg_val_loss = total_val_loss / len(val_loader)
#             print(f"Epoch {epoch + 1} | 验证平均Loss: {avg_val_loss:.4f}")
#
#         # ========== 保存模型 ==========
#         # 1. 保存检查点（用于恢复训练，每次都更）
#         checkpoint = {
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'loss': avg_train_loss,
#             'best_loss': best_loss  # 新增：保存最优loss，恢复训练时能用
#         }
#         torch.save(checkpoint, checkpoint_path)
#
#         # 2. 保存最优模型（仅当loss更优时）
#         # 优先用验证loss，无验证集则用训练loss
#         current_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
#         if current_loss < best_loss:
#             best_loss = current_loss  # 更新最优loss
#             torch.save(model.state_dict(), 'lpr_runs/lprnet_best.pth')
#             print(f"🎉 保存最优模型! 最新最优Loss: {best_loss:.4f}")
#         else:
#             print(f"⚠️ 本次Loss {current_loss:.4f} 未优于最优 {best_loss:.4f}, 不保存best模型")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, help='指定初始权重路径 (例如 lpr_runs/lprnet_best.pth)')
#     parser.add_argument('--resume', action='store_true', help='从上次中断的检查点恢复训练')
#     args = parser.parse_args()
#     train_lpr(resume=args.resume, weights=args.weights)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys
import argparse
import numpy as np

# 添加当前目录到path，以便直接运行
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LPRNet, CHARS
from dataset import LPRDataset, collate_fn


# 早停类：防止过拟合，验证Loss连续上升时停止训练
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience  # 容忍验证Loss上升的轮数
        self.min_delta = min_delta  # 最小变化值（小于此值视为无提升）
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ 早停计数器: {self.counter}/{self.patience} (验证Loss未下降)")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("🛑 触发早停，停止训练")
        return self.early_stop


def train_lpr(resume=False, weights=None):
    # ===== 1. 核心配置（适配你的原始参数）=====
    BATCH_SIZE = 32
    EPOCHS = 50
    INIT_LR = 0.001
    PATIENCE = 8  # 早停容忍轮数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== 2. 数据集加载（区分训练/验证集，适配你的路径）=====
    train_dir = 'lpr_dataset/train'
    val_dir = 'lpr_dataset/val'
    # 检查目录合法性
    for dir_path, dir_name in [(train_dir, "训练集"), (val_dir, "验证集")]:
        if not os.path.exists(dir_path):
            print(f"错误: 找不到{dir_name}目录 {dir_path}")
            return

    # 加载训练集
    train_dataset = LPRDataset(train_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0  # 兼容你的原始设置
    )
    # 加载验证集
    val_dataset = LPRDataset(val_dir)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    print(f"✅ 训练集: {len(train_dataset)} 张图片 | 验证集: {len(val_dataset)} 张图片")

    # ===== 3. 模型/损失/优化器/调度器初始化（修复verbose参数）=====
    model = LPRNet().to(DEVICE)
    # CTC Loss（和你原始代码一致）
    criterion = nn.CTCLoss(blank=len(CHARS) - 1, zero_infinity=True)
    # 优化器（加入权重衰减防过拟合，兼容你的原始设置）
    optimizer = optim.Adam(
        model.parameters(), lr=INIT_LR, weight_decay=1e-5  # L2正则
    )
    # 学习率调度器：移除verbose参数（适配低版本PyTorch）
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    # 早停实例
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    # ===== 4. 恢复训练/加载初始权重（鲁棒性优化，兼容你的逻辑）=====
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = 'lpr_runs/lprnet_last.pth'

    # 恢复训练逻辑（和你原始代码一致，增加鲁棒性）
    if resume and os.path.exists(checkpoint_path):
        print(f"📌 从 {checkpoint_path} 恢复训练...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # 兼容旧检查点（无scheduler和best_val_loss的情况）
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            start_epoch = checkpoint['epoch']
            print(f"✅ 恢复成功! 从第 {start_epoch + 1} 轮开始，最优验证Loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"❌ 恢复失败: {e}, 重新开始训练")
            start_epoch = 0
            best_val_loss = float('inf')
    elif resume:
        print(f"⚠️ 未找到检查点 {checkpoint_path}, 重新开始训练")

    # 加载初始权重（增加异常捕获，避免报错）
    model_path = weights if weights else None
    if model_path and os.path.exists(model_path):
        print(f"📌 加载初始权重: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except Exception as e:
            print(f"❌ 权重加载失败: {e}, 使用随机初始化")
    elif model_path:
        print(f"⚠️ 指定的权重文件 {model_path} 不存在，使用随机初始化")
    else:
        print("📌 使用随机初始化权重")

    # 创建保存目录（和你原始代码一致）
    os.makedirs('lpr_runs', exist_ok=True)
    print(f"🚀 开始训练 (设备: {DEVICE}) | 总轮数: {EPOCHS} | 初始LR: {INIT_LR}")

    # ===== 5. 训练+验证主循环（核心优化）=====
    for epoch in range(start_epoch, EPOCHS):
        # ---- 训练阶段（和你原始代码一致，增加梯度裁剪）----
        model.train()
        total_train_loss = 0.0
        for i, (imgs, labels, lengths) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)  # [B, W, C]
            outputs = outputs.permute(1, 0, 2)  # CTC要求 [T, B, C]
            input_lengths = torch.full(
                size=(imgs.size(0),), fill_value=outputs.size(0),
                dtype=torch.long, device=DEVICE
            )
            # 计算CTC Loss（和你原始代码一致）
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)

            # 反向传播+梯度裁剪（防止梯度爆炸）
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()

            # 打印步长日志（和你原始代码一致）
            if i % 10 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, Step {i}, Loss: {loss.item():.4f}")

        # 计算训练平均Loss
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | 训练平均Loss: {avg_train_loss:.4f}")

        # ---- 验证阶段（新增，无梯度计算）----
        model.eval()  # 切换评估模式（禁用Dropout/BatchNorm）
        total_val_loss = 0.0
        with torch.no_grad():  # 禁用梯度，节省显存
            for imgs, labels, lengths in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(imgs)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(imgs.size(0),), fill_value=outputs.size(0),
                    dtype=torch.long, device=DEVICE
                )
                loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
                total_val_loss += loss.item()

        # 计算验证平均Loss
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} | 验证平均Loss: {avg_val_loss:.4f}")

        # ===== 6. 优化策略执行 =====
        # 学习率调度（根据验证Loss调整）
        scheduler.step(avg_val_loss)
        # 打印学习率变化（替代原verbose参数）
        current_lr = optimizer.param_groups[0]['lr']
        print(f"🔧 当前学习率: {current_lr:.6f}")

        # 检查早停
        if early_stopping(avg_val_loss):
            break

        # ===== 7. 模型保存（区分最新/最优，兼容你的逻辑）=====
        # 保存最新检查点（含调度器状态，用于恢复训练）
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),  # 新增保存调度器
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, checkpoint_path)

        # 仅当验证Loss刷新最优时，保存best模型（修复你原始代码"每次都覆盖"的问题）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'lpr_runs/lprnet_best.pth')
            print(f"🎉 保存最优模型! 最新最优Loss: {best_val_loss:.4f}")
        else:
            print(f"⚠️ 本次Loss {avg_val_loss:.4f} 未优于最优 {best_val_loss:.4f}, 不保存best模型")

    # 训练结束提示
    print(f"\n🏁 训练完成! 最优验证Loss: {best_val_loss:.4f} | 最优模型路径: lpr_runs/lprnet_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='指定初始权重路径 (例如 lpr_runs/lprnet_best.pth)')
    parser.add_argument('--resume', action='store_true', help='从上次中断的检查点恢复训练')
    args = parser.parse_args()
    train_lpr(resume=args.resume, weights=args.weights)