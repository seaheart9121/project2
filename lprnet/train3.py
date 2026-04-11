# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import os
# import sys
# import argparse
# import numpy as np
#
# # 添加当前目录到path，以便直接运行
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# from model import LPRNet, CHARS
# from dataset import LPRDataset, collate_fn
#
#
# # 早停类：防止过拟合，验证Loss连续上升时停止训练
# class EarlyStopping:
#     def __init__(self, patience=5, min_delta=0.001, verbose=True):
#         self.patience = patience  # 容忍验证Loss上升的轮数
#         self.min_delta = min_delta  # 最小变化值（小于此值视为无提升）
#         self.verbose = verbose
#         self.counter = 0
#         self.best_loss = float('inf')
#         self.early_stop = False
#
#     def __call__(self, val_loss):
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0  # 重置计数器
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"⚠️ 早停计数器: {self.counter}/{self.patience} (验证Loss未下降)")
#             if self.counter >= self.patience:
#                 self.early_stop = True
#                 if self.verbose:
#                     print("🛑 触发早停，停止训练")
#         return self.early_stop
#
#
# # 新增：自动生成最新的exp文件夹（exp1/exp2/...）
# def get_latest_exp_dir(base_dir='lpr_runs'):
#     # 确保主目录存在
#     os.makedirs(base_dir, exist_ok=True)
#     # 遍历目录下所有exp开头的文件夹，提取序号
#     exp_folders = []
#     for item in os.listdir(base_dir):
#         item_path = os.path.join(base_dir, item)
#         if os.path.isdir(item_path) and item.startswith('exp'):
#             try:
#                 # 提取exp后的数字（如exp1 -> 1）
#                 exp_num = int(item.replace('exp', ''))
#                 exp_folders.append(exp_num)
#             except ValueError:
#                 continue  # 非数字后缀的exp文件夹忽略
#     # 生成下一个序号（无exp文件夹则从1开始）
#     if not exp_folders:
#         new_exp_num = 1
#     else:
#         new_exp_num = max(exp_folders) + 1
#     # 拼接新文件夹路径
#     new_exp_dir = os.path.join(base_dir, f'exp{new_exp_num}')
#     # 创建新文件夹
#     os.makedirs(new_exp_dir, exist_ok=True)
#     print(f"📁 本次训练文件将保存到: {new_exp_dir}")
#     return new_exp_dir
#
#
# def train_lpr(resume=False, weights=None):
#     # ===== 1. 核心配置（适配你的原始参数）=====
#     BATCH_SIZE = 32
#     EPOCHS = 50
#     INIT_LR = 0.001
#     PATIENCE = 8  # 早停容忍轮数
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # ===== 新增：获取本次训练的exp文件夹路径 =====
#     exp_dir = get_latest_exp_dir(base_dir='lpr_runs')
#
#     # ===== 2. 数据集加载（区分训练/验证集，适配你的路径）=====
#     train_dir = 'lpr_dataset2/train'
#     val_dir = 'lpr_dataset2/val'
#     # 检查目录合法性
#     for dir_path, dir_name in [(train_dir, "训练集"), (val_dir, "验证集")]:
#         if not os.path.exists(dir_path):
#             print(f"错误: 找不到{dir_name}目录 {dir_path}")
#             return
#
#     # 加载训练集
#     train_dataset = LPRDataset(train_dir)
#     train_loader = DataLoader(
#         train_dataset, batch_size=BATCH_SIZE, shuffle=True,
#         collate_fn=collate_fn, num_workers=0  # 兼容你的原始设置
#     )
#     # 加载验证集
#     val_dataset = LPRDataset(val_dir)
#     val_loader = DataLoader(
#         val_dataset, batch_size=BATCH_SIZE, shuffle=False,
#         collate_fn=collate_fn, num_workers=0
#     )
#     print(f"✅ 训练集: {len(train_dataset)} 张图片 | 验证集: {len(val_dataset)} 张图片")
#
#     # ===== 3. 模型/损失/优化器/调度器初始化（修复verbose参数）=====
#     model = LPRNet().to(DEVICE)
#     # CTC Loss（和你原始代码一致）
#     criterion = nn.CTCLoss(blank=len(CHARS) - 1, zero_infinity=True)
#     # 优化器（加入权重衰减防过拟合，兼容你的原始设置）
#     optimizer = optim.Adam(
#         model.parameters(), lr=INIT_LR, weight_decay=1e-5  # L2正则
#     )
#     # 学习率调度器
#     scheduler = ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
#     )
#     # 早停实例
#     early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
#
#     # ===== 4. 恢复训练/加载初始权重（鲁棒性优化，兼容你的逻辑）=====
#     start_epoch = 0
#     best_val_loss = float('inf')
#     # 修改：模型路径指向本次exp文件夹
#     checkpoint_path = os.path.join(exp_dir, 'lprnet_last.pth')
#     best_model_path = os.path.join(exp_dir, 'lprnet_best.pth')
#
#     # 恢复训练逻辑（和你原始代码一致，增加鲁棒性）
#     if resume and os.path.exists(checkpoint_path):
#         print(f"📌 从 {checkpoint_path} 恢复训练...")
#         try:
#             checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             # 兼容旧检查点（无scheduler和best_val_loss的情况）
#             if 'scheduler' in checkpoint:
#                 scheduler.load_state_dict(checkpoint['scheduler'])
#             if 'best_val_loss' in checkpoint:
#                 best_val_loss = checkpoint['best_val_loss']
#             start_epoch = checkpoint['epoch']
#             print(f"✅ 恢复成功! 从第 {start_epoch + 1} 轮开始，最优验证Loss: {best_val_loss:.4f}")
#         except Exception as e:
#             print(f"❌ 恢复失败: {e}, 重新开始训练")
#             start_epoch = 0
#             best_val_loss = float('inf')
#     elif resume:
#         print(f"⚠️ 未找到检查点 {checkpoint_path}, 重新开始训练")
#
#     # 加载初始权重（增加异常捕获，避免报错）
#     model_path = weights if weights else None
#     if model_path and os.path.exists(model_path):
#         print(f"📌 加载初始权重: {model_path}")
#         try:
#             model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#         except Exception as e:
#             print(f"❌ 权重加载失败: {e}, 使用随机初始化")
#     elif model_path:
#         print(f"⚠️ 指定的权重文件 {model_path} 不存在，使用随机初始化")
#     else:
#         print("📌 使用随机初始化权重")
#
#     print(f"🚀 开始训练 (设备: {DEVICE}) | 总轮数: {EPOCHS} | 初始LR: {INIT_LR}")
#
#     # ===== 5. 训练+验证主循环（核心优化）=====
#     for epoch in range(start_epoch, EPOCHS):
#         # ---- 训练阶段（和你原始代码一致，增加梯度裁剪）----
#         model.train()
#         total_train_loss = 0.0
#         for i, (imgs, labels, lengths) in enumerate(train_loader):
#             imgs = imgs.to(DEVICE)
#             labels = labels.to(DEVICE)
#
#             optimizer.zero_grad()
#             outputs = model(imgs)  # [B, W, C]
#             outputs = outputs.permute(1, 0, 2)  # CTC要求 [T, B, C]
#             input_lengths = torch.full(
#                 size=(imgs.size(0),), fill_value=outputs.size(0),
#                 dtype=torch.long, device=DEVICE
#             )
#             # 计算CTC Loss（和你原始代码一致）
#             loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
#
#             # 反向传播+梯度裁剪（防止梯度爆炸）
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()
#
#             total_train_loss += loss.item()
#
#             # 打印步长日志（和你原始代码一致）
#             if i % 10 == 0:
#                 print(f"Epoch {epoch + 1}/{EPOCHS}, Step {i}, Loss: {loss.item():.4f}")
#
#         # 计算训练平均Loss
#         avg_train_loss = total_train_loss / len(train_loader)
#         print(f"Epoch {epoch + 1} | 训练平均Loss: {avg_train_loss:.4f}")
#
#         # ---- 验证阶段（新增，无梯度计算）----
#         model.eval()  # 切换评估模式（禁用Dropout/BatchNorm）
#         total_val_loss = 0.0
#         with torch.no_grad():  # 禁用梯度，节省显存
#             for imgs, labels, lengths in val_loader:
#                 imgs = imgs.to(DEVICE)
#                 labels = labels.to(DEVICE)
#
#                 outputs = model(imgs)
#                 outputs = outputs.permute(1, 0, 2)
#                 input_lengths = torch.full(
#                     size=(imgs.size(0),), fill_value=outputs.size(0),
#                     dtype=torch.long, device=DEVICE
#                 )
#                 loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
#                 total_val_loss += loss.item()
#
#         # 计算验证平均Loss
#         avg_val_loss = total_val_loss / len(val_loader)
#         print(f"Epoch {epoch + 1} | 验证平均Loss: {avg_val_loss:.4f}")
#
#         # ===== 6. 优化策略执行 =====
#         # 学习率调度（根据验证Loss调整）
#         scheduler.step(avg_val_loss)
#         # 打印学习率变化（替代原verbose参数）
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f"🔧 当前学习率: {current_lr:.6f}")
#
#         # 检查早停
#         if early_stopping(avg_val_loss):
#             break
#
#         # ===== 7. 模型保存（修改：保存到本次exp文件夹）=====
#         # 保存最新检查点（含调度器状态，用于恢复训练）
#         checkpoint = {
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'scheduler': scheduler.state_dict(),  # 新增保存调度器
#             'train_loss': avg_train_loss,
#             'val_loss': avg_val_loss,
#             'best_val_loss': best_val_loss
#         }
#         torch.save(checkpoint, checkpoint_path)
#
#         # 仅当验证Loss刷新最优时，保存best模型
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), best_model_path)
#             print(f"🎉 保存最优模型! 最新最优Loss: {best_val_loss:.4f} (路径: {best_model_path})")
#         else:
#             print(f"⚠️ 本次Loss {avg_val_loss:.4f} 未优于最优 {best_val_loss:.4f}, 不保存best模型")
#
#     # 训练结束提示
#     print(f"\n🏁 训练完成!")
#     print(f"📊 最优验证Loss: {best_val_loss:.4f}")
#     print(f"📁 最优模型路径: {os.path.abspath(best_model_path)}")
#     print(f"📁 最新检查点路径: {os.path.abspath(checkpoint_path)}")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, help='指定初始权重路径 (例如 lpr_runs/exp1/lprnet_best.pth)')
#     parser.add_argument('--resume', action='store_true', help='从上次中断的检查点恢复训练（路径: 本次exp文件夹/lprnet_last.pth）')
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


# 早停类
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


# 1. 新增函数：获取上一级EXP目录
# 例如当前运行train3.py，base_dir是lpr_runs，它会寻找exp2
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
        # 返回最大的那个序号（即上一次训练的exp）
        return max(exp_folders)


# 2. 新增函数：获取当前要创建的EXP目录
# 例如上一级是exp2，这里就返回exp3
def get_current_exp_dir(prev_num, base_dir='lpr_runs'):
    if prev_num is None:
        new_num = 1
    else:
        new_num = prev_num + 1
    new_exp_dir = os.path.join(base_dir, f'exp{new_num}')
    os.makedirs(new_exp_dir, exist_ok=True)
    print(f"📁 本次训练保存到: {new_exp_dir}")
    return new_exp_dir


def train_lpr(resume=False, weights=None):
    BATCH_SIZE = 32
    EPOCHS = 50
    INIT_LR = 0.001
    PATIENCE = 8
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_DIR = 'lpr_runs'

    # ===== 核心逻辑：计算EXP序号 =====
    prev_exp_num = get_prev_exp_dir(BASE_DIR)
    exp_dir = get_current_exp_dir(prev_exp_num, BASE_DIR)

    # 路径设置
    checkpoint_path = os.path.join(exp_dir, 'lprnet_last.pth')
    best_model_path = os.path.join(exp_dir, 'lprnet_best.pth')

    # 数据集
    train_dir = 'lpr_dataset2/train'
    val_dir = 'lpr_dataset2/val'
    for dir_path, dir_name in [(train_dir, "训练集"), (val_dir, "验证集")]:
        if not os.path.exists(dir_path):
            print(f"错误: 找不到{dir_name}目录 {dir_path}")
            return

    train_dataset = LPRDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dataset = LPRDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"✅ 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    # 模型
    model = LPRNet().to(DEVICE)
    criterion = nn.CTCLoss(blank=len(CHARS) - 1, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    start_epoch = 0
    best_val_loss = float('inf')
    weights_to_load = None

    # ===== 关键修改：自动寻找上一级的最优模型 =====
    if prev_exp_num is not None:
        # 拼接上一级的最优模型路径 (如 lpr_runs/exp2/lprnet_best.pth)
        prev_best_weights = os.path.join(BASE_DIR, f'exp{prev_exp_num}', 'lprnet_best.pth')
        if os.path.exists(prev_best_weights):
            weights_to_load = prev_best_weights
        else:
            print(f"⚠️ 上级exp{prev_exp_num}未找到最优模型，尝试加载最后检查点...")
            prev_last_weights = os.path.join(BASE_DIR, f'exp{prev_exp_num}', 'lprnet_last.pth')
            if os.path.exists(prev_last_weights):
                weights_to_load = prev_last_weights
            else:
                print(f"⚠️ 均未找到，使用随机初始化权重。")
    else:
        # 如果是第一次运行(只有exp1)，就加载exp1的
        if os.path.exists(os.path.join(BASE_DIR, 'exp1', 'lprnet_best.pth')):
            weights_to_load = os.path.join(BASE_DIR, 'exp1', 'lprnet_best.pth')

    # 加载权重
    if weights_to_load:
        print(f"📌 加载上一级权重: {weights_to_load}")
        try:
            # 加载权重但不加载optimizer状态（重新训练）
            state_dict = torch.load(weights_to_load, map_location=DEVICE)
            # 如果是checkpoint文件，提取state_dict
            if 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            print(f"✅ 权重加载成功!")
        except Exception as e:
            print(f"❌ 权重加载失败: {e}, 使用随机初始化")

    print(f"🚀 开始训练 | 目标目录: {exp_dir}")

    # 训练循环
    for epoch in range(start_epoch, EPOCHS):
        # 训练
        model.train()
        total_train_loss = 0.0
        for i, (imgs, labels, lengths) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs).permute(1, 0, 2)
            input_lengths = torch.full((imgs.size(0),), outputs.size(0), dtype=torch.long, device=DEVICE)

            loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")

        # 验证
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
        print(f"🔧 LR: {current_lr:.6f}")

        if early_stopping(avg_val_loss):
            break

        # 保存
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

    print(f"\n🏁 {exp_dir} 训练完成! 最优Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    # 运行train3.py，它会自动处理exp1->exp2->exp3的继承
    train_lpr()