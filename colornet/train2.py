import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ColorNet


class ColorDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        # 0: Blue, 1: Yellow, 2: Green
        classes = {'blue': 0, 'yellow': 1, 'green': 2}

        for cls_name, cls_idx in classes.items():
            cls_dir = os.path.join(root_dir, cls_name)
            # ✅ 修复这里：os.path.exists
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith('.jpg'):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (94, 24))  # Same size as LPRNet
        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)
        return torch.tensor(img), torch.tensor(label, dtype=torch.long)


# ==================== 自动创建 exp1/exp2/exp3... 核心函数 ====================
def get_prev_exp_dir(base_dir='color_runs'):
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

def get_current_exp_dir(prev_num, base_dir='color_runs'):
    if prev_num is None:
        new_num = 1
    else:
        new_num = prev_num + 1
    new_exp_dir = os.path.join(base_dir, f'exp{new_num}')
    os.makedirs(new_exp_dir, exist_ok=True)
    print(f"📁 本次训练保存到: {new_exp_dir}")
    return new_exp_dir
# ============================================================================


def train_color_net(resume=False):
    BATCH_SIZE = 64
    EPOCHS = 200
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_DIR = 'color_runs'

    dataset_dir = './color_dataset2'
    if not os.path.exists(dataset_dir):
        print(f"错误: 找不到数据集 {dataset_dir}")
        return

    dataset = ColorDataset(dataset_dir)
    if len(dataset) == 0:
        print("数据集为空")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ColorNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ==================== 自动获取当前 exp 文件夹 ====================
    prev_exp_num = get_prev_exp_dir(BASE_DIR)
    exp_dir = get_current_exp_dir(prev_exp_num, BASE_DIR)

    checkpoint_path = os.path.join(exp_dir, "color_last.pth")
    best_model_path = os.path.join(exp_dir, "color_best.pth")
    # ================================================================

    start_epoch = 0

    # ==================== 自动加载上一轮最优权重 ====================
    if prev_exp_num is not None:
        prev_best = os.path.join(BASE_DIR, f'exp{prev_exp_num}', 'color_best.pth')
        if os.path.exists(prev_best):
            print(f"📌 自动加载上一轮模型: {prev_best}")
            state_dict = torch.load(prev_best, map_location=DEVICE)
            model.load_state_dict(state_dict)
    # ================================================================

    if resume:
        if os.path.exists(checkpoint_path):
            print(f"正在从 {checkpoint_path} 恢复训练...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                print(f"成功恢复! 将从第 {start_epoch + 1} 轮继续训练")
            except Exception as e:
                print(f"恢复失败: {e}, 将重新开始")

    print(f"开始训练颜色分类模型, 样本数: {len(dataset)}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        i = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if i % 10 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, Step {i}, Loss: {loss.item():.4f}")
            i += 1

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}, Acc: {100. * correct / total:.2f}%")

        # 保存
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        torch.save(model.state_dict(), best_model_path)
        print(f"模型已保存到 → {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='从上次中断的检查点恢复训练')
    args = parser.parse_args()
    train_color_net(resume=args.resume)