# import torch
# import cv2
# import numpy as np
# import os
# import sys
#
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from model import ColorNet
# from train2 import ColorDataset  # 直接用你的数据集类
#
# CLASSES = ['蓝色', '黄色', '绿色']
#
# # 测试两个模型
# MODEL_PATHS = [
#     "./color_runs/exp1/color_best.pth",
#     "./color_runs/exp2/color_best.pth"
# ]
#
# TEST_DATASET = "./color_dataset2"  # 你的测试集路径
#
#
# def test_model_accuracy(model_path, test_loader, device):
#     print(f"\n=====================================")
#     print(f"正在测试模型：{model_path}")
#     print(f"=====================================")
#
#     # 加载模型
#     model = ColorNet().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for imgs, labels in test_loader:
#             imgs = imgs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(imgs)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#
#     acc = 100.0 * correct / total
#     print(f"✅ 测试完成 | 正确：{correct}/{total} | 准确率：{acc:.2f}%")
#     return acc
#
#
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 加载测试集
#     test_dataset = ColorDataset(TEST_DATASET)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
#     print(f"测试集总数：{len(test_dataset)} 张")
#
#     # 测试两个模型
#     acc1 = test_model_accuracy(MODEL_PATHS[0], test_loader, device)
#     acc2 = test_model_accuracy(MODEL_PATHS[1], test_loader, device)
#
#     # 最终对比
#     print("\n" + "="*50)
#     print(f"📊 最终对比结果")
#     print(f"exp1 模型准确率：{acc1:.2f}%")
#     print(f"exp2 模型准确率：{acc2:.2f}%")
#     print("="*50)
#
#     if acc1 > acc2:
#         print("\n🏆 获胜者：exp1 模型更好！")
#     elif acc2 > acc1:
#         print("\n🏆 获胜者：exp2 模型更好！")
#     else:
#         print("\n🤝 两个模型效果一样！")

import torch
import cv2
import numpy as np
import os
import sys
from typing import Union  # 新增类型导入

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ColorNet
from train import ColorDataset  # 直接用你的数据集类

# 类型别名：统一路径类型
PathType = Union[str, os.PathLike[str]]

CLASSES = ['蓝色', '黄色', '绿色']

# 测试两个模型
MODEL_PATHS: list[PathType] = [
    "./color_runs/exp1/color_best.pth",
    "./color_runs/exp2/color_best.pth"
]

TEST_DATASET: PathType = "./color_dataset2"  # 你的测试集路径


def test_model_accuracy(model_path: PathType, test_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    print(f"\n=====================================")
    print(f"正在测试模型：{model_path}")
    print(f"=====================================")

    # 加载模型（确保路径为str）
    model = ColorNet().to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print(f"✅ 测试完成 | 正确：{correct}/{total} | 准确率：{acc:.2f}%")
    return acc


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试集
    test_dataset = ColorDataset(TEST_DATASET)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试集总数：{len(test_dataset)} 张")

    # 测试两个模型
    acc1 = test_model_accuracy(MODEL_PATHS[0], test_loader, device)
    acc2 = test_model_accuracy(MODEL_PATHS[1], test_loader, device)

    # 最终对比
    print("\n" + "="*50)
    print(f"📊 最终对比结果")
    print(f"exp1 模型准确率：{acc1:.2f}%")
    print(f"exp2 模型准确率：{acc2:.2f}%")
    print("="*50)

    if acc1 > acc2:
        print("\n🏆 获胜者：exp1 模型更好！")
    elif acc2 > acc1:
        print("\n🏆 获胜者：exp2 模型更好！")
    else:
        print("\n🤝 两个模型效果一样！")