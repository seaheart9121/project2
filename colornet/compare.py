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

# 测试模型
MODEL_PATHS: list[PathType] = [
    "./color_runs/exp1/color_best.pth",
    "./color_runs/exp2/color_best.pth",
    "./color_runs/exp3/color_best.pth",
    "./color_runs/exp4/color_best.pth",
    "./color_runs/exp5/color_best.pth"
]

TEST_DATASET: PathType = "./color_dataset5"  # 你的测试集路径


def test_model_accuracy(model_path: PathType, test_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    print(f"\n=====================================")
    print(f"正在测试模型：{model_path}")
    print(f"=======================================")

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

    # 测试多个模型
    acc1 = test_model_accuracy(MODEL_PATHS[0], test_loader, device)
    acc2 = test_model_accuracy(MODEL_PATHS[1], test_loader, device)
    acc3 = test_model_accuracy(MODEL_PATHS[2], test_loader, device)
    acc4 = test_model_accuracy(MODEL_PATHS[3], test_loader, device)
    acc5 = test_model_accuracy(MODEL_PATHS[4], test_loader, device)

    model_accs = []
    for idx, model_path in enumerate(MODEL_PATHS):
        exp_name = f"exp{idx + 1}"
        acc = test_model_accuracy(model_path, test_loader, device)
        model_accs.append((exp_name, acc))

    print("\n" + "="*50)
    print(f"📊 最终对比结果")
    for exp_name, acc in model_accs:
        print(f"{exp_name} 模型准确率：{acc:.2f}%")
    print("="*50)

    # 找出最高准确率 + 对应的模型
    max_acc = max([acc for _, acc in model_accs])
    best_models = [exp_name for exp_name, acc in model_accs if acc == max_acc]

    # 输出最优结果（支持单个/多个并列最优）
    if len(best_models) == 1:
        print(f"\n🏆 获胜者：{best_models[0]} 模型更好！（准确率：{max_acc:.2f}%）")
    else:
        best_models_str = "、".join(best_models)
        print(f"\n🏆 并列最优：{best_models_str} 模型准确率相同！（准确率：{max_acc:.2f}%）")