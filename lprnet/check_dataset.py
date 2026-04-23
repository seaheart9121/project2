# 新增数据校验脚本：check_dataset.py
import os
import cv2
import numpy as np
from model import CHARS

def check_lpr_dataset(data_dir):
    char_set = set(CHARS)
    error_count = 0
    total = 0
    for subset in ['train', 'val']:
        subset_dir = os.path.join(data_dir, subset)
        for filename in os.listdir(subset_dir):
            if not filename.endswith('.jpg'):
                continue
            total += 1
            # 1. 校验标签解析
            label_str = filename.split('_')[0]
            for c in label_str:
                if c not in char_set:
                    print(f"❌ 非法字符: {c} in {filename}")
                    error_count += 1
                    break
            # 2. 校验图片尺寸（必须是94x24）
            img_path = os.path.join(subset_dir, filename)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img.shape[:2] != (24, 94):
                print(f"❌ 尺寸错误: {img.shape} in {filename}")
                error_count += 1
    print(f"\n📊 校验完成：总数{total}，错误数{error_count}")
    return error_count == 0

if __name__ == "__main__":
    check_lpr_dataset("lpr_dataset2")  # 替换为你的数据集目录