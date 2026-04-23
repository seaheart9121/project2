import cv2
import numpy as np
import os
from tqdm import tqdm


def augment_plate_color(img):
    """优化版：鲁棒的蓝→黄→绿转换"""
    # 1. 生成黄牌（反转+亮度调整，让黄色更真实）
    img_yellow = cv2.bitwise_not(img)
    img_yellow = cv2.convertScaleAbs(img_yellow, alpha=1.2, beta=10)  # 提升亮度

    # 2. 生成绿牌（色相偏移+饱和度增强，让绿色更鲜艳）
    hsv = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_new = (h.astype(int) + 45) % 180  # 黄色(30)→绿色(75)
    h_new = h_new.astype(np.uint8)
    s_new = cv2.convertScaleAbs(s, alpha=1.5)  # 提升饱和度
    hsv_green = cv2.merge([h_new, s_new, v])
    img_green = cv2.cvtColor(hsv_green, cv2.COLOR_HSV2BGR)

    return img_yellow, img_green


def is_blue_plate(img):
    """辅助判断：是否为蓝底车牌（B通道均值远大于G/R）"""
    h, w = img.shape[:2]
    center = img[h // 4:3 * h // 4, w // 4:3 * w // 4]  # 取中心区域（避免边缘干扰）
    b, g, r = cv2.split(center)
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    return b_mean > g_mean + 50 and b_mean > r_mean + 50  # 蓝牌B通道显著更高


def generate_augmented_data(src_dir, dst_dir, subset):
    """改进版：不需要train/val拆分、只处理蓝牌、修复文件名逻辑"""
    os.makedirs(os.path.join(dst_dir, 'yellow'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'green'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'blue'), exist_ok=True)

    print(f"生成增强数据: {src_dir} -> {dst_dir}")
    for filename in tqdm(os.listdir(src_dir)):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue
        if '_yellow_' in filename or '_green_' in filename:  # 跳过已增强的图
            continue

        img_path = os.path.join(src_dir, filename)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue

        # 只处理蓝牌，过滤非蓝牌（避免错误转换）
        if not is_blue_plate(img):
            continue

        # 1. 保存原始蓝牌
        blue_dst = os.path.join(dst_dir, 'blue', filename)
        cv2.imencode('.jpg', img)[1].tofile(blue_dst)

        # 2. 生成并保存黄牌、绿牌
        img_yellow, img_green = augment_plate_color(img)
        # 黄牌
        yellow_filename = f"yellow_{filename}"
        yellow_dst = os.path.join(dst_dir, 'yellow', yellow_filename)
        cv2.imencode('.jpg', img_yellow)[1].tofile(yellow_dst)
        # 绿牌
        green_filename = f"green_{filename}"
        green_dst = os.path.join(dst_dir, 'green', green_filename)
        cv2.imencode('.jpg', img_green)[1].tofile(green_dst)

        # 3. （可选）保存回lpr_dataset（用于LPRNet训练）
        parts = filename.split('_', 1)  # 兼容任意文件名格式（只拆分第一个下划线）
        if len(parts) == 2:
            label, rest = parts
            yellow_lpr = f"{label}_yellow_{rest}"
            green_lpr = f"{label}_green_{rest}"
        else:
            yellow_lpr = f"yellow_{filename}"
            green_lpr = f"green_{filename}"
        cv2.imencode('.jpg', img_yellow)[1].tofile(os.path.join(src_dir, yellow_lpr))
        cv2.imencode('.jpg', img_green)[1].tofile(os.path.join(src_dir, green_lpr))


if __name__ == "__main__":
    LPR_ROOT = "../lprnet/lpr_dataset5"  # 原始蓝牌数据源
    DST_DIR = "./color_dataset5"  # 按颜色分类的目标目录

    for subset in ['train', 'val']:
        src_dir = os.path.join(LPR_ROOT, subset)
        if os.path.exists(src_dir):
            print(f"\n===== 处理 {subset} 集 =====")
            generate_augmented_data(src_dir, DST_DIR, subset)
        else:
            print(f"\n跳过 {subset} 集（目录不存在: {src_dir}）")

    if not os.path.exists(LPR_ROOT):
        print(f"\n错误：未找到数据源 {LPR_ROOT}，请先运行 ../lprnet/ccpd_to_lpr.py")