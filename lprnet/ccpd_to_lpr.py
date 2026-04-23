import os
import cv2
import numpy as np
from tqdm import tqdm

CHARS = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O",
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']

PROVINCE_IDX_MAP = list(range(34))
ADS_IDX_MAP = [34 + i for i in range(26)] + [59 + i for i in range(10)] + [33]


def four_point_transform(image, pts):
    width = 94
    height = 24
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (width, height))


def safe_split(s):
    return s.replace(',', '&').split('&')


def convert_ccpd_to_lpr(ccpd_root_list, save_root):
    os.makedirs(os.path.join(save_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'val'), exist_ok=True)

    image_files = []
    # 【修改点】循环遍历每一个根目录
    for ccpd_root in ccpd_root_list:
        if not os.path.exists(ccpd_root):
            print(f"⚠️ 路径不存在，跳过：{ccpd_root}")
            continue
        for root, _, files in os.walk(ccpd_root):
            for file in files:
                if file.endswith('.jpg'):
                    image_files.append(os.path.join(root, file))
    # image_files = []
    # for root, _, files in os.walk(ccpd_root):
    #     for file in files:
    #         if file.endswith('.jpg'):
    #             image_files.append(os.path.join(root, file))

    split = int(len(image_files) * 0.9)
    train_files = image_files[:split]
    val_files = image_files[split:]

    def process(files, subset):
        for file_path in tqdm(files, desc=f"{subset}"):
            try:
                filename = os.path.basename(file_path)
                parts = filename.split('-')
                if len(parts) < 5:
                    continue

                plate_indices = parts[4].split('_')

                # ====== 解析标签 ======
                province_char = CHARS[PROVINCE_IDX_MAP[int(plate_indices[0])]]
                plate_str = province_char

                for idx_str in plate_indices[1:]:
                    idx = int(idx_str)
                    plate_str += CHARS[ADS_IDX_MAP[idx]]

                # ====== ❗关键修改：不补 '-' ======
                plate_str = plate_str[:7]

                # ====== 解析坐标 ======
                pts = []
                for v in parts[3].split('_'):
                    x, y = safe_split(v)
                    pts.append([int(x), int(y)])

                pts = np.array([pts[2], pts[3], pts[0], pts[1]], dtype="float32")

                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                crop = four_point_transform(img, pts)

                save_name = f"{plate_str}_{os.path.splitext(filename)[0]}.jpg"
                save_path = os.path.join(save_root, subset, save_name)

                cv2.imencode('.jpg', crop)[1].tofile(save_path)

            except:
                continue

    process(train_files, 'train')
    process(val_files, 'val')

# if __name__ == "__main__":
#     convert_ccpd_to_lpr("../qwe", "lpr_dataset1")
if __name__ == "__main__":
    # 【修改点】把两个路径放在一个列表里传进去
    convert_ccpd_to_lpr(ccpd_root_list=["../qwertyu", "../zxcvbnm"], save_root="lpr_dataset5")