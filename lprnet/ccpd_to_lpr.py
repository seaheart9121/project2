import os
import cv2
import numpy as np
from tqdm import tqdm

# CCPD 字符映射表
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def four_point_transform(image, pts):
    width = 94  # LPRNet 输入宽
    height = 24 # LPRNet 输入高
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def convert_ccpd_to_lpr(ccpd_root, save_root):
    os.makedirs(os.path.join(save_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'val'), exist_ok=True)

    image_files = []
    for root, _, files in os.walk(ccpd_root):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))

    # 简单划分
    split = int(len(image_files) * 0.9)
    train_files = image_files[:split]
    val_files = image_files[split:]

    def process(files, subset):
        for file_path in tqdm(files, desc=f"Processing {subset}"):
            try:
                filename = os.path.basename(file_path)
                parts = filename.split('-')
                if len(parts) < 5: continue

                # 解析车牌号
                plate_indices = parts[4].split('_')
                plate_str = PROVINCES[int(plate_indices[0])] + ADS[int(plate_indices[1])]
                for idx in plate_indices[2:]:
                    plate_str += ADS[int(idx)]

                # 解析Vertices进行矫正
                vertices_str = parts[3]
                v_coords = vertices_str.split('_')
                pts = []
                for v in v_coords:
                    vx, vy = v.split('&')
                    pts.append([int(vx), int(vy)])

                # CCPD: BR, BL, TL, TR -> TL, TR, BR, BL
                pts = np.array([pts[2], pts[3], pts[0], pts[1]], dtype="float32")

                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None: continue

                crop = four_point_transform(img, pts)

                # 保存: 文件夹/车牌号.jpg
                # 注意: 车牌号可能包含中文，Windows下文件名可能乱码，建议使用索引或确保编码
                # 这里直接用车牌号作为文件名
                save_name = f"{plate_str}_{os.path.splitext(filename)[0]}.jpg"
                save_path = os.path.join(save_root, subset, save_name)

                # cv2.imwrite不支持中文路径，使用imencode
                cv2.imencode('.jpg', crop)[1].tofile(save_path)

            except Exception as e:
                continue

    process(train_files, 'train')
    process(val_files, 'val')

if __name__ == "__main__":
    CCPD_ROOT = "../zxcv" # 假设在上级目录
    SAVE_ROOT = "lpr_dataset"
    if os.path.exists(CCPD_ROOT):
        convert_ccpd_to_lpr(CCPD_ROOT, SAVE_ROOT)

# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
#
# # CCPD 字符映射表
# PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
#              "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
# ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
#        'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
#
#
# def four_point_transform(image, pts):
#     width = 94  # LPRNet 输入宽
#     height = 24  # LPRNet 输入高
#     dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
#     M = cv2.getPerspectiveTransform(pts, dst)
#     warped = cv2.warpPerspective(image, M, (width, height))
#     return warped
#
#
# def convert_ccpd_to_lpr(ccpd_root, save_root):
#     # 确保保存目录存在
#     os.makedirs(os.path.join(save_root, 'train'), exist_ok=True)
#     os.makedirs(os.path.join(save_root, 'val'), exist_ok=True)
#
#     # 1. 遍历所有jpg文件，打印总数（排查是否找到文件）
#     image_files = []
#     for root, _, files in os.walk(ccpd_root):
#         for file in files:
#             if file.endswith('.jpg'):
#                 image_files.append(os.path.join(root, file))
#     print(f"\n【调试】找到CCPD图片总数：{len(image_files)}")
#     if len(image_files) == 0:
#         print("❌ 未找到任何.jpg文件，请检查CCPD_ROOT路径是否正确！")
#         return
#
#     # 简单划分训练/验证集
#     split = int(len(image_files) * 0.9)
#     train_files = image_files[:split]
#     val_files = image_files[split:]
#     print(f"【调试】训练集数量：{len(train_files)} | 验证集数量：{len(val_files)}")
#
#     # 统计成功/失败数
#     success_count = 0
#     fail_count = 0
#     fail_reasons = []
#
#     def process(files, subset):
#         nonlocal success_count, fail_count, fail_reasons
#         for file_path in tqdm(files, desc=f"Processing {subset}"):
#             try:
#                 filename = os.path.basename(file_path)
#                 parts = filename.split('-')
#
#                 # 2. 检查文件名分割后长度（CCPD命名规则：xxx-xxx-xxx-xxx-xxx-xxx）
#                 if len(parts) < 5:
#                     fail_reasons.append(f"文件名格式错误：{filename}（分割后仅{len(parts)}部分）")
#                     fail_count += 1
#                     continue
#
#                 # 3. 解析车牌号（关键：检查索引是否越界）
#                 plate_indices = parts[4].split('_')
#                 if len(plate_indices) < 7:  # 车牌号至少7位（省+字母+5位）
#                     fail_reasons.append(f"车牌号索引不足：{filename}（仅{len(plate_indices)}位）")
#                     fail_count += 1
#                     continue
#                 # 检查索引是否在映射表范围内
#                 if int(plate_indices[0]) >= len(PROVINCES):
#                     fail_reasons.append(f"省份索引越界：{filename}（索引{plate_indices[0]} ≥ {len(PROVINCES)}）")
#                     fail_count += 1
#                     continue
#                 for idx in plate_indices[1:]:
#                     if int(idx) >= len(ADS):
#                         fail_reasons.append(f"车牌号索引越界：{filename}（索引{idx} ≥ {len(ADS)}）")
#                         fail_count += 1
#                         continue
#                 # 拼接车牌号
#                 plate_str = PROVINCES[int(plate_indices[0])] + ADS[int(plate_indices[1])]
#                 for idx in plate_indices[2:]:
#                     plate_str += ADS[int(idx)]
#
#                 # 4. 解析矫正坐标
#                 vertices_str = parts[3]
#                 v_coords = vertices_str.split('_')
#                 if len(v_coords) != 4:  # CCPD坐标是4个点
#                     fail_reasons.append(f"坐标数量错误：{filename}（仅{len(v_coords)}个坐标点）")
#                     fail_count += 1
#                     continue
#                 pts = []
#                 for v in v_coords:
#                     if '&' not in v:
#                         fail_reasons.append(f"坐标格式错误：{filename}（坐标{v}无&分隔）")
#                         fail_count += 1
#                         continue
#                     vx, vy = v.split('&')
#                     pts.append([int(vx), int(vy)])
#                 # 坐标格式转换（CCPD: BR, BL, TL, TR -> TL, TR, BR, BL）
#                 pts = np.array([pts[2], pts[3], pts[0], pts[1]], dtype="float32")
#
#                 # 5. 读取图片（支持中文路径）
#                 img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
#                 if img is None:
#                     fail_reasons.append(f"图片读取失败：{file_path}")
#                     fail_count += 1
#                     continue
#
#                 # 6. 矫正图片
#                 crop = four_point_transform(img, pts)
#
#                 # 7. 保存图片（解决中文文件名问题）
#                 save_name = f"{plate_str}_{os.path.splitext(filename)[0]}.jpg"
#                 save_path = os.path.join(save_root, subset, save_name)
#                 # 确保保存路径的目录存在（防止意外）
#                 os.makedirs(os.path.dirname(save_path), exist_ok=True)
#                 # 写入图片（imencode支持中文路径）
#                 encode_ret, encode_buf = cv2.imencode('.jpg', crop)
#                 if not encode_ret:
#                     fail_reasons.append(f"图片编码失败：{filename}")
#                     fail_count += 1
#                     continue
#                 encode_buf.tofile(save_path)
#
#                 # 统计成功
#                 success_count += 1
#
#             except Exception as e:
#                 # 打印具体异常，不再吞掉错误
#                 error_info = f"处理文件{file_path}出错：{str(e)}"
#                 fail_reasons.append(error_info)
#                 fail_count += 1
#                 continue
#
#     # 处理训练集和验证集
#     process(train_files, 'train')
#     process(val_files, 'val')
#
#     # 打印最终统计
#     print(f"\n========== 处理结果 ==========")
#     print(f"✅ 成功处理：{success_count} 张")
#     print(f"❌ 失败处理：{fail_count} 张")
#     if fail_reasons:
#         print(f"\n【失败原因Top5】")
#         for i, reason in enumerate(fail_reasons[:5], 1):
#             print(f"{i}. {reason}")
#
#
# if __name__ == "__main__":
#     # 核心修正：指向project2/qwer（脚本在lprnet下，上级目录是project2）
#     CCPD_ROOT = "../qwer"
#     SAVE_ROOT = "lpr_dataset"
#
#     # 验证CCPD路径是否存在
#     abs_ccpd_root = os.path.abspath(CCPD_ROOT)
#     print(f"【调试】CCPD_ROOT 绝对路径：{abs_ccpd_root}")
#     print(f"【调试】CCPD_ROOT 是否存在：{os.path.exists(abs_ccpd_root)}")
#
#     if os.path.exists(abs_ccpd_root):
#         convert_ccpd_to_lpr(abs_ccpd_root, SAVE_ROOT)
#     else:
#         print(f"\n❌ 致命错误：CCPD_ROOT路径不存在！")
#         print(f"请检查：{abs_ccpd_root} 是否是你的CCPD数据集目录")