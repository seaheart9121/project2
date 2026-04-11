# import torch
# import cv2
# import numpy as np
# import sys
# import os
#
# # 添加当前目录到path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from model import LPRNet, CHARS
#
# def decode(preds):
#     # preds: [T, B, C] or [B, T, C]
#     # greedy decode
#     pred_labels = []
#     for i in range(preds.size(0)):
#         pred = preds[i] # [T, C]
#         pred_indices = torch.argmax(pred, dim=1)
#
#         char_list = []
#         prev_idx = -1
#         for idx in pred_indices:
#             idx = idx.item()
#             if idx != prev_idx and idx != len(CHARS)-1:
#                 char_list.append(CHARS[idx])
#             prev_idx = idx
#         pred_labels.append("".join(char_list))
#     return pred_labels
#
# def predict_lpr(model_path, img_path):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = LPRNet().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (94, 24))
#     img = img.astype('float32') / 255.0
#     img = img.transpose(2, 0, 1)
#     img = torch.tensor(img).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         output = model(img) # [1, W, C]
#         result = decode(output)
#         return result[0]
#
# if __name__ == "__main__":
#     # Test
#     print(predict_lpr('lpr_runs/lprnet_best.pth', 'test_plate.jpg'))

import torch
import cv2
import numpy as np
import os
import time
from model import LPRNet, CHARS

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 车牌标准输入尺寸（和训练保持一致）
IMG_H, IMG_W = 24, 94


# -------------------------- 核心预测函数 --------------------------
def predict_single_image(model, img_path):
    """
    单张图片预测：输入模型和图片路径，返回预测车牌、置信度、耗时
    """
    # 1. 加载并预处理图片
    img = cv2.imread(img_path)
    if img is None:
        return f"错误：无法读取图片 {img_path}", 0, 0

    # 预处理：resize + 归一化 + 维度转换
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # (1, 3, 24, 94)

    # 2. 模型推理（计时）
    start_time = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)  # 模型输出 (1, W, C)
        # CTC解码：去重+去blank
        pred = torch.argmax(outputs, dim=2).squeeze().cpu().numpy()
        pred_chars = []
        last_char = -1
        for c in pred:
            if c != last_char and c != len(CHARS) - 1:  # 去重+去blank（最后一个字符是blank）
                pred_chars.append(c)
            last_char = c
        # 拼接车牌
        plate = ''.join([CHARS[c] for c in pred_chars])
        # 计算置信度（取平均概率）
        probs = torch.softmax(outputs, dim=2).max(dim=2).values.squeeze().cpu().numpy()
        confidence = round(np.mean(probs) * 100, 2)  # 转百分比

    end_time = time.time()
    infer_time = round((end_time - start_time) * 1000, 2)  # 转毫秒

    return plate, confidence, infer_time


# -------------------------- 多exp模型对比函数 --------------------------
def compare_all_exp_models(test_img_path, base_dir='lpr_runs'):
    """
    自动遍历所有exp文件夹，用每个exp的最优模型做预测，对比效果
    """
    # 1. 检查测试图片
    if not os.path.exists(test_img_path):
        print(f"❌ 错误：测试图片 {test_img_path} 不存在！")
        return

    # 2. 遍历所有exp文件夹
    exp_folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('exp'):
            try:
                exp_num = int(item.replace('exp', ''))
                exp_folders.append((exp_num, item_path))
            except ValueError:
                continue

    # 按exp序号排序（exp1→exp2→exp3→exp4）
    exp_folders.sort(key=lambda x: x[0])
    if not exp_folders:
        print(f"❌ 错误：在 {base_dir} 中未找到任何exp文件夹！")
        return

    print(f"\n🚀 开始对比 {len(exp_folders)} 个exp模型的预测效果")
    print(f"📸 测试图片：{test_img_path}")
    print("=" * 80)

    # 3. 逐个加载模型并预测
    results = []
    for exp_num, exp_path in exp_folders:
        model_path = os.path.join(exp_path, 'lprnet_best.pth')
        if not os.path.exists(model_path):
            print(f"⚠️ exp{exp_num} 未找到最优模型 {model_path}，跳过")
            continue

        # 加载模型
        print(f"\n🔍 正在加载 exp{exp_num} 模型...")
        model = LPRNet().to(DEVICE)
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            # 兼容checkpoint格式（带state_dict键）和纯权重格式
            if 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            model.eval()
            print(f"✅ exp{exp_num} 模型加载成功")
        except Exception as e:
            print(f"❌ exp{exp_num} 模型加载失败：{e}，跳过")
            continue

        # 预测
        plate, conf, time_cost = predict_single_image(model, test_img_path)
        results.append({
            'exp': f'exp{exp_num}',
            'plate': plate,
            'confidence': conf,
            'time_cost_ms': time_cost
        })

        # 打印当前exp结果
        print(f"📊 exp{exp_num} 预测结果：")
        print(f"  车牌号码：{plate}")
        print(f"  置信度：{conf}%")
        print(f"  推理耗时：{time_cost}ms")
        print("-" * 60)

    # 4. 汇总对比表
    print("\n" + "=" * 80)
    print("📋 所有exp模型预测结果汇总")
    print("=" * 80)
    print(f"{'exp序号':<8} {'预测车牌':<12} {'置信度(%)':<10} {'推理耗时(ms)':<12}")
    print("-" * 80)
    for res in results:
        print(f"{res['exp']:<8} {res['plate']:<12} {res['confidence']:<10} {res['time_cost_ms']:<12}")
    print("=" * 80)

    # 5. 找出最优模型（置信度最高）
    if results:
        best_res = max(results, key=lambda x: x['confidence'])
        print(f"\n🏆 最优模型：{best_res['exp']}")
        print(f"  最优预测：{best_res['plate']}（置信度 {best_res['confidence']}%）")


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 测试图片路径（你可以改成自己的测试图，默认用项目里的test_plate.jpg）
    test_img = "./test_plate.jpg"
    # 运行对比
    compare_all_exp_models(test_img)