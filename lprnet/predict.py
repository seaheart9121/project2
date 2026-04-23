import torch
import cv2
import numpy as np
import os
import time
from model import LPRNet, CHARS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BLANK_IDX = len(CHARS) - 1

# 修复：标准CTC去重解码
def ctc_decode(preds):
    pred_indices = torch.argmax(preds, dim=1).cpu().numpy()
    result = []
    prev = -1
    for c in pred_indices:
        if c != prev and c != BLANK_IDX:
            result.append(CHARS[c])
        prev = c
    return ''.join(result)

def predict_single_image(model, img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "读取失败", 0, 0

    img = cv2.resize(img, (94, 24))
    img = img.astype('float32') / 255.0
    img = img.transpose(2, 0, 1)

    img_tensor = torch.tensor(img).unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(img_tensor).squeeze(0)
        plate = ctc_decode(outputs)

        probs = torch.softmax(outputs, dim=1)
        max_probs, indices = probs.max(dim=1)
        valid_probs = []
        prev = -1
        for i, c in enumerate(indices.cpu().numpy()):
            if c != prev and c != BLANK_IDX:
                valid_probs.append(max_probs[i].item())
            prev = c
        confidence = np.mean(valid_probs) * 100 if valid_probs else 0

    cost_time = round((time.time() - start_time) * 1000, 2)
    return plate, round(confidence, 2), cost_time

# 批量测试7个模型
def predict_all_exps(img_path):
    print("🚀 开始对比exp模型的预测效果")
    print(f"📸 测试图片：{img_path}")
    print("=" * 80)

    results = []
    for exp_id in range(1, 8):
        model_path = f"./lpr_runs/exp{exp_id}/lprnet_best.pth"
        if not os.path.exists(model_path):
            print(f"❌ 模型不存在：{model_path}")
            continue

        print(f"🔍 正在加载 exp{exp_id} 模型...")
        model = LPRNet().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"✅ exp{exp_id} 模型加载成功")

        plate, conf, speed = predict_single_image(model, img_path)
        print(f"📊 exp{exp_id} 预测结果：")
        print(f"  车牌号码：{plate}")
        print(f"  置信度：{conf}%")
        print(f"  推理耗时：{speed}ms")
        print("-" * 60)
        results.append((exp_id, plate, conf, speed))

    print("\n" + "=" * 80)
    print("📋 所有exp模型预测结果汇总")
    print("=" * 80)
    print(f"{'exp序号':<8}{'预测车牌':<12}{'置信度(%)':<18}{'推理耗时(ms)':<10}")
    print("-" * 80)
    best_conf = -1
    best_item = None
    for item in results:
        exp, plate, conf, speed = item
        print(f"{f'exp{exp}':<8}{plate:<12}{conf:<18}{speed:<10}")
        if conf > best_conf:
            best_conf = conf
            best_item = item
    print("=" * 80)

    if best_item:
        print(f"\n🏆 最优模型：exp{best_item[0]}")
        print(f"  最优预测：{best_item[1]}（置信度 {best_item[2]}%）")

if __name__ == "__main__":
    predict_all_exps("./test_plate2.jpg")