from ultralytics import YOLO
import cv2
import numpy as np
import os

def order_points(pts):
    # 确保点顺序为 TL, TR, BR, BL
    # 我们的模型训练目标就是 TL, TR, BR, BL，所以直接使用即可
    # 但为了保险，可以根据坐标排序
    # 这里假设模型输出顺序正确
    return pts

def four_point_transform(image, pts):
    # 目标尺寸 (宽, 高)
    # 中国车牌标准尺寸 440mm x 140mm -> 比例约 3.14
    # 设定输出图片大小为 240 x 80
    width = 240
    height = 80
    
    # 目标点: TL, TR, BR, BL
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dst)
    
    # 进行变换
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def predict_and_rectify(model_path, image_path, output_dir='output_rectified'):
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return

    results = model(img)
    result = results[0]
    
    if result.keypoints is None:
        print("未检测到关键点")
        return
        
    # 获取关键点
    # shape: (N, 4, 2) or (N, 4, 3)
    keypoints = result.keypoints.xy.cpu().numpy()

    for i, kpts in enumerate(keypoints):
        # kpts shape: (4, 2) -> TL, TR, BR, BL
        # 转换为float32
        pts = np.array(kpts, dtype="float32")
        
        # 透视变换
        warped = four_point_transform(img, pts)
        
        # save_path = os.path.join(output_dir, f"rectified_{i}.jpg")
        base_name = f"rectified_{i}"
        save_path = os.path.join(output_dir, f"{base_name}.jpg")
        # 如果文件已存在，自动加后缀（_1、_2...）
        count = 1
        while os.path.exists(save_path):
            save_path = os.path.join(output_dir, f"{base_name}_{count}.jpg")
            count += 1

        cv2.imwrite(save_path, warped)
        print(f"已保存矫正后的车牌: {save_path}")

# if __name__ == "__main__":
#     MODEL_PATH = 'ccpd_pose_runs/exp5/weights/best.pt'
#     TEST_IMG = 'images/car4.jpg'
#     if not os.path.exists(MODEL_PATH):
#         MODEL_PATH = 'yolov8n-pose.pt' # Fallback
#
#     if os.path.exists(TEST_IMG):
#         predict_and_rectify(MODEL_PATH, TEST_IMG)

if __name__ == "__main__":
    # ========== 请修改这里的路径 ==========
    MODEL_PATH = 'ccpd_pose_runs/exp5/weights/best.pt'  # 你的训练模型路径
    TEST_IMG = 'images/car7.jpg'  # 你的测试图片路径
    # =====================================

    # 步骤1：校验模型路径
    print(f"===== 程序启动 =====")
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  未找到训练模型 {MODEL_PATH}，将使用官方预训练模型 yolov8n-pose.pt")
        MODEL_PATH = 'yolov8n-pose.pt'

    # 步骤2：校验测试图片路径
    if not os.path.exists(TEST_IMG):
        print(f"❌ 测试图片不存在：{TEST_IMG}")
    else:
        # 执行核心逻辑
        predict_and_rectify(MODEL_PATH, TEST_IMG)

    print(f"\n程序正常退出（退出代码 0）")