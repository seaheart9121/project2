import cv2
import numpy as np
import os
import sys
import torch
from ultralytics import YOLO
from lprnet.model import LPRNet, CHARS
from colornet.model import ColorNet


# 保留核心辅助函数
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def four_point_transform(image, pts):
    width = 240
    height = 80
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def decode(preds):
    pred_labels = []
    for i in range(preds.size(0)):
        pred = preds[i]
        pred_indices = torch.argmax(pred, dim=1)
        char_list = []
        prev_idx = -1
        for idx in pred_indices:
            idx = idx.item()
            if idx != prev_idx and idx != len(CHARS) - 1:
                char_list.append(CHARS[idx])
            prev_idx = idx
        pred_labels.append("".join(char_list))
    return pred_labels


# 模型加载类/函数
class PlateRecognitionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = None
        self.lpr_model = None
        self.color_model = None
        self.load_models()

    def load_models(self):
        """加载所有模型"""
        try:
            # 1. YOLO Pose 模型加载
            local_yolo = 'ccpd_pose_runs/exp7/weights/best.pt'
            if os.path.exists(local_yolo):
                self.yolo_path = local_yolo
            else:
                self.yolo_path = resource_path('best_pose.pt')

            if not os.path.exists(self.yolo_path):
                self.yolo_path = 'yolov8n-pose.pt'
                print("使用预训练YOLO模型")

            self.yolo_model = YOLO(self.yolo_path)
            print("YOLO模型加载完成")

            # 2. LPRNet 模型加载
            local_lpr = 'lpr_runs/lprnet_best.pth'
            if os.path.exists(local_lpr):
                self.lpr_path = local_lpr
            else:
                self.lpr_path = resource_path('lprnet_best.pth')

            self.lpr_model = LPRNet().to(self.device)
            if os.path.exists(self.lpr_path):
                self.lpr_model.load_state_dict(torch.load(self.lpr_path, map_location=self.device))
                self.lpr_model.eval()
                print("LPRNet模型加载完成")
            else:
                print("警告: 未找到LPRNet模型，号码识别功能不可用")
                self.lpr_model = None

            # 3. ColorNet 模型加载
            local_color = 'color_runs/color_best.pth'
            if os.path.exists(local_color):
                self.color_path = local_color
            else:
                self.color_path = resource_path('color_best.pth')

            self.color_model = ColorNet().to(self.device)
            if os.path.exists(self.color_path):
                self.color_model.load_state_dict(torch.load(self.color_path, map_location=self.device))
                self.color_model.eval()
                print("ColorNet模型加载完成")
            else:
                print("警告: 未找到ColorNet模型，颜色识别将使用HSV兜底")
                self.color_model = None

        except Exception as e:
            print(f"模型加载失败: {e}")

    def recognize_plate(self, image_path):
        """
        核心识别函数
        :param image_path: 图片路径
        :return: 识别结果列表
        """
        # 读取图片（兼容中文路径）
        img_cv = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None:
            print("错误：无法读取图片，请检查路径")
            return []

        # 1. YOLO 检测车牌关键点
        results = self.yolo_model(img_cv)
        result = results[0]
        keypoints = result.keypoints.xy.cpu().numpy()

        if len(keypoints) == 0:
            print("未检测到车牌")
            return []

        recognition_results = []

        # 处理每个检测到的车牌
        for i, kpts in enumerate(keypoints):
            # 2. 车牌矫正
            pts = np.array(kpts, dtype="float32")
            warped = four_point_transform(img_cv, pts)

            # 3. 颜色识别
            color = "未知"
            if self.color_model:
                c_input = cv2.resize(warped, (94, 24))
                c_input = c_input.astype('float32') / 255.0
                c_input = c_input.transpose(2, 0, 1)
                c_input = torch.tensor(c_input).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    c_out = self.color_model(c_input)
                    _, c_pred = c_out.max(1)
                    color_classes = ['蓝色', '黄色', '绿色']
                    color = color_classes[c_pred.item()]
            else:
                # HSV 兜底方案
                hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                mean_h = np.mean(hsv[:, :, 0])
                if 100 < mean_h < 124:
                    color = "蓝色"
                elif 35 < mean_h < 77:
                    color = "绿色"
                elif 11 < mean_h < 34:
                    color = "黄色"

            # 4. 号码识别
            plate_text = "模型未加载"
            if self.lpr_model:
                lpr_input = cv2.resize(warped, (94, 24))
                lpr_input = lpr_input.astype('float32') / 255.0
                lpr_input = lpr_input.transpose(2, 0, 1)
                lpr_input = torch.tensor(lpr_input).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.lpr_model(lpr_input)
                    plate_text = decode(output)[0]

            # 保存结果
            result = {
                "车牌序号": i + 1,
                "号码": plate_text,
                "颜色": color,
                "字符数": len(plate_text)
            }
            recognition_results.append(result)

            # 打印单车牌结果
            print(f"\n===== 车牌 {i + 1} =====")
            print(f"号码: {plate_text}")
            print(f"颜色: {color}")
            print(f"字符数: {len(plate_text)}")

        # 可选：保存矫正后的车牌图片
        # cv2.imencode('.jpg', warped)[1].tofile(f'矫正后的车牌_{i+1}.jpg')

        return recognition_results


# 测试主函数
if __name__ == "__main__":
    # 初始化模型
    print("正在加载模型...")
    plate_model = PlateRecognitionModel()

    # 输入图片路径（替换成你的测试图片路径）
    test_image_path = input("请输入测试图片路径（如：test.jpg）：").strip()

    # 检查路径是否存在
    if not os.path.exists(test_image_path):
        print("错误：图片路径不存在！")
    else:
        print("\n开始识别...")
        results = plate_model.recognize_plate(test_image_path)

        # 汇总结果
        print("\n===== 识别汇总 =====")
        for res in results:
            print(f"车牌{res['车牌序号']}: {res['颜色']} | {res['号码']}")

