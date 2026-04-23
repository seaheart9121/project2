# import torch
# import cv2
# import numpy as np
# import os
# import sys
#
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from model import ColorNet
#
# CLASSES = ['蓝色', '黄色', '绿色']
#
# def predict_color(model_path, img_cv):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = ColorNet().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     img = cv2.resize(img_cv, (94, 24))
#     img = img.astype('float32') / 255.0
#     img = img.transpose(2, 0, 1)
#     img = torch.tensor(img).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         output = model(img)
#         _, predicted = output.max(1)
#         return CLASSES[predicted.item()]


import torch
import cv2
import numpy as np
import os
import sys
from typing import Union  # 新增类型导入

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ColorNet

# 类型别名：统一路径类型
PathType = Union[str, os.PathLike[str]]

CLASSES = ['蓝色', '黄色', '绿色']


def predict_color(model_path: PathType, img_cv: np.ndarray) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ColorNet().to(device)
    # 确保路径为str类型传入torch.load
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    img = cv2.resize(img_cv, (94, 24))
    img = img.astype('float32') / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = output.max(1)
        return CLASSES[predicted.item()]