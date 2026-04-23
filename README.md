在终端中执行以下命令
```bash
pip install -r requirements.txt
```
姿势训练
ccpd_to_yolo_pose
predict_pose
train_pose

数据准备：ccpd_to_yolo_pose.py → 
模型训练：train_pose.py → 
图片采集：opencvutil.py → 
车牌矫正：predict_pose.py → 
核心识别：plate_recognizer.py（+ ocrutil.py兜底）→ 
数据管理：datautil.py → 
可视化交互：main3.py

✔ GUI负责显示
✔ OpenCV只负责“抓拍一帧”

CCPD → 数据转换 → LPRNet训练 → exp对比 → OCR识别
YOLO检测 → 车牌裁剪 → LPRNet识别 → 输出结果
车牌图片 → CNN分类 → 蓝/黄/绿