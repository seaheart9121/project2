from ultralytics import YOLO
import os
import argparse

def train_pose(resume=False, weights=None):
    last_pt_path = 'ccpd_pose_runs/exp4/weights/last.pt'
    #****注意路径****
    if resume:
        if os.path.exists(last_pt_path):
            print(f"正在从 {last_pt_path} 恢复训练...")
            try:
                model = YOLO(last_pt_path)
                model.train(resume=True)
                print("Pose模型训练已恢复并完成!")
                return
            except Exception as e:
                print(f"恢复训练失败: {e}")
                print("将重新开始训练...")
        else:
            print(f"未找到检查点 {last_pt_path}，将重新开始训练...")

    # 加载模型权重
    # model_path = weights if weights else 'yolov8n-pose.pt'
    # print(f"加载YOLOv8-Pose模型: {model_path}...")
    # model = YOLO(model_path)
    # 加载模型权重（优先使用你训练好的 best.pt）
    best_pt_path = 'ccpd_pose_runs/exp4/weights/best.pt'
    # ****注意路径****
    if weights:
        model_path = weights
    elif os.path.exists(best_pt_path):
        model_path = best_pt_path
        print("✅ 加载你自己训练好的最优模型：best.pt")
    else:
        model_path = 'yolov8n-pose.pt'
        print("⚠️ 使用官方原始模型：yolov8n-pose.pt")

    print(f"加载YOLOv8-Pose模型: {model_path}...")
    model = YOLO(model_path)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, 'data_pose.yaml')
    
    print(f"开始训练Pose模型: {data_yaml_path}")
    model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        project='ccpd_pose_runs',
        name='exp5',
        # ****注意路径****
        exist_ok=True,
        pretrained=True
    )
    print("Pose模型训练完成")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='从上次中断的检查点恢复训练')
    parser.add_argument('--weights', type=str, help='指定初始权重路径 (例如 ccpd_pose_runs/exp1/weights/best.pt)')
    args = parser.parse_args()
    train_pose(resume=args.resume, weights=args.weights)
