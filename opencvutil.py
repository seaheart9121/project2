import cv2
import os


class OpenCVUtil:
    def __init__(self):
        # 初始化摄像头，0为默认USB摄像头
        self.cam = cv2.VideoCapture(0)
        # 设置摄像头分辨率
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 创建图片保存文件夹
        if not os.path.exists("./file"):
            os.makedirs("./file")
        if not os.path.exists("./images"):
            os.makedirs("./images")

    # 摄像头实时抓拍
    def capture_plate(self, save_path="./file/captured_plate.jpg"):
        """抓拍车牌图片，返回保存路径和图片数据"""
        if not self.cam.isOpened():
            return None, "摄像头打开失败，请检查权限和连接！"

        # 窗口标题使用英文，彻底避免乱码
        window_name = "Capture Plate (S to save, Q to quit)"
        while True:
            ret, frame = self.cam.read()
            if not ret:
                return None, "摄像头读取失败！"

            # 不再叠加任何文字，保持画面干净
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # 按 's' 保存图片
            if key == ord('s'):
                cv2.imwrite(save_path, frame)
                cv2.destroyAllWindows()
                return save_path, "抓拍成功！"
            # 按 'q' 或点击窗口关闭按钮退出
            elif key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return None, "取消抓拍"

    # 本地图片导入
    def load_local_image(self, img_path):
        """加载本地图片，返回图片路径和状态"""
        if not os.path.exists(img_path):
            return None, "图片文件不存在！"
        # 复制到项目目录，统一处理
        save_path = "./file/imported_plate.jpg"
        img = cv2.imread(img_path)
        cv2.imwrite(save_path, img)
        return save_path, "图片导入成功！"

    # # 图像预处理（灰度化、二值化、噪声过滤）
    # def preprocess_image(self, img_path, save_path="./file/preprocessed_plate.jpg"):
    #     """预处理图片，返回预处理后的图片路径和状态"""
    #     # 读取图片
    #     img = cv2.imread(img_path)
    #     if img is None:
    #         return None, "图片读取失败！"
    #
    #     # 1. 灰度化
    #     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # 2. 高斯滤波去噪
    #     blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    #     # 3. 自适应二值化
    #     binary_img = cv2.adaptiveThreshold(
    #         blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    #     )
    #     # 4. 形态学操作，去除小噪点
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #     final_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    #
    #     # 保存预处理后的图片
    #     cv2.imwrite(save_path, final_img)
    #     return save_path, "预处理完成！"

    # 释放摄像头资源
    def release_cam(self):
        """强制释放摄像头资源，无论窗口是否关闭"""
        if hasattr(self, 'cam') and self.cam.isOpened():
            self.cam.release()  # 核心：释放硬件资源
        cv2.destroyAllWindows()  # 销毁所有OpenCV窗口
        cv2.waitKey(1)  # 解决部分系统窗口销毁不彻底的问题


# 测试用例
if __name__ == "__main__":
    opencv_util = OpenCVUtil()
    # 测试抓拍
    path, msg = opencv_util.capture_plate()
    print(msg)
    if path:
        # 测试预处理
        pre_path, pre_msg = opencv_util.preprocess_image(path)
        print(pre_msg)
    opencv_util.release_cam()