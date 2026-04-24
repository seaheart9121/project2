#coding:utf-8
from aip import AipOcr
import easyocr
from tkinter import simpledialog, messagebox
import os


class OCRUtil:
    def __init__(self):
        # 百度OCR配置（已填入你截图中的真实密钥，无需修改）
        self.APP_ID = "121576563"
        self.API_KEY = "JiXaJ9IEmh5edjVKFurkM7SN"
        self.SECRET_KEY = "9YaKfrNk7JWmi6sNSEfiCPo9In93U6WR"
        self.baidu_client = AipOcr(self.APP_ID, self.API_KEY, self.SECRET_KEY)

        # # 本地EasyOCR初始化（备用方案）
        # self.local_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        # ========== 【新增】一次性加载本地识别模型 ==========
        try:
            from plate_recognizer import PlateRecognitionModel
            self.plate_recognizer = PlateRecognitionModel()
            print("本地车牌识别模型加载完成")
        except Exception as e:
            print(f"本地车牌识别模型加载失败：{e}")
            self.plate_recognizer = None

    # 百度OCR车牌识别（主方案）- 无任何预处理，保持你原有逻辑
    def baidu_ocr_recognize(self, img_path):
        """调用百度OCR识别车牌，返回车牌号码和状态"""
        if not os.path.exists(img_path):
            return None, "图片不存在！"

        # 读取图片二进制数据（纯原始图片，无预处理）
        with open(img_path, 'rb') as f:
            img_data = f.read()

        # 调用百度车牌识别接口
        try:
            # 设置可选参数（保留你原有配置，仅识别多车牌）
            options = {}
            options["multi_detect"] = "true"  # 支持多车牌识别

            result = self.baidu_client.licensePlate(img_data, options)

            # 打印完整返回结果，用于调试car5/6/7的问题
            print(f"【调试信息】{os.path.basename(img_path)} 百度OCR返回结果:", result)

            if "error_code" in result:
                return None, f"百度OCR调用失败：{result['error_code']} - {result.get('error_msg', '未知错误')}"
            elif "words_result" in result and len(result["words_result"]) > 0:
                plate_num = result["words_result"][0]["number"]
                return plate_num, f"百度OCR识别成功：{plate_num}"
            else:
                return None, "百度OCR未识别到车牌"
        except Exception as e:
            return None, f"百度OCR调用异常：{str(e)}"

    # # 本地EasyOCR识别（备用方案）- 保留你原有过滤逻辑
    # def local_ocr_recognize(self, img_path):
    #     """本地OCR识别，百度OCR失败时调用"""
    #     if not os.path.exists(img_path):
    #         return None, "图片不存在！"
    #
    #     try:
    #         result = self.local_reader.readtext(img_path, detail=0)
    #         # 过滤车牌格式（7-8位，包含汉字+字母+数字）
    #         plate_list = [text.replace(" ", "").upper() for text in result if 7 <= len(text.replace(" ", "")) <= 8]
    #         if plate_list:
    #             return plate_list[0], f"本地OCR识别成功：{plate_list[0]}"
    #         else:
    #             return None, "本地OCR未识别到符合格式的车牌"
    #     except Exception as e:
    #         return None, f"本地OCR识别失败：{str(e)}"
    def full_recognize_process(self, img_path, ui_obj=None, use_baidu=False):
        """
        对外接口完全不变！main.py里的调用一行都不用改！
        :param img_path: 图片路径
        :param ui_obj: UI对象（用于更新进度条等，可选）
        :param use_baidu: 是否使用百度OCR（默认False用本地）
        :return: (plate_num, ocr_msg)
        """
        try:
            # if use_baidu:
            #     # ========== 保留：百度OCR分支（用原始图） ==========
            #     plate_num = self.baidu_ocr_recognize(img_path)
            #     if plate_num != "未识别车牌":
            #         ocr_msg = f"百度OCR识别成功：{plate_num}"
            #     else:
            #         ocr_msg = "百度OCR未识别到车牌"
            #     return plate_num, ocr_msg
            if use_baidu:
                # ========== 保留：百度OCR分支（用原始图） ==========
                plate_num, baidu_msg = self.baidu_ocr_recognize(img_path)  # 修复：接收两个返回值
                if plate_num:  # 百度识别到车牌
                    ocr_msg = f"百度OCR识别成功：{plate_num}"
                else:  # 百度未识别到
                    plate_num = "未识别车牌"
                    ocr_msg = baidu_msg  # 复用百度返回的错误信息
                return plate_num, ocr_msg
            else:
                # ========== 【新增/修改】本地OCR分支（调用plate_recognizer，无重复预处理） ==========
                # 检查plate_recognizer是否已初始化（如果没有，在这里补一个，但最好放在__init__里）
                if not hasattr(self, "plate_recognizer"):
                    from plate_recognizer import PlateRecognitionModel
                    self.plate_recognizer = PlateRecognitionModel()

                # 直接调用，不做任何预处理！
                results = self.plate_recognizer.recognize_plate(img_path)

                # 适配返回值（取第一个识别到的车牌，符合停车场单车牌场景）
                if results:
                    plate_num = results[0]["号码"]
                    color = results[0]["颜色"]
                    ocr_msg = f"本地O识别成功：{color} | {plate_num}"
                else:
                    plate_num = "未识别车牌"
                    ocr_msg = "本地未检测到车牌"

                return plate_num, ocr_msg
        except Exception as e:
            return "未识别车牌", f"识别失败：{str(e)}"

    # 手动录入车牌（最终备用方案）- 完全保留你原有逻辑
    def manual_input_plate(self, parent_window):
        """手动录入车牌，返回录入的车牌号码"""
        while True:
            plate_num = simpledialog.askstring("手动录入车牌", "请输入车牌号码：", parent=parent_window)
            if plate_num is None:
                return None, "取消手动录入"
            # 简单格式校验
            plate_num = plate_num.strip().upper()
            if 7 <= len(plate_num) <= 8:
                return plate_num, f"手动录入成功：{plate_num}"
            else:
                messagebox.showwarning("格式错误", "车牌号码格式不正确，请重新输入（7-8位）！")

    # # 完整识别流程（百度用原始图 → 本地用预处理图 → 手动录入）
    # def full_recognize_process(self, raw_img_path, parent_window, pre_img_path=None):
    #     """完整识别流程：百度OCR用原始图，本地OCR用预处理图"""
    #     # 1. 百度OCR：使用原始图片（不做预处理）
    #     plate_num, msg = self.baidu_ocr_recognize(raw_img_path)
    #     if plate_num:
    #         return plate_num, msg
    #
    #     # 2. 百度失败，本地OCR：使用预处理后的图片（如果有）
    #     messagebox.showwarning(title="识别失败", message=f"{msg}，将启动本地备用识别方案！")
    #     # 优先用预处理图，没有则用原始图
    #     ocr_img_path = pre_img_path if pre_img_path else raw_img_path
    #     plate_num, msg = self.local_ocr_recognize(ocr_img_path)
    #     if plate_num:
    #         return plate_num, msg
    #
    #     # 3. 本地也失败，手动录入
    #     messagebox.showwarning(title="识别失败", message=f"{msg}，请手动录入车牌！")
    #     plate_num, msg = self.manual_input_plate(parent_window)
    #     return plate_num, msg
# 测试用例（保留你原有测试，可直接运行验证）
if __name__ == "__main__":
    ocr_util = OCRUtil()
    # 测试识别car4.jpg（可替换为car5/6/7测试）
    plate, msg = ocr_util.baidu_ocr_recognize("./file/car6.png")
    print(msg)