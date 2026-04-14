import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.simpledialog import askfloat
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from datetime import datetime
import threading
import cv2
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont  # 修复：直接导入ImageFont，统一处理

from opencvutil import OpenCVUtil
from ocrutil import OCRUtil
from datautil import DataUtil

# 解决matplotlib中文显示问题
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


class ParkingSystem(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("智能停车场车牌识别计费系统")
        self.geometry("1200x800")
        self.resizable(True, True)

        self.opencv_util = OpenCVUtil()
        self.ocr_util = OCRUtil()
        self.data_util = DataUtil()

        self.current_plate = tk.StringVar(value="未识别车牌")
        self.current_img_path = None
        self.log_list = []

        self.init_ui()
        self.refresh_parking_data()
        self.refresh_system_time()

    def init_ui(self):
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(self.title_frame, text="智能停车场车牌识别计费系统", font=("黑体", 18, "bold")).pack(side=tk.LEFT)
        self.time_label = ttk.Label(self.title_frame, text="系统时间：", font=("宋体", 10))
        self.time_label.pack(side=tk.RIGHT, padx=10)

        self.stats_frame = ttk.Frame(self)
        self.stats_frame.pack(fill=tk.X, padx=10, pady=5)
        self.warn_label = tk.Label(
            self.stats_frame,
            text="根据数据分析，今天可能出现车位紧张的情况，请做好调度！",
            bg="yellow", fg="red", font=("宋体", 12, "bold"), height=2
        )
        self.warn_label.pack(fill=tk.X, pady=2)
        self.warn_label.pack_forget()

        self.card_frame = ttk.Frame(self.stats_frame)
        self.card_frame.pack(fill=tk.X, pady=5)

        self.total_card = tk.LabelFrame(self.card_frame, text="总车位", font=("宋体", 10))
        self.total_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.total_label = ttk.Label(self.total_card, text="0", font=("黑体", 20, "bold"), foreground="blue")
        self.total_label.pack(pady=5)

        self.used_card = tk.LabelFrame(self.card_frame, text="已用车位", font=("宋体", 10))
        self.used_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.used_label = ttk.Label(self.used_card, text="0", font=("黑体", 20, "bold"), foreground="orange")
        self.used_label.pack(pady=5)

        self.free_card = tk.LabelFrame(self.card_frame, text="剩余车位", font=("宋体", 10))
        self.free_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.free_label = ttk.Label(self.free_card, text="0", font=("黑体", 20, "bold"), foreground="green")
        self.free_label.pack(pady=5)

        self.rate_card = tk.LabelFrame(self.card_frame, text="车位占用率", font=("宋体", 10))
        self.rate_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.rate_label = ttk.Label(self.rate_card, text="0%", font=("黑体", 20, "bold"), foreground="red")
        self.rate_label.pack(pady=5)

        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.operate_frame = tk.LabelFrame(self.main_frame, text="核心操作区", font=("宋体", 10))
        self.operate_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, ipadx=10, ipady=10)

        ttk.Label(self.operate_frame, text="车牌识别操作", font=("宋体", 10, "bold")).pack(pady=5)
        self.capture_btn = ttk.Button(self.operate_frame, text="摄像头抓拍", command=self.capture_plate, width=20)
        self.capture_btn.pack(pady=3)
        self.import_btn = ttk.Button(self.operate_frame, text="本地导入图片", command=self.import_image, width=20)
        self.import_btn.pack(pady=3)
        self.local_recognize_btn = ttk.Button(self.operate_frame, text="本地识别车牌", command=self.local_recognize,width=20)
        self.local_recognize_btn.pack(pady=3)
        self.manual_btn = ttk.Button(self.operate_frame, text="手动录入车牌", command=self.manual_input, width=20)
        self.manual_btn.pack(pady=3)

        ttk.Label(self.operate_frame, text="识别结果", font=("宋体", 10, "bold")).pack(pady=10)
        self.plate_label = ttk.Label(self.operate_frame, textvariable=self.current_plate, font=("黑体", 14, "bold"),
                                     foreground="blue")
        self.plate_label.pack(pady=3)

        ttk.Label(self.operate_frame, text="车辆管理操作", font=("宋体", 10, "bold")).pack(pady=10)
        self.entry_btn = ttk.Button(self.operate_frame, text="车辆入场", command=self.car_entry, width=20,
                                    state=tk.DISABLED)
        self.entry_btn.pack(pady=3)
        self.exit_btn = ttk.Button(self.operate_frame, text="车辆出场", command=self.car_exit, width=20,
                                   state=tk.DISABLED)
        self.exit_btn.pack(pady=3)
        self.rate_btn = ttk.Button(self.operate_frame, text="费率设置", command=self.set_rate, width=20)
        self.rate_btn.pack(pady=3)

        # 新增：增加车位按钮
        self.manage_spaces_btn = ttk.Button(self.operate_frame, text="车位管理", command=self.manage_parking_spaces, width=20)
        self.manage_spaces_btn.pack(pady=3)

        self.query_btn = ttk.Button(self.operate_frame, text="记录查询", command=self.query_record, width=20)
        self.query_btn.pack(pady=3)

        ttk.Label(self.operate_frame, text="数据分析操作", font=("宋体", 10, "bold")).pack(pady=10)
        self.space_chart_btn = ttk.Button(self.operate_frame, text="车位状态图表", command=self.show_space_chart,
                                          width=20)
        self.space_chart_btn.pack(pady=3)
        self.fee_chart_btn = ttk.Button(self.operate_frame, text="收入统计图表", command=self.show_fee_chart, width=20)
        self.fee_chart_btn.pack(pady=3)

        # 改造：实时预览与识别区（替换原有静态label为canvas）
        self.preview_frame = tk.LabelFrame(self.main_frame, text="实时预览与识别区", font=("宋体", 10))
        self.preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.preview_canvas = tk.Canvas(self.preview_frame, background="#f0f0f0")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # 新增提示文字（无内容时显示）
        self.preview_tip = ttk.Label(
            self.preview_canvas,
            text="\n1. 点击「摄像头抓拍」可实时预览摄像头画面\n2. 抓拍/导入图片后将显示图片+识别结果",
            font=("宋体", 11),
            foreground="#666666"
        )
        self.preview_tip.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.car_list_frame = tk.LabelFrame(self.main_frame, text="当前在场车辆", font=("宋体", 10))
        self.car_list_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.car_tree = ttk.Treeview(self.car_list_frame, columns=["车位ID", "车牌号码", "入场时间"], show="headings",
                                     height=20)
        self.car_tree.heading("车位ID", text="车位ID")
        self.car_tree.heading("车牌号码", text="车牌号码")
        self.car_tree.heading("入场时间", text="入场时间")
        self.car_tree.column("车位ID", width=80, anchor=tk.CENTER)
        self.car_tree.column("车牌号码", width=100, anchor=tk.CENTER)
        self.car_tree.column("入场时间", width=150, anchor=tk.CENTER)
        self.car_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_frame = tk.LabelFrame(self, text="操作日志", font=("宋体", 10))
        self.log_frame.pack(fill=tk.X, padx=10, pady=5, ipady=5)
        self.log_text = tk.Text(self.log_frame, height=5, font=("宋体", 9))
        self.log_text.pack(fill=tk.X, padx=5)
        self.log_text.config(state=tk.DISABLED)

    # 新增：实时显示摄像头画面
    def start_camera_preview(self):
        """在预览区实时显示摄像头画面"""
        if not self.opencv_util.cam.isOpened():
            self.add_log("摄像头预览失败：摄像头未打开")
            return

        def update_frame():
            ret, frame = self.opencv_util.cam.read()
            if ret:
                # 转换颜色空间（OpenCV BGR → PIL RGB）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 调整画面大小适配预览区
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                if canvas_width <= 1 or canvas_height <= 1:  # 避免初始化时尺寸为0
                    canvas_width, canvas_height = 640, 480

                # 等比例缩放图片
                img = Image.fromarray(frame_rgb)
                img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                # 转换为tkinter可用的格式
                self.preview_img = ImageTk.PhotoImage(image=img)

                # 清空画布并显示新画面
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(
                    canvas_width / 2, canvas_height / 2,
                    image=self.preview_img, anchor=tk.CENTER
                )
                # 持续刷新（10ms/帧）
                self.preview_after_id = self.after(10, update_frame)
            else:
                self.add_log("摄像头预览失败：无法读取画面")

        # 先隐藏提示文字
        self.preview_tip.place_forget()
        # 启动帧刷新
        update_frame()

    # 新增：停止摄像头预览
    def stop_camera_preview(self):
        """停止预览并清空画布"""
        if hasattr(self, 'preview_after_id'):
            self.after_cancel(self.preview_after_id)
        self.preview_canvas.delete("all")
        # 恢复提示文字
        self.preview_tip.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # 新增：预览抓拍/导入的图片 + 叠加识别结果
    def show_image_preview(self, img_path, plate_num=""):
        """显示图片预览，并叠加识别到的车牌文字"""
        if not os.path.exists(img_path):
            self.add_log(f"图片预览失败：{img_path}不存在")
            return

        # 读取图片并转换格式
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 适配预览区尺寸
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 640, 480

        # 等比例缩放
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        # 叠加识别结果文字（如果有）
        if plate_num and plate_num != "未识别车牌":
            draw = ImageDraw.Draw(pil_img)
            # 设置文字样式（兼容不同系统字体）
            try:
                font = ImageFont.truetype("simhei.ttf", 24)
            except Exception as e:
                self.add_log(f"加载字体失败，使用默认字体：{e}")
                font = ImageFont.load_default(size=24)
            # 计算文字位置（底部居中，加背景）
            text = f"识别结果：{plate_num}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (pil_img.width - text_width) // 2
            text_y = pil_img.height - text_height - 10
            # 绘制文字背景（半透明黑）
            draw.rectangle(
                [text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5],
                fill=(0, 0, 0, 128)  # 黑色半透明
            )
            # 绘制白色文字
            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

        # 转换为tkinter格式并显示
        self.preview_img = ImageTk.PhotoImage(image=pil_img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            canvas_width / 2, canvas_height / 2,
            image=self.preview_img, anchor=tk.CENTER
        )
        # 隐藏提示文字
        self.preview_tip.place_forget()

    def refresh_system_time(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=f"系统时间：{now}")
        self.after(1000, self.refresh_system_time)

    def refresh_parking_data(self):
        total, used, free, rate = self.data_util.get_parking_stats()
        self.total_label.config(text=str(total))
        self.used_label.config(text=str(used))
        self.free_label.config(text=str(free))
        self.rate_label.config(text=f"{round(rate, 1)}%")
        if rate >= 80:
            self.warn_label.pack(fill=tk.X, pady=2)
        else:
            self.warn_label.pack_forget()
        self.refresh_car_list()
        self.after(5000, self.refresh_parking_data)

    def refresh_car_list(self):
        for item in self.car_tree.get_children():
            self.car_tree.delete(item)
        car_list = self.data_util.get_in_car_list()
        for car in car_list:
            self.car_tree.insert("", tk.END, values=car)

    def add_log(self, msg):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{now}] {msg}\n"
        self.log_list.append(log_msg)
        if len(self.log_list) > 10:
            self.log_list.pop(0)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "".join(self.log_list))
        self.log_text.config(state=tk.DISABLED)

    # 改造：摄像头抓拍集成实时预览
    def capture_plate(self):
        def task():
            try:
                self.add_log("开始摄像头抓拍...")
                # 启动实时预览
                self.after(0, self.start_camera_preview)
                # 调用抓拍方法
                img_path, msg = self.opencv_util.capture_plate()
                # 停止预览
                self.after(0, self.stop_camera_preview)
                self.add_log(msg)

                if img_path:
                    self.current_img_path = img_path
                    pre_path, pre_msg = self.opencv_util.preprocess_image(img_path)
                    self.add_log(pre_msg)
                    # 百度OCR用原始图，本地OCR用预处理图
                    plate_num, ocr_msg = self.ocr_util.full_recognize_process(img_path, self)
                    self.add_log(ocr_msg)

                    if plate_num:
                        self.after(0, lambda: self.current_plate.set(plate_num))
                        self.after(0, lambda: self.entry_btn.config(state=tk.NORMAL))
                        self.after(0, lambda: self.exit_btn.config(state=tk.NORMAL))
                    # 预览抓拍的图片 + 识别结果
                    self.after(0, lambda: self.show_image_preview(img_path, plate_num))
            except Exception as e:
                self.add_log(f"抓拍异常：{str(e)}")
                self.after(0, self.stop_camera_preview)
            finally:
                self.opencv_util.release_cam()
                self.opencv_util.cam = cv2.VideoCapture(0)
                self.add_log("摄像头已释放")

        threading.Thread(target=task, daemon=True).start()

    # 改造：本地导入图片集成预览功能
    def import_image(self):
        try:
            # 强制清理所有OpenCV残留，防止冲突
            cv2.destroyAllWindows()

            img_path = filedialog.askopenfilename(
                title="选择车牌图片",
                filetypes=[("图片文件", "*.jpg;*.png;*.jpeg;*.bmp")]
            )
            if not img_path:
                return

            self.add_log(f"导入图片：{img_path}")
            self.current_img_path = img_path

            # 重新加载，避免缓存问题
            pre_path, pre_msg = self.opencv_util.preprocess_image(img_path)
            self.add_log(pre_msg)

            # 核心修改：use_baidu=True 强制使用百度OCR
            plate_num, ocr_msg = self.ocr_util.full_recognize_process(img_path, self, use_baidu=True)
            self.add_log(ocr_msg)

            if plate_num and plate_num != "未识别车牌":
                self.current_plate.set(plate_num)
                self.entry_btn.config(state=tk.NORMAL)
                self.exit_btn.config(state=tk.NORMAL)
            else:
                self.current_plate.set("未识别车牌")
                self.entry_btn.config(state=tk.DISABLED)
                self.exit_btn.config(state=tk.DISABLED)

            # 预览导入的图片 + 识别结果
            self.show_image_preview(img_path, plate_num)

        except Exception as e:
            self.add_log(f"导入失败：{str(e)}")
            messagebox.showerror("错误", f"导入图片失败：{str(e)}")

    # local_recognize 函数
    def local_recognize(self):
        """本地识别车牌：替代百度OCR，使用本地模型识别已导入的图片"""
        # 1. 检查是否已导入图片
        if not self.current_img_path:
            messagebox.showwarning("警告", "请先点击「本地导入图片」选择要识别的图片！")
            return

        # 2. 调用本地OCR识别（use_baidu=False）
        try:
            self.add_log("开始本地模型识别车牌...")
            plate_num, ocr_msg = self.ocr_util.full_recognize_process(
                self.current_img_path,
                self,
                use_baidu=False  # 强制使用本地识别
            )
            self.add_log(ocr_msg)

            # 3. 更新识别结果和UI状态
            if plate_num and plate_num != "未识别车牌":
                self.current_plate.set(plate_num)
                self.entry_btn.config(state=tk.NORMAL)
                self.exit_btn.config(state=tk.NORMAL)
                # 更新预览区的识别结果
                self.show_image_preview(self.current_img_path, plate_num)
            else:
                self.current_plate.set("未识别车牌")
                self.entry_btn.config(state=tk.DISABLED)
                self.exit_btn.config(state=tk.DISABLED)
                messagebox.showinfo("提示", "本地识别未检测到有效车牌！")

        except Exception as e:
            self.add_log(f"本地识别失败：{str(e)}")
            messagebox.showerror("错误", f"本地识别异常：{str(e)}")

    def manual_input(self):
        plate_num, msg = self.ocr_util.manual_input_plate(self)
        if plate_num:
            self.current_plate.set(plate_num)
            self.add_log(msg)
            self.entry_btn.config(state=tk.NORMAL)
            self.exit_btn.config(state=tk.NORMAL)
            # 如果有已导入/抓拍的图片，更新预览区的识别结果
            if self.current_img_path:
                self.show_image_preview(self.current_img_path, plate_num)

    def car_entry(self):
        plate_num = self.current_plate.get()
        if plate_num == "未识别车牌":
            messagebox.showwarning("警告", "请先识别或录入车牌！")
            return
        success, msg = self.data_util.car_entry(plate_num)
        self.add_log(msg)
        if success:
            messagebox.showinfo("成功", msg)
            self.refresh_parking_data()
        else:
            messagebox.showwarning("失败", msg)

    def car_exit(self):
        plate_num = self.current_plate.get()
        if plate_num == "未识别车牌":
            messagebox.showwarning("警告", "请先识别或录入车牌！")
            return
        success, msg = self.data_util.car_exit(plate_num)
        self.add_log(msg)
        if success:
            messagebox.showinfo("成功", msg)
            self.refresh_parking_data()
            # 修复：确保重置所有关联状态
            self.current_plate.set("未识别车牌")
            self.current_img_path = None  # 清空图片路径
            self.entry_btn.config(state=tk.DISABLED)
            self.exit_btn.config(state=tk.DISABLED)
            # 清空预览区并恢复提示文字
            self.stop_camera_preview()
            self.preview_tip.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        else:
            messagebox.showwarning("失败", msg)

    def set_rate(self):
        new_rate = askfloat("费率设置", "请输入每小时收费标准（元）：", initialvalue=self.data_util.hour_rate, minvalue=0)
        if new_rate is not None:
            self.data_util.hour_rate = new_rate
            self.add_log(f"费率设置成功：{new_rate}元/小时")
            messagebox.showinfo("成功", f"费率已设置为{new_rate}元/小时")

    def manage_parking_spaces(self):
        """车位管理弹窗：同时支持增加和删除车位"""
        manage_win = tk.Toplevel(self)
        manage_win.title("车位管理")
        manage_win.geometry("400x300")
        manage_win.resizable(False, False)

        # 增加车位区域
        add_frame = tk.LabelFrame(manage_win, text="增加车位", font=("宋体", 10, "bold"))
        add_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(add_frame, text="增加数量：").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        add_count_var = tk.IntVar(value=1)
        add_entry = ttk.Entry(add_frame, textvariable=add_count_var, width=10)
        add_entry.grid(row=0, column=1, padx=5, pady=5)

        def do_add():
            count = add_count_var.get()
            if count <= 0:
                messagebox.showwarning("提示", "请输入大于0的数量！")
                return
            success, msg = self.data_util.add_parking_spaces(count)
            self.add_log(msg)
            if success:
                self.refresh_parking_data()
                messagebox.showinfo("成功", msg)
                manage_win.destroy()
            else:
                messagebox.showerror("失败", msg)

        ttk.Button(add_frame, text="确认增加", command=do_add).grid(row=0, column=2, padx=10, pady=5)

        # 删除车位区域
        remove_frame = tk.LabelFrame(manage_win, text="删除车位", font=("宋体", 10, "bold"))
        remove_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(remove_frame, text="车位ID（逗号分隔）：").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        remove_ids_var = tk.StringVar()
        remove_entry = ttk.Entry(remove_frame, textvariable=remove_ids_var, width=25)
        remove_entry.grid(row=0, column=1, padx=5, pady=5)

        def do_remove():
            ids_str = remove_ids_var.get().strip()
            if not ids_str:
                messagebox.showwarning("提示", "请输入要删除的车位ID！")
                return
            space_ids = [s.strip() for s in ids_str.split(",")]
            success, msg = self.data_util.remove_parking_spaces(space_ids)
            self.add_log(msg)
            if success:
                self.refresh_parking_data()
                messagebox.showinfo("成功", msg)
                manage_win.destroy()
            else:
                messagebox.showerror("失败", msg)

        ttk.Button(remove_frame, text="确认删除", command=do_remove).grid(row=0, column=2, padx=10, pady=5)

        # 提示信息
        tip_label = ttk.Label(manage_win, text="注意：删除的车位必须是空闲状态！", foreground="red")
        tip_label.pack(pady=10)

        manage_win.mainloop()

    def query_record(self):
        query_win = tk.Toplevel(self)
        query_win.title("停车记录查询")
        query_win.geometry("800x500")
        tree = ttk.Treeview(query_win, columns=["车牌号码", "入场时间", "出场时间", "车位ID", "停车状态"],
                            show="headings")
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=150)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        record_df = pd.read_excel(self.data_util.excel_path, sheet_name="车辆记录")
        for _, row in record_df.iterrows():
            tree.insert("", tk.END, values=row.tolist())

    def show_space_chart(self):
        """修复：将get_widget()改为get_tk_widget()"""
        chart_win = tk.Toplevel(self)
        chart_win.title("车位状态统计图表")
        chart_win.geometry("600x500")
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        total, used, free, rate = self.data_util.get_parking_stats()
        labels = ["已用车位", "剩余车位"]
        sizes = [used, free]
        colors = ["#ff9999", "#66b3ff"]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title(f"车位状态统计（总车位：{total}）")
        ax.axis("equal")
        canvas = FigureCanvasTkAgg(fig, master=chart_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # 关键修复：get_tk_widget()

    def show_fee_chart(self):
        """修复：将get_widget()改为get_tk_widget()"""
        chart_win = tk.Toplevel(self)
        chart_win.title("收入统计图表")
        chart_win.geometry("800x500")
        fee_df = pd.read_excel(self.data_util.excel_path, sheet_name="收费记录")
        if len(fee_df) == 0:
            messagebox.showwarning("提示", "暂无收费数据！")
            chart_win.destroy()
            return
        fee_df["收费日期"] = pd.to_datetime(fee_df["收费时间"]).dt.date
        daily_fee = fee_df.groupby("收费日期")["收费金额(元)"].sum()
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        daily_fee.plot(kind="bar", ax=ax, color="#66b3ff")
        ax.set_title("每日收入统计")
        ax.set_xlabel("日期")
        ax.set_ylabel("收入（元）")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=chart_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # 关键修复：get_tk_widget()

    # 改造：关闭时停止预览
    def on_closing(self):
        self.opencv_util.release_cam()
        self.stop_camera_preview()  # 停止预览
        self.add_log("程序即将退出")
        if messagebox.askokcancel("退出", "确定要退出系统吗？"):
            self.destroy()

# 主程序入口
if __name__ == "__main__":
    app = ParkingSystem()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()