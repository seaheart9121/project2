# import pandas as pd
# import os
# from datetime import datetime
#
# # 数据工具类（完整保留+适配）
# class DataUtil:
#     def __init__(self, excel_path="./datafile/parking_data.xlsx"):
#         self.excel_path = excel_path
#         # 创建数据文件夹
#         if not os.path.exists("./datafile"):
#             os.makedirs("./datafile")
#         # 初始化Excel数据表
#         self.init_excel()
#
#     # 初始化Excel数据表（确保所有工作表都存在）
#     def init_excel(self):
#         """创建3张核心表：车位信息、车辆记录、收费记录"""
#         if not os.path.exists(self.excel_path):
#             # 1. 车位信息表（可自行修改总车位数）
#             total_space = 10  # 和你界面的总车位对应，可修改
#             parking_space_df = pd.DataFrame({
#                 "车位ID": [f"P{i:03d}" for i in range(1, total_space + 1)],
#                 "车位状态": ["空闲"] * total_space,
#                 "车牌号码": [""] * total_space,
#                 "入场时间": [""] * total_space
#             })
#             # 2. 车辆进出场记录表
#             car_record_df = pd.DataFrame(
#                 columns=["车牌号码", "入场时间", "出场时间", "车位ID", "停车状态"]
#             )
#             # 3. 收费记录表
#             fee_record_df = pd.DataFrame(
#                 columns=["车牌号码", "停车时长(分钟)", "收费金额(元)", "入场时间", "出场时间", "收费时间"]
#             )
#             # 写入Excel
#             with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:
#                 parking_space_df.to_excel(writer, sheet_name="车位信息", index=False)
#                 car_record_df.to_excel(writer, sheet_name="车辆记录", index=False)
#                 fee_record_df.to_excel(writer, sheet_name="收费记录", index=False)
#         # 初始化费率（默认2元/小时，可在界面修改）
#         self.hour_rate = 2
#
#     # 获取车位统计数据
#     def get_parking_stats(self):
#         """返回总车位、已用车位、剩余车位、占用率"""
#         parking_df = pd.read_excel(self.excel_path, sheet_name="车位信息")
#         total = len(parking_df)
#         used = len(parking_df[parking_df["车位状态"] == "占用"])
#         free = total - used
#         occupancy_rate = (used / total) * 100 if total > 0 else 0
#         return total, used, free, occupancy_rate
#
#     # 车辆入场
#     def car_entry(self, plate_num):
#         """车辆入场，分配车位，写入记录"""
#         # 车牌标准化：去除空格、全角空格，转大写
#         plate_num = plate_num.strip().upper().replace(" ", "").replace("　", "")
#         if not plate_num:
#             return False, "车牌号码不能为空！"
#
#         try:
#             # 读取车位信息
#             parking_df = pd.read_excel(self.excel_path, sheet_name="车位信息")
#             # 检查车辆是否已在场
#             if plate_num in parking_df["车牌号码"].values:
#                 return False, "该车辆已在场，无需重复入场！"
#             # 查找空闲车位
#             free_spaces = parking_df[parking_df["车位状态"] == "空闲"]
#             if len(free_spaces) == 0:
#                 return False, "车位已满，无法入场！"
#             # 分配第一个空闲车位
#             space_id = free_spaces.iloc[0]["车位ID"]
#             entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             # 更新车位信息
#             parking_df.loc[parking_df["车位ID"] == space_id, ["车位状态", "车牌号码", "入场时间"]] = ["占用", plate_num,
#                                                                                                   entry_time]
#             # 写入车辆记录
#             car_record_df = pd.read_excel(self.excel_path, sheet_name="车辆记录")
#             new_record = pd.DataFrame({
#                 "车牌号码": [plate_num],
#                 "入场时间": [entry_time],
#                 "出场时间": [""],
#                 "车位ID": [space_id],
#                 "停车状态": ["在场"]
#             })
#             car_record_df = pd.concat([car_record_df, new_record], ignore_index=True)
#             # 读取收费记录
#             fee_record_df = pd.read_excel(self.excel_path, sheet_name="收费记录")
#             # 保存到Excel（使用mode='w'重写整个文件，确保所有表都存在）
#             with pd.ExcelWriter(self.excel_path, engine="openpyxl", mode="w") as writer:
#                 parking_df.to_excel(writer, sheet_name="车位信息", index=False)
#                 car_record_df.to_excel(writer, sheet_name="车辆记录", index=False)
#                 fee_record_df.to_excel(writer, sheet_name="收费记录", index=False)
#             return True, f"入场成功！车牌{plate_num}，分配车位{space_id}，入场时间{entry_time}"
#         except Exception as e:
#             return False, f"入场失败：{str(e)}"
#
#     # 车辆出场+计费（核心修复）
#     def car_exit(self, plate_num):
#         """车辆出场，计算费用，更新记录"""
#         # 车牌标准化：去除空格、全角空格，转大写
#         plate_num = plate_num.strip().upper().replace(" ", "").replace("　", "")
#         if not plate_num:
#             return False, "车牌号码不能为空！"
#
#         try:
#             # 读取数据
#             parking_df = pd.read_excel(self.excel_path, sheet_name="车位信息")
#             car_record_df = pd.read_excel(self.excel_path, sheet_name="车辆记录")
#             fee_record_df = pd.read_excel(self.excel_path, sheet_name="收费记录")
#
#             # 检查车辆是否在场
#             if plate_num not in parking_df["车牌号码"].values:
#                 return False, "该车辆未在场，无法出场！"
#
#             # 获取车辆信息（修复：时间格式匹配 %Y-%m-%d %H:%M:%S）
#             car_info = parking_df[parking_df["车牌号码"] == plate_num].iloc[0]
#             space_id = car_info["车位ID"]
#             entry_time_str = car_info["入场时间"]
#             # 兼容不同时间格式（容错处理）
#             try:
#                 entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
#             except:
#                 entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M")
#
#             exit_time = datetime.now()
#             # 计算停车时长和费用（修复：时长计算逻辑）
#             duration_sec = (exit_time - entry_time).total_seconds()
#             duration_min = round(duration_sec / 60, 2)
#             # 不足1小时按1小时计算
#             hour_count = int(duration_min // 60) if duration_min % 60 == 0 else int(duration_min // 60) + 1
#             fee = hour_count * self.hour_rate
#
#             # 格式化时间
#             exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M:%S")
#             entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M:%S")
#
#             # 更新车位信息
#             parking_df.loc[parking_df["车位ID"] == space_id, ["车位状态", "车牌号码", "入场时间"]] = ["空闲", "", ""]
#
#             # 更新车辆记录
#             car_record_df.loc[
#                 (car_record_df["车牌号码"] == plate_num) & (car_record_df["停车状态"] == "在场"),
#                 ["出场时间", "停车状态"]
#             ] = [exit_time_str, "离场"]
#
#             # 写入收费记录
#             new_fee = pd.DataFrame({
#                 "车牌号码": [plate_num],
#                 "停车时长(分钟)": [duration_min],
#                 "收费金额(元)": [fee],
#                 "入场时间": [entry_time_str],
#                 "出场时间": [exit_time_str],
#                 "收费时间": [exit_time_str]
#             })
#             fee_record_df = pd.concat([fee_record_df, new_fee], ignore_index=True)
#
#             # 保存到Excel（使用mode='w'重写整个文件，确保所有表都存在）
#             with pd.ExcelWriter(self.excel_path, engine="openpyxl", mode="w") as writer:
#                 parking_df.to_excel(writer, sheet_name="车位信息", index=False)
#                 car_record_df.to_excel(writer, sheet_name="车辆记录", index=False)
#                 fee_record_df.to_excel(writer, sheet_name="收费记录", index=False)
#
#             return True, f"出场成功！车牌{plate_num}，停车时长{duration_min}分钟，收费{fee}元"
#         except Exception as e:
#             return False, f"出场失败：{str(e)}"
#
#     # 获取在场车辆列表
#     def get_in_car_list(self):
#         """返回当前在场的车辆列表"""
#         parking_df = pd.read_excel(self.excel_path, sheet_name="车位信息")
#         in_car_df = parking_df[parking_df["车位状态"] == "占用"][["车位ID", "车牌号码", "入场时间"]]
#         return in_car_df.values.tolist()
#
#     # 新增：增加车位
#     def add_parking_spaces(self, new_spaces_count):
#         """在现有车位信息表中追加新的车位"""
#         try:
#             # 读取现有车位信息
#             parking_df = pd.read_excel(self.excel_path, sheet_name="车位信息")
#             # 获取当前最大车位ID，生成新的车位ID
#             existing_ids = parking_df["车位ID"].tolist()
#             if existing_ids:
#                 # 从最大ID数字开始递增，例如 P003 → P004
#                 max_id_num = max([int(id[1:]) for id in existing_ids])
#             else:
#                 max_id_num = 0
#
#             # 生成新的车位数据
#             new_data = []
#             for i in range(1, new_spaces_count + 1):
#                 new_id_num = max_id_num + i
#                 new_id = f"P{new_id_num:03d}"
#                 new_data.append({
#                     "车位ID": new_id,
#                     "车位状态": "空闲",
#                     "车牌号码": "",
#                     "入场时间": ""
#                 })
#             new_df = pd.DataFrame(new_data)
#
#             # 追加到现有表
#             parking_df = pd.concat([parking_df, new_df], ignore_index=True)
#
#             # 读取另外两张表，避免丢失数据
#             car_record_df = pd.read_excel(self.excel_path, sheet_name="车辆记录")
#             fee_record_df = pd.read_excel(self.excel_path, sheet_name="收费记录")
#
#             # 重写整个Excel文件（mode="w"），确保所有表都更新
#             with pd.ExcelWriter(self.excel_path, engine="openpyxl", mode="w") as writer:
#                 parking_df.to_excel(writer, sheet_name="车位信息", index=False)
#                 car_record_df.to_excel(writer, sheet_name="车辆记录", index=False)
#                 fee_record_df.to_excel(writer, sheet_name="收费记录", index=False)
#
#             return True, f"成功添加 {new_spaces_count} 个车位，当前总车位：{len(parking_df)}"
#         except Exception as e:
#             return False, f"添加车位失败：{str(e)}"
#
#     # 新增：删除车位
#     def remove_parking_spaces(self, space_ids):
#         """
#         删除指定车位ID的车位
#         :param space_ids: 要删除的车位ID列表，如 ["P001", "P002"]
#         """
#         try:
#             # 读取现有数据
#             parking_df = pd.read_excel(self.excel_path, sheet_name="车位信息")
#             car_record_df = pd.read_excel(self.excel_path, sheet_name="车辆记录")
#             fee_record_df = pd.read_excel(self.excel_path, sheet_name="收费记录")
#
#             # 检查要删除的车位是否存在且为空闲
#             for space_id in space_ids:
#                 if space_id not in parking_df["车位ID"].values:
#                     return False, f"车位 {space_id} 不存在，无法删除！"
#                 if parking_df[parking_df["车位ID"] == space_id]["车位状态"].values[0] != "空闲":
#                     return False, f"车位 {space_id} 正在被占用，无法删除！"
#
#             # 删除指定车位
#             parking_df = parking_df[~parking_df["车位ID"].isin(space_ids)]
#
#             # 重写整个Excel文件，确保所有表都更新
#             with pd.ExcelWriter(self.excel_path, engine="openpyxl", mode="w") as writer:
#                 parking_df.to_excel(writer, sheet_name="车位信息", index=False)
#                 car_record_df.to_excel(writer, sheet_name="车辆记录", index=False)
#                 fee_record_df.to_excel(writer, sheet_name="收费记录", index=False)
#
#             return True, f"成功删除 {len(space_ids)} 个车位，当前总车位：{len(parking_df)}"
#         except Exception as e:
#             return False, f"删除车位失败：{str(e)}"
#
# # 测试用例
# if __name__ == "__main__":
#     data_util = DataUtil()
#     # 测试车位统计
#     total, used, free, rate = data_util.get_parking_stats()
#     print(f"总车位：{total}，已用：{used}，剩余：{free}，占用率：{rate}%")

import pymysql
from datetime import datetime

class DataUtil:
    def __init__(self):
        self.host = "localhost"
        self.user = "root"
        self.password = "123456"  # 改成你自己的密码
        self.database = "parking_db"
        self.hour_rate = 2

    def get_conn(self):
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn, conn.cursor()

    def close(self, conn, cursor):
        cursor.close()
        conn.close()

    def get_parking_stats(self):
        conn, cursor = self.get_conn()
        cursor.execute("SELECT COUNT(*) AS total FROM parking_space")
        total = cursor.fetchone()["total"]
        cursor.execute("SELECT COUNT(*) AS used FROM parking_space WHERE space_status='占用'")
        used = cursor.fetchone()["used"]
        free = total - used
        rate = (used / total) * 100 if total > 0 else 0
        self.close(conn, cursor)
        return total, used, free, rate

    def car_entry(self, plate_num):
        plate_num = plate_num.strip().upper().replace(" ", "")
        if not plate_num:
            return False, "车牌不能为空"

        try:
            conn, cursor = self.get_conn()
            cursor.execute("SELECT * FROM parking_space WHERE plate_num=%s", (plate_num,))
            if cursor.fetchone():
                self.close(conn, cursor)
                return False, "车辆已在场"

            cursor.execute("SELECT * FROM parking_space WHERE space_status='空闲' LIMIT 1")
            space = cursor.fetchone()
            if not space:
                self.close(conn, cursor)
                return False, "车位已满"

            space_id = space["space_id"]
            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute(
                "UPDATE parking_space SET space_status='占用', plate_num=%s, entry_time=%s WHERE space_id=%s",
                (plate_num, entry_time, space_id)
            )
            cursor.execute(
                "INSERT INTO car_record(plate_num, entry_time, space_id, park_status) VALUES(%s,%s,%s,'在场')",
                (plate_num, entry_time, space_id)
            )
            conn.commit()
            self.close(conn, cursor)
            return True, f"入场成功！车牌：{plate_num}，车位：{space_id}"
        except Exception as e:
            return False, f"入场失败：{str(e)}"

    # 修复3：彻底解决strptime类型不匹配问题
    def car_exit(self, plate_num):
        plate_num = plate_num.strip().upper().replace(" ", "")
        if not plate_num:
            return False, "车牌不能为空"

        try:
            conn, cursor = self.get_conn()
            cursor.execute("SELECT * FROM parking_space WHERE plate_num=%s", (plate_num,))
            car = cursor.fetchone()
            if not car:
                self.close(conn, cursor)
                return False, "车辆不在场"

            space_id = car["space_id"]
            entry_time = car["entry_time"]  # 已经是datetime类型，无需strptime！
            exit_time = datetime.now()

            # 直接计算时长，无需类型转换
            duration_sec = (exit_time - entry_time).total_seconds()
            duration_min = round(duration_sec / 60, 2)
            hour_count = int(duration_min // 60) + (0 if duration_min % 60 == 0 else 1)
            fee = hour_count * self.hour_rate

            exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M:%S")
            entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute(
                "UPDATE parking_space SET space_status='空闲', plate_num=NULL, entry_time=NULL WHERE space_id=%s",
                (space_id,)
            )
            cursor.execute(
                "UPDATE car_record SET exit_time=%s, park_status='离场' WHERE plate_num=%s AND park_status='在场' LIMIT 1",
                (exit_time_str, plate_num)
            )
            cursor.execute(
                "INSERT INTO fee_record(plate_num, park_duration, fee, entry_time, exit_time, fee_time) VALUES(%s,%s,%s,%s,%s,%s)",
                (plate_num, duration_min, fee, entry_time_str, exit_time_str, exit_time_str)
            )
            conn.commit()
            self.close(conn, cursor)
            return True, f"出场成功！车牌：{plate_num}，停车时长{duration_min}分钟，收费{fee}元"
        except Exception as e:
            return False, f"出场失败：{str(e)}"

    def get_in_car_list(self):
        conn, cursor = self.get_conn()
        cursor.execute("SELECT space_id, plate_num, entry_time FROM parking_space WHERE space_status='占用'")
        rows = cursor.fetchall()
        res = [[row["space_id"], row["plate_num"], row["entry_time"].strftime("%Y-%m-%d %H:%M:%S")] for row in rows]
        self.close(conn, cursor)
        return res

    def add_parking_spaces(self, count):
        try:
            conn, cursor = self.get_conn()
            cursor.execute("SELECT MAX(space_id) AS last_id FROM parking_space")
            last = cursor.fetchone()["last_id"]
            num = int(last[1:]) if last else 0

            for i in range(1, count + 1):
                new_num = num + i
                new_id = f"P{new_num:03d}"
                cursor.execute("INSERT INTO parking_space(space_id) VALUES(%s)", (new_id,))

            conn.commit()
            self.close(conn, cursor)
            return True, f"成功添加{count}个车位"
        except Exception as e:
            return False, f"添加失败：{str(e)}"

    def remove_parking_spaces(self, space_ids):
        try:
            conn, cursor = self.get_conn()
            for sid in space_ids:
                cursor.execute("SELECT * FROM parking_space WHERE space_id=%s AND space_status='占用'", (sid,))
                if cursor.fetchone():
                    return False, f"车位{sid}已占用，无法删除"

            placeholders = ",".join(["%s"] * len(space_ids))
            cursor.execute(f"DELETE FROM parking_space WHERE space_id IN ({placeholders})", space_ids)
            conn.commit()
            self.close(conn, cursor)
            return True, f"成功删除{len(space_ids)}个车位"
        except Exception as e:
            return False, f"删除失败：{str(e)}"