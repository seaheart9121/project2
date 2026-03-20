#coding:utf-8
import datetime
def DtCale(date1,date2):
    date_format='%Y-%m-%d %H:%M'
    date1=datetime.datetime.strptime(date1,date_format) # 将字符串转成datetime
    # date2是系统时间
    date2=datetime.datetime(*date2[:5]) # 昨天讲的，获取到的是年，月，日，时，分
    # 时间差
    time_diff=date2-date1 # 秒
    # 将秒转成小时
    hours_diff=time_diff.total_seconds()/3600
    return round(hours_diff) # 采用的计费方式是四舍五入
    # 四舍五入，向上取整，还是向下取整
 #round() 四舍五入    # math.ceil()向上取整

def get_week_number(k): # 根据给定的日期计算星期几?
    # 将字符串类型转成datetime类型
    date_time_obj=datetime.datetime.strptime(k,'%Y-%m-%d %H:%M')
    # 将datetime类型转成structtime对象
    # structtime对象中包含9个元素，年，月，日，时，分，秒，星期，一年中的第几天，夏令时
    struct_time_obj=date_time_obj.timetuple()
    return struct_time_obj.tm_wday # 返回的就是星期几


