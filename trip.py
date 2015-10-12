#coding:utf-8
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import sys
import os
import time
import datetime
def time2stamp(t):
    time_arr = time.strptime(t, "%Y-%m-%d %H:%M:%S")
    time_stamp = int(time.mktime(time_arr))
    return time_stamp
def stamp2time(s):
    # timestamp转换为UTC时间字符串
    time_str = datetime.datetime.utcfromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S')
    return time_str
def get_invalid_bool(df):
    invalid_number1 = df['Field_Mask'] == '1F'
    invalid_number2 = df['Field_Mask'] == '07'
    invalid_number3 = df['Field_Mask'] == '18'
    invalid_number4 = df['Field_Mask'] == '0'
    bool_invalid = invalid_number1 | invalid_number2 | invalid_number3 | invalid_number4
    return bool_invalid
# # 计算阳光数据校验规则
# def tower_rule(datasrc):
#     data1 = datasrc.copy()
#     # # 过滤掉行程
#     # abnormal_trip = []
#     # for t in trip_duration_list:
#     #     if t[1] < 10:
#     #         abnormal_trip.append(t[0])
#     #         data1 = data1[data1["Trip_Number"] != t[0]]
#     R1 = data1["Accel_Longitudinal"].mean()/10
#     R2 = data1["Accel_Lateral"].mean()/10
#     R3 = data1["Accel_Vertical"].mean()/10

#     # 计算GPS_Speed变化值和GPS_Heading变化值，以及处理GPS_Heading值
#     data1["GPSSpeed_diff"] = data1["GPS_Speed"].diff()
#     # 去掉不连续的数据，计算GPS_Speed变化值
#     data1["TimeStamp_diff1"] = data1["Time_Stamp"].diff()
#     GPS_Speed_diff = data1["GPSSpeed_diff"]
#     GPS_Speed_diff = data1[data1["TimeStamp_diff1"]==1]["GPSSpeed_diff"]

#     R4 = data1["Accel_Longitudinal"].corr(GPS_Speed_diff)
#     R5 = data1["Accel_Lateral"].corr(GPS_Speed_diff)
#     R6 = data1["Accel_Vertical"].corr(GPS_Speed_diff)

#     data2 = data1[data1["GPS_Speed"]>=200].copy()
#     # GPSHeading_diff处理
#     data2["GPSHeading_diff"] = data2["GPS_Heading"].diff()
#     data2.loc[data2["GPSHeading_diff"]<-25000, "GPSHeading_diff"] += 36000
#     data2.loc[data2["GPSHeading_diff"]>25000, "GPSHeading_diff"] -= 36000
#     # 去掉不连续数据，计算GPS_Heading变化值
#     data2["TimeStamp_diff2"] = data2["Time_Stamp"].diff()
#     GPS_Heading_diff = data2["GPSHeading_diff"]
#     GPS_Heading_diff = data2[data2["TimeStamp_diff2"]==1]["GPSHeading_diff"]
#     R7 = data2["Accel_Longitudinal"].corr(GPS_Heading_diff)
#     R8 = data2["Accel_Lateral"].corr(GPS_Heading_diff)
#     R9 = data2["Accel_Vertical"].corr(GPS_Heading_diff)
#     bool_list = [R1 >= -0.3 and R1 <= 0.3, R2 >= -0.2 and R2 <= 0.2, R3 >= 9.5 and R3 <= 10.1,\
#                  R4 >= 0.2, R5 >= -0.05 and R5 <= 0.05, R6 >= -0.05 and R6 <= 0.05, \
#                  R7 >= -0.05 and R7 <= 0.05, R8 >= 0.2, R9 >= -0.05 and R9 <= 0.05]
#     rule_result = {"rule_value":[R1, R2, R3, R4, R5, R6, R7, R8, R9],
#                    "rule_bool": bool_list,
#                    "group1_middata": data1,
#                    "group2_middata": data2}
#     return rule_result
def get_valid_df(df, bitpos):
    df_cp = df.copy()
    df_cp["bin_Field_Mask"] = df_cp["Field_Mask"].apply(lambda x: bin(int(x,16)))
    select_bool_1 = df_cp["bin_Field_Mask"].apply(lambda x: len(x)) > bitpos+1+2
    df_cp = df_cp[select_bool_1]
    select_bool_2 = df_cp["bin_Field_Mask"].apply(lambda x: x[bitpos+1]) == '1' 
    df_cp = df_cp[select_bool_2]
    valid_GPS_Speed_df = df_cp.copy()
    return valid_GPS_Speed_df
# 计算阳光数据校验规则
def tower_rule(datasrc):
    data1 = datasrc.copy()
    data2 = datasrc.copy()
    
    R1 = data1["Accel_Longitudinal"].mean()/10
    R2 = data1["Accel_Lateral"].mean()/10
    R3 = data1["Accel_Vertical"].mean()/10

    # 1. 先计算Time_Stamp的间隔
    # 2. 过滤掉Field_Mask不为3FF的数据
    # 3. 计算GPS_Speed的变化值
    # 4. 过滤掉Time_Stamp的间隔不为1的数据
    # 5. 过滤完后得到GPS_Speed的变化值
    data1["TimeStamp_diff1"] = data1["Time_Stamp"].diff()
    data1 = get_valid_df(data1,4)
    # data1 = data1[data1["Field_Mask"] == "3FF"]
    data1["GPSSpeed_diff"] = data1["GPS_Speed"].diff()
    time_lianxu_df1 = data1[data1["TimeStamp_diff1"]==1]
    GPS_Speed_diff = time_lianxu_df1["GPSSpeed_diff"]
    R4 = time_lianxu_df1["Accel_Longitudinal"].corr(GPS_Speed_diff)
    R5 = time_lianxu_df1["Accel_Lateral"].corr(GPS_Speed_diff)
    R6 = time_lianxu_df1["Accel_Vertical"].corr(GPS_Speed_diff)


    # 1. 过滤掉GPS_Speed小于20公里/时的数据
    # 2. 计算GPS_Heading的变化值（每个值减去相邻的上一个值）
    # 3. 处理GPS_Heading变化值
    # 4. 计算Time_Stamp的间隔
    # 5. 过滤Time_Stamp的间隔不为1的数据
    data2 = data2[data2["GPS_Speed"]>=200].copy()
    # GPSHeading_diff处理
    data2["GPSHeading_diff"] = data2["GPS_Heading"].diff()
    data2.loc[data2["GPSHeading_diff"]<-25000, "GPSHeading_diff"] += 36000
    data2.loc[data2["GPSHeading_diff"]>25000, "GPSHeading_diff"] -= 36000
    data2["TimeStamp_diff2"] = data2["Time_Stamp"].diff()
    time_lianxu_df2 = data2[data2["TimeStamp_diff2"]==1]
    GPS_Heading_diff = time_lianxu_df2["GPSHeading_diff"]
    R7 = time_lianxu_df2["Accel_Longitudinal"].corr(GPS_Heading_diff)
    R8 = time_lianxu_df2["Accel_Lateral"].corr(GPS_Heading_diff)
    R9 = time_lianxu_df2["Accel_Vertical"].corr(GPS_Heading_diff)
    bool_list = [R1 >= -0.3 and R1 <= 0.3, R2 >= -0.2 and R2 <= 0.2, R3 >= 9.5 and R3 <= 10.1,\
                 R4 >= 0.2, R5 >= -0.05 and R5 <= 0.05, R6 >= -0.05 and R6 <= 0.05, \
                 R7 >= -0.05 and R7 <= 0.05, R8 >= 0.2, R9 >= -0.05 and R9 <= 0.05]
    rule_result = {"rule_value":[R1, R2, R3, R4, R5, R6, R7, R8, R9],
                   "rule_bool": bool_list
                  }
    return rule_result
# root_path = sys.path[0]+"/"
# root_path = "/Users/alanhu/dataset/20150722_datacheck/20150701000000,20150721235959/"
# root_path = "/Users/alanhu/dataset/new_Chainway_20150707/SUNSJRN_20150707.00/"
# root_path = "/Users/alanhu/dataset/new_chainway_20150716/SUNSJRN_20150716.00/"
# root_path = "/Users/alanhu/dataset/20150724_daochu/"
root_path = "/Users/alanhu/dataset/chainway_20150716/SUNSJRN_20150716.00/"
# root_path = "/Users/alanhu/dataset/20150702000000,20150728235959/"
# root_path = "/Users/alanhu/dataset/20150808000000,20150818235959/"
# root_path = "/Users/alanhu/dataset/20150818000000,20150820235959/"
all_file = os.listdir(root_path)
csv_file = [v for v in all_file if ".csv" in v.lower() ]
# 获取行程统计数据文件列表
trip_stat_file = [v for v in csv_file if "sum" in v.lower()]
# 获得行程统计数据的所有行程详细数据文件列表
detail_file = [v for v in csv_file if (("sum" not in v.lower()) & ("evt" not in v.lower()))]
# 参数定义
# new_col_name = ["data_count", "interval_gt1_count", "interval_ltet0_count", "interval_outlier_rate", "interval_max", "invalid_count", "invalid_rate", "GPS_Speed_max", "GPS_Speed_average", "VSS_Speed_max", "VSS_Speed_average", "head_invalid_count", "tail_invalid_count", "interval_gt3_count"]
additional_list = ["开始时间", "结束时间", "行程时长(分)", "数据总量", "头定位无效点数量", "尾定位无效点数量", "中间定位无效点数量", \
                   "总定位无效点数量", "总定位无效占比",\
                   "间隔小于0的数量", "间隔等于0的数量", "间隔大于1的数量", "间隔在[2,3]范围的数量", "间隔在[4,10]范围的数量",\
                   "间隔在[10,+∞)范围的数量", "间隔异常率", "间隔最大值", \
                   "最大10%的GPS速度平均值", "GPS最大速度", "GPS平均速度", "最大10%的VSS速度平均值", "VSS最大速度", "VSS平均速度",\
                   "[1,5]的定位无效段数", "[6,30]的定位无效段数", "[31,60]的定位无效段数", "[61,+∞)的定位无效段数", \
                   "发动机转速非零非空值占比", "[162,166]间数据的占比", "总线连接不上和不稳定的占比", \
                   "VSS_Speed为0的数量", "VSS_Speed不为0的数量", "VSS_Speed不为0的占比", \
                   "Engine_RPM为0的数量", "Engine_RPM不为0的数量", "Engine_RPM不为0的占比"]
sunshine_rule = ["(20)[Y]正负0.3m/s2", "(21.1)[X]正负0.2m/s2", "(21.2)[Z]9.5m/s2~10.1m/s2", \
                 "(22)[Y]大于0.2", "(23)[X]正负0.05之间", "(24)[Z]正负0.05之间", \
                 "(25)[Y]正负0.05之间", "(26)[X]大于0.2", "(27)[Z]正负0.05之间", \
                 "(20)是否通过检测", "(21.1)是否通过检测", "(21.2)是否通过检测", \
                 "(22)是否通过检测", "(23)是否通过检测", "(24)是否通过检测", \
                 "(25)是否通过检测", "(26)是否通过检测", "(27)是否通过检测"]
new_col_name = additional_list + sunshine_rule
total_result = DataFrame()
result_folder = root_path + "by_trip_total_result/"
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
result_file_path = result_folder +"trip_result.xlsx"
# arranged_data_folder = root_path + result_folder + "arranged_detaildata/"
# os.mkdir(arranged_data_folder)

# 获取行程统计数据
trip_stat_src = pd.read_csv(root_path+trip_stat_file[0], sep = '|', header = None)
colname = ["Device_ID", "Trip_Number", "Time_Stamp", "Enrolled_VIN", "Detected_VIN", "Time_Zone"]
trip_stat_src.columns = colname
for val in new_col_name:
    trip_stat_src[val] =np.nan
# 根据每个行程的详细数据进行二次筛选
for idx,fname in enumerate(detail_file):
    device_pn = int(fname.split('_')[0])
    s2t = lambda x: stamp2time(x)
    detaildata_src = pd.read_csv(root_path + fname, dtype = {'Field_Mask': object}, sep = '|')
    detaildata_src.Device_ID.astype(str)
    data_count = len(detaildata_src)
    if data_count == 0:
        continue
    trip_id = detaildata_src.irow(0)['Trip_Number']

    # 定位有效无效相关指标
    invalid_count = len(detaildata_src[get_invalid_bool(detaildata_src)])
    valid_data = detaildata_src[~get_invalid_bool(detaildata_src)].copy()
    valid_count = len(valid_data)
    if valid_count > 0:        
        start_ix = valid_data.icol(0).index[0]
        end_ix = valid_data.icol(0).index[len(valid_data)-1]
        head_invalid_count = start_ix
        tail_invalid_count = data_count - end_ix - 1
        middle_data = detaildata_src[start_ix:end_ix+1]
        middle_invalid_count = len(middle_data[get_invalid_bool(middle_data)])
    elif valid_count == 0:
        head_invalid_count = invalid_count
        tail_invalid_count = 0
        middle_invalid_count = 0
    # 生成数据指标    
    time_diff = detaildata_src['Time_Stamp'].diff()
    interval_gt1_count = len(time_diff[time_diff > 1])
    interval_lt0_count = len(time_diff[time_diff < 0])
    interval_et0_count = len(time_diff[time_diff == 0])
    interval_between_2_3 = len(time_diff[(time_diff >= 2) & (time_diff <= 3)])
    interval_between_4_10 = len(time_diff[(time_diff >= 4) & (time_diff <= 10)])
    interval_gt10 = len(time_diff[time_diff >= 11])
    interval_max = time_diff.max()
    invalid_rate = invalid_count / data_count
    if data_count > 1:        
        interval_outlier_rate = (interval_gt1_count + interval_lt0_count + interval_et0_count) / (data_count - 1)
    else:
        interval_outlier_rate = 0
    trip_duration = (detaildata_src["Time_Stamp"].iloc[len(detaildata_src)-1] - detaildata_src["Time_Stamp"].iloc[0])/60
    # valid_bejing_time = detaildata_src['Time_Stamp'].apply(s2t)
    time_stamp= detaildata_src['Time_Stamp']
    start_time = time_stamp.iloc[0]
    end_time = time_stamp.iloc[len(time_stamp)-1]
    GPS_Speed_10per_average = detaildata_src.sort("GPS_Speed",ascending=False).iloc[0:int(data_count * 0.1 + 1)]["GPS_Speed"].mean()/10
    GPS_Speed_max = detaildata_src["GPS_Speed"].max()/10
    GPS_Speed_average = detaildata_src[detaildata_src["GPS_Speed"]!=0]["GPS_Speed"].mean()/10 
    VSS_Speed_10per_average = detaildata_src.sort("VSS_Speed",ascending=False).iloc[0:int(data_count * 0.1 + 1)]["VSS_Speed"].mean()/10
    VSS_Speed_max = detaildata_src["VSS_Speed"].max()/10
    VSS_Speed_average = detaildata_src[detaildata_src["VSS_Speed"]!=0]["VSS_Speed"].mean()/10
    EngineRPM_not0_notnull_count = len(detaildata_src[(detaildata_src["Engine_RPM"]!=0) & (detaildata_src["Engine_RPM"].notnull())])
    EngineRPM_not0_notnull_value_rate = EngineRPM_not0_notnull_count / float(data_count)

    VSS_Speed_et_0_count = len(detaildata_src[detaildata_src["VSS_Speed"]==0])
    VSS_Speed_not_et_0_count = len(detaildata_src[detaildata_src["VSS_Speed"]!=0])
    VSS_Speed_not_et_0_rate = len(detaildata_src[detaildata_src["VSS_Speed"]!=0])/float(data_count)

    Engine_RPM_et_0_count = len(detaildata_src[detaildata_src["Engine_RPM"]==0])
    Engine_RPM_not_et_0_count = len(detaildata_src[detaildata_src["Engine_RPM"]!=0])
    Engine_RPM_not_et_0_rate = len(detaildata_src[detaildata_src["Engine_RPM"]!=0])/float(data_count)
    # Ending_Timestamp = detaildata_src["Time_Stamp"].iloc[data_count-1]
    # Backward180_Start = Ending_Timestamp - 180
    # Final180s_data = detaildata_src[detaildata_src['Time_Stamp']>Backward180_Start]
    # if len(Final180s_data)==180:
    # Final180s_0_count = len(Final180s_data[(Final180s_data["GPS_Speed"]==0) & (Final180s_data["VSS_Speed"]==0) & (Final180s_data["Engine_RPM"]==0)])
    # if(data_count==Final180s_0_count):
        # EngineRPM_not0_notnull_value_rate = 0
    # else:
        # EngineRPM_not0_notnull_value_rate = EngineRPM_not0_notnull_count / float(data_count - Final180s_0_count)
    # elif len(Final180s_data)<180:
    #     Engine_abnormal_rate = EngineRPM_not0_notnull_count/ float(data_count)
    #     if(Final180s_data["Trip_Number"].iloc[0]==1432695380):
    #         debug_data = Final180s_data
    GPS_Heading_rate = len(detaildata_src[(detaildata_src["GPS_Heading"] >= 16200) & (detaildata_src["GPS_Heading"] <= 16600)]) / float(data_count)
    Bus_unconnect_instability = len(detaildata_src[((detaildata_src["Engine_RPM"]==0) | (detaildata_src["Engine_RPM"].isnull())) & (detaildata_src["GPS_Speed"]!=0) ] ) / float(data_count)

    valid_data.loc[:,"index_value"] = valid_data.index
    valid_index_diff = valid_data["index_value"].diff()-1
    invalid_segment = valid_index_diff[valid_index_diff > 0]
    range_1_5_invalid_segment_count = len(invalid_segment[(invalid_segment >= 1) & (invalid_segment <= 5)])
    range_6_30_invalid_segment_count = len(invalid_segment[(invalid_segment >= 6) & (invalid_segment <= 30)])
    range_31_60_invalid_segment_count = len(invalid_segment[(invalid_segment >= 31) & (invalid_segment <= 60)])
    range_gtet_61_invalid_segment_count = len(invalid_segment[invalid_segment >= 61])
    if valid_count > 0:
        if head_invalid_count > 0 or tail_invalid_count > 0:
            if head_invalid_count>=1 and head_invalid_count <= 5:
                range_1_5_invalid_segment_count += 1
            elif head_invalid_count>=6 and head_invalid_count <= 30:
                range_6_30_invalid_segment_count += 1
            elif head_invalid_count>=31 and head_invalid_count <= 60:
                range_31_60_invalid_segment_count += 1
            elif head_invalid_count >= 61:
                range_gtet_61_invalid_segment_count += 1

            if tail_invalid_count>=1 and tail_invalid_count <= 5:
                range_1_5_invalid_segment_count += 1
            elif tail_invalid_count>=6 and tail_invalid_count <= 30:
                range_6_30_invalid_segment_count += 1
            elif tail_invalid_count>=31 and tail_invalid_count <= 60:
                range_31_60_invalid_segment_count += 1
            elif tail_invalid_count>=61:
                range_gtet_61_invalid_segment_count += 1
    elif valid_count == 0:
        if head_invalid_count>=1 and head_invalid_count <= 5:
            range_1_5_invalid_segment_count += 1
        elif head_invalid_count>=6 and head_invalid_count <= 30:
            range_6_30_invalid_segment_count += 1
        elif head_invalid_count>=31 and head_invalid_count <= 60:
            range_31_60_invalid_segment_count += 1
        elif head_invalid_count>=61:
            range_gtet_61_invalid_segment_count += 1
    # head_invalid_index = invalid_segment[]
    # tail_invalid_index = 

    tower_result = tower_rule(detaildata_src)
    # # 计算阳光数据校验规则
    # R1 = detaildata_src["Accel_Longitudinal"].mean()/10
    # R2 = detaildata_src["Accel_Lateral"].mean()/10
    # R3 = detaildata_src["Accel_Vertical"].mean()/10
    # GPS_Speed_diff = detaildata_src["GPS_Speed"].diff()
    # # R4 = detaildata_src[detaildata_src["Accel_Longitudinal"]==0 ]
    # R4 = detaildata_src["Accel_Longitudinal"].corr(GPS_Speed_diff)
    # R5 = detaildata_src["Accel_Lateral"].corr(GPS_Speed_diff)
    # R6 = detaildata_src["Accel_Vertical"].corr(GPS_Speed_diff)
    # filter_src = detaildata_src[detaildata_src["GPS_Speed"] >=200]
    # GPS_Heading_diff = filter_src["GPS_Heading"].diff()
    # R7 = filter_src["Accel_Longitudinal"].corr(GPS_Heading_diff)
    # R8 = filter_src["Accel_Lateral"].corr(GPS_Heading_diff)
    # R9 = filter_src["Accel_Vertical"].corr(GPS_Heading_diff)

    bool_trip_id = trip_stat_src["Trip_Number"] == trip_id
    bool_device_id = trip_stat_src["Device_ID"] == device_pn
    row_select = bool_trip_id & bool_device_id

    # 新增字段列表
    value_list = [start_time, end_time, trip_duration, data_count, head_invalid_count, tail_invalid_count, middle_invalid_count,\
                  invalid_count, invalid_rate, interval_lt0_count, interval_et0_count, interval_gt1_count,\
                  interval_between_2_3, interval_between_4_10, interval_gt10,\
                  interval_outlier_rate, interval_max, \
                  GPS_Speed_10per_average, GPS_Speed_max, GPS_Speed_average,\
                  VSS_Speed_10per_average, VSS_Speed_max, VSS_Speed_average,\
                  range_1_5_invalid_segment_count, range_6_30_invalid_segment_count,\
                  range_31_60_invalid_segment_count, range_gtet_61_invalid_segment_count,\
                  EngineRPM_not0_notnull_value_rate, GPS_Heading_rate, Bus_unconnect_instability, \
                  VSS_Speed_et_0_count, VSS_Speed_not_et_0_count, VSS_Speed_not_et_0_rate, \
                  Engine_RPM_et_0_count, Engine_RPM_not_et_0_count, Engine_RPM_not_et_0_rate] + tower_result["rule_value"]
    len_value_list = len(value_list)
    for index, value in enumerate(value_list):
        trip_stat_src.loc[row_select, new_col_name[index]] = value_list[index]
    for index, value in enumerate(tower_result["rule_bool"]):
        i = len_value_list + index
        if tower_result["rule_bool"][index]:
            trip_stat_src.loc[row_select, new_col_name[i]] = "通过"
        else:
            trip_stat_src.loc[row_select, new_col_name[i]] = "不通过"
    # detaildata_src.to_csv(arranged_data_folder + fname, index = False)
    print(fname + ": ", idx+1, " ok!")
# new_col_order= [trip_stat_src.columns[0], trip_stat_src.columns[1], trip_stat_src.columns[2], \
#                 trip_stat_src.columns[3], trip_stat_src.columns[4], trip_stat_src.columns[5], \
#                 "开始时间", "结束时间", "行程时长(分)", "数据总量", "头定位无效点数量", "尾定位无效点数量", "中间定位无效点数量", \
#                 "总定位无效点数量", "总定位无效占比",\
#                 "间隔小于0的数量", "间隔等于0的数量", "间隔大于1的数量", "间隔在[2,3]范围的数量", "间隔在[4,10]范围的数量",\
#                 "间隔在[10,+∞)范围的数量", "间隔异常率", "间隔最大值", \
#                 "最大10%的GPS速度平均值", "GPS最大速度", "GPS平均速度", "最大10%的VSS速度平均值", "VSS最大速度", "VSS平均速度",\
#                 "[1,5]的定位无效段数", "[6,30]的定位无效段数", "[31,60]的定位无效段数", "[61,+∞)的定位无效段数",\
#                 "发动机转速非零非空值占比", "[162,166]间数据的占比", "总线连接不上和不稳定的占比",\
#                 "(20)[Y]正负0.3m/s2", "(20)是否通过检测", "(21.1)[X]正负0.2m/s2", "(21.1)是否通过检测",\
#                 "(21.2)[Z]9.5m/s2~10.1m/s2", "(21.2)是否通过检测", "(22)[Y]大于0.2", "(22)是否通过检测",\
#                 "(23)[X]正负0.05之间", "(23)是否通过检测", "(24)[Z]正负0.05之间", "(24)是否通过检测",\
#                 "(25)[Y]正负0.05之间", "(25)是否通过检测", "(26)[X]大于0.2", "(26)是否通过检测",\
#                 "(27)[Z]正负0.05之间", "(27)是否通过检测"]
# new_col_order = [trip_stat_src.columns[0], trip_stat_src.columns[1], trip_stat_src.columns[2], \
#                  trip_stat_src.columns[3], trip_stat_src.columns[4], trip_stat_src.columns[5]] \
#                  + additional_list + sunshine_rule
# trip_stat_src = trip_stat_src[new_col_order]
# trip_stat_src.to_csv(result_file_path, index = False)
trip_stat_src.to_excel(result_file_path, sheet_name="Sheet1", engine='xlsxwriter', index = False)
print("Complete!")



