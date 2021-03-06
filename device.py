#coding=utf-8
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import sys,os

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
# root_path = sys.path[0] + "/"
# root_path = "/Users/alanhu/dataset/20150722_datacheck/20150701000000,20150721235959/"
# root_path = "/Users/alanhu/dataset/阳光数据导出20150604/SZCWJRN_20150604.00/"
# root_path = "/Users/alanhu/dataset/chainway_20150716/SUNSJRN_20150716.00/"
# root_path = "/Users/alanhu/dataset/20150724_daochu/"
# root_path = "/Users/alanhu/dataset/new_Chainway_20150707/SUNSJRN_20150707.00/"
# root_path = "/Users/alanhu/dataset/20150702000000,20150728235959/"
# root_path = "/Users/alanhu/dataset/SZCWJRN_20150813.00/"
root_path = "/Users/alanhu/dataset/20150917000000,20150917235959/"
result_folder = root_path + "by_device_total_result/"
merge_trip_folder = result_folder + "merge_trips_of_device/"
# mid_result_folder = result_folder + "middle_result/"
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
if not os.path.exists(merge_trip_folder):
    os.mkdir(merge_trip_folder)
# if not os.path.exists(mid_result_folder):
#     os.mkdir(mid_result_folder)
all_file = os.listdir(root_path)
csv_file = [v for v in all_file if ".csv" in v.lower()]
new_field = ["设备编号", "GPSSpeed大于0且VSSSpeed等于0的占比", "VSSSpeed大于0的占比", \
             "发动机转速不为0的占比", "发动机转速为32767的占比", "总行程时长(分)", "(第22-27项)不通过数量"]
sunshine_rule = ["(20)[Y]正负0.3m/s2", "(21.1)[X]正负0.2m/s2", "(21.2)[Z]9.5m/s2~10.1m/s2", \
                 "(22)[Y]大于0.2", "(23)[X]正负0.05之间", "(24)[Z]正负0.05之间", \
                 "(25)[Y]正负0.05之间", "(26)[X]大于0.2", "(27)[Z]正负0.05之间", \
                 "(20)是否通过检测", "(21.1)是否通过检测", "(21.2)是否通过检测", \
                 "(22)是否通过检测", "(23)是否通过检测", "(24)是否通过检测", \
                 "(25)是否通过检测", "(26)是否通过检测", "(27)是否通过检测"]
new_col_name = new_field + sunshine_rule
total_result = DataFrame()
# 获得行程统计数据的所有行程详细数据文件列表
detail_file = [v for v in csv_file if (("sum" not in v.lower()) & ("evt" not in v.lower()))]
dictfile = {}
for f in detail_file:
    flist = f.split('_')
    if flist[0] in dictfile:
        dictfile[flist[0]].append(f)
    else:
        dictfile[flist[0]]=[f]
device_idx = 0
for k,v in dictfile.items():
    print("Processing: " + k)
    df_src = DataFrame()
    for fname in v:
        df = pd.read_csv(root_path+fname, dtype = {'Field_Mask': object, 'Device_ID': int},sep='|')
        df_src = df_src.append(df, ignore_index=True)
    df_src.to_csv(merge_trip_folder + k + ".csv", index = False)
    data_count = len(df_src)
    # 计算行程时长
    tripID_list = df_src["Trip_Number"].unique()
    trip_duration_list = []
    total_trip_duration = 0
    for tripIndex,tripID in enumerate(tripID_list):
        single_trip = df_src[df_src["Trip_Number"] == tripID]
        single_trip_duration = single_trip["Time_Stamp"].iloc[len(single_trip)-1] - single_trip["Time_Stamp"].iloc[0]
        total_trip_duration += single_trip_duration
        trip_duration_list.append([tripID, single_trip_duration])


    tower_result = tower_rule(df_src)
    # tower_result["group1_middata"].to_csv(mid_result_folder + "group1_middata" + k + ".csv")
    # tower_result["group2_middata"].to_csv(mid_result_folder + "group2_middata" + k + ".csv")
    
    GPSSpeed_gt0_and_VSSSpeed_et0_rate = len(df_src[(df_src["GPS_Speed"]>0) & (df_src["VSS_Speed"]==0)]) / float(data_count)
    VSSSpeed_gt0_rate = len(df_src[df_src["VSS_Speed"]>0]) / float(data_count)
    Engine_RPM_not_et_0_rate = len(df_src[df_src["Engine_RPM"]!=0]) / float(data_count)
    # Engine_RPM_et_0_rate = len(df_src[df_src["Engine_RPM"]==0]) / float(data_count)
    Engine_RPM_et_32767_rate = len(df_src[df_src["Engine_RPM"]==32767]) / float(data_count)
    total_trip_duration = total_trip_duration / 60
    no_pass_count = tower_result["rule_bool"][3:].count(False)
    # 新字段
    value_list = [df_src["Device_ID"].iloc[0], GPSSpeed_gt0_and_VSSSpeed_et0_rate, VSSSpeed_gt0_rate, \
                  Engine_RPM_not_et_0_rate, Engine_RPM_et_32767_rate, total_trip_duration, no_pass_count] \
                  + tower_result["rule_value"] 
                 
    len_value_list = len(value_list)
    for index, value in enumerate(value_list):
        total_result.loc[device_idx, new_col_name[index]] = value
    for index, value in enumerate(tower_result["rule_bool"]):
        i = len_value_list + index
        if tower_result["rule_bool"][index]:
            total_result.loc[device_idx, new_col_name[i]] = "通过"
        else:
            total_result.loc[device_idx, new_col_name[i]] = "不通过"
    device_idx += 1
total_result.to_excel(result_folder + "device_result.xlsx", sheet_name="Sheet1", engine='xlsxwriter', index = False)