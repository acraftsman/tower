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
# 过滤行程
def filter_trip(datasrc):
    tripID_list = datasrc["Trip_Number"].unique()
    trip_duration_list = []
    for tripIndex,tripID in enumerate(tripID_list):
        single_trip = datasrc[datasrc["Trip_Number"] == tripID]
        trip_duration_list.append([tripID, single_trip["Time_Stamp"].iloc[len(single_trip)-1] - single_trip["Time_Stamp"].iloc[0]])
    abnormal_trip = []
    for t in trip_duration_list:
        # 单位是秒
        if t[1] > 300*60:
            abnormal_trip.append(t[0])
            datasrc = datasrc[datasrc["Trip_Number"] != t[0]]
    return [datasrc, abnormal_trip]
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
# ======程序开始======
# root_path = "/Users/alanhu/dataset/20150724_daochu/"
# root_path = "/Users/alanhu/dataset/new_Chainway_20150707/SUNSJRN_20150707.00/"
root_path = "/Users/alanhu/dataset/new_chainway_20150716/SUNSJRN_20150716.00/"
result_folder = root_path + "data_diagnose_result/"
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
all_file = os.listdir(root_path)
csv_file = [v for v in all_file if ".csv" in v.lower()]
new_field = ["设备编号", "Accel_Longitudinal平均值","Accel_Longitudinal标准差","Accel_Lateral平均值","Accel_Lateral标准差",\
             "Accel_Vertical平均值", "Accel_Vertical标准差", "设备总行程时长", "平均行程时长"]
sunshine_rule = ["(20)[Y]正负0.3m/s2", "(21.1)[X]正负0.2m/s2", "(21.2)[Z]9.5m/s2~10.1m/s2", \
                 "(22)[Y]大于0.2", "(23)[X]正负0.05之间", "(24)[Z]正负0.05之间", \
                 "(25)[Y]正负0.05之间", "(26)[X]大于0.2", "(27)[Z]正负0.05之间", \
                 "(20)是否通过检测", "(21.1)是否通过检测", "(21.2)是否通过检测", \
                 "(22)是否通过检测", "(23)是否通过检测", "(24)是否通过检测", \
                 "(25)是否通过检测", "(26)是否通过检测", "(27)是否通过检测"]
new_col_name = new_field + sunshine_rule
detail_file = [v for v in csv_file if (("sum" not in v.lower()) & ("evt" not in v.lower()))]
dictfile = {}
for f in detail_file:
    flist = f.split('_')
    if flist[0] in dictfile:
        dictfile[flist[0]].append(f)
    else:
        dictfile[flist[0]]=[f]
total_result = DataFrame()
device_idx = 0
for k,v in dictfile.items():
    print("Processing: " + k)
    deivce_df = DataFrame()
    total_trip_duration = 0
    for fname in v:
        df = pd.read_csv(root_path+fname, dtype = {'Field_Mask': object, 'Device_ID': int},sep='|')
        total_trip_duration += (df["Time_Stamp"].iloc[len(df)-1]-df["Time_Stamp"].iloc[0])/60
        deivce_df = deivce_df.append(df, ignore_index=True)
    mean_accel_longitudinal = deivce_df["Accel_Longitudinal"].mean()
    std_accel_longitudinal = deivce_df["Accel_Longitudinal"].std()
    mean_accel_lateral = deivce_df["Accel_Lateral"].mean()
    std_accel_lateral = deivce_df["Accel_Lateral"].std()
    mean_accel_vertical = deivce_df["Accel_Vertical"].mean()
    std_accel_vertical = deivce_df["Accel_Vertical"].std()

    trip_duration_mean = total_trip_duration/len(v)
    tower_result = tower_rule(deivce_df)

    new_value = [mean_accel_longitudinal,std_accel_longitudinal,mean_accel_lateral,\
                 std_accel_lateral, mean_accel_vertical, std_accel_vertical, total_trip_duration,\
                 trip_duration_mean
                ] \
                + tower_result["rule_value"]

    value_list = [deivce_df["Device_ID"].iloc[0]] + new_value
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
total_result.to_excel(result_folder + "data_diagnose_result.xlsx", sheet_name="Sheet1", engine='xlsxwriter', index = False)