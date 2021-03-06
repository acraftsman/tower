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
        if t[1] > 300000*60:
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

df = pd.read_csv("/Users/alanhu/Desktop/merge_device.csv")
 
bool_filter1 = (df["Device_ID"] != 863158020786267) & (df["Device_ID"] != 863158020785004) & (df["Device_ID"] != 863158020756880) & (df["Device_ID"] != 863158020758589) & (df["Device_ID"] != 863158020787299) & \
               (df["Device_ID"] != 863158020784775) & (df["Device_ID"] != 863158020786887) & (df["Device_ID"] != 863158020758175) & (df["Device_ID"] != 863158020758431) & \
               (df["Device_ID"] != 863158020783579) & (df["Device_ID"] != 863158020757961) & (df["Device_ID"] != 863158020784460)
df_correct = df[bool_filter1].copy()
result = tower_rule(df_correct)