"""制作季节性的频率采样数据集"""

import sys
import os
import numpy as np
import glob
import timestampToTime as t2ts


"""参数调整"""
start_year = 2016
end_year = 2022

start_month = 4
start_day = 1
end_month = 7
end_day = 31

frequency = 1 # 采样频率(h)
train_d_p = 3
val_d_p = 1
test_d_p = val_d_p
dataset_divide_prop = train_d_p + val_d_p + test_d_p

"""文件位置"""
# 存储所需位置(按照所需更改)
save_need_location = '/home/gnn/HaojyTest/forcast_tp_3.8/data/'
featureNpy_save_location = save_need_location + 'feature_2160nodes.npy'
timeNpy_save_location = save_need_location + 'time_2160nodes.npy'

# 读取各年npy位置(不用更改)
featureNpy_everyYear_load_location = '/public/npydata/178feature/feature_178nodes_20*'
timeNpy_everyYear_load_location = '/public/npydata/178time/time_178nodes_20*'

start_idxArr = []
end_idxArr = []
year_Arr = []

"""合并or读取 各年npy"""
# time all year
for file in sorted(glob.glob(timeNpy_everyYear_load_location)):
    print('load:', file)
    year = int(file[-8:-4])
    year_Arr.append(year)
    time_oldNpy = np.load(file) # (time,)
    hour_sum = time_oldNpy.shape[0]

    # 验证npy是否损坏
    if (year % 100 != 0 and year % 4 == 0) or (year % 400 == 0):
        if 366 * 24 != hour_sum:
            print(year, "Npy is Error")
            sys.exit()
    else:
        if 365 * 24 != hour_sum:
            if year != 2022:
                print(year, "Npy is Error")
                sys.exit()

    everyY_start_timestamp = t2ts.timeToTimestamp(year, start_month, start_day, 0)
    everyY_end_timestamp = t2ts.timeToTimestamp(year, end_month, end_day, 0)

    for i in range(time_oldNpy.shape[0]):
        if time_oldNpy[i] == everyY_start_timestamp:
            start_idx = i
            start_idxArr.append(i)
        if time_oldNpy[i] == everyY_end_timestamp:
            end_idx = i
            end_idxArr.append(i)

    locals()[f'var_time_{year}'] = time_oldNpy[start_idx:end_idx]

# feature all year
i = 0
for file in sorted(glob.glob(featureNpy_everyYear_load_location)):
    print('load:', file)
    year = int(file[-8:-4])
    feature_oldNpy = np.load(file)  # (node,hour,feature)
    hour_sum = feature_oldNpy.shape[1]

    # 验证npy是否损坏
    if (year % 100 != 0 and year % 4 == 0) or (year % 400 == 0):
        if 366 * 24 != hour_sum:
            print(year, "Npy is Error")
            sys.exit()
    else:
        if 365 * 24 != hour_sum:
            if year != 2022:
                print(year, "Npy is Error")
                sys.exit()

    locals()[f'var_feat_{year}'] = feature_oldNpy[:,start_idxArr[i]:end_idxArr[i],:]
    i = i + 1


"""npy操作"""
is_first_need = 0
for i in range(len(year_Arr)):
    year = year_Arr[i]
    is_first_concat = 0
    if locals()[f'var_feat_{year}'].shape[1] != locals()[f'var_time_{year}'].shape[0]:
        print("数据集维度有误!")
        sys.exit()
    else:
        # 对每年个性化采样
        hour_sum = locals()[f'var_time_{year}'].shape[0]
        range_bound = int(hour_sum / frequency) + 1
        for i in range(range_bound):
            site = frequency * i
            node_feature = locals()[f'var_feat_{year}'][:, site:site + 1, :]
            node_time = locals()[f'var_time_{year}'][site:site + 1]
            if is_first_need == 0:
                feature_needY = node_feature
                time_needY = node_time
                is_first_need = 1
            else:
                feature_needY = np.concatenate((feature_needY, node_feature), axis=1)
                time_needY = np.concatenate((time_needY, node_time), axis=0)

        # 合并各年做完个性化采样的数据集
        if is_first_concat == 0:
            feature_all_needS = feature_needY
            time_all_needS = time_needY
            is_first_concat = 1
        else:
            feature_all_needS = np.concatenate((feature_all_needS,feature_needY),axis=1)
            time_all_needS = np.concatenate((time_all_needS,time_needY),axis=0)


# save needY
np.save(featureNpy_save_location,feature_all_needS)
print('==================feature needed year saved Success==================')
np.save(timeNpy_save_location,time_all_needS)
print('==================time needed year saved Success==================')

# divide border
allData_num = time_all_needS.shape[0]
batch_num = int(allData_num/dataset_divide_prop)
train_start_idx = 0
train_end_idx = batch_num * train_d_p - 1
val_start_idx = train_end_idx + 1
val_end_idx = train_end_idx + batch_num * val_d_p - 1
test_start_idx = val_end_idx + 1
test_end_idx = -1

# get train date
train_start_ts = time_all_needS[train_start_idx]
train_end_ts = time_all_needS[train_end_idx]
print('train_start_ts:')
t2ts.timestampToDate(train_start_ts)
print('train_end_ts:')
t2ts.timestampToDate(train_end_ts)

# get val date
val_start_ts = time_all_needS[val_start_idx]
val_end_ts = time_all_needS[val_end_idx]
print('val_start_ts:')
t2ts.timestampToDate(val_start_ts)
print('train_end_ts:')
t2ts.timestampToDate(val_end_ts)

# get test date
test_start_ts = time_all_needS[test_start_idx]
test_end_ts = time_all_needS[test_end_idx]
print('test_start_ts:')
t2ts.timestampToDate(test_start_ts)
print('test_end_ts:')
t2ts.timestampToDate(test_end_ts)