"""制作连续时间的频率采样数据集"""
import sys
import os
import numpy as np
import glob
import timestampToTime as t2ts


"""参数调整"""
start_year = 2018
start_month = 1
start_day = 1

end_year = 2022
end_month = 11
end_day = 30

frequency = 3 # 采样频率(h)
train_d_p = 3
val_d_p = 1
test_d_p = val_d_p
dataset_divide_prop = train_d_p + val_d_p + test_d_p

"""文件位置"""
# 存储所需位置(按照所需更改)
# xbw
# save_need_location = '/workplace/HaojyTest/forcast_4Gmodel_laterVar_3.17/data/'
# 3090
save_need_location = '/public/ecai/uvq_eur/useData/'
featureNpy_save_location = save_need_location + 'feature.npy'
timeNpy_save_location = save_need_location + 'time.npy'
# 合并各个年份的npy位置(不新增数据集不用更改 目前为 2016-1-1 00:00~2022-11-30 23:00)
load_allYear_location = '/public/ecai/uvq_eur/allYearData/'
featureNpy_allYear_load_location = load_allYear_location + 'feature_all_year.npy'
timeNpy_allYear_load_location = load_allYear_location + 'time_all_year.npy'
# 读取各年npy位置
featureNpy_everyYear_load_location = '/public/ecai/uvq_eur/everyYearData/feat/*'
timeNpy_everyYear_load_location = '/public/ecai/uvq_eur/everyYearData/time/*'

"""合并or读取 各年npy"""
if os.path.exists(featureNpy_allYear_load_location) or \
        os.path.exists(timeNpy_allYear_load_location):
    print("读取存储的全年数据集！")
    # load all year saved npy
    time_allY = np.load(timeNpy_allYear_load_location)
    feature_allY = np.load(featureNpy_allYear_load_location)

else:
    # time all year
    is_first = 0
    for file in sorted(glob.glob(timeNpy_everyYear_load_location)):
        print('load:', file)
        year = int(file[-17:-13])
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

        if is_first == 0:
            time_allY = time_oldNpy
            is_first = 1
        else:
            time_allY = np.concatenate((time_allY, time_oldNpy), axis=0)
    np.save(timeNpy_allYear_load_location,time_allY)
    print('==================time all year saved Success==================')

    # feature all year
    is_first = 0
    for file in sorted(glob.glob(featureNpy_everyYear_load_location)):
        print('load:', file)
        year = int(file[-17:-13])
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

        if is_first == 0:
            feature_allY = feature_oldNpy
            is_first = 1
        else:
            feature_allY = np.concatenate((feature_allY, feature_oldNpy), axis=1)
    np.save(featureNpy_allYear_load_location,feature_allY)
    print('==================feature all year saved Success==================')


"""判断数据集是否健康"""
if feature_allY.shape[1] != time_allY.shape[0]:
    print("数据集维度有误！")
    sys.exit()


"""npy操作"""
# get all dataset [everyHour]
start_timestamp = t2ts.timeToTimestamp(start_year,start_month,start_day,0)
end_timestamp = t2ts.timeToTimestamp(end_year,end_month,end_day,0)

for i in range(time_allY.shape[0]):
    if time_allY[i] == start_timestamp:
        start_idx = i
    if time_allY[i] == end_timestamp:
        end_idx = i

time_allY = time_allY[start_idx:end_idx] # start day 0点 - end day 23点
feature_allY = feature_allY[:,start_idx:end_idx,:]

# get all dataset [already sample]
is_first = 0
hour_sum = time_allY.shape[0]
if hour_sum % 24 != 0:
    print('error')
    sys.exit()
range_bound = int(hour_sum / frequency) + 1
for i in range(range_bound):
    site = frequency * i
    node_feature = feature_allY[:,site:site+1,:]
    node_time = time_allY[site:site+1]
    if is_first == 0:
        feature_needY = node_feature
        time_needY = node_time
        is_first = 1
    else:
        feature_needY = np.concatenate((feature_needY,node_feature),axis=1)
        time_needY = np.concatenate((time_needY,node_time),axis=0)

# save needY
np.save(featureNpy_save_location,feature_needY)
print('==================feature needed year saved Success==================')
np.save(timeNpy_save_location,time_needY)
print('==================time needed year saved Success==================')

# divide border
allData_num = time_needY.shape[0]
batch_num = int(allData_num/dataset_divide_prop)
train_start_idx = 0
train_end_idx = batch_num * train_d_p - 1
val_start_idx = train_end_idx + 1
val_end_idx = train_end_idx + batch_num * val_d_p - 1
test_start_idx = val_end_idx + 1
test_end_idx = -1

# get train date
train_start_ts = time_needY[train_start_idx]
train_end_ts = time_needY[train_end_idx]
print('train_start_ts:')
t2ts.timestampToDate(train_start_ts)
print('train_end_ts:')
t2ts.timestampToDate(train_end_ts)

# get val date
val_start_ts = time_needY[val_start_idx]
val_end_ts = time_needY[val_end_idx]
print('val_start_ts:')
t2ts.timestampToDate(val_start_ts)
print('train_end_ts:')
t2ts.timestampToDate(val_end_ts)

# get test date
test_start_ts = time_needY[test_start_idx]
test_end_ts = time_needY[test_end_idx]
print('test_start_ts:')
t2ts.timestampToDate(test_start_ts)
print('test_end_ts:')
t2ts.timestampToDate(test_end_ts)