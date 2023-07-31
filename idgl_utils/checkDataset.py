import os
import sys
import numpy as np

"""需改参数"""
node_num = '133'
start_years = 2018

"""不变参数"""
dataset_fp = '/public/ecai/uvq_us/everyYearData/'
# feature_fp = dataset_fp + 'feat/' + node_num + 'feature'
# time_fp = dataset_fp + 'time/' + node_num + 'time'
feature_fp = dataset_fp + 'feat'
time_fp = dataset_fp + 'time'
# feature_fn = 'feature_' + node_num + 'nodes_' # 2021.npy
# time_fn = 'time_' + node_num + 'nodes_'
feature_fn = '_US_feat'
time_fn = '_US_time'
length = len(os.listdir(feature_fp))
for i in range(length):
    start_year = start_years
    start_year = start_year + i
    feature_site = feature_fp + '/' + str(start_year) + feature_fn + '.npy'
    time_site = time_fp + '/' + str(start_year) + time_fn + '.npy'
    feature_file = np.load(feature_site)
    time_file = np.load(time_site)
    if time_file.shape[0] != feature_file.shape[1]:
        print('Error,',start_year,' time_file.shape[0]=',time_file.shape[0],',feature_file.shape[1]=',feature_file.shape[1])
        sys.exit()
    else:
        print("Success!")
