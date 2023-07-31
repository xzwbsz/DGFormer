import glob
import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
from netCDF4 import Dataset
from utils import config, file_dir
import numpy as np
import glob as gl

# dataset_fp = file_dir['dataset1_fp']
# for fdir in sorted(glob.glob(dataset_fp)):
#     print(fdir)

array_feature = np.load('data/feature_2160nodes.npy')
array_time = np.load('data/time_2160nodes.npy')
# for i in range(2116):
#     for j in range(31):
#         for z in range(8):
#             if np.isnan(array1[i][j][z]):
#                 # print("[",i,",",j,",",z,"]",array1[i][j][z])
#                 print(i)
#                 break;
num = array_time.shape[0]
for i in range(int(array_time[0].shape)):
    print(1)
print(array_feature)
print(array_time)

# print(nc_file.variables.keys())

# nc_file = Dataset('/climatedata/data2/spring_data_2016_2022/2016-3.nc')
# print(nc_file)

# from decimal import Decimal
# a = 1.346
# a = "1.345"
# a_t = Decimal(a).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
# print(float(a_t))
#
# print(round(1.345,2))