"""
    对野生数据集进行处理 按照节点的经纬度坐标将数据集剪裁
    野生数据集(time,lon,lat,attr_num) -> 节点数据集(time,node_idx,attr_num)
    (365,721,1440,3) -> (365,2160,3) 那么每个节点的数据集为(365,3)

    网格化:
    longtitude : [0,0.25, ... ,359.5,359.75] 1440
    latitude : [90,89.75, ... ,0.25,0,-0.25, ... ,-89.75,-90] 721
"""
import os
import sys

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
from netCDF4 import Dataset
from utils import config, file_dir
import numpy as np
import glob

save_feat = '/public/ecai/temp_eur/everyYearData/feat/'
save_time = '/public/ecai/temp_eur/everyYearData/time/'

class Dataset_pre():
    def __init__(self, nodes_dir, nodes_num, fdir, year):
        self.node_attr = []  # 节点特征矩阵
        # 得到字典并读取nc文件
        nc_file = Dataset(fdir)

        feat_file = save_feat + str(year) + '_EUR_feat.npy'
        time_file = save_time + str(year) + '_EUR_time.npy'

        """feature npy"""
        # var_q = nc_file.variables['q'][:]
        # var_crwc = nc_file.variables['crwc'][:]
        # var_t = nc_file.variables['t'][:]
        # var_u = nc_file.variables['u'][:]
        # var_v = nc_file.variables['v'][:]

        var_d2m = nc_file.variables['d2m'][:]
        var_t2m = nc_file.variables['t2m'][:]
        var_skt = nc_file.variables['skt'][:]
        var_stl1 = nc_file.variables['stl1'][:]
        var_sp = nc_file.variables['sp'][:]
        var_tp = nc_file.variables['tp'][:]

        # 通过每个node的经纬度得到对应特征 每个node应该得到(time.shape,attr_num)
        addNode_attr = []
        for i in range(nodes_num):
            node_lon = nodes_dir[i][1]
            node_lat = nodes_dir[i][2]

            lon_start = float(nc_file.variables['longitude'][0])
            lon_end = float(nc_file.variables['longitude'][-1])
            if node_lon <= lon_end and node_lon >= lon_start:
                lon_idx = (node_lon - lon_start) / 0.25
                lon_idx = int(lon_idx)
            else:
                print("array out of bounds!")

            lat_start = float(nc_file.variables['latitude'][0])
            lat_end = float(nc_file.variables['latitude'][-1])
            if node_lat <= lat_start and node_lat >= lat_end:
                lat_idx = (lat_start - node_lat) / 0.25
                lat_idx = int(lat_idx)
            else:
                print("array out of bounds!")

            # q = var_q[:, lat_idx, lon_idx]
            # crwc = var_crwc[:, lat_idx, lon_idx]
            # t = var_t[:, lat_idx, lon_idx]
            # u = var_u[:, lat_idx, lon_idx]
            # v = var_v[:,lat_idx,lon_idx]

            t2m = var_t2m[:, lat_idx, lon_idx]
            d2m = var_d2m[:, lat_idx, lon_idx]
            skt = var_skt[:, lat_idx, lon_idx]
            stl1 = var_stl1[:, lat_idx, lon_idx]
            sp = var_sp[:, lat_idx, lon_idx]
            tp = var_tp[:, lat_idx, lon_idx]

            # oneNode_attr = np.array(list(zip(q, crwc, t, u, v))) # (time.shape,attr_num)
            oneNode_attr = np.array(list(zip(sp, d2m, t2m, skt, stl1, tp)))
            addNode_attr.append(oneNode_attr)

        self.node_attr = np.array(addNode_attr)

        if os.path.exists(feat_file):
            # load - append - save
            feature_2160nodes = np.load(feat_file)
            feature_2160nodes_new = np.concatenate([feature_2160nodes, self.node_attr], axis=1)
            np.save(feat_file, feature_2160nodes_new)
        else:
            np.save(feat_file, self.node_attr)

        """time npy"""
        var_time = nc_file.variables['time'][:]
        self.time = np.array(var_time)
        if os.path.exists(time_file):
            # load - append - save
            time_2160nodes = np.load(time_file)
            time_2160nodes = np.concatenate([time_2160nodes, self.time], axis=0)
            np.save(time_file, time_2160nodes)
        else:
            np.save(time_file, self.time)
