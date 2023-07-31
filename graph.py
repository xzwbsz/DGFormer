"""
    初始图构建:
    根据经纬网格欧式距离小于3格 & (可选)采用海拔高度差小于1200m 进行节点间的边的连接
    分别得到稀疏邻接矩阵和稠密邻接矩阵 并且 得到一个【idx:{station_name,lon,lat,altitude}】字典
"""
import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
import numpy as np
import csv
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj

# 真实节点文件位置
site_fp = '/public/ecai/graph/eur/102nodes_EUR.CSV'
site_dist = '/public/ecai/graph/eur/102nodes_dist.npy'
site_angle = '/public/ecai/graph/eur/102nodes_angle.npy'

class Graph():
    def __init__(self):
        self.dist_thres = 7  # 300km=3
        self.alti_thres = 1200 # 1200m 1200
        self.use_altitude = True

        self.dist_adj = np.load(site_dist)
        self.angle_adj = np.load(site_angle)

        self.lonlat = self._latlon_nodes()
        self.lonlat = self.lonlat.T
        self.nodes = self._gen_nodes() # 形成字典表 idx:{station_name,lon,lat,altitude}
        self.node_num = len(self.nodes) # 节点总数
        self.edge_index, self.edge_attr = self._gen_edges() # edge_index [起始节点,到达节点]  edge_attr 节点间的欧式距离
        # 对超过海拔范围的边进行剪枝
        if self.use_altitude:
            self._update_edges() # 更新edge_index和edge_attr
        self.edge_attr = self.edge_attr[:,np.newaxis]
        self.edge_num = self.edge_index.shape[1] # 边总数
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]
        # self.nodes = self._grad_lat_lon_010()
        self.nodes = self._grad_lat_lon_025()


    # 加载2160nodes.CSV文件 得到nodes字典
    def _gen_nodes(self):
        nodes = OrderedDict()
        with open(site_fp, 'r') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                idx = int(line[0])
                city = str(line[1]+line[2])
                lon, lat, alt = float(line[3]), float(line[4]), float(line[5])
                nodes.update({idx: {'city':city, 'lon':lon, 'lat':lat, 'alt':alt}})
        return nodes

    # 加载142nodes文件 得到lat-lon数据集
    def _latlon_nodes(self):
        with open(site_fp, 'r') as f:
            csv_reader = csv.reader(f)
            list1,list2 = [],[]
            for line in csv_reader:
                lon = float(line[3])
                list1.append(lon)
                lat = float(line[4])
                list2.append(lat)
            nd1 = np.array(list1)
            nd2 = np.array(list2)
            return np.stack([nd1,nd2])


    # 获取稀疏矩阵和边权值
    def _gen_edges(self):
        coords = []
        # 获取每个节点经纬度
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        self.dist = self.dist_adj # 得到各点间的欧式距离 shape=(node_num，node_num)
        adj = np.zeros((self.node_num,self.node_num),dtype=np.uint8)
        adj[self.dist <= self.dist_thres] = 1 # 若dist ≤ 300km(此时还是按照经纬网格计算的) 则置对应位为1 表示节点间有边相连A+I
        assert adj.shape == self.dist.shape
        dist = self.dist * adj # 仅保留adj中对应位置为1的项 *哈达玛积

        # 稠密矩阵转稀疏矩阵
        edge_index,dist = dense_to_sparse(torch.tensor(dist))  # edge_index edge_index = {Tensor: (2, 408492)} tensor([[   0,    0,    0,  ..., 2047, 2047, 2047],\n        [   1,    2,    3,  ..., 2044, 2045, 2046]])[起始节点,到达节点]  dist 节点间的欧式距离
        edge_index,edge_attr = edge_index.numpy(),dist.numpy()
        return edge_index,edge_attr

    # 对超过海拔范围的边进行剪枝 更新self.edge_index和self.edge_attr
    def _update_edges(self):
        edge_index = []
        edge_attr = []
        # 遍历稀疏矩阵 进行剪枝
        for i in range(self.edge_index.shape[1]):
            src,dest = self.edge_index[0,i], self.edge_index[1,i]
            src_alt = self.nodes[src]['alt']
            dest_alt = self.nodes[dest]['alt']
            poor_alt = int(src_alt-dest_alt)
            poor_alt = abs(poor_alt)
            if poor_alt < self.alti_thres:
                edge_index.append(self.edge_index[:,i])
                edge_attr.append(self.edge_attr[i])

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)

    # 遍历字典中的所有节点 并将不规范的经纬度网格化(最顶注释解释) 以获取每个节点的特征 标度：0.25度
    # lat,lon -> node_idx
    def _grad_lat_lon_025(self):
        nodes_new = []
        for i in self.nodes:
            node_lon = self.nodes[i]['lon']
            node_lon_mod = node_lon % 0.25
            if abs(node_lon_mod) <= 0.125:
                node_lon_grid = int(node_lon / 0.25)
            else:
                node_lon_grid = int((node_lon + 0.25) / 0.25)
            node_lon_grid = node_lon_grid * 0.25

            node_lat = self.nodes[i]['lat']
            node_lat_mod = node_lat % 0.25
            if abs(node_lat_mod) <= 0.125:
                node_lat_grid = int(node_lat / 0.25)
            else:
                if node_lat > 0:
                    node_lat_grid = int((node_lat + 0.25) / 0.25)
                else:
                    node_lat_grid = int((node_lat - 0.25) / 0.25)
            node_lat_grid = node_lat_grid * 0.25

            if node_lon_grid == 360.0:
                node_lon_grid = 0.0

            nodes_new.append([i, node_lon_grid, node_lat_grid])
        return nodes_new

    # 标度为0.1
    def _grad_lat_lon_010(self):
        nodes_new = []
        for i in self.nodes:
            node_lon = self.nodes[i]['lon']
            node_lat = self.nodes[i]['lat']

            node_lon_grid = round(node_lon,1)
            node_lat_grid = round(node_lat,1)

            nodes_new.append([i, node_lon_grid, node_lat_grid])

        return nodes_new

if __name__ == '__main__':
    graph = Graph()