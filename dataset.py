"""
    对于dataset_preprocessing处理好的数据 ndarray (node_num,time_num,attr_num) 这样整理好的全部数据集特征进行再处理
    目的 : 分割整体数据集为 训练集 验证集 测试集 并且按照时间滑窗拓展维度 并norm化 另外后期可以增加对于node_attr的处理环节
"""
import gc
import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
from utils import config, file_dir
import datetime
import numpy as np
import arrow
from torch.utils import data

site = '/public/ecai/graph/eur/'

class HaveData(data.Dataset):
    def __init__(self, graph,
                 node_attr,
                 timestamp,
                 hist_len=1,
                 pred_len=24,
                 dataset_num=1,
                 flag='Train',):
        # 获取config文件中的参数
        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')

        self.start_time = self._get_time(config['dataset'][dataset_num][start_time_str])
        self.end_time = self._get_time(config['dataset'][dataset_num][end_time_str])
        self.data_start = self._get_time(config['dataset']['data_start'])
        self.data_end = self._get_time(config['dataset']['data_end'])

        self.graph = graph
        self.node_attr = node_attr
        self.timestamp = timestamp
        self.jet_lat = self.timestamp[1] - self.timestamp[0] # 每个索引间的时间戳差值

        # new var
        self.acc_u = np.float32(np.expand_dims(np.load(site + 'acc_u.npy'), axis=-1))
        self.acc_v = np.float32(np.expand_dims(np.load(site + 'acc_v.npy'), axis=-1))
        self.uv_angleACC = np.float32(np.expand_dims(np.load(site + 'uv_angleACC.npy'), axis=-1))
        self.uv_angle = np.float32(np.expand_dims(np.load(site + 'uv_angle.npy'), axis=-1))
        self.uv_speed = np.float32(np.expand_dims(np.load(site + 'uv_speed.npy'), axis=-1))

        self._overall_time() # 得到所需整体数据集 (train+val+test dataset)
        self._process_time() # 得到所需部分数据集 (train|val|test dataset)
        self.node_attr = np.float32(self.node_attr)

        self.node_attr = np.concatenate((self.node_attr,self.acc_u,self.acc_v,self.uv_angleACC,self.uv_angle,self.uv_speed),axis=-1)

        self.node_attr = self.node_attr.transpose((1,0,2)) # (node_num,time_num,attr_num) -> (time_num,node_num,attr_num)

        self.feature = self.node_attr[:,:,1:]
        self.t2m = self.node_attr[:,:,:1]

        self._calc_mean_std()  # 求node_attr标准差和均值
        seq_len = hist_len + pred_len # 总的窗口长度 e.g.一天预测十天
        # if config['idgl']['graph_model'] == 'ls-gnn':
        self._add_time_dim(seq_len) # 类似滑动窗口 增加数据维度
        self._norm() # 数据标准化


    # 获取config.yaml中的时间信息转换为arrow形式
    def _get_time(self, time_yaml):
        from datetime import datetime
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])  # 获取config文件中时间戳信息
        return arrow_time

    # 获取整体(data_start,data_end)时间戳
    def _overall_time(self):
        self.all_start_timestamp = self._get_timestamp(self.data_start)
        self.all_end_timestamp = self._get_timestamp(self.data_end)
        start_idx = self._get_index(self.all_start_timestamp)
        end_idx = self._get_index(self.all_end_timestamp)
        self.node_attr = self.node_attr[:,start_idx:end_idx+1,:]
        self.acc_u = self.acc_u[:, start_idx:end_idx + 1, :]
        self.acc_v = self.acc_v[:, start_idx:end_idx + 1, :]
        self.uv_angle = self.uv_angle[:, start_idx:end_idx + 1, :]
        self.uv_angleACC = self.uv_angleACC[:, start_idx:end_idx + 1, :]
        self.uv_speed = self.uv_speed[:, start_idx:end_idx + 1, :]
        self.timestamp = self.timestamp[start_idx:end_idx + 1]

    # 获取部分(dataset_num.start_time,dataset_num.end_time)时间戳
    def _process_time(self):
        start_timestamp = self._get_timestamp(self.start_time)
        end_timestamp = self._get_timestamp(self.end_time)
        start_idx = self._get_index(start_timestamp)
        end_idx = self._get_index(end_timestamp)
        self.node_attr = self.node_attr[:,start_idx:end_idx+1,:]
        self.acc_u = self.acc_u[:,start_idx:end_idx+1,:]
        self.acc_v = self.acc_v[:,start_idx:end_idx+1,:]
        self.uv_angle = self.uv_angle[:,start_idx:end_idx+1,:]
        self.uv_angleACC = self.uv_angleACC[:,start_idx:end_idx+1,:]
        self.uv_speed = self.uv_speed[:,start_idx:end_idx+1,:]
        self.timestamp = self.timestamp[start_idx:end_idx+1]

    # 获取时间戳(这里的1900根据数据集时间戳开始时间查看)
    def _get_timestamp(self,t):
        t = t.datetime
        t_year = t.year
        t_month = t.month
        t_day = t.day
        t_hour = t.hour
        d1 = datetime.date(1900,1,1)
        d2 = datetime.date(t_year,t_month,t_day)
        timestamp = (d2 - d1).days * 24 + t_hour
        return timestamp

    # 获取时间索引
    # def _get_index(self,t_old,t_new):
    #     t_new = t_new - t_old
    #     t_index = t_new / self.jet_lat
    #     return int(t_index)
    def _get_index(self,time_stamp_now):
        for i in range(self.timestamp.shape[0]):
            if time_stamp_now == self.timestamp[i]:
                return i
        print("Timestamp None Peer!")

    # 获取标准差和均值
    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0,1))
        self.feature_std = self.feature.std(axis=(0,1))
        self.t2m_mean = self.t2m.mean(axis=(0,1))
        self.t2m_std = self.t2m.std(axis=(0,1))

    # slide window 增加数据维度
    def _add_time_dim(self,seq_len):

        def _add_t(arr,seq_len):
            t_len = arr.shape[0]
            assert t_len > seq_len
            arr_ts = []
            # 类似滑动窗口
            for i in range(seq_len, t_len):
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.feature = _add_t(self.feature,seq_len)
        self.t2m = _add_t(self.t2m,seq_len)
        self.timestamp = _add_t(self.timestamp,seq_len)

    # 标准化
    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.t2m = (self.t2m - self.t2m_mean) / self.t2m_std

    def __len__(self):
        return len(self.t2m)

    # DataLoader调用时候会自动获取这个函数 因此其他参数并不发出
    def __getitem__(self, index):
        return self.feature[index], self.t2m[index], self.timestamp[index]


if __name__ == '__main__':
    from graph import Graph
    from dataset_preprocessing import dataset_pre
    graph = Graph()
    station_num = graph.node_num
    station_node = graph.nodes
    dataset_pre = dataset_pre(station_node, station_num)
    node_attr = dataset_pre.node_attr
    timestamp = dataset_pre.time
    train_data = HaveData(graph,node_attr,timestamp,flag='Train')
    val_data = HaveData(graph,node_attr,timestamp,flag='Val')
    test_data = HaveData(graph,node_attr,timestamp,flag='Test')

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))




