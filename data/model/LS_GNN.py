import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add#, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter

# 定义GraphGNN类 继承Module
class GraphGNN(nn.Module):
    def __init__(self, device, edge_index, edge_attr, in_dim, out_dim):
        super(GraphGNN, self).__init__()
        self.device = device
        self.edge_index = torch.LongTensor(edge_index).to(self.device)  # 获取图连接的稀疏矩阵并转化为tensor并运行在设备上
        self.edge_attr = torch.Tensor(np.float32(edge_attr)) # 获取图节点间欧式距离转化为tensor
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)
        self.w = Parameter(torch.rand([1])) # w和b
        self.b = Parameter(torch.rand([1]))
        e_h = 32 # edge_mlp hidden
        e_out = 30 # edge_mlp out
        n_out = out_dim # node_mlp out
        self.edge_mlp = Sequential(Linear(in_dim * 2 + 1, e_h), # 输入维度7,输出维度32
                                   Sigmoid(),
                                   Linear(e_h, e_out), # 32 30
                                   Sigmoid(),
                                   )
        self.node_mlp = Sequential(Linear(e_out, n_out),
                                   Sigmoid(),
                                   )

    def forward(self, x): # (batch_size=32,station_num=2160,attr_num=3)
        self.edge_index = self.edge_index.to(self.device) # 节点索引 class中的参数都传入设备 (2,edges_num)
        self.edge_attr = self.edge_attr.to(self.device)
        self.w = self.w.to(self.device)
        self.b = self.b.to(self.device)

        edge_src, edge_target = self.edge_index # {2,edge_num} -> src {1,edges_num} 和 target {1,edges_num}
        node_src = x[:, edge_src] # {32,station_num,3} -> {32,edges_num,3}
        node_target = x[:, edge_target] # {32,nodes_num,3} -> {32,edges_num,3}

        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device) # (edges_num,1) -> (32,edges_num,1)
        out = torch.cat([node_src, node_target, edge_attr_norm], dim=-1) #在最后一个维度进行累加 -> (32,edges_num,3+3+1=7)
        out = self.edge_mlp(out) # out传入edge_mlp更新边属性(32,edges_num,30) e_h = 30

        # 汇聚入度的边特征 and 刨除出度的边特征 最后得到本节点的特征
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1)) # For higher version of PyG.
        out = out_add + out_sub
        out = self.node_mlp(out) # 将out传入node_mlp

        return out


class LS_GNN(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, station_num, batch_size, device, edge_index, edge_attr):
        super(LS_GNN, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.station_num = station_num
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = 64
        self.out_dim = 1
        self.gnn_out = 13

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out)
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)


    def forward(self, t2m_hist, feature):
        t2m_pred = []
        h0 = torch.zeros(self.batch_size * self.station_num, self.hid_dim).to(self.device)
        hn = h0
        xn = t2m_hist[:, -1] # (32,1,station_num,1) -> (32,station_num,1)
        for i in range(self.pred_len): # 5
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1) # 对最后一维进行cat拼接操作 (32,station_num,1+2)

            xn_gnn = x # (32,2160,3)
            xn_gnn = xn_gnn.contiguous() # (32,station_num,3)
            xn_gnn = self.graph_gnn(xn_gnn) # 调用GraphGNN()的 (32,station_num,3) -> (32,station_num,13)
            x = torch.cat([xn_gnn, x], dim=-1)  # (32,station_num,13+3) -> (32,station_num,16)

            hn = self.gru_cell(x, hn) # 调用GRU层的forward (batch_size*station_num , his_dim)
            xn = hn.view(self.batch_size, self.station_num, self.hid_dim) # view(x1,x2,x3)调整维度为(batch_size*station_num , hid_dim)->(batch_size , station_num , hid_dim)
            xn = self.fc_out(xn) # 调用Linear层 (32,station_num,64) -> (32,station_num,1)
            t2m_pred.append(xn) # 每次增加(32,station_num,1)

        t2m_pred = torch.stack(t2m_pred, dim=1) # (32,5,station_num,1)

        return t2m_pred
