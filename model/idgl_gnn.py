import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add#, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter

class LS_GNN_GCN(nn.Module):
    def __init__(self, nfeat, nhidden, npred, graph_hops, dropout,hist_len,station_num, batch_size, device, batch_norm=False):
        # nfeat=in_dim, nhidden=hidden_size, npred=pred_len
        super(LS_GNN_GCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList() # 在列表中保存子模块
        self.graph_encoders.append(GCNLayer(nfeat, nhidden, batch_norm=batch_norm)) # 添加第一层GCN 输入8 输出8
        self.graph_encoders.append(LS_GNN(hist_len,npred,nfeat,station_num,batch_size,device))
        # self.graph_encoders.append(GCNLayer(nhidden, npred, batch_norm=False)) # 添加第二层GCN 输入3 输出1


    def forward(self, x, node_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_adj)
        return x


class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight)) # 随机初始权重矩阵

        # 偏差这里没用到
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True): # GCN1 : input-原始特征X adj-当前邻接矩阵cur_adj
        support = torch.matmul(input, self.weight) # MP(X,W) = W × X  e.g.1 (2160,3) × (3,3) = (2160,3)
        output = torch.matmul(adj, support) # MP(cur_adj,support) = cur_adj × support = cur_adj × W × X（or Z(t)) | e.g. (2160,2160) × (2160,3) = (2160,3)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class LS_GNN(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, station_num, batch_size, device):
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
        self.graph_gnn = GraphGNN(self.device, self.in_dim, self.gnn_out)
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)


    def forward(self, node_vec , cur_adj , feature):
        t2m_pred = []
        h0 = torch.zeros(self.batch_size * self.station_num, self.hid_dim).to(self.device)
        hn = h0
        edge_src_target,edge_weight = dense_to_sparse(cur_adj)

        for i in range(self.pred_len): # 5
            if i == 0 :
                x = node_vec
            else:
                x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1) # 对最后一维进行cat拼接操作 (batch_size,station_num,1+2)

            xn_gnn = x # (batch_size,2160,3)
            xn_gnn = xn_gnn.contiguous() # (batch_size,station_num,3)
            xn_gnn = self.graph_gnn(xn_gnn,edge_src_target,edge_weight) # 调用GraphGNN()的 (batch_size,station_num,3) -> (batch_size,station_num,13)
            x = torch.cat([xn_gnn, x], dim=-1)  # (batch_size,station_num,13+3) -> (batch_size,station_num,16)

            hn = self.gru_cell(x, hn) # 调用GRU层的forward (batch_size*station_num , his_dim)
            xn = hn.view(self.batch_size, self.station_num, self.hid_dim) # view(x1,x2,x3)调整维度为(batch_size*station_num , hid_dim)->(batch_size , station_num , hid_dim)
            xn = self.fc_out(xn) # 调用Linear层 (batch_size,station_num,64) -> (batch_size,station_num,1)
            t2m_pred.append(xn) # 每次增加(batch_size,station_num,1)

        t2m_pred = torch.stack(t2m_pred, dim=1) # (batch_size,pred_len,station_num,1)

        return t2m_pred


# 定义GraphGNN类 继承Module
class GraphGNN(nn.Module):
    def __init__(self, device, in_dim, out_dim):
        super(GraphGNN, self).__init__()
        self.device = device
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

    def forward(self, x, edge_src_target, edge_weight): # (batch_size=32,station_num=2160,attr_num=3)
        edge_src_target = edge_src_target.to(self.device) # 节点索引 class中的参数都传入设备 (2,edges_num)
        edge_weight = edge_weight.to(self.device)
        self.w = self.w.to(self.device)
        self.b = self.b.to(self.device)

        edge_src, edge_target = edge_src_target # {2,edge_num} -> src {1,edges_num} 和 target {1,edges_num}
        node_src = x[:, edge_src] # {batch_size,station_num,feature_num} -> {batch_size,edges_num,feature_num}
        node_target = x[:, edge_target] # {batch_size,station_num,3} -> {batch_size,edges_num,feature_num}

        edge_w = edge_weight.unsqueeze(-1)
        edge_w = edge_w[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device) # (edges_num,1) -> (32,edges_num,1)
        out = torch.cat([node_src, node_target, edge_w], dim=-1) #在最后一个维度进行累加 -> (32,edges_num,3+3+1=7)
        out = self.edge_mlp(out) # out传入edge_mlp更新边属性(32,edges_num,30) e_h = 30

        # 汇聚入度的边特征 and 刨除出度的边特征 最后得到本节点的特征
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1)) # For higher version of PyG.
        out = out_add + out_sub
        out = self.node_mlp(out) # 将out传入node_mlp

        return out # y_hat


"""
    LS_GNN —— Transformer, Reformer
"""
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    Reformer Version
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output