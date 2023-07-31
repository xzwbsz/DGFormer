import os
import random
import numpy as np
import torch
import torch.nn as nn

from idgl_utils.constants import VERY_SMALL_NUMBER
from idgl_utils.generic_utils import normalize_adj, to_cuda
from model.LS_GNN import LS_GNN
from model.graphLearn import GraphLearner
from model.idgl_gnn import LS_GNN_GCN
from ablation.agcrn.AGCRN import AGCRN
from ablation.dcrnn.dcrnn_model import DCRNNModel
from ablation.stgcn.models import STGCNGraphConv
from lib_clcrn.dataloader import KernelGenerator
from ablation.clcnn.recurrent.seq2seq_model import CLCRNModel


class IDGL(nn.Module):
    """
        model:
            layer 2: ls-gnn → predict
            layer 1: gcn
            layer 0: Graph-Learning
    """

    def __init__(self, config, adj0=None, gso=None, lonlat=None):
        """
            1 config中信息赋值给私有化变量
        """
        super(IDGL, self).__init__()
        self.config = config
        self.pred_var = config['experiments']['pred_var']
        self.name = "IDGL"
        self.graph_learn = config['idgl']['graph_learn']
        self.graph_module = config['idgl']['graph_model']
        self.device = config['device']
        feature_num = len(config['experiments']['metero_use'])
        hidden_size = config['idgl']['hidden_size']
        hist_len = int(config['train']['hist_len'])
        pred_len = int(config['train']['pred_len'])
        self.dropout = config['idgl']['dropout']
        self.graph_skip_conn = config['idgl']['graph_skip_conn']
        self.graph_include_self = config['idgl'].get('graph_include_self', True)
        self.scalable_run = config['idgl'].get('scalable_run', False)
        self.adj = adj0
        self.isPhy = bool(config['idgl']['physics_guidance'])

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.config['device'] = self.device

        # 通过untils中的函数读取config文件 并获取相应的参数
        in_dim = config['in_dim']  # 3
        station_num = config['station_num']
        edge_index = config['edge_index']
        edge_attr = config['edge_attr']
        batch_size = config['train']['batch_size']  # 32
        hist_len = int(config['train']['hist_len'])  # 1 # 隐藏层大小
        pred_len = int(config['train']['pred_len'])  # 24 预测天数
        criterion = nn.MSELoss()  # 定义MSE为loss

        """
            2 select graph_module
        """
        if self.graph_module == 'ls-gnn':
            self.encoder = LS_GNN_GCN(nfeat=in_dim,
                                      nhidden=hidden_size,
                                      npred=pred_len,
                                      graph_hops=config.get('graph_hops', 2),
                                      dropout=self.dropout,
                                      hist_len=hist_len,
                                      station_num=station_num,
                                      batch_size=batch_size,
                                      device=self.device,
                                      batch_norm=config.get('batch_norm', False))
        elif self.graph_module == 'agrcn':
            self.encoder = AGCRN(config)
        elif self.graph_module == 'dcrnn':
            self.encoder = DCRNNModel(self.adj, config)
        elif self.graph_module == 'stgcn':
            Ko = hist_len - (int(config['stgcn']['Kt']) - 1) * 2 * int(config['stgcn']['stblock_num'])
            # blocks: settings of channel size in st_conv_blocks and output layer,
            # using the bottleneck design in st_conv_blocks
            blocks = []
            blocks.append([1])
            for l in range(int(config['stgcn']['stblock_num'])):
                blocks.append([64, 16, 64])
            if Ko == 0:
                blocks.append([128])
            elif Ko > 0:
                blocks.append([128, 128])
            blocks.append([pred_len])
            self.encoder = STGCNGraphConv(config, blocks, config['stgcn']['n_vertex'], gso)
        elif self.graph_module == 'clcrn':
            kernel_generator = KernelGenerator(lonlat)
            kernel_info = {'sparse_idx': kernel_generator.sparse_idx,
                           'MLP_inputs': kernel_generator.MLP_inputs,
                           'geodesic': kernel_generator.geodesic.flatten(),
                           'angle_ratio': kernel_generator.ratio_lists.flatten()}
            self.sparse_idx = torch.from_numpy(kernel_info['sparse_idx']).long().to(self.device)
            self.loc_info = torch.from_numpy(kernel_info['MLP_inputs']).float().to(self.device)
            self.geodesic = torch.from_numpy(kernel_info['geodesic']).float().to(self.device)
            self.angle_ratio = torch.from_numpy(kernel_info['angle_ratio']).float().to(self.device)
            self.encoder = CLCRNModel(loc_info=self.loc_info, sparse_idx=self.sparse_idx, geodesic=self.geodesic,
                                      angle_ratio=self.angle_ratio, config=config)
        else:
            raise RuntimeError('Unknown graph_module: {}'.format(self.graph_module))

        """
            3 graph-learning network model
        """
        if self.graph_learn:
            graph_learn_fun = GraphLearner
            self.graph_learner = graph_learn_fun(feature_num, config['idgl']['graph_learn_hidden_size'],
                                                 # input_size=3 | hidden_size = 70
                                                 topk=config['idgl']['graph_learn_topk'],  # none
                                                 epsilon=config['idgl']['graph_learn_epsilon'],  # ε = 0
                                                 num_pers=config['idgl']['graph_learn_num_pers'],  # 4
                                                 metric_type=config['idgl']['graph_metric_type'],  # weight_cosine
                                                 device=self.device,
                                                 pred_var=self.pred_var)

            # print('[ Graph Learner ]')
            # if config['idgl']['graph_learn_regularization']:
            #     print('[ Graph Regularization]')
        else:
            self.graph_learner = None

    """
        来自 module_handler.py 调用
            输入 node_features为原始输入特征X或者Z(t-1) | init_adj为A(0)初始邻接矩阵 | graph_learner为哪种维度的模型
            输出 A(t) 和 A_hat(t)
    """

    def learn_graph(self, graph_learner, gl_feature, acc_u, acc_v, uv_angleACC, uv_angle, speed, graph_skip_conn=None,
                    graph_include_self=False, init_adj=None, dist=None, angle=None,
                    anchor_features=None, ):  # node_features 原始特征X或者Z(t-1) | init_adj A(0)初始邻接矩阵
        # 是否进行图学习
        if self.graph_learn:
            # 是否采用anchor
            if self.scalable_run:
                node_anchor_adj = graph_learner(gl_feature, anchor_features)
                return node_anchor_adj
            else:
                # graph_learner(node_features) 执行graphlearn.py的forward()函数
                raw_adj = graph_learner(gl_feature, dist=dist, angle=angle, acc_u=acc_u, acc_v=acc_v, uv_angle=uv_angle,
                                        uv_angleACC=uv_angleACC, speed=speed, isPhy=self.isPhy)  # A(t) = GL(Z(t-1))

                # assert raw_adj.min().item() >= 0 # 确定权重矩阵中没有负数项
                adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True),
                                            min=VERY_SMALL_NUMBER)  # f(A(t)) = f(A_ij) = A_ij/Σ_j(A_ij) 行归一化 (2708,2708)

                if graph_skip_conn in (0, None):
                    if graph_include_self:
                        adj = adj + to_cuda(torch.eye(adj.size(0)), self.device)
                else:
                    adj = graph_skip_conn * to_cuda(init_adj, self.device) + (
                            1 - graph_skip_conn) * adj  # λA(0) + (1-λ)f(A(t))

                return raw_adj, adj  # 返回A(t) 和 A_hat(t)
        else:
            raw_adj = None
            adj = init_adj

            return raw_adj, adj
