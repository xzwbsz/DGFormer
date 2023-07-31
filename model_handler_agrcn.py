import os
import json
import glob
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from idgl_utils.generic_utils import to_cuda
from idgl_utils.constants import VERY_SMALL_NUMBER
import torch.nn.functional as F
import shutil

import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir) # 将路径添加到环境目录
from idgl_utils.idgl_utils import AverageMeter
from model.model_idgl import IDGL

class ModelHandler_agrcn(object):
    def __init__(self,config,train_loader,val_loader,test_loader,adj0,dist):
        self.config = config
        """
            1 创建评价指标 并指定数据结构
        """
        # 训练集损失 验证集损失 测试集损失 均方根误差列表
        self.train_loss_list, self.val_loss_list, self.test_loss_list, self.rmse_list = [], [], [], []

        """
            2 确定运行设备
        """
        use_cuda = torch.cuda.is_available()
        # if use_cuda:
        #     print('[Using CUDA]')
        # else:
        #     print("[Using CPU]")
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.config['device'] = self.device

        """
            3 设置随机种子应用到设备
        """
        seed = self.config['idgl'].get('seed',42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device:
            torch.cuda.manual_seed(seed)

        """
            4 准备数据集
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.adj0 = adj0
        self.dist = dist
        np.save("result/adj0.npy",self.adj0.cpu().detach().numpy())
        """
            5 model初始化并打印模型信息和参数总数
        """
        self.model = IDGL(config)
        self.model = self.model.to(self.device)
        self.config = self.model.config
        self.is_test = False
        self.firstModelLoad = True
        # print(self.model)

        # num_params = 0
        # for name,p in self.model.named_parameters():
        #     print('{}: {}'.format(name, str(p.size())))
        #     num_params += p.numel()
        # print('#Parameters = {}\n'.format(num_params))

        """
            6 确定使用哪种优化器 定义计算RMSE函数
        """
        self.init_optimizer()
        self.criterion = nn.MSELoss()

    # 初始化优化器
    def init_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.config['train']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['train']['lr'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['train']['weight_decay'])
        elif self.config['train']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['train']['lr'],
                                        weight_decay=self.config['train']['weight_decay'])
        elif self.config['train']['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, lr=self.config['train']['lr'])
        elif self.config['train']['optimizer'] == 'rmsprop':
            self.optimizer = optim.RMSprop(parameters,lr=self.config['train']['lr'],
                                           weight_decay=self.config['train']['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['train']['optimizer'])
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['train']['lr_reduce_factor'], \
        #                                    patience=self.config['train']['lr_patience'], verbose=True) # 动态调整学习率


    # 训练过程
    def train(self,epoch):
        if self.train_loader is None:
            print("No training set specified -- skipped training")
            return
        self.epoch_num = epoch
        train_loss = self.run_whole_epoch(self.train_loader,task_type="train",verbose=self.config['idgl']['verbose'])
        # self.train_loss_list.append(train_loss.item())
        torch.cuda.empty_cache()

        return train_loss


    # 验证过程
    def val(self):
        if self.val_loader is None:
            print("No val set specified -- skipped training")
            return

        val_loss = self.run_whole_epoch(self.val_loader,task_type="val",verbose=self.config['idgl']['verbose'])
        torch.cuda.empty_cache()

        return val_loss

    # 测试过程
    def test(self,t2m_mean,t2m_std):

        self.t2m_mean = t2m_mean
        self.t2m_std = t2m_std
        test_loss = self.run_whole_epoch(self.test_loader,task_type="test",verbose=self.config['idgl']['verbose'])
        torch.cuda.empty_cache()

        return test_loss, self.predict_epoch,self.label_epoch,self.time_epoch



    # graph learn + train phase
    def run_whole_epoch(self,data_loader,task_type,verbose=None,out_predictions=False):

        if task_type == "train":
            self.model.train()
            if os.path.exists("./save_train_param/Param0.pth") :
                load_data = torch.load("./save_train_param/Param0.pth")
                self.model.load_state_dict(load_data)
        else:
            self.model.eval()

        if task_type == "test":
            self.predict_list = []
            self.label_list = []
            self.time_list = []

        loss = 0
        total_loss = torch.tensor(0.,requires_grad=False)
        hist_len = self.config['train']['hist_len']
        pth_idx = 0
        save_range = self.config['train']['batch_epoch']
        load_modelParam = False
        is_mean_curA = True
        pth_idx_str = "./save_train_param/Param"

        # 每个batch分别做训练
        for batch_idx,data in enumerate(data_loader):
            self.optimizer.zero_grad()  # 梯度置0
            feature,t2m,timestamp_arr = data
            feature = feature.to(self.device) # (7,6,2160,3)
            t2m = t2m.to(self.device)
            t2m_label = t2m[:,hist_len:] # (batch_size,pred_len=5,station_num,attr_num)
            t2m_hist = t2m[:,:hist_len] # (batch_size,hist_len=1,station_num,attr_num)
            t2m_trueData = t2m_hist[:,-1]
            feature_trueData = feature[:,:hist_len]
            feature_trueData = feature_trueData[:,-1]
            slideWindow_first_day_feature = torch.cat((t2m_trueData,feature_trueData),dim=-1)
            max_iter = self.config['idgl'].get('max_iter', 10)
            self.adj0 = self.adj0.to(self.device)

            # 重新初始化模型并进行读取数值
            if load_modelParam:
                modelA = IDGL(self.config)
                self.model = modelA.to(self.device)
                load_data = torch.load(pth_idx_str) if task_type == "train" else torch.load("./save_train_param/Param0.pth")
                self.model.load_state_dict(load_data)
                pth_idx_str = "./save_train_param/Param"
                loss = torch.tensor(0)
                load_modelParam = False

            cur_adj = self.adj0
            graphLoss_update_continue = False

            # batch_size,hist_len,node_sensor,input_dim
            t2m_pred = self.model.encoder(t2m_hist)

            loss = loss + self.criterion(t2m_pred, t2m_label)

            # 每轮都进行存储
            if task_type == "train":
                total_loss = loss.clone().detach() + total_loss.item()
                pth_idx += 1
                pth_idx_str = pth_idx_str + str(pth_idx) + ".pth"
                loss.backward()
                self.optimizer.step()
                torch.save(self.model.state_dict(),pth_idx_str)
                load_modelParam = True
                self.optimizer.zero_grad()

            if task_type == "val" or task_type == "test":
                total_loss = loss.clone().detach() + total_loss.item()
                load_modelParam = True


            if task_type == "test":
                t2m_pred_val = np.concatenate([t2m_hist.cpu().detach().numpy(), t2m_pred.cpu().detach().numpy()],axis=1) * self.t2m_std + self.t2m_mean
                t2m_label_val = t2m.cpu().detach().numpy() * self.t2m_std + self.t2m_mean
                self.predict_list.append(t2m_pred_val)
                self.label_list.append(t2m_label_val)
                self.time_list.append(timestamp_arr.cpu().detach().numpy())

        # 复制最后模型参数为Param0.pth
        if task_type == "train":
            # 绘邻接矩阵热点图
            if self.config['idgl']['graph_learn']:
                print('start imaging')
            else :
                print('start imaging')
                cur_add = 255 * cur_adj
                save_adj_new = "result/epoch=" + str(self.epoch_num) + "_adj.npy"
                np.save(save_adj_new, cur_add.cpu().detach().numpy())

            # 存储参数
            pth_new = "./save_train_param/Param0.pth"
            shutil.copyfile(pth_idx_str,pth_new)

        if task_type == "test":
            pth_new = "./save_train_param/Param0.pth"
            pth_save_final = "./save_final_Parameter/Param_new_final.pth"
            shutil.copyfile(pth_new,pth_save_final)


        if task_type == "val" or task_type=="test":
            total_loss = loss.clone().detach() + total_loss.item()

        if task_type == "test" :
            self.predict_epoch = np.concatenate(self.predict_list,axis=0)
            self.label_epoch = np.concatenate(self.label_list, axis=0)
            self.time_epoch = np.concatenate(self.time_list, axis=0)
            self.predict_epoch[self.predict_epoch < 0] = 0

        total_loss = total_loss.item() / (batch_idx + 1)

        return total_loss


    def add_graph_loss(self, out_adj, features):  # 输入 A(t) 和 features:X
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1, keepdim=True)) - out_adj  # (2708,2708) diagflat对角扩展 L = D-A
        graph_loss += self.config['idgl']['smoothness_ratio'] * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))  # Ω(A,X) = 1/n^2 tr(X^T L X) 最小化平滑损失 | tr() 表示对角线元素总和
        ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
        graph_loss += -self.config['idgl']['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]  # -β/n 1^T log(A1)
        graph_loss += self.config['idgl']['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))  # f(A) = ↑ + γ/n^2 ||A||_2_F
        return graph_loss  # loss_graph



    def get_loss_list(self):
        return self.train_loss_list,self.val_loss_list,self.test_loss_list