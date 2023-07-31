import gc
import os
import shutil
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir) # 将路径添加到环境目录1
from utils import config, file_dir
from graph import Graph
from dataset import HaveData
from dataset_preprocessing import Dataset_pre
from model.LS_GNN import LS_GNN
from model_handler import ModelHandler
from model_handler_agrcn import ModelHandler_agrcn
from model_handler_dcrnn import ModelHandler_dcrnn
from model_handler_stgcn import ModelHandler_stgcn
from model_handler_clcrn import ModelHandler_clcrn

import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import glob
import psutil

feat_site = '/public/ecai/uvq_eur/useData/'
time_site = '/public/ecai/uvq_eur/useData/'
feat_file = feat_site + 'feature.npy'
time_file = time_site + 'time.npy'

# 设置运行设备
torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# device = torch.device('cpu')

# 实例化 图类
graph = Graph()
station_num = graph.node_num # 节点数
station_node = graph.nodes # 字典 idx:{station_name,lon,lat,altitude}
config['station_num'] = station_num
config['edge_index'] = graph.edge_index
config['edge_attr'] = graph.edge_attr
adj0 = graph.adj
lonlat = graph.lonlat
dist = graph.dist_adj
angle_adj = graph.angle_adj

"""nc文件预处理/读取npy"""
if os.path.exists(feat_file) and os.path.exists(time_file):
    print("Dataset use save_npy")
    node_attr = np.load(feat_file)
    timestamp = np.load(time_file)
else:
    dataset_fp = file_dir['dataset1_fp']  # 数据集上级目录
    start_month = int(config['dataset']['start_month'])
    end_month = int(config['dataset']['end_month'])
    for fdir in sorted(glob.glob(dataset_fp)):
        month = int(fdir[-2] + fdir[-1])
        year = int(fdir[-7] + fdir[-6] + fdir[-5] + fdir[-4])
        if month >= start_month and month <= end_month:
            fdir = fdir + "/*"
            for f in sorted(glob.glob(fdir)):
                print(f)
                dataset_pre = Dataset_pre(station_node,station_num,f,year)

    node_attr = np.load(feat_file) # get type:ndarray | var_name:node_attr | shape:(node_num,time_num,attr_num)
    timestamp = np.load(time_file) # get type:ndarray | var_name:timestamp | shape:(time_num,)


"""更改变量维度次序 确保要预测的变量在第一维"""
use_var_list = config['experiments']['metero_use']
pred_var_name = config['experiments']['pred_var']
use_var_num = config['idgl']['hidden_size']
pred_var_site = -1
for i in range(use_var_num):
    if pred_var_name[0] == use_var_list[i]:
        pred_var_site = i
if pred_var_site == -1:
    print("Not get pred_var_num value")
var_feature1 = node_attr[:,:,:pred_var_site]
var_pred = node_attr[:,:,pred_var_site:pred_var_site+1]
var_feature2 = node_attr[:,:,pred_var_site+1:]
node_attr = np.concatenate((var_pred,var_feature1,var_feature2),axis=-1)


"""通过untils中的函数读取config文件 并获取相应的参数"""
batch_size = config['train']['batch_size'] # 32
epochs = config['train']['epochs'] # 50
hist_len = config['train']['hist_len'] # 1 # 隐藏层大小
pred_len = config['train']['pred_len'] # 24 预测天数
weight_decay = config['train']['weight_decay'] # 0.0005 L2正则化的目的就是为了让权重衰减到更小的值，在一定程度上减少模型过拟合的问题，所以权重衰减也叫L2正则化。
early_stop = config['train']['early_stop'] # 10 当early stop被激活(如发现loss相比上一个epoch训练没有下降)，则经过10(此处为10)个epoch后停止训练
lr = config['train']['lr'] # 0.0005 学习率
results_dir = file_dir['results_dir'] # util下的file_dir下的results_dir路径
dataset_num = config['experiments']['dataset_num'] # 1 采用数据集配置1
exp_model = config['experiments']['model'] # 根据model选择 LS_GNN
exp_repeat = config['train']['exp_repeat'] # 10次实验 减少实验误差
save_npy = config['experiments']['save_npy'] # True 是否保存模型为.npy文件
seed = config['idgl']['seed'] # 420
idgl_hid_len = config['idgl']['hidden_size']
graph_learn = config['idgl']['graph_learn']
criterion = nn.MSELoss() # 定义MSE为loss

"""实例化dataset.py文件对数据进行所需要的处理"""
train_data = HaveData(graph, node_attr, timestamp, hist_len, pred_len, dataset_num, flag='Train')
val_data = HaveData(graph, node_attr, timestamp, hist_len, pred_len, dataset_num, flag='Val')
test_data = HaveData(graph, node_attr, timestamp, hist_len, pred_len, dataset_num, flag='Test')
in_dim = int(idgl_hid_len) # 3
config['in_dim'] = in_dim
t2m_mean,t2m_std = test_data.t2m_mean,test_data.t2m_std
print("attr_std:",t2m_std[0])
print("pred_var:",config['experiments']['pred_var'])


def get_exp_info():
    exp_info = '============== Train Info ==============\n' + \
               'Dataset number: %s\n' % dataset_num + \
               'Model: %s\n' % exp_model + \
               'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
               'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
               'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
               'Station number: %s\n' % station_num + \
               'Use metero: %s\n' % config['experiments']['metero_use'] + \
               'batch_size: %s\n' % batch_size + \
               'epochs: %s\n' % epochs + \
               'hist_len: %s\n' % hist_len + \
               'pred_len: %s\n' % pred_len + \
               'weight_decay: %s\n' % weight_decay + \
               'early_stop: %s\n' % early_stop + \
               'lr: %s\n' % lr + \
               '=============== IDGL Info ==============\n' + \
               'seed: %s\n' % seed + \
               'idgl_hid_len: %s\n' % idgl_hid_len + \
               '========================================\n'
    return exp_info


def get_model():
    if exp_model == 'LS_GNN':
        return LS_GNN(hist_len, pred_len, in_dim, station_num, batch_size, device, graph.edge_index, graph.edge_attr)
    else:
        raise Exception('Wrong model name!')

def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(),data.std()


def get_metric(predict_epoch, label_epoch):
    predict = predict_epoch[:, :, :, 0].transpose((0, 2, 1))
    label = label_epoch[:, :, :, 0].transpose((0, 2, 1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    return rmse, mae


def main():
    if True:
        exp_info = get_exp_info()
        exp_time = arrow.now().format('YYYYMMDDHHmmss')

        # 调用DataLoader 返回dataset.py的__getitem__函数值 ()
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        # exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), exp_model,str(exp_time))

        if os.path.exists("./save_train_param"):
            shutil.rmtree("./save_train_param")
            os.mkdir("./save_train_param")
        else:
            os.mkdir("./save_train_param")

        file_exit = os.path.exists("./save_final_Parameter")
        if file_exit == False:
            os.mkdir("./save_final_Parameter")

        if os.path.exists("./result"):
            shutil.rmtree("./result")
            os.mkdir("./result")
        else:
            os.mkdir("./result")

        if os.path.exists("./png"):
            shutil.rmtree("./png")
            os.mkdir("./png")
        else:
            os.mkdir("./png")

        best_epoch = 0
        val_loss_min = 100000
        train_loss_list , val_loss_list , test_loss_list , rmse_list , mae_list = [],[],[],[],[]

        for epoch in range(config['idgl']['max_epochs']):
            if config['idgl']['graph_model'] == 'ls-gnn':
                model = ModelHandler(config, train_loader, val_loader, test_loader, adj0, dist, angle_adj)
            elif config['idgl']['graph_model'] == 'agrcn':
                model = ModelHandler_agrcn(config, train_loader, val_loader, test_loader, adj0, dist)
            elif config['idgl']['graph_model'] == 'dcrnn':
                model = ModelHandler_dcrnn(config, train_loader, val_loader, test_loader, adj0, dist)
            elif config['idgl']['graph_model'] == 'stgcn':
                model = ModelHandler_stgcn(config, train_loader, val_loader, test_loader, adj0, dist)
            elif config['idgl']['graph_model'] == 'clcrn':
                model = ModelHandler_clcrn(config, train_loader, val_loader, test_loader, adj0, dist, lonlat)

            epoch += 1

            print('\nTrain epoch %s:' % (epoch))

            # train
            train_loss = model.train(epoch)
            print('train_loss: %.4f' % train_loss)
            train_loss_ = train_loss
            train_loss_list.append(train_loss_)

            # time.sleep(10)

            # val
            val_loss = model.val()
            print('val_loss: %.4f' % val_loss)
            val_loss_ = val_loss
            val_loss_list.append(val_loss_)

            if epoch - best_epoch > early_stop:
                break

            if val_loss_ < val_loss_min:
                val_loss_min = val_loss_
                best_epoch = epoch
                print('Minimum val loss!!!')

                # time.sleep(20)
                test_loss_, predict_epoch_, label_epoch_, time_epoch_ = model.test(t2m_mean,t2m_std)
                test_loss_list.append(test_loss_)

                rmse, mae = get_metric(predict_epoch_, label_epoch_)
                print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.8f , MAE:%0.8f' % (train_loss_, val_loss_, test_loss_, rmse, mae))

                """加入NMSE"""
                nmse = rmse/t2m_std
                print('NMSE:',nmse[0])

                rmse_list.append(rmse)
                mae_list.append(mae)

                if save_npy == True:
                    print('save numpy!')
                    save_predicate = "result/epoch="+str(epoch)+"_predicateList"
                    save_label = "result/epoch=" + str(epoch) + "_labelList"
                    save_time = "result/epoch=" + str(epoch) + "_timeList"
                    np.save(save_predicate,predict_epoch_)
                    np.save(save_label,label_epoch_)
                    np.save(save_time,time_epoch_)

            del model
            gc.collect()
            torch.cuda.empty_cache()

        exp_epoch_str = '---------------------------------------\n' + \
                         'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                         'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                         'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                         'RMSE       | mean: %0.8f std: %0.8f\n' % (get_mean_std(rmse_list)) + \
                         'MAE        | mean: %0.8f std: %0.8f\n' % (get_mean_std(mae_list))

        # metric_fp = os.path.join(os.path.dirname(exp_model_dir),'metric.txt')

        print('=========================\n')
        print(exp_info)
        print(exp_epoch_str)
        print(str(model))
        # print(metric_fp)

if __name__ == '__main__':
    main()