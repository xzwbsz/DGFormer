import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True): # 16 64
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias) # 输入映射 w*x+b
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias) # 隐藏映射 w*x+b
        self.reset_parameters() # 权重初始化

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(-1))

        gate_x = self.x2h(x) # 用全连接层给x乘上w，加上bias
        gate_h = self.h2h(hidden) # 用全连接层给hidden乘上w，加上bias

        gate_x = gate_x.squeeze() # 对数据的维度进行压缩 去掉维数为1的的维度
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1) # 将输入gate_x分割成3个子Tensor
        h_r, h_i, h_n = gate_h.chunk(3, 1) # 将输入gate_h分割成3个子Tensor

        resetgate = F.sigmoid(i_r + h_r) # 重置门
        inputgate = F.sigmoid(i_i + h_i) # 更新门
        newgate = F.tanh(i_n + (resetgate * h_n)) # 当前记忆内容 ht_hat

        hy = newgate + inputgate * (hidden - newgate) # 当前时间步的最终记忆 ht

        return hy


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden

        x = x.view(-1, x.size(-1))

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))

        return (hy, cy)