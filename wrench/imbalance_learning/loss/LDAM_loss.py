import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, device=None):
        super(LDAMLoss, self).__init__()
        self.device = device
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))  # nj的四次开方
        m_list = m_list * (max_m / np.max(m_list))  # 常系数 C
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        assert s > 0
        self.s = s  # 这个参数的作用论文里提过么？
        self.weight = weight  # 和频率相关的 re-weight

    def forward(self, x, target, **kwargs):
        index = torch.zeros_like(x, dtype=torch.uint8)  # 和 x 维度一致全 0 的tensor
        index.scatter_(1, target.data.view(-1, 1), 1)  # dim idx input
        index_float = index.type(torch.FloatTensor).to(self.device)
        ''' 以上的idx指示的应该是一个batch的y_true '''
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m  # y 的 logit 减去 margin
        output = torch.where(index, x_m, x)  # 按照修改位置合并
        return F.cross_entropy(self.s * output, target, weight=self.weight)
