import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def set_beta():
    # return np.linspace(1, 2*n_class, num=10)[int(str(step)[0])]
    return 1.2


def loss_cores(step, logits, target, noise_prior=None):
    beta = set_beta()

    loss = F.cross_entropy(logits, target, reduction='none')
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_ = -torch.log(F.softmax(logits, dim=-1) + 1e-8)
    # sel metric
    loss_sel = loss - torch.mean(loss_, 1)
    if noise_prior is None:
        loss = loss - beta * torch.mean(loss_, 1)
    else:
        loss = loss - beta * torch.sum(torch.mul(noise_prior.cuda(), loss_), 1)

    loss_div_numpy = loss_sel.data.cpu().numpy()

    for i in range(len(loss_numpy)):
        if step <= 500:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_) / 100000000, loss_sel
    else:
        return torch.sum(loss_) / sum(loss_v), loss_sel


class CoreLoss(nn.Module):
    def __init__(self,
                 n_class,
                 noise_prior=None):
        super(CoreLoss, self).__init__()
        self.prior = noise_prior.cuda()
        self.n_class = n_class

    @torch.no_grad()
    def _data_sel(self, loss, loss_copy):
        return loss - torch.mean(loss_copy, 1)

    def forward(self, inputs, target, step):
        loss = F.cross_entropy(inputs, target, reduction='none')
        loss_ = -torch.log(F.softmax(inputs, dim=1) + 1e-8)
        score = self._data_sel(loss, loss_)
        v = torch.lt(score, 0).cpu().numpy()
        v = Variable(torch.from_numpy(v)).cuda()

        loss_cr = set_beta() * torch.sum(torch.mul(self.prior, loss_), 1)
        loss = loss - loss_cr

        if step <= 500:
            v[:] = 1.0

        if sum(v) == 0.0:
            return torch.mean(torch.mul(v, loss)) / 100000000, score

        return torch.sum(v * loss) / torch.sum(v), score

