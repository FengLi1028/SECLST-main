import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        result = torch.mean(nb_final)
        return result



class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
       return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
