import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from PointNet.PointnetUtils import *

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")

class PositionalEmbedding(nn.Module):

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2 #N 8
        emb = math.log(10000) / half_dim #N 1.15
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # e^[[0,1,2...7]*(-1.15)]  [8]
        emb = torch.outer(x * self.scale, emb) #[len(x), len(emb)]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        #[len(x), len(emb)*2]
        return emb


class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channel, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.conv1(x)
        x = F.relu(self.bn(x))
        print(x.shape)
        x = x.permute(0, 3, 2, 1)
        return x



class DownLayer(nn.Module):
    def __init__(self,
                 in_channels, mlp_channels, out_channels, npoint, radius, nsample
                 ):
        """
        :param in_channels:points的channels
        :param mlp_channels: 中间态channels
        :param out_channels: 输出的channels
        :param npoint:
        :param radius:
        :param nsample:
        """
        super().__init__()
        self.radius = radius
        self.npoint = npoint
        self.nsample = nsample
        self.in_channels = in_channels
        self.mlp_channels = mlp_channels
        self.out_channels = out_channels

        self.mlp_layer1 = nn.Linear(in_channels, mlp_channels)
        self.conv_layer1 = GroupConv(mlp_channels, out_channels)

    def forward(self, xyz, t, points=None):
        device = xyz.device
        if points is not None:
            B, N, C = points.shape
        else:
            B, N, C = xyz.shape



if __name__ == '__main__':
    B, npoint, nsample, C = 1, 64, 32, 32
    inchannels = 32
    outchannels = 128
    convtest = GroupConv(inchannels, outchannels)
    data = torch.randn(B, npoint, nsample, C)
    result = convtest(data)
    print(result.shape)















