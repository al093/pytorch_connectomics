import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial as sp
import numpy as np

from torch_connectomics.model.utils import *
from torch_connectomics.model.blocks import *


class ClassificationNet(nn.Module):

    def __init__(self, in_channel=1, filters=(16, 16, 32, 32, 64), non_linearity=torch.sigmoid):
        super().__init__()

        # encoding path
        self.layer1_E = nn.Sequential(
            conv3d_bn_elu(in_planes=in_channel, out_planes=filters[0], 
                          kernel_size=(1,5,5), stride=1, padding=(0,2,2)),
            conv3d_bn_elu(in_planes=filters[0], out_planes=filters[0], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_2d(filters[0], filters[0], projection=False)  
        )
        self.layer2_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[0], out_planes=filters[1], 
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            residual_block_3d(filters[1], filters[1], projection=False),
            residual_block_3d(filters[1], filters[1], projection=False)
        )
        self.layer3_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[1], out_planes=filters[2], 
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            residual_block_3d(filters[2], filters[2], projection=False),
            residual_block_3d(filters[2], filters[2], projection=False)
        )
        self.layer4_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[2], out_planes=filters[3], 
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            residual_block_3d(filters[3], filters[3], projection=False),
            residual_block_3d(filters[3], filters[3], projection=False)
        )

        self.layer5_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[3], out_planes=filters[4],
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            residual_block_3d(filters[4], filters[4], projection=False),
            residual_block_3d(filters[4], filters[4], projection=False)
        )

        # pooling & upsample blocks
        self.down = nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.down_z = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.linear_layer_in_sz = filters[-1]*4*6*6
        self.fc_1 = nn.Linear(self.linear_layer_in_sz, 64)
        self.fc_2 = nn.Linear(64, 1)

        self.dropout = nn.modules.Dropout(p=0.33)
        self.non_linearity = non_linearity

        #initialization
        ortho_init(self)

    def forward(self, x):

        # encoding path
        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down_z(z2)
        x = self.dropout(x)
        z3 = self.layer3_E(x)
        x = self.down_z(z3)
        x = self.dropout(x)
        z4 = self.layer4_E(x)
        x = self.down_z(z4)
        x = self.dropout(x)
        z5 = self.layer5_E(x)
        x = self.down_z(z5)
        x = self.dropout(x)
        x = x.view(-1, self.linear_layer_in_sz)
        x = self.fc_1(x)
        x = self.fc_2(x)
        # x = self.non_linearity(x)

        return x

def get_distance_feature(size):
    z = np.arange(size[0], dtype=np.int16)
    y = np.arange(size[1], dtype=np.int16)
    x = np.arange(size[2], dtype=np.int16)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    Z =  Z.astype(np.int32) - (size[0] // 2)
    Y =  Y.astype(np.int32) - (size[1] // 2)
    X =  X.astype(np.int32) - (size[2] // 2)
    return np.sqrt(Z**2 + Y**2 + X**2, dtype=np.float32)