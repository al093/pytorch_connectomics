import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial as sp
import numpy as np

from torch_connectomics.model.utils import *
from torch_connectomics.model.blocks import *

class DirectionNet(nn.Module):

    def __init__(self, input_sz, in_channel=1, filters=[8, 16, 24, 32, 40]):
        super().__init__()

        # encoding path
        self.layer1_E = nn.Sequential(
            conv3d_bn_elu(in_planes=in_channel, out_planes=filters[0],
                          kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            conv3d_bn_elu(in_planes=filters[0], out_planes=filters[0],
                          kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            residual_block_2d(filters[0], filters[0], projection=False)
        )
        self.layer2_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[0], out_planes=filters[1],
                          kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            residual_block_3d(filters[1], filters[1], projection=False)
        )
        self.layer3_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[1], out_planes=filters[2],
                          kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            residual_block_3d(filters[2], filters[2], projection=False)
        )
        self.layer4_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[2], out_planes=filters[3],
                          kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            residual_block_3d(filters[3], filters[3], projection=False)
        )

        self.convLSTMCell3D = ConvLSTMCell3D(input_sz, filters[3], filters[3])

        self.layer5_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[3], out_planes=filters[4],
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            residual_block_3d(filters[4], filters[4], projection=True)
        )

        self.linear_layer_in_sz = (filters[-1]*np.prod(np.array(input_sz) / np.array([8, 16, 16]))).astype(np.int32)
        self.fc_1 = nn.Linear(self.linear_layer_in_sz, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 3)

        self.down = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down_z = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.dropout = nn.Dropout3d(p=0.25)
        self.act = nn.Tanh()

        # initialization
        ortho_init(self)

    def forward(self, x, lstm_hidden_state=None, lstm_cell_state=None):

        # encoding path
        z1 = self.layer1_E(x)
        x = self.down(z1)

        z2 = self.layer2_E(x)
        x = self.down_z(z2)

        z3 = self.layer3_E(x)
        x = self.down_z(z3)
        x = self.dropout(x)

        z4 = self.layer4_E(x)
        x = self.down_z(z4)
        x = self.dropout(x)

        lstm_hidden_state_next, lstm_cell_state_next = self.convLSTMCell3D(x, lstm_hidden_state, lstm_cell_state)

        x = self.layer5_E(lstm_hidden_state_next)

        x = x.view(-1, self.linear_layer_in_sz)
        x = self.fc_1(x)
        x = self.act(x)

        x = self.fc_2(x)
        x = self.act(x)

        x = self.fc_3(x)
        x = self.act(x)

        return x, lstm_hidden_state_next, lstm_cell_state_next
