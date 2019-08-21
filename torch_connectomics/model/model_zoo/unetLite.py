import os,sys

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch_connectomics.model.blocks import *
from torch_connectomics.model.utils import *
from torch_connectomics.libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d


class unetLite(nn.Module):
    """Light Lightweight U-net with residual blocks .

    Args:
        in_channel (int): number of input channels.
        out_channel (int): number of output channels.
        filters (list): number of filters at each u-net stage.
    """
    def __init__(self, in_channel=1, out_channel=3, filters=[8, 12, 16, 20, 24]):
        super().__init__()

        # encoding path
        self.layer1_E = nn.Sequential(
            conv3d_bn_lrelu(in_planes=in_channel, out_planes=filters[0],
                            kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            conv3d_bn_lrelu(in_planes=filters[0], out_planes=filters[0],
                            kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        )

        self.layer2_E = nn.Sequential(
            conv3d_bn_lrelu(in_planes=filters[0], out_planes=filters[1],
                            kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        )

        self.layer3_E = nn.Sequential(
            conv3d_bn_lrelu(in_planes=filters[1], out_planes=filters[2],
                            kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        )

        self.layer4_E = nn.Sequential(
            conv3d_bn_lrelu(in_planes=filters[2], out_planes=filters[3],
                            kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        )

        # center block
        self.center = nn.Sequential(
            conv3d_bn_lrelu(in_planes=filters[3], out_planes=filters[4],
                            kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
        )

        # decoding path
        self.layer1_D = nn.Sequential(
            conv3d_bn_lrelu(in_planes=filters[0], out_planes=filters[0],
                            kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            conv3d_bn_non(in_planes=filters[0], out_planes=out_channel, 
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        )
        self.layer2_D = nn.Sequential(
            conv3d_bn_lrelu(in_planes=filters[1], out_planes=filters[1],
                            kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
        )
        self.layer3_D = nn.Sequential(
            conv3d_bn_lrelu(in_planes=filters[2], out_planes=filters[2],
                            kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
        )

        self.layer4_D = nn.Sequential(
            conv3d_bn_lrelu(in_planes=filters[3], out_planes=filters[3],
                            kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
        )

        # pooling & upsample
        self.down = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.up = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False)
        # self.dropout = nn.Dropout3d(p=0.25)

        # conv + upsample
        self.conv1 = conv3d_bn_lrelu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv2 = conv3d_bn_lrelu(filters[2], filters[1], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_lrelu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv4 = conv3d_bn_lrelu(filters[4], filters[3], kernel_size=(1,1,1), padding=(0,0,0))

        #initialization
        ortho_init(self)

    def forward(self, x):

        # encoding path
        z1 = self.layer1_E(x)
        x = self.down(z1)

        z2 = self.layer2_E(x)
        x = self.down(z2)

        z3 = self.layer3_E(x)
        x = self.down(z3)

        z4 = self.layer4_E(x)
        x = self.down(z4)

        x = self.center(x)

        # decoding path
        x = self.up(self.conv4(x))
        x = x + z4
        x = self.layer4_D(x)

        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = torch.sigmoid(x)
        return x