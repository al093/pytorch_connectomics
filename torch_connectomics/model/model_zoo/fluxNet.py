import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import scipy.spatial as sp
import numpy as np

from torch_connectomics.model.utils import *
from torch_connectomics.model.blocks import *


class FluxNet(nn.Module):

    def __init__(self, input_sz, batch_sz, in_channel=1, out_channel=3, filters=[8, 16, 24, 32, 64],
                 non_linearity=(torch.sigmoid), aspp_dilation_ratio=1, symmetric=True):
        super().__init__()

        # encoding path
        self.symmetric = symmetric
        if self.symmetric:
            self.layer1_E = nn.Sequential(
                conv3d_bn_relu(in_planes=in_channel, out_planes=filters[0],
                               kernel_size=(5,5,5), stride=1, padding=(2,2,2)),
                conv3d_bn_relu(in_planes=filters[0], out_planes=filters[0],
                              kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
                residual_block_3d(filters[0], filters[0], projection=False)
            )
        else:
            self.layer1_E = nn.Sequential(
                conv3d_bn_relu(in_planes=in_channel, out_planes=filters[0],
                               kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
                conv3d_bn_relu(in_planes=filters[0], out_planes=filters[0],
                               kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                residual_block_2d(filters[0], filters[0], projection=False)
            )

        self.layer2_E = nn.Sequential(
            conv3d_bn_relu(in_planes=filters[0], out_planes=filters[1],
                           kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            residual_block_3d(filters[1], filters[1], projection=False)
        )
        self.layer3_E = nn.Sequential(
            conv3d_bn_relu(in_planes=filters[1], out_planes=filters[2],
                           kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            residual_block_3d(filters[2], filters[2], projection=False)
        )
        self.layer4_E = nn.Sequential(
            conv3d_bn_relu(in_planes=filters[2], out_planes=filters[3],
                           kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            residual_block_3d(filters[3], filters[3], projection=False)
        )

        # center ASPP block
        self.center = ASPP(filters[3], filters[4], [[2, int(2*aspp_dilation_ratio), int(2*aspp_dilation_ratio)],
                                                    [3, int(3*aspp_dilation_ratio), int(3*aspp_dilation_ratio)],
                                                    [5, int(5*aspp_dilation_ratio), int(5*aspp_dilation_ratio)]])

        # decoding path
        self.layer1_D_flux = nn.Sequential(
            conv3d_bn_relu(in_planes=filters[1], out_planes=filters[0],
                           kernel_size=(1,1,1), stride=1, padding=(0,0,0)),
            residual_block_3d(filters[0], filters[0], projection=False),
            conv3d_bn_non(in_planes=filters[0], out_planes=3,
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        )

        self.layer1_D_skeleton = nn.Sequential(
            conv3d_bn_relu(in_planes=filters[1], out_planes=filters[0],
                           kernel_size=(1,1,1), stride=1, padding=(0,0,0)),
            residual_block_3d(filters[0], filters[0], projection=False),
            conv3d_bn_non(in_planes=filters[0], out_planes=1,
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        )

        self.layer2_D = nn.Sequential(
            conv3d_bn_relu(in_planes=filters[2], out_planes=filters[1],
                           kernel_size=(1,1,1), stride=1, padding=(0,0,0)),
            residual_block_3d(filters[1], filters[1], projection=False)
        )
        self.layer3_D = nn.Sequential(
            conv3d_bn_relu(in_planes=filters[3], out_planes=filters[2],
                           kernel_size=(1,1,1), stride=1, padding=(0,0,0)),
            residual_block_3d(filters[2], filters[2], projection=False)
        )
        self.layer4_D = nn.Sequential(
            conv3d_bn_relu(in_planes=filters[4], out_planes=filters[3],
                           kernel_size=(1,1,1), stride=1, padding=(0,0,0)),
            residual_block_3d(filters[3], filters[3], projection=False)
        )

        # downsample pooling
        self.down = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.down_aniso = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # upsampling
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
        self.up_aniso = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')

        if len(non_linearity) == 2 and out_channel > 1:
            self.non_linearity_1 = non_linearity[0]
            self.non_linearity_2 = non_linearity[1]
        elif len(non_linearity) == 1:
            self.non_linearity_1 = non_linearity[0]
            self.non_linearity_2 = None
        else:
            raise Exception('Undefined Network configuration, More than one nonlinearities but output channels are 1')

        #initialization
        ortho_init(self)

    def forward(self, x, get_penultimate_layer=False):

        # encoding path
        z1 = self.layer1_E(x)
        x = self.down(z1) if self.symmetric else self.down_aniso(z1)

        z2 = self.layer2_E(x)
        x = self.down(z2)

        z3 = self.layer3_E(x)
        x = self.down(z3)

        z4 = self.layer4_E(x)

        #center ASPP block
        x = self.center(z4)

        # decoding path
        x = self.layer4_D(x)

        x = self.up(x)
        x = self.layer3_D(x)
        x = x + z3

        x = self.up(x)
        x = self.layer2_D(x)
        x = x + z2

        x = self.up(x) if self.symmetric else self.up_aniso(x)

        flux = self.layer1_D_flux(x)
        flux = self.non_linearity_1(flux)

        skeleton = self.layer1_D_skeleton(x)
        skeleton = nn.functional.sigmoid(skeleton)

        return flux, skeleton
