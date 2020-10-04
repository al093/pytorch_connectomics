import torch
import torch.nn as nn

from torch_connectomics.model.utils import *
from torch_connectomics.model.blocks import *

class FluxToSkeletonHead(nn.Module):

    def __init__(self, xy_z_factor: int = 1):
        super().__init__()

        modules = []

        half_kernel_size = [2, 2*xy_z_factor, 2*xy_z_factor]
        padding, kernel_size = self._get_padding_kernel_size(half_kernel_size)
        modules.append(conv3d_bn_relu(in_planes=4, out_planes=1, kernel_size=kernel_size, stride=1, padding=padding))

        dilation = [2, 2*xy_z_factor, 2*xy_z_factor]
        padding, kernel_size = self._get_padding_kernel_size(half_kernel_size, dilation)
        modules.append(conv3d_bn_relu(in_planes=4, out_planes=1, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation,))

        half_kernel_size = [3, 3*xy_z_factor, 3*xy_z_factor]
        padding, kernel_size = self._get_padding_kernel_size(half_kernel_size)
        modules.append(conv3d_bn_relu(in_planes=4, out_planes=1, kernel_size=kernel_size, stride=1, padding=padding))

        dilation = [2, 2*xy_z_factor, 2*xy_z_factor]
        padding, kernel_size = self._get_padding_kernel_size(half_kernel_size, dilation)
        modules.append(conv3d_bn_relu(in_planes=4, out_planes=1, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(nn.Conv3d(4, 1, 1, bias=False),
                                     nn.Sigmoid())

        self.pad_fn = nn.ReplicationPad3d(1)

        #initialization
        ortho_init(self)

    def forward(self, x):
        divergence = self._3d_divergence(x)
        divergence_x = torch.cat([divergence, x], dim=1)

        res = []
        for conv in self.convs:
            res.append(conv(divergence_x))
        res = torch.cat(res, dim=1)
        return self.project(res)

    def _3d_divergence(self, input):
        input_pad = self.pad_fn(input)
        dx = input_pad[:, 0, 1:-1, 1:-1, 0:-2] - input[:, 0]
        dy = input_pad[:, 1, 1:-1, 0:-2, 1:-1] - input[:, 1]
        dz = input_pad[:, 2, 0:-2, 1:-1, 1:-1] - input[:, 2]
        return (dz + dy + dx).unsqueeze(1)

    def _get_padding_kernel_size(self, half_kernel_size, dilation=(1, 1, 1)):
        kernel_size = tuple([k * 2 + 1 for k in half_kernel_size])
        padding = tuple([k*d for k, d in zip(half_kernel_size, dilation)])
        return padding, kernel_size
