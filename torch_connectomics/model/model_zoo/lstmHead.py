import torch
import torch.nn as nn
import numpy as np

from torch_connectomics.model.utils import *
from torch_connectomics.model.blocks import *

class LSTMHead(nn.Module):
    def __init__(self, input_sz=None):
        super().__init__()

        self.slice_lstm = SliceLSTM(input_size=int(input_sz[2]))
        # self.conv_lstm = ConvLSTMCenter(input_sz[1:], input_dim=1, hidden_dim=1, kernel_size=(5, 5), num_layers=1)

    def forward(self, x):

        # run sliceLSTM on middle slice
        center_slice_idx = x.shape[2] // 2
        x[:, 0, center_slice_idx, :, :] = self.slice_lstm(x[:, 0, center_slice_idx, :, :])

        #run convLSTM along Z now
        # x = self.conv_lstm(x)

        return x