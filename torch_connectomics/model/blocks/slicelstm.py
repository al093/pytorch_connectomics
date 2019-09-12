import torch.nn as nn
import torch

from torch_connectomics.model.utils import *
from torch_connectomics.model.blocks import *

class SliceLSTM(nn.Module):
    def __init__(self, input_size, num_layers=1, bias=True, dropout=0):
        super(SliceLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size,
                            num_layers=num_layers, bias=bias, dropout=dropout)
        self.conv_layer = conv2d_pad(in_planes=2, out_planes=1, kernel_size=(3, 3), stride=1,
                                     dilation=1, padding=(1, 1), bias=bias)

        ortho_init(self)

    def forward(self, x):
        # input x will be of [batch, height, width]
        # which then must be arranged into 2 tensors in the format: (seq, batch, features)
        # [width//2 to width, batch, height]
        # [width//2 to 0, batch, height]for lstm computation along X
        # the final output should be of shape [batch, 1, height, width]

        batchSize = x.shape[0]
        height = x.shape[1]
        width = x.shape[2]

        # Run along X
        # split x from the center and invert first half
        x = x.clone()
        x_ = x.permute(2, 0, 1)  # [width, batch, height]
        x1 = x_[0:width//2]
        x1 = torch.flip(x1, [0])
        x2 = x_[width//2:]

        x1_h, _ = self.lstm(x1)  # output [width//2-1:[0], batch, height]
        # change output  0:width//2 and reorder
        x1_h = torch.flip(x1_h, [0])
        x1_h = x1_h.permute(1, 2, 0)  # output [batch, height, width//2-1:[0]]

        x2_h, _ = self.lstm(x2)  # output [width//2:end, batch, height]
        x2_h = x2_h.permute(1, 2, 0)  # output [batch, height, width//2:end]
        result_along_x = torch.cat((x1_h, x2_h), 2).unsqueeze(1)

        # Run along Y
        x_ = x.permute(1, 0, 2)  # [height, batch, width]
        x1 = x_[0:height // 2]
        x1 = torch.flip(x1, [0])
        x2 = x_[height // 2:]

        x1_h, _ = self.lstm(x1)  # output [height//2-1:[0], batch, width]
        x1_h = torch.flip(x1_h, [0])
        x1_h = x1_h.permute(1, 0, 2)

        x2_h, _ = self.lstm(x2)  # output [height//2:end, batch, width]
        x2_h = x2_h.permute(1, 0, 2)  # output [batch, height//2:end, width]
        result_along_y = torch.cat((x1_h, x2_h), 1).unsqueeze(1)

        result = self.conv_layer(torch.cat((result_along_x, result_along_y), 1))
        result = torch.sigmoid(result)

        return result.squeeze()
