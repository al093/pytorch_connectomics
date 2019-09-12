import torch.nn as nn
import torch

from torch_connectomics.model.utils import *
from torch_connectomics.model.blocks import *

class SliceLSTMConv(nn.Module):
    def __init__(self, input_size, num_layers=1, bias=True, dropout=0):
        super(SliceLSTMConv, self).__init__()

        self.lstm = ConvLSTM(input_size=(1, input_size), input_dim=3, hidden_dim=1, kernel_size=(1, 3), num_layers=1,
                             batch_first=True)

        self.conv_layer = conv2d_pad(in_planes=2, out_planes=1, kernel_size=(3, 3), stride=1,
                                     dilation=1, padding=(1, 1), bias=bias)
        self.pad = nn.ReflectionPad2d((1, 1, 0, 0))
        ortho_init(self)

    def forward(self, x):
        # input x will be of [batch, height, width]
        # which then must be arranged into 2 tensors in the format: (seq, batch, features)

        batchSize = x.shape[0]
        height = x.shape[1]
        width = x.shape[2]

        x = x.clone()
        x = x.unsqueeze(1)  # [batch, height, width]
        x_padded = self.pad(x)

        x = torch.cat((x_padded[:, :, :, :-2], x, x_padded[:, :, :, 2:]), 1)  # [batch, 3, h, w]

        # Run along X
        x_ = x.permute(0, 3, 1, 2)  # [batch, width, 3, height]
        x_ = x_.unsqueeze(3)  # [batch, width, 3, 1, height]
        x1 = x_[:, 0:width//2 + 1]
        x1 = torch.flip(x1, [1])
        x2 = x_[:, width//2:]

        x1_h, _ = self.lstm(x1)
        x1_h = x1_h[0]  # [batch, width//2, 1, 1, height]
        x1_h = torch.flip(x1_h, [1])
        x1_h = x1_h[:, :-1]
        x1_h = x1_h.squeeze(dim=2)
        x1_h = x1_h.squeeze(dim=2)  # [batch, width//2 - 1, height]
        x1_h = x1_h.permute(0, 2, 1)  # [batch, height, width//2 - 1]

        x2_h, _ = self.lstm(x2)
        x2_h = x2_h[0]
        x2_h = x2_h.squeeze(dim=2)
        x2_h = x2_h.squeeze(dim=2)  # [batch, width//2, height]
        x2_h = x2_h.permute(0, 2, 1)  # [batch, height, width//2]
        result_along_x = torch.cat((x1_h, x2_h), 2).unsqueeze(1)

        # Run along Y
        x_ = x.permute(0, 2, 1, 3)  # [batch, height, 3, width]
        x_ = x_.unsqueeze(3) # [batch, height, 3, 1, width]
        x1 = x_[:, 0:height//2 + 1]
        x1 = torch.flip(x1, [1])
        x2 = x_[:, height//2:]

        x1_h, _ = self.lstm(x1)
        x1_h = x1_h[0]  # [batch, height//2, 1, 1, width]
        x1_h = torch.flip(x1_h, [1])
        x1_h = x1_h[:, :-1]
        x1_h = x1_h.squeeze(dim=2)
        x1_h = x1_h.squeeze(dim=2)  # [batch, height//2, width]

        x2_h, _ = self.lstm(x2)
        x2_h = x2_h[0]  # [batch, height//2, 1, 1, width]
        x2_h = x2_h.squeeze(dim=2)
        x2_h = x2_h.squeeze(dim=2)  # [batch, height//2, width]
        result_along_y = torch.cat((x1_h, x2_h), 1).unsqueeze(1)

        result = self.conv_layer(torch.cat((result_along_x, result_along_y), 1))
        result = torch.sigmoid(result)

        return result.squeeze()
