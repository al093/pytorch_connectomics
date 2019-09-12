import torch.nn as nn
import torch

from torch_connectomics.model.blocks.convlstm import ConvLSTM

class ConvLSTMCenter(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTMCenter, self).__init__()

        self.conv_lstm = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input x will be of [batch, 1, z, h, w]
        # which then must be arranged into 2 tensors in the format: (seq, batch, features, h, w)
        # [z//2 to z, batch, 1, h, w]
        # [z//2 to 0, batch, 1, h, w] for lstm computation along z
        z_sz = x.shape[2]
        assert x.shape[1] == 1

        x = x.permute(2, 0, 1, 3, 4)  # [z, batch, 1, h, w]
        # split z from the center and invert first half
        x1 = x[0:z_sz//2 + 1]  # must include the center
        x1 = torch.flip(x1, [0])
        x2 = x[z_sz//2:]

        x1_h, _ = self.conv_lstm(x1)
        x1_h = x1_h[0]  # output [z/2, batch, 1, h, w]
        x1_h = torch.transpose(x1_h, 0, 1)
        # change output  0:width//2 and reorder
        x1_h = torch.flip(x1_h, [0])
        x1_h = x1_h.permute(1, 2, 0, 3, 4)
        x1_h = x1_h[:, :, :-1, :, :]  # removing the center z slice as that is taken from the other half

        x2_h, _ = self.conv_lstm(x2)
        x2_h = x2_h[0]
        x2_h = torch.transpose(x2_h, 0, 1)
        x2_h = x2_h.permute(1, 2, 0, 3, 4)

        x = self.relu(torch.cat((x1_h, x2_h), 2))
        return x
