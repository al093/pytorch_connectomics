import torch.nn as nn
from torch.autograd import Variable
import torch

from torch_connectomics.model.utils import *

class ConvLSTMCell3D(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size=(3,3,3), bias=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int, int)
            Height and width of input tensor as (depth, height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell3D, self).__init__()

        self.depth, self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.bias = bias

        self.padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2

        self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        ortho_init(self)

    def forward(self, input, h, c):

        output_shape = (input.shape[0], self.hidden_dim, input.shape[2], input.shape[3], input.shape[4])
        if h is None:
            h = torch.zeros(output_shape, dtype=input.dtype, device=input.device)
        if c is None:
            c = torch.zeros(output_shape, dtype=input.dtype, device=input.device)

        combined = torch.cat([input, h], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next