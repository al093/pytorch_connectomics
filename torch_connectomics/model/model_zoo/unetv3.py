import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial as sp
import numpy as np

from torch_connectomics.model.utils import *
from torch_connectomics.model.blocks import *



class unetv3(nn.Module):
    """Lightweight U-net with residual blocks (based on [Lee2017]_ with modifications).

    .. [Lee2017] Lee, Kisuk, Jonathan Zung, Peter Li, Viren Jain, and 
        H. Sebastian Seung. "Superhuman accuracy on the SNEMI3D connectomics 
        challenge." arXiv preprint arXiv:1706.00120, 2017.
        
    Args:
        in_channel (int): number of input channels.
        out_channel (int): number of output channels.
        filters (list): number of filters at each u-net stage.
    """
    def __init__(self, input_sz, batch_sz, in_channel=1, out_channel=3, filters=[8, 12, 16, 20, 24], non_linearity=torch.sigmoid):
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
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[1], filters[1], projection=False)
        )
        self.layer3_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[1], out_planes=filters[2], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[2], filters[2], projection=False)
        )
        self.layer4_E = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[2], out_planes=filters[3], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[3], filters[3], projection=False)
        )

        # center block
        self.center = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[3], out_planes=filters[4], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[4], filters[4], projection=True)
        )

        # decoding path
        self.layer1_D = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[0]+1, out_planes=filters[0],
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_2d(filters[0], filters[0], projection=False),
            conv3d_bn_non(in_planes=filters[0], out_planes=out_channel, 
                          kernel_size=(1,5,5), stride=1, padding=(0,2,2))
        )
        self.layer2_D = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[1]+1, out_planes=filters[1],
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[1], filters[1], projection=False)
        )
        self.layer3_D = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[2]+1, out_planes=filters[2],
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[2], filters[2], projection=False)
        )
        self.layer4_D = nn.Sequential(
            conv3d_bn_elu(in_planes=filters[3]+1, out_planes=filters[3],
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            residual_block_3d(filters[3], filters[3], projection=False)
        )

        # pooling & upsample
        self.down = nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        self.dropout = nn.Dropout3d(p=0.25)

        # conv + upsample
        self.conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv2 = conv3d_bn_elu(filters[2], filters[1], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_elu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0)) 
        self.conv4 = conv3d_bn_elu(filters[4], filters[3], kernel_size=(1,1,1), padding=(0,0,0))

        self.attention_layer1D = get_distance_feature(input_sz)
        self.attention_layer1D = np.tile(self.attention_layer1D, (batch_sz, 1, 1, 1, 1))
        self.attention_layer1D = torch.from_numpy(self.attention_layer1D).cuda()

        sz = list(input_sz)
        sz[1:3] = [x // 2 for x in sz[1:3]]
        self.attention_layer2D = get_distance_feature(sz)
        self.attention_layer2D = np.tile(self.attention_layer2D, (batch_sz, 1, 1, 1, 1))
        self.attention_layer2D = torch.from_numpy(self.attention_layer2D).cuda()

        sz[1:3] = [x // 2 for x in sz[1:3]]
        self.attention_layer3D = get_distance_feature(sz)
        self.attention_layer3D = np.tile(self.attention_layer3D, (batch_sz, 1, 1, 1, 1))
        self.attention_layer3D = torch.from_numpy(self.attention_layer3D).cuda()

        sz[1:3] = [x // 2 for x in sz[1:3]]
        self.attention_layer4D = get_distance_feature(sz)
        self.attention_layer4D = np.tile(self.attention_layer4D, (batch_sz, 1, 1, 1, 1))
        self.attention_layer4D = torch.from_numpy(self.attention_layer4D).cuda()

        self.non_linearity = non_linearity
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
        x = self.dropout(x)
        z4 = self.layer4_E(x)
        x = self.down(z4)

        x = self.dropout(x)
        x = self.center(x)

        # decoding path
        x = self.up(self.conv4(x))
        x = x + z4
        x = self.dropout(x)
        x = self.layer4_D(torch.cat((x, self.attention_layer4D[0:x.shape[0]]), 1))

        x = self.up(self.conv3(x))
        x = x + z3
        x = self.dropout(x)
        x = self.layer3_D(torch.cat((x, self.attention_layer3D[0:x.shape[0]]), 1))

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.dropout(x)
        x = self.layer2_D(torch.cat((x, self.attention_layer2D[0:x.shape[0]]), 1))

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(torch.cat((x, self.attention_layer1D[0:x.shape[0]]), 1))

        x = self.non_linearity(x)

        if torch.isnan(x).any() or torch.isinf(x).any():
            import pdb; pdb.set_trace()
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

def normalize(x):
    norm = torch.sum(x**2, dim=1)
    return x / (norm.unsqueeze(1) + 1e-10)

def test():
    model = unetv3()
    print('model type: ', model.__class__.__name__)
    num_params = sum([p.data.nelement() for p in model.parameters()])
    print('number of trainable parameters: ', num_params)
    x = torch.randn(8, 1, 4, 128, 128)
    y = model(x)
    print(x.size(), y.size())

if __name__ == '__main__':
    test()