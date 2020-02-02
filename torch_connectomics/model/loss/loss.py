from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# 0. main loss functions

class DiceLoss(nn.Module):
    """DICE loss.
    """

    def __init__(self, size_average=True, reduce=True, smooth=100.0):
        super(DiceLoss, self).__init__(size_average, reduce)
        self.smooth = smooth
        self.reduce = reduce

    def dice_loss(self, input, target):
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + self.smooth) / 
                    ( (iflat**2).sum() + (tflat**2).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(input.size()[0])

    def dice_loss_batch(self, input, target):

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        loss = 1 - ((2. * intersection + self.smooth) / 
               ( (iflat**2).sum() + (tflat**2).sum() + self.smooth))
        return loss

    def forward(self, input, target):
        #_assert_no_grad(target)
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        if self.reduce:
            loss = self.dice_loss(input, target)
        else:    
            loss = self.dice_loss_batch(input, target)
        return loss

class WeightedMSE(nn.Module):
    """Weighted mean-squared error.
    """

    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, input, target, weight, norm_term):
        if norm_term is None:
            s1 = torch.prod(torch.tensor(input.size()[2:]).float())
            s2 = input.size()[0]
            norm_term = (s1 * s2).cuda()

        if weight is not None:
            return torch.sum(weight * (input - target) ** 2) / norm_term
        else:
            return torch.sum((input - target) ** 2) / norm_term

    def forward(self, input, target, weight=None, norm_term=None):
        #_assert_no_grad(target)
        return self.weighted_mse_loss(input, target, weight, norm_term)

class WeightedL1(nn.Module):
    def __init__(self):
        super().__init__()

    def weighted_l1_loss(self, input, target, weight):
        s1 = torch.prod(torch.tensor(input.size()[2:]).float())
        s2 = input.size()[0]
        norm_term = (s1 * s2).cuda()
        if weight is not None:
            return torch.sum(weight*torch.abs(input - target)) / norm_term
        else:
            return torch.sum(torch.abs(input - target)) / norm_term

    def forward(self, input, target, weight=None):
        #_assert_no_grad(target)
        return self.weighted_l1_loss(input, target, weight)

class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """
    def __init__(self, size_average=True, reduce=True, focal_loss=False):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.focal_loss = focal_loss

    def forward(self, input, target, weight=None):
        if self.focal_loss: #TODO CHECK IF CORRECT
            if input.max() > 1.0 or input.min() < 0.0:
                print('Output Values are outside [1,0]. Clipping them')
            fl_weight = torch.clamp(input, min=0.0, max=1.0)
            fl_weight[target == 0] = 1 - fl_weight[target == 0]
            fl_weight = torch.pow((1 - fl_weight), 2)

            if weight is not None:
                weight = fl_weight * weight
            else:
                weight = fl_weight

        return F.binary_cross_entropy(input, target, weight.detach(), reduction='mean')

# class WeightedCosineLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#
#     def forward(self, input, target, weight):
#         # _assert_no_grad(target)
#         s1 = torch.prod(torch.tensor(input.size()[2:]).float())
#         s2 = input.size()[0]
#         norm_term = (s1 * s2).cuda()
#
#         cosine_similarity = self.cos(input, target)
#         cosine_loss = 1 - cosine_similarity
#         cosine_loss = cosine_loss*weight/norm_term
#
#         return cosine_loss

class AngularAndScaleLoss(nn.Module):
    def __init__(self, alpha, dim=1):
        super().__init__()
        self.w_mse = WeightedMSE()
        self.alpha = alpha
        self.cos = nn.CosineSimilarity(dim=dim, eps=1e-6)

    def get_norm(self, input):
        # input b, c, z, y, x
        scale = torch.sqrt((input**2).sum(dim=1, keepdim=True))
        return scale

    # def angular_loss(self, input, target, norm_i, norm_t, weight, batch_norm_fac):
    #     inner_p = (input*target).sum(dim=1, keepdim=True)
    #     den = norm_i*norm_t + 1e-10
    #     inner_p = inner_p/den
    #     if torch.isnan(inner_p).any() or torch.isinf(inner_p).any():
    #         import pdb; pdb.set_trace()
    #
    #     inner_p[inner_p > 1.0] = 1.0
    #     inner_p[inner_p < -1.0] = -1.0
    #     loss = torch.acos(inner_p)
    #
    #     if torch.isnan(loss).any() or torch.isinf(loss).any():
    #         import pdb; pdb.set_trace()
    #
    #     if weight is not None:
    #         loss = weight * loss
    #
    #     return loss.sum()/batch_norm_fac

    def scale_loss(self, norm_i, norm_t, weight, norm_term):
        return self.w_mse(norm_i, norm_t, weight, norm_term)

    def forward(self, input, target, weight=None):
        scale_i = self.get_norm(input)
        scale_t = self.get_norm(target)

        cosine_similarity = self.cos(input, target)
        cosine_loss = 1 - cosine_similarity
        if weight is not None:
            cosine_loss = weight*cosine_loss
            norm_term = (weight>0).sum()
            a_loss = cosine_loss.sum()/norm_term
        else:
            norm_term = torch.prod(torch.tensor(cosine_loss.size(), dtype=torch.float32))
            a_loss = cosine_loss.sum()/norm_term

        # a_loss = self.angular_loss(input, target, scale_i, scale_t, weight, norm_term)
        s_loss = self.scale_loss(scale_i, scale_t, weight, norm_term)

        return self.alpha*a_loss + (1.0-self.alpha)*s_loss, self.alpha*a_loss, (1.0-self.alpha)*s_loss

#. 1. Regularization

class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, input):
        diff = input - 0.5
        diff = torch.clamp(torch.abs(diff), min=1e-2)
        loss = 1.0 / diff.sum()
        return self.alpha * loss