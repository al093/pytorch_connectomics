import numpy as np
from .augmentor import DataAugment

from torch_connectomics.utils.vis import *

class Flip(DataAugment):
    """
    Randomly flip along z-, y- and x-axes as well as swap y- and x-axes.

    Args:
        p (float): probability of applying the augmentation.
    """
    def __init__(self, p=0.5):
        super(Flip, self).__init__(p)

    def set_params(self):
        # No change in sample size
        pass

    def flip_and_swap(self, data, rule):
        assert data.ndim==3 or data.ndim==4
        if data.ndim == 3:
            # z reflection.
            if rule[0]:
                data = data[::-1, :, :]
            # y reflection.
            if rule[1]:
                data = data[:, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 2, 1)
        else:
            # z reflection.
            if rule[0]:
                data = data[:, ::-1, :, :]
            # y reflection.
            if rule[1]:
                data = data[:, :, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 1, 3, 2)
        return data

    def flip_and_swap_vectors(self, data, rule):
        assert data.ndim == 4
        # z reflection.
        if rule[0]:
            data[0] = -data[0]
        # y reflection.
        if rule[1]:
            data[1] = -data[1]
        # x reflection.
        if rule[2]:
            data[2] = -data[2]
        # Transpose in xy.
        if rule[3]:
            data = data[(0, 2, 1), :]
        return data

    def __call__(self, data, random_state):
        if random_state is None:
            random_state = np.random.RandomState()

        rule = random_state.randint(2, size=4)
        output = {}
        for key, val in data.items():
            output[key] = self.flip_and_swap(val, rule)
            if key == 'flux': # extra step to flip the vectors
                output[key] = self.flip_and_swap_vectors(output[key], rule)
        return output
