import numpy as np
from .augmentor import DataAugment

class SwapZ(DataAugment):
    """
    Randomly sway along z- with y- or x-axes.

    Args:
        p (float): probability of applying the augmentation.
    """
    def __init__(self, p=0.5):
        super(SwapZ, self).__init__(p)

    def set_params(self):
        # No change in sample size
        pass

    def swap(self, data, rule):
        assert data.ndim==3 or data.ndim==4
        if rule:
            data = data.transpose(1, 0, 2) if data.ndim == 3 else data.transpose(0, 2, 1, 3)
        else:
            data = data.transpose(2, 1, 0) if data.ndim == 3 else data.transpose(0, 3, 2, 1)
        return data

    def swap_vectors(self, data, rule):
        assert data.ndim == 4
        if rule:
            data = data[(1, 0, 2), :]
        else:
            data = data[(2, 1, 0), :]
        return data

    def __call__(self, data, random_state):
        # sanity check for one volume
        i_shape = data['image'].shape
        assert i_shape[-1] == i_shape[-2] == i_shape[-3]

        if random_state is None:
            random_state = np.random.RandomState(1234)

        rule = random_state.randint(2)
        output = {}
        for key, val in data.items():
            output[key] = self.swap(val, rule)
            if key == 'flux':  # extra step to flip the vectors
                output[key] = self.swap_vectors(output[key], rule)

        return output