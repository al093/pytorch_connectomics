import numpy as np
from .augmentor import DataAugment

class SwapZ(DataAugment):
    """
    Randomly flip along z-, y- and x-axes as well as swap y- and x-axes.

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
        if data.ndim == 3:
            if rule:
                data = data.transpose(1, 0, 2)
            else:
                data = data.transpose(2, 1, 0)
        else:
            if rule:
                data = data.transpose(0, 2, 1, 3)
            else:
                data = data.transpose(0, 3, 2, 1)
        return data

    def __call__(self, data, random_state):
        i_shape = data['image'].shape
        l_shape = data['label'].shape
        assert i_shape[-1] == i_shape[-2] == i_shape[-3]
        assert l_shape[-1] == l_shape[-2] == l_shape[-3]

        if random_state is None:
            random_state = np.random.RandomState(1234)

        output = {}
        rule = random_state.randint(2)
        augmented_image = self.swap(data['image'], rule)
        augmented_label = self.swap(data['label'], rule)
        output['image'] = augmented_image
        output['label'] = augmented_label

        if 'input_label' in data and data['input_label'] is not None:
            augmented_input_label = self.swap(data['input_label'], rule)
            output['input_label'] = augmented_input_label

        return output