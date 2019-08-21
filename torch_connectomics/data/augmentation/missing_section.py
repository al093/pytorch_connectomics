import math
import numpy as np
from .augmentor import DataAugment

class MissingSection(DataAugment):
    """Missing-section augmentation of image stacks
    
    Args:
        num_sections (int): number of missing sections.
        p (float): probability of applying the augmentation.
    """
    def __init__(self, num_sections=2, p=0.5):
        super(MissingSection, self).__init__(p=p)
        # ensure the equal number of sections are removed from the top and bottom wrt center
        assert num_sections % 2 == 0
        self.num_sections = num_sections
        self.set_params()

    def set_params(self):
        self.sample_params['add'] = [int(math.ceil(self.num_sections / 2.0)), 0, 0]

    def missing_section(self, data, random_state):
        images, labels = data['image'], data['label']

        if 'input_label' in data and data['input_label'] is not None:
            mask = data['input_label']
        else:
            mask = None

        new_images = images.copy()   
        new_labels = labels.copy()

        idx1 = random_state.choice(np.array(range(1, (images.shape[0]//2)-1)), self.num_sections//2, replace=False)
        idx2 = random_state.choice(np.array(range((images.shape[0]//2)+1, images.shape[0]-1)),self.num_sections//2, replace=False)
        idx = np.concatenate((idx1, idx2))

        new_images = np.delete(new_images, idx, 0)
        new_labels = np.delete(new_labels, idx, 0)

        data = {}
        data['image'] = new_images
        data['label'] = new_labels

        if mask is not None:
            new_mask = mask.copy()
            new_mask = np.delete(new_mask, idx, 0)
            data['input_label'] = new_mask

        return data
    
    def __call__(self, data, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(1234)

        return self.missing_section(data, random_state)