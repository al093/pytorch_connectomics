import numpy as np
from .augmentor import DataAugment
from scipy import ndimage

class Blur(DataAugment):
    """
    Apply blur transformation, to augment for defocussed images.
    """

    def __init__(self, min_sigma=2, max_sigma=8, min_slices=1, max_slices=4, p=0.5):
        """Initialize parameters.
        Args:
            sigma is the std of the Gaussian Kernel
        """
        super(Blur, self).__init__(p=p)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_slices = min_slices
        self.max_slices = max_slices

    def set_params(self):
        # No change in sample size
        pass

    def blur(self, img, slices, sigma):
        """
        blur image
        """
        for z in slices:
            img[z] = ndimage.filters.gaussian_filter(img[z], sigma=sigma)

        return img

    def __call__(self, data, random_state):
        if random_state is None:
            random_state = np.random.RandomState()
        img = data['image']
        sigma = random_state.randint(low=self.min_sigma, high=self.max_sigma + 1, dtype=np.uint8)
        n_slices = random_state.randint(low=self.min_slices, high=self.max_slices + 1, dtype=np.uint8)
        slices = random_state.randint(low=0, high=img.shape[0], size=n_slices, dtype=np.uint8)
        uniq_slices = np.unique(slices)
        # print('Slices: ', slices)
        # print('Sigma: ', sigma)
        data['image'] = self.blur(img, uniq_slices, sigma)
        return data

