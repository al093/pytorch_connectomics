import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import warp

from .augmentor import DataAugment

class Elastic(DataAugment):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5.

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

    Args:
        alpha (float): maximum pixel-moving distance of elastic transformation.
        sigma (float): standard deviation of the Gaussian filter.
        p (float): probability of applying the augmentation.
    """
    def __init__(self,
                 alpha=10.0,
                 sigma=4.0,
                 p=0.5):
        
        super(Elastic, self).__init__(p)
        self.alpha = alpha
        self.sigma = sigma
        self.image_interpolation = 1
        self.label_interpolation = 0
        self.border_mode = cv2.BORDER_CONSTANT
        self.set_params()

    def set_input_sz(self, image_sz):
        self.z, self.y, self.x = np.mgrid[:image_sz[0], :image_sz[1], :image_sz[2]]
        self.mapz = self.z.astype(np.float32)

    def set_params(self):
        max_margin = int(self.alpha) + 1
        self.sample_params['add'] = [0, max_margin, max_margin]

    def __call__(self, data, random_state=None):

        image = data['image']
        depth, height, width = image.shape[-3:]  # (z, y, x)

        if random_state is None:
            random_state = np.random.RandomState(1234)
        dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha)
        dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha)
        mapy, mapx = np.float32(self.y + dy), np.float32(self.x + dx)

        output = {}
        for key, image in data.items():
            if key == 'flux' or key == 'skeleton' or key == 'context':
                output[key] = image
            elif key == 'image':
                output[key] = warp(image, np.array([self.mapz, mapy, mapx]), order=self.image_interpolation)
            elif key == 'label' or key == 'mask' or key == 'weight':
                output[key] = warp(image, np.array([self.mapz, mapy, mapx]), order=self.label_interpolation)
            else:
                raise Exception('Input data key not identified, Key was: ' + key)

        return output