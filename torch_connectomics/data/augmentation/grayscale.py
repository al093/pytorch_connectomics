import numpy as np
from .augmentor import DataAugment

class Grayscale(DataAugment):
    """
    Grayscale value augmentation.

    Randomly adjust contrast/brightness, randomly invert
    and apply random gamma correction.
    """

    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, mode='mix', p=0.5):
        """Initialize parameters.

        Args:
            contrast_factor (float): intensity of contrast change.
            brightness_factor (float): intensity of brightness change.
            mode (string): '2D', '3D' or 'mix'.
            p (float): probability of applying the augmentation.
        """
        super(Grayscale, self).__init__(p=p)
        self.set_mode(mode)
        self.CONTRAST_FACTOR   = contrast_factor
        self.BRIGHTNESS_FACTOR = brightness_factor

    def set_params(self):
        # No change in sample size
        pass

    def __call__(self, data, random_state):
        if random_state is None:
            random_state = np.random.RandomState()

        if self.mode == 'mix':
            mode = '3D' if random_state.rand() > 0.5 else '2D'
        else:
            mode = self.mode

        # apply augmentations  
        if mode is '2D': data = self.augment2D(data, random_state)
        if mode is '3D': data = self.augment3D(data, random_state)
        return data

    def augment2D(self, data, random_state):
        t_imgs = data['image'].copy()
        for z in range(t_imgs.shape[-3]):
            if random_state.rand() > 0.5:
                t_imgs[z, :, :] = self._contrast_brightness_gamma_scaling(t_imgs[z, :, :], random_state)
        data['image'] = t_imgs
        return data

    def augment3D(self, data, random_state=None):
        data['image'] = self._contrast_brightness_gamma_scaling(data['image'], random_state)
        return data

    def _contrast_brightness_gamma_scaling(self, input, random_state):
        t_input = np.copy(input)
        t_input *= 1 + (random_state.rand() - 0.5) * self.CONTRAST_FACTOR
        t_input += (random_state.rand() - 0.5) * self.BRIGHTNESS_FACTOR

        # gamma adjustment
        abs_t_input = np.abs(t_input)
        abs_t_input **= 2.0 ** (random_state.rand() * 2 - 1)
        t_input = abs_t_input * np.sign(t_input)

        # clipping
        t_input = np.clip(t_input, -1, 1)
        return t_input

    ####################################################################
    ## Setters.
    ####################################################################

    def set_mode(self, mode):
        """Set 2D/3D/mix greyscale value augmentation mode."""
        assert mode=='2D' or mode=='3D' or mode=='mix'
        self.mode = mode
