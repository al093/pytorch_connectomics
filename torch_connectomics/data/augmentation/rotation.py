import cv2
import numpy as np
from .augmentor import DataAugment
import math

class Rotate(DataAugment):
    """
    Continuous rotatation.

    The sample size for x- and y-axes should be at least sqrt(2) times larger
    than the input size to make sure there is no non-valid region after center-crop.
    
    Args:
        p (float): probability of applying the augmentation
    """
    def __init__(self, p=0.5):
        super(Rotate, self).__init__(p=p) 
        self.image_interpolation = cv2.INTER_LINEAR
        self.label_interpolation = cv2.INTER_NEAREST
        self.border_mode = cv2.BORDER_CONSTANT
        self.set_params()

    def set_params(self):
        self.sample_params['ratio'] = [1.0, 1.42, 1.42]

    def rotate(self, imgs, M, interpolation):
        height, width = imgs.shape[-2:]
        if imgs.ndim == 4:
            channels = imgs.shape[-4]
            slices = imgs.shape[-3]
        if imgs.ndim == 3:
            channels = 1
            slices = imgs.shape[-3]

        transformedimgs = np.copy(imgs)

        for z in range(slices):
            if channels == 1:
                img = transformedimgs[z, :, :]
                dst = cv2.warpAffine(img, M, (height, width), 1.0, flags=interpolation, borderMode=self.border_mode)
                transformedimgs[z, :, :] = dst
            elif channels == 3:
                img = transformedimgs[:, z, :, :]
                img = np.moveaxis(img, 0, -1)
                dst = cv2.warpAffine(img, M, (height, width), 1.0, flags=interpolation, borderMode=self.border_mode)
                transformedimgs[:, z, :, :] = np.moveaxis(dst, -1, 0)
            else:
                raise Exception('Unknown number of channels in 2d slice')

        return transformedimgs

    def rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta degrees.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        theta = float(theta) * np.pi / 180.0
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def __call__(self, data, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()

        image = data['image']
        height, width = image.shape[-2:]
        angle = random_state.rand()*360.0
        M = cv2.getRotationMatrix2D((height/2, width/2), angle, 1)

        output = {}
        for key, val in data.items():
            if key == 'label' or key == 'skeleton' or key == 'weight' or key == 'context':
                output[key] = self.rotate(val, M, self.label_interpolation)
            elif key == 'flux':
                r_img = self.rotate(val, M, self.image_interpolation)
                r_mat = self.rotation_matrix((1, 0, 0), angle)
                r_field = np.matmul(r_mat, r_img.reshape((3, -1)))
                output[key] = r_field.reshape(val.shape)
            else:
                output[key] = self.rotate(val, M, self.image_interpolation)


        return output