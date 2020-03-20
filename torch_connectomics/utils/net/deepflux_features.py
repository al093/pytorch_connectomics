import numpy as np
from enum import Enum
import torch
from torch_connectomics.data.dataset.misc import crop_volume, crop_volume_mul, check_cropable
import collections

class DeepFluxFeatures:
    def __init__(self, max_size):
        self.max_size = max_size
        self.feature_vols = collections.deque([])

    def add(self, feature_vol, flux_vol, bounds):
        if len(self.feature_vols) >= self.max_size:
            self.feature_vols.popleft()
        self.feature_vols.append((bounds, feature_vol.detach().cpu().numpy(), flux_vol.detach().cpu().numpy()))

    def get(self, bounds):
        for vol in self.feature_vols:
            extent = vol[0]
            if np.all(bounds[0] >= extent[0]) and np.all(bounds[1] <= extent[1]):
                return vol[0][0], vol[1], vol[2]
        return None, None, None