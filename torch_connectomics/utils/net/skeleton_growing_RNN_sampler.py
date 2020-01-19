import numpy as np
from enum import Enum
import torch
from torch_connectomics.data.dataset.misc import crop_volume, crop_volume_mul, check_cropable

# class State(Enum):
#     STOP = 0.0
#     CONTINUE = 1.0

path_state = {'STOP':0.0, 'CONTINUE':1.0}

class SkeletonGrowingRNNSampler:

    def __init__(self, image, skeleton, flux, path,
                 start_pos, stop_pos, start_sid, stop_sid,
                 sample_input_size, stride, anisotropy, d_avg):
        self.image = image
        self.skeleton = skeleton
        self.flux = flux
        self.path = path
        self.start_pos = start_pos
        self.stop_pos = stop_pos
        self.start_sid = start_sid
        self.stop_sid = stop_sid
        self.sample_input_size = sample_input_size
        self.stride = stride
        self.anisotropy = np.array(anisotropy, dtype=np.float32)
        self.d_avg = d_avg
        # samples, channels, depths, rows, cols
        self.image_size = np.array(self.image.shape)
        self.sample_input_size = np.array(sample_input_size)  # model input size
        self.half_input_sz = self.sample_input_size//2
        self.current_pos = np.array(self.start_pos, copy=True, dtype=np.float32)
        self.predicted_path = [self.current_pos.copy()]

    def get_next_step(self):
        '''
        Sample a region around the start_pos, check if that is possible first
        Return all the data needed for the RNN:
        Cropped Tensor Image, cropped Flux, Cropped Skeleton masks, GT direction, stopping state
        '''

        input_size = self.sample_input_size

        center_pos = self.current_pos.astype(np.int32) #rounding from float to integer
        corner_pos = center_pos - self.half_input_sz
        if check_cropable(self.image, input_size, corner_pos) == False:
            return False, None, None, None, None, None, None, None

        input_image = crop_volume(self.image, input_size, corner_pos)
        input_flux = crop_volume_mul(self.flux, input_size, corner_pos)
        cropped_skeleton = crop_volume(self.skeleton, input_size, corner_pos)
        start_skeleton_mask = (cropped_skeleton == self.start_sid).astype(np.float32)
        other_skeleton_mask = ((cropped_skeleton != self.start_sid) & (cropped_skeleton != 0)).astype(np.float32)

        input_image = torch.from_numpy(input_image).unsqueeze(0)
        input_flux = torch.from_numpy(input_flux)
        start_skeleton_mask = torch.from_numpy(start_skeleton_mask).unsqueeze(0)
        other_skeleton_mask = torch.from_numpy(other_skeleton_mask).unsqueeze(0)

        # calculate direction
        direction = self.calculate_direction()
        direction = torch.from_numpy(direction)

        # stop state occurs
        # if the current position is on the correct skeleton fragment
        # if the next direction gt is zero, which occurs when the current position is very near to the end of the GT path
        if (torch.abs(direction).sum() == 0) or self.skeleton[tuple(center_pos)] == self.stop_sid:
            state = torch.tensor(path_state['STOP'], dtype=torch.float32)
        else:
            state = torch.tensor(path_state['CONTINUE'], dtype=torch.float32)

        return True, input_image, input_flux, start_skeleton_mask, other_skeleton_mask, direction, state, center_pos

    def calculate_next_position(self, p_direction):
        '''
        internal function for calulating the next position
        '''
        #normalize direction and take stride in that direction, if it falls outside return the boundary
        next_pos = self.current_pos + (p_direction.detach().cpu().numpy() * self.stride)
        next_pos[next_pos < 0] = 0
        next_pos[next_pos >= (self.image_size - 1)] = self.image_size[next_pos >= (self.image_size - 1)] - 1
        return next_pos.astype(np.float32)

    def jump_to_next_position(self, p_direction):
        '''
        based on the predicted direction get the next position
        '''
        #normalize direction and take stride in that direction, if it falls outside return the boundary
        self.current_pos = self.calculate_next_position(p_direction)
        self.predicted_path.append(self.current_pos.copy())
        return self.current_pos

    def get_predicted_path(self):
        return np.vstack(self.predicted_path)

    def get_cropped_image(self, pos):
        out_input = crop_volume(self.input[pos[0]], self.sample_input_size, pos[1:])
        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)
        return out_input

    def calculate_direction(self):
        '''
        Based on the current_pos get the next direction which can allow growing of the skeleton
        '''
        #find the closest point to the skeleton and the distance

        distances = np.sqrt(((self.anisotropy*(self.current_pos - self.path))**2).sum(axis=1))
        nearest_skel_node_idx = np.argmin(distances)
        closest_distance = distances[nearest_skel_node_idx]

        #calculate lateral adjustment direction
        if closest_distance <= (2*self.anisotropy[0] - 0.1): #z is the coarsest so keeping one Z slice as the limit
            #the current point is close to the skeleton, so we can safely move using directions from the skeleton
            lateral_dir = np.zeros(3, dtype=np.float32)
        else: # if they are not close force the points to be closer to the skeleton path
            lateral_dir = (self.path[nearest_skel_node_idx] - self.current_pos)
            lateral_dir = self.normalize(lateral_dir)

        #calculate growing direction
        if (self.path.shape[0] - nearest_skel_node_idx ) < self.d_avg + 1:
            # there are less than d_avg steps to the closest skeleton, so point the direction directly to the stop_pos
            direction = (self.stop_pos - self.current_pos)
            direction = self.normalize(direction)
        else:
            # there are more than or equal to d_avg points after the nearest_skel_node
            # averaging the directions and normalize
            directions = self.path[nearest_skel_node_idx+1:nearest_skel_node_idx+self.d_avg+1] - self.path[nearest_skel_node_idx]
            norms = np.sqrt((directions**2).sum(axis=1))
            zero_mask = (norms < 1e-5)
            directions = directions[~zero_mask]/norms[~zero_mask, np.newaxis]
            direction = directions.sum(axis=0)
            direction = self.normalize(direction)

        direction += lateral_dir
        direction = self.normalize(direction)

        return direction

    def normalize(self, vec):
        norm = np.sqrt((vec**2).sum())
        if norm < 1e-3:
            return np.array([0,0,0], dtype=np.float32)
        else:
            return np.array(vec/norm, copy=False, dtype=np.float32)