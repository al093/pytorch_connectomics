import numpy as np
from enum import Enum
import torch
from torch_connectomics.data.dataset.misc import crop_volume, crop_volume_mul, check_cropable

# class State(Enum):
#     STOP = 0.0
#     CONTINUE = 1.0

path_state = {'STOP':0.0, 'CONTINUE':1.0}


def get_ar_in_window(ar, center, sz):
    sz = np.array(sz).astype(np.int64)
    center = np.array(center).astype(np.int64)
    ar_sz = np.array(ar.shape).astype(np.int64)
    h_sz = sz // 2
    shift = np.abs(((center - h_sz) < 0) * (center - h_sz))
    center += shift
    # if the size is divisible by 2 then reduce the half size by 1
    h_sz_2 = h_sz.copy()
    h_sz_2[(h_sz_2 % 2) == False] -= 1
    shift = (((center + h_sz_2) > (ar_sz - 1)) * (center + h_sz_2 - ar_sz + 1))
    center -= shift
    sub_ar = ar[center[0] - h_sz[0]: center[0] - h_sz[0] + sz[0],
             center[1] - h_sz[1]: center[1] - h_sz[1] + sz[1],
             center[2] - h_sz[2]: center[2] - h_sz[2] + sz[2]]
    return sub_ar, center - h_sz

class SkeletonGrowingRNNSampler:

    def __init__(self, image, skeleton, flux, start_pos, start_sid,
                 sample_input_size, stride, anisotropy, mode='train',
                 continue_growing_th=0.5,
                 stop_pos=None, stop_sid=None, path=None, ft_params=None,
                 path_state_loss_weight=None, d_avg=None, train_flux_model=None):

        self.image = image
        self.skeleton = skeleton
        self.flux = flux
        self.mode = mode
        self.start_pos = start_pos
        self.start_sid = start_sid
        self.stride = stride
        self.anisotropy = np.array(anisotropy, dtype=np.float32)
        self._sampling_idxs = np.linspace(0.0, 1.0, num=10, endpoint=True).astype(np.float32)
        self.continue_growing_th = continue_growing_th
        self.global_features = None

        self.image_size = np.array(self.image.shape)
        self.sample_input_size = np.array(sample_input_size)  # model input size
        self.half_input_sz = self.sample_input_size // 2

        # initialize predictied path with the start position
        self.current_pos = np.array(self.start_pos, copy=True, dtype=np.float32)
        self.predicted_path = [self.current_pos.copy()]
        self.predicted_state = []
        self.ft_params = ft_params
        self.stop_sid = -1

        if self.mode == 'train':
            self.path = path
            self.stop_pos = stop_pos
            self.stop_sid = stop_sid
            self.path_state_loss_weight = torch.from_numpy(np.array([path_state_loss_weight], dtype=np.float32))
            self.sample_input_size = sample_input_size
            self.d_avg = d_avg

            # flip all data except flux
            self.flip_transpose_volumes()

            # Initialize current position and predicted path again, because flip augmentation
            self.current_pos = np.array(self.start_pos, copy=True, dtype=np.float32)
            self.predicted_path = [self.current_pos.copy()]
            self.train_flux_model = train_flux_model

    def init_global_feature_models(self, flux_model, flux_model_branch, flux_model_input_size, device):
        self.flux_model = flux_model
        self.flux_model_branch = flux_model_branch
        self.flux_model_input_size = np.array(flux_model_input_size)
        self.flux_model_half_input_size = self.flux_model_input_size // 2
        self.device = device

    def get_global_features(self, input_image):
        if self.mode == 'train' and self.train_flux_model == True:
            _, p_layer = self.flux_model(torch.from_numpy(input_image[np.newaxis, np.newaxis, ...].copy()).to(self.device),
                                         get_penultimate_layer=True)
        else:
            with torch.no_grad():
                _, p_layer = self.flux_model(torch.from_numpy(input_image[np.newaxis, np.newaxis, ...].copy()).to(self.device),
                                             get_penultimate_layer=True)
        # out_features = self.flux_branch_model(p_layer)[0]
        out_features = p_layer[0]
        return out_features

    def get_roi_from_global_features(self, center_pos_growing, corner_pos_growing):
        '''
        A logic for running the global features models is implemented
        depending on the current position, the global features may need to be calculated again
        a state needs to be maintained, defining where the current global features were calculated from
        and can a roi be extracted
        '''

        compute_new_global_features = False
        if self.global_features is None:
            compute_new_global_features = True
        elif np.any(corner_pos_growing - self.corner_pos_global < 10 ) or \
                np.any((corner_pos_growing + self.sample_input_size + 10) > (self.corner_pos_global + self.flux_model_input_size)):
            # the roi exceeds the bounds of the global features pre-calculated
            compute_new_global_features = True

        if compute_new_global_features:
            # we may have to shift the corner position because the flux_model_input_size is larger than the growing model input size
            input_image, corner_pos_global = get_ar_in_window(self.image, center_pos_growing, self.flux_model_input_size)
            self.corner_pos_global = corner_pos_global

            #run model and get global features
            self.global_features = self.get_global_features(input_image)

        # find the relative corner_pos_growing wrt global features volumes and crop an roi
        corner_pos_growing_relative = corner_pos_growing - self.corner_pos_global
        roi_global_features = crop_volume_mul(self.global_features, self.sample_input_size, corner_pos_growing_relative)

        return roi_global_features

    def get_next_step(self):
        if self.mode == 'train': return self.get_next_step_train()
        elif self.mode == 'test': return self.get_next_step_test()

    def get_next_step_test(self):
        '''
        Sample a region around the start_pos, check if that is possible first
        Return all the data needed for the RNN:
        Cropped Tensor Image, cropped Flux, Cropped Skeleton masks, GT direction, stopping state ...
        '''

        input_size = self.sample_input_size

        center_pos = self.current_pos.astype(np.int32) #rounding from float to integer
        corner_pos = center_pos - self.half_input_sz
        if check_cropable(self.image, input_size, corner_pos) == False:
            return False, None, None, None, None, None, None, None

        input_image = crop_volume(self.image, input_size, corner_pos)
        cropped_skeleton = crop_volume(self.skeleton, input_size, corner_pos)
        start_skeleton_mask = (cropped_skeleton == self.start_sid).astype(np.float32)
        other_skeleton_mask = ((cropped_skeleton != self.start_sid) & (cropped_skeleton != 0)).astype(np.float32)
        input_flux = crop_volume_mul(self.flux, input_size, corner_pos)

        input_image = torch.from_numpy(input_image.copy()).unsqueeze(0)
        input_flux = torch.from_numpy(input_flux.copy())
        start_skeleton_mask = torch.from_numpy(start_skeleton_mask).unsqueeze(0)
        other_skeleton_mask = torch.from_numpy(other_skeleton_mask).unsqueeze(0)

        # stop state occurs if :
        # any of the points between the current position and the previous position is on any skeleton fragment
        # or the last predicted state was a stop
        if len(self.predicted_path) >= 2:
            # check if the last state was predicted to be a stop
            if self.predicted_state[-1] < self.continue_growing_th:
                state = torch.tensor(path_state['STOP'], dtype=torch.float32)
            else:
                path_section = self.interpolate_linear(self.predicted_path[-2], self.current_pos)
                skeletons_hit = self.skeleton[path_section]
                idx_mask = (skeletons_hit > 0) & (skeletons_hit != self.start_sid)
                first_hit_idx = np.argmax(idx_mask)
                if idx_mask[first_hit_idx] == True:
                    self.stop_sid = self.skeleton[path_section[0][first_hit_idx], path_section[1][first_hit_idx], path_section[2][first_hit_idx]]
                    state = torch.tensor(path_state['STOP'], dtype=torch.float32)
                    predicted_end = np.array([path_section[0][first_hit_idx], path_section[1][first_hit_idx], path_section[2][first_hit_idx]], dtype=np.float32)
                    self.current_pos = predicted_end
                    self.predicted_path[-1] = self.current_pos.copy()
                    # update the center and corner positions again
                    center_pos = self.current_pos.astype(np.int32)
                    corner_pos = center_pos - self.half_input_sz
                else:
                    state = torch.tensor(path_state['CONTINUE'], dtype=torch.float32)
        else:
            state = torch.tensor(path_state['CONTINUE'], dtype=torch.float32)

        global_features = self.get_roi_from_global_features(center_pos, corner_pos)
        return True, input_image, input_flux, start_skeleton_mask, other_skeleton_mask, state, center_pos, global_features

    def get_next_step_train(self):
        '''
        Sample a region around the start_pos, check if that is possible first
        Return all the data needed for the RNN:
        Cropped Tensor Image, cropped Flux, Cropped Skeleton masks, GT direction, stopping state ...
        '''

        input_size = self.sample_input_size

        center_pos = self.current_pos.astype(np.int32) #rounding from float to integer
        corner_pos = center_pos - self.half_input_sz
        if check_cropable(self.image, input_size, corner_pos) == False:
            return False, None, None, None, None, None, None, None, None

        input_image = crop_volume(self.image, input_size, corner_pos)
        cropped_skeleton = crop_volume(self.skeleton, input_size, corner_pos)
        start_skeleton_mask = (cropped_skeleton == self.start_sid).astype(np.float32)
        other_skeleton_mask = ((cropped_skeleton != self.start_sid) & (cropped_skeleton != 0)).astype(np.float32)

        input_flux = crop_volume_mul(self.flux, input_size, corner_pos)
        # flux vectors need to be flipped or transposed as per the ft_params.
        # ensure a copy is passed.
        input_flux = self.flip_transpose_flux_vectors(input_flux.copy())

        input_image = torch.from_numpy(input_image.copy()).unsqueeze(0)
        input_flux = torch.from_numpy(input_flux.copy())
        start_skeleton_mask = torch.from_numpy(start_skeleton_mask).unsqueeze(0)
        other_skeleton_mask = torch.from_numpy(other_skeleton_mask).unsqueeze(0)

        # calculate direction
        direction = self.calculate_direction()
        direction = torch.from_numpy(direction)

        # stop state occurs
        # 1) if any of the points between the current position and the previous position is on the correct skeleton fragment
        # 2) if the next direction gt is zero, which occurs when the current position is very near to the end of the GT path
        if (torch.abs(direction).sum() == 0):
            state = torch.tensor(path_state['STOP'], dtype=torch.float32)
            path_state_loss_weight = len(self.predicted_path)*torch.ones_like(self.path_state_loss_weight)
        elif len(self.predicted_path) >= 2 and \
                np.any(self.skeleton[self.interpolate_linear(self.current_pos, self.predicted_path[-2])] == self.stop_sid):
            state = torch.tensor(path_state['STOP'], dtype=torch.float32)
            path_state_loss_weight = len(self.predicted_path)*torch.ones_like(self.path_state_loss_weight)
        else:
            state = torch.tensor(path_state['CONTINUE'], dtype=torch.float32)
            path_state_loss_weight = torch.ones_like(self.path_state_loss_weight)

        global_features = self.get_roi_from_global_features(center_pos, corner_pos)
        return True, input_image, input_flux, start_skeleton_mask, other_skeleton_mask, \
               direction, state, center_pos, path_state_loss_weight, global_features

    def calculate_next_position(self, p_direction):
        '''
        internal function for calculating the next position
        '''
        #normalize direction and take stride in that direction, if it falls outside return the boundary
        next_pos = self.current_pos + (p_direction.detach().cpu().numpy() * self.stride)
        next_pos[next_pos < 0] = 0
        next_pos[next_pos >= (self.image_size - 1)] = self.image_size[next_pos >= (self.image_size - 1)] - 1
        return next_pos.astype(np.float32)

    def jump_to_next_position(self, p_direction, p_state):
        '''
        based on the predicted direction get the next position
        '''
        self.predicted_state.append(p_state.detach().cpu().item())

        if self.mode == 'train':
            self.current_pos = self.calculate_next_position(p_direction)
            self.predicted_path.append(self.current_pos.copy())
        elif self.mode == 'test':
            if p_state >= self.continue_growing_th:
                self.current_pos = self.calculate_next_position(p_direction)
                self.predicted_path.append(self.current_pos.copy())

        return self.current_pos

    def get_predicted_path(self):
        path = np.vstack(self.predicted_path)
        path = self.transpose_flip_path(path)
        state = np.array(self.predicted_state, dtype=np.float32)
        return path, state, np.array([self.start_sid, self.stop_sid], dtype=np.int32)

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

        direction = 0.75*direction + 0.25*lateral_dir
        direction = self.normalize(direction)

        return direction

    def normalize(self, vec):
        norm = np.sqrt((vec**2).sum())
        if norm < 1e-3:
            return np.array([0, 0, 0], dtype=np.float32)
        else:
            return np.array(vec/norm, copy=False, dtype=np.float32)

    def flip_transpose_volumes(self):
        # ensure that all volumes are views of the data
        # path, start_pos, stop_pos can be copied
        if self.ft_params is not None:
            self.path = self.path.copy()
            self.stop_pos = self.stop_pos.copy()
            self.start_pos = self.start_pos.copy()
            if self.ft_params['xflip']:
                self.image = self.image[:, :, ::-1]
                self.skeleton = self.skeleton[:, :, ::-1]
                self.flux = self.flux[:, :, :, ::-1]
                self.path[:, 2] = self.image.shape[2] - self.path[:, 2] - 1
                self.stop_pos[2] = self.image.shape[2] - self.stop_pos[2] - 1
                self.start_pos[2] = self.image.shape[2] - self.start_pos[2] - 1
            if self.ft_params['yflip']:
                self.image = self.image[:, ::-1, :]
                self.skeleton = self.skeleton[:, ::-1, :]
                self.flux = self.flux[:, :, ::-1, :]
                self.path[:, 1] = self.image.shape[1] - self.path[:, 1] - 1
                self.stop_pos[1] = self.image.shape[1] - self.stop_pos[1] - 1
                self.start_pos[1] = self.image.shape[1] - self.start_pos[1] - 1
            if self.ft_params['zflip']:
                self.image = self.image[::-1, :, :]
                self.skeleton = self.skeleton[::-1, :, :]
                self.flux = self.flux[:, ::-1, :, :]
                self.path[:, 0] = self.image.shape[0] - self.path[:, 0] - 1
                self.stop_pos[0] = self.image.shape[0] - self.stop_pos[0] - 1
                self.start_pos[0] = self.image.shape[0] - self.start_pos[0] - 1
            if self.ft_params['xytranspose']:
                self.image = self.image.transpose(0, 2, 1)
                self.skeleton = self.skeleton.transpose(0, 2, 1)
                self.flux = self.flux.transpose(0, 1, 3, 2)
                self.path = self.path[:, [0, 2, 1]]
                self.stop_pos = self.stop_pos[[0, 2, 1]]
                self.start_pos = self.start_pos[[0, 2, 1]]

    def transpose_flip_path(self, path):
        if self.ft_params is not None:
            path = path.copy()
            if self.ft_params['xytranspose']:
                path = path[:, [0, 2, 1]]
            if self.ft_params['zflip']:
                path[:, 0] = self.image.shape[0] - path[:, 0] - 1
            if self.ft_params['yflip']:
                path[:, 1] = self.image.shape[1] - path[:, 1] - 1
            if self.ft_params['xflip']:
                path[:, 2] = self.image.shape[2] - path[:, 2] - 1
        return path

    def flip_transpose_flux_vectors(self, data):
        assert data.ndim == 4
        if self.ft_params is not None:
            if self.ft_params['xflip']:
                data[2] = -data[2]
            if self.ft_params['yflip']:
                data[1] = -data[1]
            if self.ft_params['zflip']:
                data[0] = -data[0]
            # Transpose in xy.
            if self.ft_params['xytranspose']:
                data = data[[0, 2, 1], :]
        return data

    def interpolate_linear(self, start_pos, stop_pos):
        sampled_points = start_pos + self._sampling_idxs[:, np.newaxis] * (stop_pos - start_pos)
        sampled_points = sampled_points.astype(np.int32)
        return (tuple(sampled_points[:, 0]), tuple(sampled_points[:, 1]), tuple(sampled_points[:, 2]))