from collections import deque
import numpy as np

import torch

from torch_connectomics.data.dataset.misc import crop_volume
from torch_connectomics.data.utils.functional_collate import collate_fn_test_2, collate_fn_test

class SerialSampler():

    def __init__(self, dataset, batch_size, pad_size, init_seed_points, in_channel):
        self.dataset = dataset
        self.pos_queue = deque()
        self.pos_processed = deque()
        self.batch_size = batch_size
        self.pad_size = pad_size
        self.sel = np.ones((3, 3, 3), dtype=bool)
        for sp_idx in range(init_seed_points.shape[0]):
            self.pos_queue.append(init_seed_points[sp_idx])
        if in_channel == 1:
            self.need_past_pred = False
        else:
            self.need_past_pred = True

    def set_out_array(self, segmentation):
        self.seg = segmentation
        self.seg_shape = segmentation.shape

    def compute_new_pos(self, mask, edge_pos, pos):
        # mask_eroded = binary_erosion(mask, structure=self.sel).astype(bool)
        # mask[mask_eroded] = 0

        print("Num Prospective Positions: ", edge_pos.shape[0])
        for new_pos_id in range(edge_pos.shape[0]):
            new_pos = edge_pos[new_pos_id]

            is_edge = False
            if np.all(new_pos == self.dataset.sample_input_size - 1):
                is_edge = True

            new_pos += pos  # now point is wrt to the origin of the entire padded input vol

            # print('----------------For position: ', new_pos)
            # check if the new_pos is already inside the fov of some pos in the queue
            already_covered = False
            for old_pos in reversed(self.pos_queue):
                if np.all(np.abs(new_pos - old_pos, dtype=np.int32) < (self.dataset.half_input_sz // 6)):
                    already_covered = True
                    # print('A point exists in the NON Processed Queue')
                    break
            if not already_covered:  # check in the processed queue
                for old_pos in reversed(self.pos_processed):
                    if np.all(np.abs(new_pos - old_pos, dtype=np.int32) < (self.dataset.half_input_sz // 6)):
                        already_covered = True
                        # print('A point exists in the Processed Queue')
                        break

            if not already_covered:
                # check if the new pos is inside the unpadded region
                if np.all(new_pos >= self.pad_size) and np.all(new_pos < (self.seg_shape - self.pad_size)):
                    # print('Is outside padded region.')

                    # check if the new pos has some neighbouring pixels which are not marked as object?
                    # if not then it may be a voxel inside the segmentation
                    if (not np.all(self.seg[new_pos[0]-7:new_pos[0]+8,
                                            new_pos[1]-7:new_pos[1]+8,
                                            new_pos[2]-7:new_pos[2]+8])) \
                                            or is_edge:

                        # Pos is the center around which the input should be sampled
                        self.pos_queue.append(new_pos)
                        # print('Position Added')
                    # else:
                        # print('All points in the surrounding are 1')
                # else:
                    # print('Pos inside the padded region')
        print('..Done')
    def get_input_data(self):
        input_batch = []
        pos_batch = []
        past_pred_batch = []

        if len(self.pos_queue) > self.batch_size:
            num_data_points = self.batch_size
        else:
            num_data_points = len(self.pos_queue)

        for _ in range(num_data_points):
            pos = np.array([0, 0, 0, 0], dtype=np.uint32)
            pos[1:] = self.pos_queue.pop()
            self.pos_processed.append(pos[1:].copy())  # add pos into the processed queue
            pos[1:] = pos[1:] - self.dataset.half_input_sz  # since pos was the center shifting it to origin of sampling volume
            # print('Position: ', pos)
            input_batch.append(self.dataset.get_vol(pos))
            pos_batch.append(pos)
            if self.need_past_pred:
                past_pred_cpu = crop_volume(self.seg, self.dataset.sample_input_size, pos[1:])
                past_pred_batch.append((torch.from_numpy(past_pred_cpu.copy().astype(np.float32))).unsqueeze(0))
                return collate_fn_test_2(zip(pos_batch, input_batch, past_pred_batch))
            else:
                return collate_fn_test(zip(pos_batch, input_batch))

    def remaining_pos(self):
        return len(self.pos_queue)