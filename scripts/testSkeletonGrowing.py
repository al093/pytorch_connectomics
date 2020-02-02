import os, sys, traceback
import h5py, time, itertools, datetime
import numpy as np
from argparse import Namespace
from tqdm import tqdm

import torch

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import *

def test(args, test_loader, model, flux_model, device, logger, model_io_size):

    predicted_path_count = 0
    dataset_length = len(test_loader.dataset)
    with h5py.File(args.output + 'predicted_paths.h5', 'w') as predicted_h5, torch.no_grad():
        for iteration, data in tqdm(enumerate(test_loader)):

            sys.stdout.flush()
            image, flux, skeleton, start_pos, start_sid = data
            # initialize samplers
            batch_size = len(image)
            samplers = []
            for i in range(batch_size):
                samplers.append(SkeletonGrowingRNNSampler(image=image[i], skeleton=skeleton[i], flux=flux[i],
                                          start_pos=start_pos[i], start_sid=start_sid[i], continue_growing_th=0.60,
                                          sample_input_size=model_io_size, stride=2.0, anisotropy=[30.0, 6.0, 6.0], mode='test'))
                samplers[-1].init_global_feature_models(flux_model, None, np.array([64, 192, 192], dtype=np.int32), device)

            continue_samplers = list(range(batch_size))
            no_data_for_forward = False

            for t in range(18):
                # if no forward pass could be made in last iteration then break
                if no_data_for_forward:
                    break

                input_image, input_flux = [], []
                start_skeleton_mask, other_skeleton_mask = [], []
                global_features, temp_c_samplers = [], []
                prev_hidden_state, prev_cell_state = [], []

                for i, s_idx in enumerate(continue_samplers):
                    next_step_data = samplers[s_idx].get_next_step()
                    if next_step_data[0] == True and next_step_data[5] == torch.tensor(path_state['CONTINUE'], dtype=torch.float32):
                        # Only compute if the current state is continue
                        # if its not continue or no data could be sampled means we have reached the end
                        temp_c_samplers.append(s_idx)

                        input_image.append(next_step_data[1])
                        input_flux.append(next_step_data[2])
                        start_skeleton_mask.append(next_step_data[3])
                        other_skeleton_mask.append(next_step_data[4])
                        global_features.append(next_step_data[7].cpu())

                        if t == 0:
                            prev_hidden_state, prev_cell_state = None, None
                        else:
                            prev_hidden_state.append(output_hidden_state[i])
                            prev_cell_state.append(output_cell_state[i])

                continue_samplers = temp_c_samplers

                # stack data and train
                if len(input_image) > 0:
                    input_image, input_flux = torch.stack(input_image, 0), torch.stack(input_flux, 0)
                    start_skeleton_mask = torch.stack(start_skeleton_mask, 0)
                    other_skeleton_mask = torch.stack(other_skeleton_mask, 0)
                    global_features = torch.stack(global_features, 0)

                    if prev_hidden_state is not None: prev_hidden_state = torch.stack(prev_hidden_state, 0).to(device)
                    if prev_cell_state is not None: prev_cell_state = torch.stack(prev_cell_state, 0).to(device)

                    # concatenate image + flux + masks and compute forward pass
                    input = torch.cat((input_image, input_flux, start_skeleton_mask,
                                       other_skeleton_mask, global_features), 1).to(device)
                    with torch.no_grad():
                        output_direction, output_path_state, output_hidden_state, output_cell_state = model(input, prev_hidden_state, prev_cell_state)
                else:
                    break

                # Using the predicted directions calculate next positions, samplers will update their state
                for i, sampler_idx in enumerate(continue_samplers):
                    samplers[sampler_idx].jump_to_next_position(output_direction[i], output_path_state[i])

            # save the predcited path
            for sampler in samplers:
                hg = predicted_h5.create_group(str(predicted_path_count))
                predicted_path_count += 1
                path, state, end_ids = sampler.get_predicted_path()
                hg.create_dataset('vertices', data=path)
                hg.create_dataset('states', data=state)
                hg.create_dataset('sids', data=end_ids)
                edges = np.zeros(2 * (path.shape[0] - 1), dtype=np.uint16)
                edges[1::2] = np.arange(1, path.shape[0])
                edges[2:-1:2] = np.arange(1, path.shape[0] - 1)
                hg.create_dataset('edges', data=edges)

def main():
    args = get_args(mode='test')
    save_cmd_line(args)  # Saving the command line args with machine name and time for later reference
    args.output = args.output + args.exp_name + '/'

    print('Initial setup')
    torch.backends.cudnn.enabled = False
    model_io_size, device = init(args)

    if args.disable_logging is not True:
        logger, _ = get_logger(args)
    else:
        logger, writer = None, None
        print('No log file would be created.')

    print('Setup model')
    model = setup_model(args, device, model_io_size)
    model.eval()

    flux_model_args = Namespace(architecture='fluxNet', task=4, out_channel=3, in_channel=1,
                                batch_size=1, load_model=True, pre_model=args.pre_model_second, num_gpu=args.num_gpu)
    flux_model = setup_model(flux_model_args, device, np.array([64, 192, 192], dtype=np.int32), non_linearity=(torch.tanh,))
    flux_model.eval()

    print('Setup data')
    test_loader, _, _ = get_input(args, model_io_size, 'test', model=None)

    print('Start testing')
    test(args, test_loader, model, flux_model, device, logger, model_io_size)

    print('Finished testing')
    if args.disable_logging is not True:
        logger.close()

if __name__ == "__main__":
    main()