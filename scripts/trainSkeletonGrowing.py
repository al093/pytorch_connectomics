import os, sys
import h5py, time, itertools, datetime
import numpy as np

import torch

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import *

def train(args, train_loader, val_loader, model, device, criterion, optimizer, scheduler, logger, writer, regularization, model_io_size):

    model.train()

    start = time.time()
    iteration = 0
    for epoch in range(100000):
        for _, data in enumerate(train_loader):
            iteration += 1
            sys.stdout.flush()
            iteration_loss = 0
            image, flux, skeleton, path, start_pos, stop_pos, start_sid, stop_sid = data

            # initialize samplers
            batch_size = len(image)
            samplers = []
            for i in range(batch_size):
                samplers.append(SkeletonGrowingRNNSampler(image[i], skeleton[i], flux[i], path[i],
                                          start_pos[i], stop_pos[i], start_sid[i], stop_sid[i],
                                          sample_input_size=model_io_size, stride=2.0, anisotropy=[30.0, 6.0, 6.0], d_avg=3))

            # Get data from samplers and drop sampler which are not required to continue
            loss = torch.zeros((1,), dtype=torch.float32, device=device, requires_grad=True)
            continue_samplers = list(range(batch_size))
            no_data_for_forward = False
            do_not_log = False
            for t in range(100):

                if no_data_for_forward:
                    # if no forward pass could be made in last iteration then break
                    break

                input_image, input_flux = [], []
                start_skeleton_mask, other_skeleton_mask = [], []
                gt_direction, gt_state, center_pos = [], [], []

                temp_c_samplers = []
                sampler_idx_matching = {}
                prev_hidden_state = []
                prev_cell_state = []
                i = 0
                for s_idx in continue_samplers:
                    next_step_data = samplers[s_idx].get_next_step()
                    if next_step_data[0] == True: # if it could fetch data add the data to the lists
                        if next_step_data[6] == torch.tensor(path_state['CONTINUE'], dtype=torch.float32):
                            # Only continue predicting in the next step if the current state is Continue, otherwise we have reached the end
                            temp_c_samplers.append(s_idx)

                        input_image.append(next_step_data[1])
                        input_flux.append(next_step_data[2])
                        start_skeleton_mask.append(next_step_data[3])
                        other_skeleton_mask.append(next_step_data[4])
                        gt_direction.append(next_step_data[5])
                        gt_state.append(next_step_data[6])
                        center_pos.append(next_step_data[7])

                        # save_volumes_in_dict({'input_image':input_image[-1][0],
                        #                       'input_flux':input_flux[-1],
                        #                       'start_skeleton_mask':start_skeleton_mask[-1][0],
                        #                       'other_skeleton_mask':other_skeleton_mask[-1][0]}, base_path=args.output)

                        # print('GT  Direction:', gt_direction)
                        # print('GT State:', gt_state)
                        # print('GT_position:', center_pos)

                        if t == 0:
                            prev_hidden_state = None
                            prev_cell_state = None
                        else:
                            prev_hidden_state.append(output_hidden_state[directions_idx_for_next_step[i]])
                            prev_cell_state.append(output_cell_state[directions_idx_for_next_step[i]])

                        sampler_idx_matching[s_idx] = i
                        i += 1

                continue_samplers = temp_c_samplers
                directions_idx_for_next_step = [sampler_idx_matching[x] for x in continue_samplers]

                # stack data and train
                if len(input_image) > 0:
                    input_image = torch.stack(input_image, 0)
                    input_flux = torch.stack(input_flux, 0)
                    start_skeleton_mask = torch.stack(start_skeleton_mask, 0)
                    other_skeleton_mask = torch.stack(other_skeleton_mask, 0)
                    gt_direction = torch.stack(gt_direction, 0).to(device)
                    gt_state = torch.stack(gt_state, 0).to(device)
                    if prev_hidden_state is not None:
                        prev_hidden_state = torch.stack(prev_hidden_state, 0).to(device)
                    if prev_cell_state is not None:
                        prev_cell_state = torch.stack(prev_cell_state, 0).to(device)

                    # concatenate image + flux + masks and compute forward pass
                    input = torch.cat((input_image, input_flux, start_skeleton_mask, other_skeleton_mask), 1).to(device)

                    #forward pass
                    output_direction, output_hidden_state, output_cell_state = model(input, prev_hidden_state, prev_cell_state)

                    # loss
                    flux_loss, angular_l, scale_l = criterion(output_direction, gt_direction)
                    loss = loss + flux_loss
                    do_backpropagate = True
                else:
                    # no data from any sampler
                    no_data_for_forward = True
                    if t == 0:
                        # do not log this if in the first iteration no data could be collected for forward pass
                        do_not_log = True

                #after every 5 steps backpropagate or do it before exiting the for loop because no forward passes could be made
                if (t + 1) % 10 == 0 or (do_backpropagate and no_data_for_forward):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    do_backpropagate = False
                    print('- [Steps: %d] train_loss=%0.4f lr=%.6f' % ( t+1, loss.item(), optimizer.param_groups[0]['lr']))

                    # Remove computation graph
                    iteration_loss += loss.detach().item()
                    loss = torch.zeros((1,), dtype=torch.float32, device=device, requires_grad=True)
                    output_direction = output_direction.detach()
                    output_hidden_state = output_hidden_state.detach()
                    output_cell_state = output_cell_state.detach()

                # Using the predicted directions calculate next positions, samplers will update their state
                for i, sampler_idx in enumerate(continue_samplers):
                    next_pos = samplers[sampler_idx].jump_to_next_position(output_direction[i])
                    # print('--------------------')
                    # print('Previous pos: ', center_pos[i])
                    # print('Predicted Direction: ', output_direction[i])
                    # print ('New Pos:', next_pos)

            # normalize logged loss by the number of steps taken
            iteration_loss /= t
            if logger and writer and ~do_not_log:
                print('[Iteration %d] train_loss=%0.4f lr=%.6f' % (
                iteration, iteration_loss, optimizer.param_groups[0]['lr']))
                logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration,
                                                                         iteration_loss,
                                                                         optimizer.param_groups[0]['lr']))
                writer.add_scalars('Loss', {'Overall Loss': iteration_loss}, iteration)

            # save the predcited path for debugging
            if iteration % 20 == 0:
                try:
                    with h5py.File(args.output + 'predicted_paths.h5', 'w') as predicted_h5:
                        count = 0
                        for sampler in samplers:
                            hg = predicted_h5.create_group(str(count))
                            count += 1
                            path = sampler.get_predicted_path()
                            hg.create_dataset('vertices', data=path)
                            edges = np.zeros(2 * (path.shape[0] - 1), dtype=np.uint16)
                            edges[1::2] = np.arange(1, path.shape[0])
                            edges[2:-1:2] = np.arange(1, path.shape[0] - 1)
                            hg.create_dataset('edges', data=edges)
                        print('Saved paths: ', np.arange(count))
                except:
                    print('Exception catched while writing path to h5')

            #save model
            if iteration % args.iteration_save == 0 or iteration >= args.iteration_total:
                torch.save(model.state_dict(), args.output + (args.exp_name + '_%d.pth' % iteration))

            # Log time for first 100 iterations
            if iteration < 100:
                print('Time taken: ', time.time() - start)
                start = time.time()

            # Terminate
            if iteration >= args.iteration_total:
                break


def main():
    args = get_args(mode='train')
    save_cmd_line(args)  # Saving the command line args with machine name and time for later reference
    args.output = args.output + args.exp_name + '/'

    print('Initial setup')
    torch.backends.cudnn.enabled = False
    model_io_size, device = init(args)

    if args.disable_logging is not True:
        logger, writer = get_logger(args)
    else:
        logger, writer = None, None
        print('No log file would be created.')

    print('Setup model')
    model = setup_model(args, device, model_io_size)

    print('Setup data')
    train_loader = get_input(args, model_io_size, 'train', model=None)

    print('Setup loss function')
    criterion = AngularAndScaleLoss(alpha=0.08)
    # criterion = WeightedMSE()

    print('Setup optimizer')
    model_parameters = list(model.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                           patience=1000, verbose=False, threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0,
                                                           min_lr=1e-7, eps=1e-08)

    print('4. start training')
    train(args, train_loader, None, model, device, criterion, optimizer, scheduler, logger, writer, None, model_io_size)

    print('5. finish training')
    if args.disable_logging is not True:
        logger.close()
        writer.close()


if __name__ == "__main__":
    main()
