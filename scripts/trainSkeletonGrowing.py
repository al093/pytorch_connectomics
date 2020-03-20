import os, sys, traceback
import h5py, time, itertools, datetime
import numpy as np
from argparse import Namespace

import torch

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import *

def train(args, train_loader, model, flux_model, device, criterion, criterion_bce, criterion_2, optimizer, scheduler,
          logger, writer, regularization, model_io_size, train_end_to_end, num_volumes, resolution, flux_model_io_size):
    total_steps = args.tracking_steps
    backprop_frequency = total_steps // 2
    debug = False
    if debug:
        print('DEBUG MODE IS ON!!, No training will be performed')
        total_epochs = 1
    else:
        total_epochs = 100000
    model.train()
    if train_end_to_end: flux_model.train()
    else: flux_model.eval()
    output_dicts = [{} for i in range(num_volumes)]
    start = time.time()
    iteration = 0
    for epoch in range(total_epochs):
        for _, data in enumerate(train_loader):
            iteration += 1
            sys.stdout.flush()
            iteration_loss = 0
            partwise_iteraton_loss = {'direction':0.0, 'flux':0.0, 'state':0.0}

            image, flux, flux_gt, skeleton, divergence, path, start_pos, stop_pos, start_sid, stop_sid, ft_params, path_state_loss_weight, first_split_node, did, sid = data

            # initialize samplers
            batch_size = len(image)
            samplers = []
            for i in range(batch_size):
                samplers.append(SkeletonGrowingRNNSampler(image=image[i], skeleton=skeleton[i], flux=flux[i], divergence=divergence[i],
                                                          path=path[i], start_pos=start_pos[i], stop_pos=stop_pos[i],
                                                          start_sid=start_sid[i], stop_sid=stop_sid[i], first_split_node=first_split_node[i],
                                                          ft_params=ft_params[i], path_state_loss_weight=path_state_loss_weight[i],
                                                          did=did[i], sid=sid[i], sample_input_size=model_io_size, stride=2.0,
                                                          anisotropy=resolution, d_avg=24, mode='train',
                                                          train_flux_model=train_end_to_end, flux_gt=flux_gt[i], debug=True))

                samplers[-1].init_global_feature_models(flux_model, None, np.array(flux_model_io_size, dtype=np.int32), device)

            # Get data from samplers and drop sampler which are not required to continue
            loss = torch.zeros((1,), dtype=torch.float32, device=device, requires_grad=True)
            continue_samplers = list(range(batch_size))
            no_data_for_forward = False
            do_not_log = False
            do_backpropagate = False
            for t in range(total_steps):
                if no_data_for_forward:
                    # if no forward pass could be made in last iteration then break
                    break

                input_image, input_flux, input_divergence = [], [], [] # they are precomputed
                start_skeleton_mask, other_skeleton_mask = [], []
                gt_direction, gt_path_state, center_pos = [], [], []
                path_state_loss_weight, global_features= [], []
                p_flux, p_divergence = [], [], # they are on the GPU
                gt_flux = []

                temp_c_samplers = []
                sampler_idx_matching = {}
                prev_hidden_state = []
                prev_cell_state = []
                i = 0
                for s_idx in continue_samplers:
                    next_step_data = samplers[s_idx].get_next_step()
                    if next_step_data[0] == True: # if it could fetch data add the data to the lists
                        if next_step_data[7] == torch.tensor(path_state['CONTINUE'], dtype=torch.float32):
                            # Only continue predicting in the next step if the current state is Continue, otherwise we have reached the end
                            temp_c_samplers.append(s_idx)

                        input_image.append(next_step_data[1])
                        input_flux.append(next_step_data[2])
                        start_skeleton_mask.append(next_step_data[3])
                        other_skeleton_mask.append(next_step_data[4])
                        input_divergence.append(next_step_data[5])
                        gt_direction.append(next_step_data[6])
                        gt_path_state.append(next_step_data[7])
                        center_pos.append(next_step_data[8])
                        path_state_loss_weight.append(next_step_data[9])
                        global_features.append(next_step_data[10]) # on GPU
                        p_flux.append(next_step_data[11]) # on GPU
                        p_divergence.append(next_step_data[12]) # on GPU
                        if train_end_to_end is True:
                            gt_flux.append(next_step_data[13])

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
                    input_divergence = torch.stack(input_divergence, 0)
                    start_skeleton_mask = torch.stack(start_skeleton_mask, 0)
                    other_skeleton_mask = torch.stack(other_skeleton_mask, 0)
                    global_features = torch.stack(global_features, 0) # on GPU
                    p_flux = torch.stack(p_flux, 0) # on GPU
                    p_divergence = torch.stack(p_divergence, 0) # on GPU

                    gt_direction = torch.stack(gt_direction, 0).to(device)
                    gt_path_state = torch.stack(gt_path_state, 0).to(device)
                    path_state_loss_weight = torch.cat(path_state_loss_weight).to(device)

                    if train_end_to_end is True:
                        gt_flux = torch.stack(gt_flux, 0).to(device)

                    if prev_hidden_state is not None:
                        prev_hidden_state = torch.stack(prev_hidden_state, 0).to(device)
                    if prev_cell_state is not None:
                        prev_cell_state = torch.stack(prev_cell_state, 0).to(device)

                    # concatenate image + flux + masks and compute forward pass
                    if args.use_precomputed is True:
                        input = torch.cat((input_image, start_skeleton_mask, other_skeleton_mask, input_flux, input_divergence), 1).to(device)
                    else:
                        input = torch.cat((input_image, start_skeleton_mask, other_skeleton_mask), 1).to(device)
                        input = torch.cat((input, p_flux, p_divergence), 1)

                    # forward pass
                    if debug is not True:
                        output_direction, output_path_state, output_hidden_state, output_cell_state = model(input, prev_hidden_state, prev_cell_state)
                    else:
                        output_direction = gt_direction
                        output_path_state = gt_path_state
                        output_hidden_state = torch.empty_like(gt_direction)
                        output_cell_state = torch.empty_like(gt_direction)

                    direction_loss, _, _ = criterion(output_direction, gt_direction)  # direction loss
                    state_loss = criterion_bce(output_path_state, gt_path_state, path_state_loss_weight)  # state loss
                    state_loss_alpha = 0.60
                    state_loss = state_loss_alpha*state_loss
                    direction_loss = (1.0-state_loss_alpha)*direction_loss

                    flux_loss_alpha = 0.0
                    if train_end_to_end is True:
                        flux_loss_alpha = 0.50
                        flux_loss, _, _ = criterion_2(p_flux, gt_flux)
                        flux_loss = flux_loss_alpha*flux_loss
                        loss = loss + flux_loss + (1-flux_loss_alpha)*(direction_loss + state_loss)
                        partwise_iteraton_loss['flux'] += flux_loss.detach().item()
                    else:
                        loss = loss + direction_loss + state_loss

                    partwise_iteraton_loss['direction'] += (1.0-flux_loss_alpha)*direction_loss.detach().item()
                    partwise_iteraton_loss['state'] += (1.0-flux_loss_alpha)*state_loss.detach().item()

                    do_backpropagate = True
                else:
                    # no data from any sampler
                    no_data_for_forward = True
                    if t == 0:
                        # do not log this if in the first iteration no data could be collected for forward pass
                        do_not_log = True

                #after every n steps backpropagate or do it before exiting the for loop because no forward passes could be made
                if debug is not True:
                    if (t + 1) % backprop_frequency == 0 or (do_backpropagate and no_data_for_forward):
                        optimizer.zero_grad()
                        if t == total_steps -1 or (do_backpropagate and no_data_for_forward) or train_end_to_end is False:
                            retain_graph = False
                        else:
                            retain_graph = True
                        loss.backward(retain_graph=retain_graph)
                        optimizer.step()
                        do_backpropagate = False
                        print('- [Steps: %d] train_loss=%0.4f lr=%.6f' % (t+1, loss.item(), optimizer.param_groups[0]['lr']))

                        # Remove computation graph
                        iteration_loss += loss.detach().item()
                        loss = torch.zeros((1,), dtype=torch.float32, device=device, requires_grad=True)
                        flux_loss = None
                        output_direction = output_direction.detach()
                        output_hidden_state = output_hidden_state.detach()
                        output_cell_state = output_cell_state.detach()

                # Using the predicted directions calculate next positions, samplers will update their state
                # evaluate next step only for the ones which had continue state
                for i, sampler_idx in enumerate(list(sampler_idx_matching.keys())):
                    if sampler_idx in continue_samplers:
                        samplers[sampler_idx].jump_to_next_position(output_direction[i], output_path_state[i])

            if debug is True: #save output for debugging,
                for sampler in samplers:
                    p_path, p_state, p_end_ids = sampler.get_predicted_path()
                    p_path = p_path - (np.array(model_io_size).astype(np.int32) // 2)
                    edges = np.zeros(2 * (p_path.shape[0] - 1), dtype=np.uint16)
                    edges[1::2] = np.arange(1, p_path.shape[0])
                    edges[2:-1:2] = np.arange(1, p_path.shape[0] - 1)
                    output_dicts[sampler.did][sampler.sid] = {'vertices': p_path, 'states': p_state, 'sids': p_end_ids, 'edges': edges}

            # normalize logged loss by the number of steps taken
            iteration_loss /= t
            for key in partwise_iteraton_loss.keys():
                partwise_iteraton_loss[key] /= t

            if logger and writer and ~do_not_log:
                print('[Iteration %d] train_loss=%0.4f lr=%.6f' % (
                iteration, iteration_loss, optimizer.param_groups[0]['lr']))
                logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration,
                                                                         iteration_loss,
                                                                         optimizer.param_groups[0]['lr']))
                writer.add_scalars('Loss', {'Overall Loss': iteration_loss}, iteration)
                writer.add_scalars('Partwise Loss', partwise_iteraton_loss, iteration)

            #save model
            if iteration % args.iteration_save == 0:
                torch.save(model.state_dict(), args.output + (args.exp_name + '_%d.pth' % iteration))
                if train_end_to_end is True:
                    torch.save(flux_model.state_dict(), args.output + (args.exp_name + '_fluxNet_%d.pth' % iteration))

            # Log time for first 100 iterations
            if iteration < 100:
                print('Time taken: ', time.time() - start)
                start = time.time()

            # Terminate
            if iteration >= args.iteration_total:
                break

        # for debugging
        if debug is True:
            for i, output_dict in enumerate(output_dicts):
                with open(args.output + str(i) + '_predicted_paths.pkl', 'wb') as pfile:
                    pickle.dump(output_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    args = get_args(mode='train')
    save_cmd_line(args)  # Saving the command line args with machine name and time for later reference
    args.output = args.output + args.exp_name + '/'

    print('Initial setup')
    torch.backends.cudnn.enabled = False
    model_io_size, device = init(args)

    resolution = np.array([x for x in args.resolution.split(',')], dtype=np.float32)
    print('Data resolution: ', resolution)

    if args.disable_logging is not True:
        logger, writer = get_logger(args)
    else:
        logger, writer = None, None
        print('No log file would be created.')

    print('Setup model')
    model = setup_model(args, device, model_io_size)

    flux_model_io_size = np.array([96, 256, 256], dtype=np.int32)
    flux_model_args = Namespace(architecture='fluxNet', task=4, out_channel=3, in_channel=1,
                                batch_size=1, load_model=True, pre_model=args.pre_model_second, num_gpu=args.num_gpu)
    flux_model = setup_model(flux_model_args, device, flux_model_io_size, non_linearity=(torch.tanh,))

    train_end_to_end = args.train_end_to_end
    print('Train end to end: ', train_end_to_end)

    print('Setup data')
    train_loader = get_input(args, model_io_size, 'train', model=None)

    print('Setup loss function')
    criterion = AngularAndScaleLoss(alpha=1.0)
    criterion_bce = WeightedBCE()
    criterion_2 = AngularAndScaleLoss(alpha=0.08)

    model_parameters = list(model.parameters())
    if train_end_to_end == True:
        model_parameters += list(flux_model.parameters())
    print('Setup optimizer')
    optimizer = torch.optim.Adam(model_parameters, lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                           patience=1000, verbose=False, threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0,
                                                           min_lr=1e-7, eps=1e-08)
    num_volumes = len(args.img_name.split('@'))
    print('4. start training')
    train(args, train_loader, model, flux_model, device, criterion, criterion_bce, criterion_2, optimizer, scheduler, logger, writer,
          None, model_io_size, train_end_to_end, num_volumes, resolution, flux_model_io_size)

    print('5. finish training')
    if args.disable_logging is not True:
        logger.close()
        writer.close()

if __name__ == "__main__":
    main()
