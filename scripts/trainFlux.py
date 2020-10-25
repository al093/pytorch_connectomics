import os,sys
import h5py, time, itertools, datetime
import numpy as np
import copy

import torch

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import *


def train(args, train_loader, models, device, loss_fns, optimizer, scheduler, logger, writer, regularization=None):
    for m in models: m.train()
    last_iteration_num, loss = restore_state(optimizer, scheduler, args, device)

    start = time.time()
    for iteration, data in enumerate(train_loader, start=last_iteration_num+1):
        if args.local_rank is None or args.local_rank == 0:
            sys.stdout.flush()
            if iteration < 50:
                print('time taken for iteration: ', time.time() - start)
                start = time.time()

        _, volume, label, flux, flux_weight, skeleton = data
        flux_weight_gpu = flux_weight.to(device)
        volume_gpu = volume.to(device)

        model_output_dict = models[0](volume_gpu)

        losses_dict = dict()
        loss = 0.0
        for output_type, model_output in model_output_dict.items():
            if output_type == 'flux':
                if isinstance(loss_fns[0], AngularAndScaleLoss):
                    flux_loss, angular_l, scale_l = loss_fns[0](model_output, flux.to(device), weight=flux_weight_gpu)
                    loss += flux_loss
                    losses_dict.update({'Angular': angular_l.item(), 'Scale': scale_l.item()})
                else:
                    loss += loss_fns[0](model_output, flux.to(device), weight=flux_weight_gpu)
            elif output_type == 'skeleton':
                skeleton_loss = loss_fns[1](model_output, skeleton.to(device), weight=flux_weight_gpu)
                loss += 2*skeleton_loss
                losses_dict['Skeleton'] = 2*skeleton_loss.item()
            else:
                raise ValueError('Unknown model type')

        if args.local_rank is None or args.local_rank == 0 and writer and losses_dict:
            writer.add_scalars('Part-wise Losses', losses_dict, iteration)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if args.local_rank is None or args.local_rank == 0:
            print('[Iteration %d] train_loss=%0.4f lr=%.6f' % (iteration,
                                                               loss.item(), optimizer.param_groups[0]['lr']))

            if logger and writer:
                logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration,
                                                                         loss.item(), optimizer.param_groups[0]['lr']))
                writer.add_scalars('Loss', {'Overall Loss': loss.item()}, iteration)

            if iteration % 500 == 0:
                title = "Image/GT_Skeleton"
                if writer:
                    if 'skeleton' in model_output_dict.keys():
                        vis_bw = model_output_dict['skeleton'].cpu()
                        title += '/Skeleton'
                    else:
                        vis_bw = flux_weight
                        title += '/Weight'

                    if 'flux' in model_output_dict.keys():
                        vis_color = torch.cat((vec_to_RGB(model_output_dict['flux'].cpu()), vec_to_RGB(flux)), 1)
                        title += '/Predicted_Flux/GT_Flux'
                    else:
                        vis_color = vec_to_RGB(flux)
                        title += '/GT_Flux'

                    visualize(volume, skeleton, vis_bw, iteration, writer, title=title, color_data=vis_color)

            #Save model, update lr
            if iteration % args.iteration_save == 0 or iteration >= args.iteration_total:
                save_dict = {models[0].module.__class__.__name__ + '_state_dict': models[0].state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss':loss,
                            'iteration':iteration}
                torch.save(save_dict, args.output+(args.exp_name + '_%d.pth' % iteration))

        # Terminate
        if iteration >= args.iteration_total:
            break

def setup_ddp(local_rank):
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

def main():
    args = get_args(mode='train')

    save_cmd_line(args)  # Saving the command line args with machine name and time for later reference
    args.output = args.output + args.exp_name + '/'

    print('Initial setup')
    if args.local_rank is not None:
        setup_ddp(args.local_rank)
        print(f'Local rank: {args.local_rank}')

    torch.backends.cudnn.enabled = True
    model_io_size, device = init(args)

    logger, writer = None, None
    if args.disable_logging is not True:
        if args.local_rank in [None, 0]:
            logger, writer = get_logger(args)

    print('Setup model.')
    model = setup_model(args, device, model_io_size, non_linearity=(torch.tanh,))
    models = [model]

    print('Setup data.')
    train_loader = get_input(args, model_io_size, 'train', model=None)

    print('Setup loss function.')
    loss_fns = [AngularAndScaleLoss(alpha=0.25)]

    loss_fns.append(WeightedL1())

    print('Setup optimizer')
    model_parameters = list(model.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)

    # TODO(alok) maybe we need this back
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
    #             patience=1000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
    #             min_lr=1e-7, eps=1e-08)

    if args.lr_scheduler is 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=round(args.iteration_total/5), gamma=0.75)
    else:
        print("Learning rate scheduler is not defined to any known types.")
        return

    print('Start training')
    train(args, train_loader, models, device, loss_fns, optimizer, scheduler, logger, writer)

    print('Training finished')
    if args.disable_logging is not True:
        if args.local_rank in [None, 0]:
            logger.close()
            writer.close()

if __name__ == "__main__":
    main()
