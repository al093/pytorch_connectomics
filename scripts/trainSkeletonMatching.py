import os,sys
import h5py, time, itertools, datetime
import numpy as np

import torch

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import *


def train(args, train_loader, models, device, loss_fns, optimizer, scheduler, logger, writer, regularization=None):
    models[0].eval()
    models[1].train()

    last_iteration_num, _ = restore_state(optimizer, scheduler, args, device)

    start = time.time()
    epochs = int(np.ceil(args.iteration_total / len(train_loader.dataset)))
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for iteration, data in enumerate(train_loader, start=last_iteration_num+1):
            sys.stdout.flush()

            if iteration < 50:
                print('time taken for itr: ', time.time() - start)
                start = time.time()

            sample, volume, out_skeleton_1, out_skeleton_2, out_flux, match = data

            volume_gpu = volume.to(device)
            out_skeleton_1_gpu, out_skeleton_2_gpu = out_skeleton_1.to(device), out_skeleton_2.to(device)
            # out_flux = out_flux.to(device)
            match = match.to(device)

            with torch.no_grad():
                pred_flux = models[0](volume_gpu)['flux']

            out_match = models[1](torch.cat((volume_gpu, out_skeleton_1_gpu, out_skeleton_2_gpu, pred_flux), dim=1))

            loss = loss_fns[0](out_match, match)

            # compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            print('[Iteration %d] train_loss=%0.4f lr=%.6f' % (iteration,
                                                               loss.item(), optimizer.param_groups[0]['lr']))
            if logger and writer:
                logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration,
                                                                         loss.item(), optimizer.param_groups[0]['lr']))
                writer.add_scalars('Loss', {'Overall Loss': loss.item()}, iteration)

            if iteration % 500 == 0:
                title = "Image/GT_Skeleton"
                if writer:
                    visualize(volume, out_skeleton_1, out_skeleton_2, iteration, writer, title=title)

            # Save model
            if iteration % args.iteration_save == 0 or iteration >= args.iteration_total:
                save_dict = {models[0].module.__class__.__name__ + '_state_dict': models[0].state_dict(),
                             models[1].module.__class__.__name__ + '_state_dict': models[1].state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'scheduler_state_dict': scheduler.state_dict(),
                             'loss': loss,
                             'iteration': iteration}
                torch.save(save_dict, args.output + (args.exp_name + '_%d.pth' % iteration))

            last_iteration_num = iteration

            # Terminate
            if iteration >= args.iteration_total:
                break

def main():
    args = get_args(mode='train')

    save_cmd_line(args)  # Saving the command line args with machine name and time for later reference
    args.output = args.output + args.exp_name + '/'

    print('Initial setup')
    torch.backends.cudnn.enabled = True
    model_io_size, device = init(args)

    if args.disable_logging is not True:
        logger, writer = get_logger(args)
    else:
        logger, writer = None, None
        print('No log file would be created.')

    classification_model = setup_model(args, device, model_io_size, non_linearity=(torch.sigmoid,))

    print('Setting up Second model')

    class ModelArgs(object):
        pass
    args2 = ModelArgs()
    args2.task = 4
    args2.architecture = 'fluxNet'
    args2.in_channel = 1
    args2.out_channel = 3
    args2.num_gpu = args.num_gpu
    args2.pre_model = args.pre_model
    args2.load_model = args.load_model
    args2.use_skeleton_head = args.use_skeleton_head
    args2.use_flux_head = args.use_flux_head
    args2.aspp_dilation_ratio = args.aspp_dilation_ratio
    args2.resolution = args.resolution
    args2.symmetric = args.symmetric
    args2.batch_size = args.batch_size
    args2.local_rank = args.local_rank
    flux_model = setup_model(args2, device, model_io_size, non_linearity=(torch.tanh,))
    models = [flux_model, classification_model]

    print('Setup data')
    train_loader = get_input(args, model_io_size, 'train', model=None)

    print('Setup loss function')
    loss_fns = [nn.BCEWithLogitsLoss()]

    print('Setup optimizer')
    model_parameters = list()
    [model_parameters.extend(model.parameters()) for model in models[1:]]

    optimizer = torch.optim.Adam(model_parameters, lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=1e-6, amsgrad=True)

    if args.lr_scheduler is 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=round(args.iteration_total / 5), gamma=0.75)
    else:
        print("Learning rate scheduler is not defined to any known types.")
        return

    print('Start training.')
    train(args, train_loader, models, device, loss_fns, optimizer, scheduler, logger, writer)

    print('Training finished.')
    if args.disable_logging is not True:
        logger.close()
        writer.close()


if __name__ == "__main__":
    main()
