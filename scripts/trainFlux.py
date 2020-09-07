import os,sys
import h5py, time, itertools, datetime
import numpy as np

import torch

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import *


def train(args, train_loader, model, device, criterion, optimizer, scheduler, logger, writer, regularization=None):
    model.train()
    start = time.time()

    for iteration, data in enumerate(train_loader):
        sys.stdout.flush()

        if iteration < 50:
            print('time taken for itr: ', time.time() - start)
            start = time.time()

        _, volume, label, flux, flux_weight, _, _= data

        volume = volume.to(device)
        output_flux = model(volume)
        flux, flux_weight = flux.to(device), flux_weight.to(device)

        if isinstance(criterion, AngularAndScaleLoss):
            flux_loss, angular_l, scale_l = criterion(output_flux, flux, weight=flux_weight)
            loss = flux_loss
            if writer:
                writer.add_scalars('Part-wise Losses',
                                   {'Angular': angular_l.item(),
                                    'Scale': scale_l.item()}, iteration)
        else:
            loss = criterion(output_flux, flux, weight=flux_weight)

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
            if writer:
                visualize(volume.cpu(), flux_weight.cpu() / flux_weight.max().cpu(), label,
                          iteration, writer, mode='Train',
                          color_data=torch.cat((vec_to_RGB(output_flux.cpu()), vec_to_RGB(flux.cpu())), 1))

        #Save model, update lr
        if iteration % args.iteration_save == 0 or iteration >= args.iteration_total:
            torch.save(model.state_dict(), args.output+(args.exp_name + '_%d.pth' % iteration))

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

    print('Setup model')
    model = setup_model(args, device, model_io_size, non_linearity=(torch.tanh,))

    print('Setup data')
    train_loader = get_input(args, model_io_size, 'train', model=None)

    print('Setup loss function')
    # TODO | NOTE change loss fn and weight, alpha = 0.08 for all ours experiement
    criterion = AngularAndScaleLoss(alpha=0.08)
    # criterion = WeightedMSE()

    print('Setup optimizer')
    model_parameters = list(model.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)

    # TODO(alok) maybe we need this back
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
    #             patience=1000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
    #             min_lr=1e-7, eps=1e-08)

    if args.lr_scheduler is 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=round(args.iteration_total/5), gamma=0.5)
    else:
        print("Learning rate scheduler is not defined to any known types.")
        return

    print('4. start training')
    train(args, train_loader, model, device, criterion, optimizer, scheduler, logger, writer)

    print('5. finish training')
    if args.disable_logging is not True:
        logger.close()
        writer.close()

if __name__ == "__main__":
    main()