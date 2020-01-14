import os,sys
import h5py, time, itertools, datetime
import numpy as np

import torch

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import *


def train(args, train_loader, val_loader, model, device, criterion, optimizer, scheduler, logger, writer, regularization=None):
    record = AverageMeter()
    val_record = AverageMeter()

    model.train()

    if val_loader is not None:
        val_loader_itr = iter(val_loader)

    start = time.time()

    for iteration, data in enumerate(train_loader):
        sys.stdout.flush()

        if iteration < 100:
            print('time taken for itr: ', time.time() - start)
            start = time.time()

        _, volume, label, flux, flux_weight = data

        volume = volume.to(device)
        flux, flux_weight = flux.to(device), flux_weight.to(device)

        output = model(volume)
        output_flux = output[:, :]

        flux_loss, angular_l, scale_l = criterion(output_flux, flux, weight=flux_weight)
        loss = flux_loss
        if writer:
            writer.add_scalars('Part-wise Losses',
                               {'Angular': angular_l.item(),
                                'Scale': scale_l.item()}, iteration)

        # For L2 uncomment
        # loss = criterion(output_flux, flux, weight=flux_weight)

        record.update(loss, args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[Iteration %d] train_loss=%0.4f lr=%.6f' % (iteration,
                                                           loss.item(), optimizer.param_groups[0]['lr']))

        if logger and writer:
            logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration,
                                                                     loss.item(), optimizer.param_groups[0]['lr']))
            writer.add_scalars('Loss', {'Overall Loss': loss.item()}, iteration)

        if iteration % 200 == 0:
            if writer:
                visualize(volume.cpu(), flux_weight.cpu() / flux_weight.max().cpu(), label,
                          iteration, writer, mode='Train',
                          color_data=torch.cat((vec_to_RGB(output_flux.cpu()), vec_to_RGB(flux.cpu())), 1))

            scheduler.step(record.avg)
            record.reset()

            if val_loader is not None:
                model.eval()
                val_record.reset()

                #for better coverage of validation dataset running multiple batches of val
                for _ in range(5):
                    try:
                        (_, volume, label, class_weight, _) = next(val_loader_itr)
                    except StopIteration:
                        val_loader_itr = iter(val_loader)
                        (_, volume, label, class_weight, _) = next(val_loader_itr)

                    with torch.no_grad():
                        volume, label = volume.to(device), label.to(device)
                        class_weight = class_weight.to(device)
                        output = model(volume)

                        if regularization is not None:
                            val_loss = criterion(output, label, class_weight) + regularization(output)
                        else:
                            val_loss = criterion(output, label, class_weight)

                        val_record.update(val_loss, args.batch_size)

                writer.add_scalars('Loss', {'Val': val_loss.item()}, iteration)
                print('[Iteration %d] val_loss=%0.4f lr=%.6f' % (iteration,
                      val_loss.item(), optimizer.param_groups[0]['lr']))

                if args.task == 0:
                    visualize_aff(volume, label, output, iteration, writer, mode='Validation')
                elif args.task == 1 or args.task == 3:
                    visualize(volume, label, output, iteration, writer)

                model.train()

        #Save model
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
    torch.backends.cudnn.enabled = False
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
    criterion = AngularAndScaleLoss(alpha=0.08)
    # criterion = WeightedMSE()

    print('Setup optimizer')
    model_parameters = list(model.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                patience=1000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, 
                min_lr=1e-7, eps=1e-08)

    print('4. start training')
    train(args, train_loader, None, model, device, criterion, optimizer, scheduler, logger, writer)

    print('5. finish training')
    if args.disable_logging is not True:
        logger.close()
        writer.close()

if __name__ == "__main__":
    main()
