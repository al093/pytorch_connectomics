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

    iteration = -1
    for epoch in range(1000):
        for _, data in enumerate(train_loader):
            iteration += 1
            sys.stdout.flush()

            if iteration < 100:
                print('time taken for itr: ', time.time() - start)
                start = time.time()

            _, volume, out_skeleton_1, out_skeleton_2, out_flux, match = data

            volume = volume.to(device)
            out_skeleton_1, out_skeleton_2 = out_skeleton_1.to(device), out_skeleton_2.to(device)
            out_flux, match = out_flux.to(device), match.to(device)

            output = model(torch.cat((volume, out_skeleton_1, out_skeleton_2, out_flux), dim=1))
            loss = criterion(output, match)
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
                with torch.no_grad():
                    if writer:
                        volume_cpu = volume.cpu()
                        vis_result_gt = torch.ones_like(volume_cpu)
                        vis_result_gt *= match.cpu().view(match.shape[0], 1, 1, 1, 1)
                        vis_result_out = torch.ones_like(volume_cpu)
                        vis_result_out *= output.cpu().view(match.shape[0], 1, 1, 1, 1)

                        visualize(volume_cpu, vis_result_gt, vis_result_out, iteration, writer, mode='Train',
                                  color_data=torch.cat((out_skeleton_1.cpu(), out_skeleton_2.cpu(), out_skeleton_2.cpu(), vec_to_RGB(out_flux.cpu())), 1))
                    scheduler.step(record.avg)
                    record.reset()

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

    model = setup_model(args, device, model_io_size, non_linearity=(torch.sigmoid,))

    print('Setup data')
    train_loader = get_input(args, model_io_size, 'train', model=None)

    print('Setup loss function')
    criterion = nn.BCELoss()

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
