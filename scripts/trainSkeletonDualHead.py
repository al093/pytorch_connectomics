import os,sys
import h5py, time, itertools, datetime
import numpy as np

import torch

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import *


def train(args, train_loader, val_loader, model, model_2, device, criterion,
          criterion_2, optimizer, scheduler, logger, writer, regularization=None):
    record = AverageMeter()
    val_record = AverageMeter()

    model.train()
    if model_2:
        model_2.train()

    if val_loader is not None:
        val_loader_itr = iter(val_loader)

    start = time.time()
    loss_ratio_1 = 0.90
    loss_ratio_2 = 0.90

    softmax = nn.Softmax(dim=1)

    for iteration, data in enumerate(train_loader):
        sys.stdout.flush()

        if iteration < 100:
            print('time taken for itr: ', time.time() - start)
            start = time.time()

        _, volume, label, flux, skeleton, skeleton_weight, flux_weight = data

        if args.train_grad_dtx:
            volume = volume.to(device)
            flux, flux_weight = flux.to(device), flux_weight.to(device)
            skeleton_weight = skeleton_weight.to(device)

            output = model(volume)
            output_flux = output[:, 0:3]
            output_skel = output[:, 3:]
            # import pdb; pdb.set_trace()
            flux_loss, angular_l, scale_l = criterion(output_flux, flux, weight=flux_weight)

            if str(criterion_2) == str(nn.CrossEntropyLoss()):
                skeleton = skeleton.to(device)
                skel_loss = criterion_2(output_skel, skeleton)
                skel_loss = (skel_loss * skeleton_weight).sum() / ((skeleton_weight > 0).sum().float() + 1e-10)
            else:
                skeleton = skeleton.float() / 4.0
                skeleton = skeleton.to(device).unsqueeze(1)
                skel_loss = criterion_2(output_skel, skeleton, skeleton_weight)

            if model_2:
                pass
                # skeleton, skeleton_weight = skeleton.to(device), skeleton_weight.to(device)
                # output_skel = model_2(torch.cat((output_flux, output_b_mask), 1))
                # skel_loss = criterion_2(output_skel, skeleton, weight=skeleton_weight)
                # loss = (1-loss_ratio_2)*flux_loss + (loss_ratio_2/2)*boundary_loss + (loss_ratio_2/2)*skel_loss
                # if writer:
                #     writer.add_scalars('Part-wise Losses',
                #                        {'Flux': (1-loss_ratio_2)*flux_loss.item(),
                #                         'Skeleton': (loss_ratio_2/2)*skel_loss.item(),
                #                         'Boundary': (loss_ratio_2/2)*boundary_loss}, iteration)
            else:
                loss = loss_ratio_1 * skel_loss + (1 - loss_ratio_1) * flux_loss
                if writer:
                    writer.add_scalars('Part-wise Losses',
                                       {'Flux': (1-loss_ratio_1)*flux_loss.item(),
                                        'Skel': loss_ratio_1*skel_loss.item()}, iteration)

        else:
            pass
            # volume = volume.to(device)
            # flux = flux.to(device)
            # skeleton, skeleton_weight = skeleton.to(device), skeleton_weight.to(device)
            # output_skel = model(torch.cat((flux, volume), 1))
            # skel_loss = criterion_2(output_skel, skeleton, weight=skeleton_weight)
            # loss = skel_loss

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

        if iteration % 500 == 0:
            with torch.no_grad():
                if writer:
                    if str(criterion_2) == str(nn.CrossEntropyLoss()):
                        _, output_skel = softmax(output_skel).max(dim=1)
                        output_skel = output_skel.float() / 4.0
                        output_skel = output_skel.unsqueeze(1)
                        skeleton = skeleton.cpu().float().unsqueeze(1) / 4.0

                    visualize(volume.cpu(), skeleton.cpu(), output_skel.cpu(), iteration, writer, mode='Train',
                              input_label=torch.cat((flux_weight.cpu()/flux_weight.max().cpu(), skeleton_weight.cpu()/skeleton_weight.max().cpu()), 1),
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
            if model_2:
                torch.save(model_2.state_dict(), args.output + (args.exp_name + '_second_%d.pth' % iteration))

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
    if not args.train_grad_dtx:
        print('Will only train from gt gradients to skeleton.')
        model = setup_model(args, device, model_io_size, non_linearity=(torch.tanh, torch.sigmoid))
        model_2 = None
    else:
        model = setup_model(args, device, model_io_size, non_linearity=(torch.tanh, torch.sigmoid))
        if args.init_second_model is True:
            print('Setting up Second model')
            class ModelArgs(object):
                pass
            args2 = ModelArgs()
            args2.task = args.task
            args2.num_gpu = args.num_gpu
            args2.architecture = args.architecture
            args2.pre_model = args.pre_model_second
            args2.load_model = args.load_model_second
            args2.in_channel = args.out_channel
            args2.out_channel = 1
            args2.batch_size = args.batch_size
            model_2 = setup_model(args2, device, model_io_size, non_linearity=(torch.sigmoid,))
        else:
            model_2 = None

    print('Setup data')
    train_loader = get_input(args, model_io_size, 'train', model=None)
            
    print('Setup loss function')
    criterion = AngularAndScaleLoss(alpha=0.20)
    # criterion_2 = nn.CrossEntropyLoss(weight=None, reduction='none')
    criterion_2 = WeightedL1()
    # regularization = BinaryReg(alpha=10.0)
 
    print('Setup optimizer')
    model_parameters = list(model.parameters())
    if model_2:
        model_parameters += list(model_2.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                patience=1000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, 
                min_lr=1e-7, eps=1e-08)

    print('4. start training')
    train(args, train_loader, None, model, model_2, device, criterion, criterion_2, optimizer, scheduler, logger, writer)

    print('5. finish training')
    if args.disable_logging is not True:
        logger.close()
        writer.close()

if __name__ == "__main__":
    main()
