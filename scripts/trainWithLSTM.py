import os,sys
import h5py, time, itertools, datetime
import numpy as np

import torch

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize, visualize_aff
import torch.multiprocessing as t_mp

def train(args, train_loader, val_loader, model, model_cpu, device, criterion,
          optimizer, scheduler, logger, writer, regularization=None, model_lstm=None):
    record = AverageMeter()
    val_record = AverageMeter()

    model.train()

    if val_loader is not None:
        val_loader_itr = iter(val_loader)

    start = time.time()
    for iteration, data in enumerate(train_loader):
        print('time taken for itr: ', time.time() - start)
        start = time.time()
        sys.stdout.flush()

        if args.in_channel == 2:
            (_, volume, input_label, label, class_weight, _) = data
            volume, input_label, label = volume.to(device), input_label.to(device), label.to(device)
            output = model(torch.cat((volume, input_label), 1))
        else:
            (_, volume, label, class_weight, _) = data

            volume, label = volume.to(device), label.to(device)
            output = model(volume)
            if model_lstm is not None:
                output_pre_lstm = output
                output = model_lstm(output.clone())

        class_weight = class_weight.to(device)
        if regularization is not None:
            loss_unet = criterion(output_pre_lstm, label, class_weight) + regularization(output_pre_lstm)
            loss = loss_unet + criterion(output, label, torch.ones_like(class_weight)) + regularization(output)
        else:
            loss_unet = criterion(output_pre_lstm, label, class_weight)
            loss = loss_unet + criterion(output, label, torch.ones_like(class_weight))

        record.update(loss, args.batch_size) 

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration,
                                                                 loss.item(), optimizer.param_groups[0]['lr']))
        print('[Iteration %d] train_loss=%0.4f lr=%.6f' % (iteration,
                                                           loss.item(), optimizer.param_groups[0]['lr']))
        writer.add_scalars('Loss', {'Train': loss.item()}, iteration)

        if iteration % 50 == 0 and iteration >= 1:

            params_gpu = model.named_parameters()
            params_cpu = model_cpu.named_parameters()
            dict_p = dict(params_gpu)
            dict_params_cpu = dict(params_cpu)
            for name, param in dict_p.items():
                dict_params_cpu[name].data.copy_(param.data)

            if args.task == 0:
                visualize_aff(volume, label, output, iteration, writer, mode='Train')
            elif args.task == 1 or args.task == 3:
                if args.in_channel == 2:
                    visualize(volume, label, output, iteration, writer, input_label=input_label)
                else:
                    visualize(volume, label, output, iteration, writer, input_label=output_pre_lstm)  #TODO used input label to show the pre LSTM output, change this later into more elegant format

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

            #print('weight factor: ', weight_factor) # debug
            # debug
            # if iteration < 50:
            #     fl = h5py.File('debug_%d_h5' % (iteration), 'w')
            #     output = label[0].cpu().detach().numpy().astype(np.uint8)
            #     print(output.shape)
            #     fl.create_dataset('main', data=output)
            #     fl.close()

        #Save model
        if iteration % args.iteration_save == 0 or iteration >= args.iteration_total:
            torch.save(model.state_dict(), args.output+(args.exp_name + '_%d.pth' % iteration))
            if model_lstm is not None:
                torch.save(model_lstm.state_dict(), args.output + (args.exp_name + 'head_%d.pth' % iteration))

        # Terminate
        if iteration >= args.iteration_total:
            break

def main():
    args = get_args(mode='train')

    print('Initial setup')
    model_io_size, device = init(args)

    if args.enable_logging:
        logger, writer = get_logger(args)
    else:
        print('No log file would be created.')

    print('Setup model')
    model, model_cpu = setup_model(args, device, model_io_size)

    model_lstm = setup_lstm_model(args, device, model_io_size)

    print('Setup data')
    train_loader = get_input(args, model_io_size, 'train', model=model_cpu)

    # val_loader = get_input(args, model_io_size, 'validation')
            
    print('Setup loss function')
    criterion = WeightedBCE()   
    regularization = BinaryReg(alpha=10.0)
 
    print('Setup optimizer')
    torch.autograd.set_detect_anomaly(True)
    # list(model.parameters()) +
    optimizer = torch.optim.Adam(list(model_lstm.parameters()), lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                patience=1000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, 
                min_lr=1e-7, eps=1e-08)

    print('4. start training')
    train(args, train_loader, None, model, model_cpu, device, criterion, optimizer, scheduler, logger, writer, model_lstm=model_lstm)

    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()
