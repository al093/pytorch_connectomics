import os,sys

import h5py, time, itertools, datetime

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize, visualize_aff
import torch.multiprocessing as t_mp

def train(args, train_loader, val_loader, model, model_cpu, device, criterion,
          optimizer, scheduler, logger, writer, regularization=None):
    record = AverageMeter()
    val_record = AverageMeter()

    model.train()

    if val_loader is not None:
        val_loader_itr = iter(val_loader)

    for iteration, (_, volume, input_label, label, class_weight, _) in enumerate(train_loader):
        sys.stdout.flush()
        volume, input_label, label = volume.to(device), input_label.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output = model(torch.cat((volume, input_label), 1))

        if regularization is not None:
            loss = criterion(output, label, class_weight) + regularization(output)
        else:
            loss = criterion(output, label, class_weight)

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

            dict_params2 = dict(params_cpu)
            for name1, param1 in params_gpu:
                dict_params2[name1].data.copy_(param1.data.cpu())

            if args.task == 0:
                visualize_aff(volume, label, output, iteration, writer, mode='Train')
            elif args.task == 1 or args.task == 3:
                visualize(volume, label, output, iteration, writer, input_label=input_label)

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
            torch.save(model.state_dict(), args.output+('/m_32_192_192_noBN_Dout_dualChan_Nearest%d.pth' % (iteration)))

        # Terminate
        if iteration >= args.iteration_total:
            break    #     

def main():
    args = get_args(mode='train')

    print('0. initial setup')
    model_io_size, device = init(args) 
    logger, writer = get_logger(args)


    print('2.0 setup model')
    model, model_cpu = setup_model(args, device)

    params_gpu = model.named_parameters()
    params_cpu = model_cpu.named_parameters()
    dict_params2 = dict(params_cpu)
    for name1, param1 in params_gpu:
        dict_params2[name1].data.copy_(param1.data.cpu())

    print('1. setup data')
    train_loader = get_input(args, model_io_size, 'train', model=model_cpu)

    # val_loader = get_input(args, model_io_size, 'validation')
            
    print('2.1 setup loss function')
    criterion = WeightedBCE()   
    regularization = BinaryReg(alpha=10.0)
 
    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                patience=1000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, 
                min_lr=1e-7, eps=1e-08)
    

    print('4. start training')
    train(args, train_loader, None, model, model_cpu, device, criterion, optimizer, scheduler, logger, writer)
  
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":

    os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
    import numpy as np
    import torch
    torch.set_num_threads(1)

    main()
