import os,sys
import numpy as np
import h5py, time, itertools, datetime
from scipy.ndimage import label as scipy_label
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.interpolation import shift

import torch
import torch.nn.functional as F

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize, visualize_aff
from torch_connectomics.data.utils.functional_collate import collate_fn_2

def train(args, train_loader, val_loader, model, device, criterion,
          optimizer, scheduler, logger, writer, regularization=None):
    record = AverageMeter()
    val_record = AverageMeter()

    # connectivity and sel for erosion and connected components
    sel_cpu = np.ones((3, 3, 3), dtype=bool)
    sel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=device)

    model.train()
    visualize_fq = 1

    # need to pad the output of the first sampling inference to match the size required by augmentor for second sampling
    sz_diff = train_loader.dataset.sample_input_size - train_loader.dataset.model_input_size
    pad_sz = sz_diff // 2
    pad_adj = sz_diff % 2
    pad_sz[pad_adj > 0] += 1
    sample1_c = tuple(train_loader.dataset.model_input_size // 2)

    if val_loader is not None:
        val_loader_itr = iter(val_loader)

    for iteration, (pos, volume, input_label, label, class_weight, _) in enumerate(train_loader):
        sys.stdout.flush()
        prediction_points = []
        volume, input_label, label = volume.to(device), input_label.to(device), label.to(device)

        # First sampling inference
        with torch.no_grad():
            output = model(torch.cat((volume, input_label), 1))

        # display the first sampling input and output
        if iteration % visualize_fq == 0 and iteration >= 1:
            visualize(volume, label, output, iteration, writer, input_label=input_label, mode='step1_train')
        # Thereshold output to make a mask
        output = output > 0.85

        out_mask_cpu = []

        # for all volumes in the batch find the next possible seed locations
        for idx in range(output.shape[0]):
            # if the center pixel is not thresholded then use the
            # partial predictions to get seed points for second sample
            if output[idx][0][sample1_c] == True:
                out_mask = output[idx][0].cpu().detach().numpy().astype(bool)
            else:
                out_mask = input_label[idx][0].cpu().detach().numpy().astype(bool)

            # Binary erosion and edge detection
            cc_out_mask, _ = scipy_label(out_mask)
            out_mask = (cc_out_mask == cc_out_mask[sample1_c])
            out_mask_cpu.append(out_mask)
            out_mask = torch.from_numpy(binary_erosion(out_mask, sel_cpu).copy().astype(np.float32)).to(device)
            out_mask = out_mask.unsqueeze(0).unsqueeze(0)
            edge = (F.conv3d(out_mask, sel, padding=1))[0, 0]
            edge = (edge > 0) * (edge < 9)
            edge = F.interpolate(edge.unsqueeze(0).unsqueeze(0).float(), scale_factor=1 / 4, mode='trilinear')
            edge = edge > .50
            edge_pos = (torch.nonzero(edge[0, 0]) * 4).cpu().detach().numpy().astype(np.uint32)

            # appending center points wrt the sample volume to perform next prediction step
            prediction_points.append(np.hstack((np.full((edge_pos.shape[0], 1), idx, dtype=np.uint32), edge_pos)))

        prediction_points = np.vstack(prediction_points)
        # choose 'batch_size' number of prediction points and perform optimization
        choosen_idx = np.random.randint(prediction_points.shape[0], size=train_loader.batch_size, dtype=np.uint32)
        sample1_pos = pos

        pos, volume, input_label, label, class_weight, weight_factor = [], [], [], [], [], []
        for idx in choosen_idx:
            int_pos = prediction_points[idx]
            global_pos = int_pos[1:] + sample1_pos[int_pos[0]][1:] - train_loader.dataset.half_input_sz

            # shift the previous predictions to align with the next sample, also pad to match the augmentation size
            past_pred = shift(out_mask_cpu[int_pos[0]],
                              -(int_pos[1:].astype(np.int64) - (train_loader.dataset.augmentor_1.input_size // 2).astype(np.int64)),
                              order=0, prefilter=False)
            past_pred = past_pred.astype(np.float32)
            past_pred = np.pad(past_pred, ((pad_sz[0], pad_sz[0]),
                                           (pad_sz[1], pad_sz[1]),
                                           (pad_sz[2], pad_sz[2])), 'constant')

            (_pos, _volume, _input_label, _label, _class_weight, _weight_factor) = \
                train_loader.dataset.__getitem__(index=None, pos=np.append(0, global_pos), past_pred=past_pred)
            pos.append(_pos)
            volume.append(_volume)
            input_label.append(_input_label)
            label.append(_label)
            class_weight.append(_class_weight)
            weight_factor.append(_weight_factor)

        # stack data and copy to gpu
        pos, volume, input_label, label, class_weight, _ = \
            collate_fn_2(zip(pos, volume, input_label, label, class_weight, weight_factor))
        volume, input_label, label, class_weight \
            = volume.to(device), input_label.to(device), label.to(device), class_weight.to(device)

        # Run inference for the 2nd step
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

        if iteration % visualize_fq == 0 and iteration >= 1:
            if args.task == 0:
                visualize_aff(volume, label, output, iteration, writer, mode='Train')
            elif args.task == 1 or args.task == 3:
                visualize(volume, label, output, iteration, writer, input_label=input_label, mode='step2_train')

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
            torch.save(model.state_dict(), args.output+('/m_32_192_192_noBN_Dout_dualChan_DistanceTx%d.pth' % (iteration)))

        # Terminate
        if iteration >= args.iteration_total:
            break

def main():
    args = get_args(mode='train')

    print('0. initial setup')
    model_io_size, device = init(args) 
    logger, writer = get_logger(args)

    print('1. setup data')
    train_loader = get_input(args, model_io_size, 'train')

    # val_loader = get_input(args, model_io_size, 'validation')

    print('2.0 setup model')
    model = setup_model(args, device)
            
    print('2.1 setup loss function')
    criterion = WeightedMSE()
    regularization = BinaryReg(alpha=10.0)
 
    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                patience=1000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, 
                min_lr=1e-7, eps=1e-08)
    

    print('4. start training')
    train(args, train_loader, None, model, device, criterion, optimizer, scheduler, logger, writer)
  
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()
