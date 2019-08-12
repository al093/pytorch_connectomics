import os,sys
import numpy as np
import h5py, time, itertools, datetime
from scipy.ndimage import label as scipy_label
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.interpolation import shift
import time
import logging
import signal

import torch.multiprocessing as t_mp 

import torch
import torch.nn.functional as F
import ctypes

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize, visualize_aff
from torch_connectomics.data.utils.functional_collate import collate_fn_2

def to_numpy_array(mp_arr, shape, dtype):
    ar = np.frombuffer(mp_arr.get_obj(), dtype=dtype, count=np.prod(shape))
    ar.shape = tuple([s for s in shape])
    return ar

def mp_init(shared_ar_):
    global shared_ar
    shared_ar = shared_ar_

def get_predictions_pos(idx, sample1_c, shape, shift_sz, dtype):
    # connectivity and sel for erosion and connected components
    sel_cpu = np.ones((3, 3, 3), dtype=bool)
    sel = (torch.ones((1, 1, 3, 3, 3), dtype=torch.float32))
    shared_arr_np = to_numpy_array(shared_ar, shape, dtype)
    # Binary erosion and edge detection
    cc_out_mask, _ = scipy_label(shared_arr_np[idx, 0])
    out_mask = (cc_out_mask == cc_out_mask[sample1_c])
    out_mask_cpu = out_mask
    out_mask = torch.from_numpy(out_mask.copy().astype(np.float32))
    out_mask = out_mask.unsqueeze(0).unsqueeze(0)
    edge = (F.conv3d(out_mask, sel, padding=1))[0, 0]
    edge = (edge > 0) * (edge < 9)
    edge = F.interpolate(edge.unsqueeze(0).unsqueeze(0).float(), scale_factor=1 / 4, mode='trilinear', align_corners=True)
    edge = edge > .50
    edge_pos = (torch.nonzero(edge[0, 0]) * 4).cpu().detach().numpy().astype(np.uint32)
    c_pos = np.random.randint(edge_pos.shape[0], size=1, dtype=np.uint32)
    int_pos = edge_pos[c_pos]
    # shift the previous predictions to align with the next sample, also pad to match the augmentation size
    past_pred = shift(out_mask_cpu,
                      -(int_pos.astype(np.int64)[0] - shift_sz.astype(np.int64)),
                      order=0, prefilter=False)
    shared_arr_np[idx, 0] = past_pred

    return int_pos

def train(args, train_loader, val_loader, model, device, criterion,
          optimizer, scheduler, logger, writer, regularization=None):
    model_ip_sz = train_loader.dataset.model_input_size

    #create a shared memory for the numpy arrays of size batch_size * model input size
    shared_ar = t_mp.Array('f', int(train_loader.batch_size*np.prod(model_ip_sz)))
    shared_ar_np = to_numpy_array(shared_ar, dtype=np.float32, shape=np.array([train_loader.batch_size, 1, model_ip_sz[0], model_ip_sz[1], model_ip_sz[2]]))

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = t_mp.Pool(processes=train_loader.batch_size, initializer=mp_init, initargs=(shared_ar,))
    signal.signal(signal.SIGINT, original_sigint_handler)

    vz_freq = 50

    record = AverageMeter()
    val_record = AverageMeter()
    model.train()

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

        # start = time.time()
        volume, input_label, label = volume.to(device), input_label.to(device), label.to(device)

        # First sampling inference
        with torch.no_grad():
            output = model(torch.cat((volume, input_label), 1))

        # display the first sampling input and output
        if iteration % vz_freq == 0 and iteration >= 1:
            visualize(volume, label, output, iteration, writer, input_label=input_label, mode='step1_train')
        # Thereshold output to make a mask
        output = output > 0.85

        # print('Time taken for 1st inference and thresholding: ', time.time() - start)

        mp_data = []
        # for all volumes in the batch find the next possible seed locations
        # start = time.time()
        for idx in range(output.shape[0]):
            # if the center pixel is not thresholded then use the
            # partial predictions to get seed points for second sample
            if output[idx][0][sample1_c] == True:
                shared_ar_np[idx] = output[idx].cpu().detach()
            else:
                shared_ar_np[idx] = input_label[idx].cpu().detach()
            mp_data.append((idx, sample1_c,
                            np.array([train_loader.batch_size, 1, model_ip_sz[0], model_ip_sz[1], model_ip_sz[2]]),
                            np.array(train_loader.dataset.augmentor_1.input_size // 2), np.float32))

        # import pdb; pdb.set_trace()
        results = pool.starmap_async(get_predictions_pos, mp_data)
        results = results.get()
        prediction_points = np.vstack(results)

        # print('Time taken for CC and edge detection: ', time.time() - start)

        sample1_pos = pos
        pos, volume, input_label, label, class_weight, weight_factor = [], [], [], [], [], []
        # start = time.time()
        past_pred = np.pad(shared_ar_np, ((0, 0), (0, 0),
                                          (pad_sz[0], pad_sz[0]),
                                          (pad_sz[1], pad_sz[1]),
                                          (pad_sz[2], pad_sz[2])), 'constant')
        # print('Time taken for padding: ', time.time() - start)

        for idx in range(prediction_points.shape[0]):
            int_pos = prediction_points[idx]
            global_pos = int_pos + sample1_pos[idx][1:] - train_loader.dataset.half_input_sz
            (_pos, _volume, _input_label, _label, _class_weight, _weight_factor) = \
                train_loader.dataset.__getitem__(index=None, pos=np.append(0, global_pos), past_pred=past_pred[idx, 0])
            pos.append(_pos)
            volume.append(_volume)
            input_label.append(_input_label)
            label.append(_label)
            class_weight.append(_class_weight)
            weight_factor.append(_weight_factor)

        # print('Time taken for padding and fetching data: ', time.time() - start)

        # start = time.time()
        # stack data and copy to gpu
        pos, volume, input_label, label, class_weight, _ = \
            collate_fn_2(zip(pos, volume, input_label, label, class_weight, weight_factor))
        volume, input_label, label, class_weight \
            = volume.to(device), input_label.to(device), label.to(device), class_weight.to(device)

        # Run inference for the 2nd step
        output = model(torch.cat((volume, input_label), 1))

        # print('Time taken for copy to GPU and 2nd inference: ', time.time() - start)

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

        if iteration % vz_freq == 0 and iteration >= 1:
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
            torch.save(model.state_dict(), args.output+('/m_32_192_192_noBN_Dout_dualChan_%d.pth' % (iteration)))

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
    # t_mp.set_start_method('spawn', force=True)
    mpl = t_mp.log_to_stderr()
    mpl.setLevel(logging.INFO)
    main()
