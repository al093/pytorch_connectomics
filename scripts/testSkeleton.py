import os,sys
import numpy as np
import torch
import h5py, time, itertools, datetime
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize_aff

def test(args, test_loader, model, model_2, device, model_io_size, volume_shape, pad_size):
    # switch to eval mode
    model.eval()
    volume_id = 0
    ww = blend(model_io_size)

    result_grad = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(3)]) for x in volume_shape]
    if model_2 is not None:
        result_skel = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(1)]) for x in volume_shape]
    weight = [np.zeros(x, dtype=np.float32) for x in volume_shape]

    start = time.time()
    with torch.no_grad():
        for i, (pos, volume) in enumerate(test_loader):
            volume_id += args.batch_size
            print('volume_id:', volume_id)

            # for gpu computing
            volume = volume.to(device)

            output_grad = model(volume)
            if model_2 is not None:
                output_skel = model_2(output_grad)

            sz = tuple([3]+list(model_io_size))
            sz_skel = tuple([1] + list(model_io_size))
            for idx in range(output_grad.size()[0]):
                st = pos[idx]
                result_grad[st[0]][:, st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]] \
                    += output_grad[idx].cpu().detach().numpy().reshape(sz) * np.expand_dims(ww, axis=0)
                weight[st[0]][st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]]\
                    += ww
                if model_2 is not None:
                    result_skel[st[0]][:, st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]] \
                        += output_skel[idx].cpu().detach().numpy().reshape(sz_skel) * np.expand_dims(ww, axis=0)

    end = time.time()
    print("prediction time:", (end-start))

    for vol_id in range(len(result_grad)):
        data_grad = result_grad[vol_id]

        data_grad = data_grad[:,
                    pad_size[0]:-pad_size[0],
                    pad_size[1]:-pad_size[1],
                    pad_size[2]:-pad_size[2]]

        hf = h5py.File(args.output + '/skeleton_' + str(vol_id) + '.h5', 'w')
        hf.create_dataset('main', data=data_grad, compression='gzip')
        hf.close()

def get_augmented(volume):
    # perform 16 Augmentations as mentioned in Kisuks thesis
    vol0    = volume
    vol90   = torch.rot90(vol0, 1, [3, 4])
    vol180  = torch.rot90(vol90,  1,  [3, 4])
    vol270  = torch.rot90(vol180, 1, [3, 4])

    vol0f   = torch.flip(vol0,   [3])
    vol90f  = torch.flip(vol90,  [3])
    vol180f = torch.flip(vol180, [3])
    vol270f = torch.flip(vol270, [3])

    vol0z   = torch.flip(vol0,   [2])
    vol90z  = torch.flip(vol90,  [2])
    vol180z = torch.flip(vol180, [2])
    vol270z = torch.flip(vol270, [2])

    vol0fz   = torch.flip(vol0f,   [2])
    vol90fz  = torch.flip(vol90f,  [2])
    vol180fz = torch.flip(vol180f, [2])
    vol270fz = torch.flip(vol270f, [2])

    augmented_volumes = [vol0,  vol90,  vol180,  vol270,  vol0f,  vol90f,  vol180f,  vol270f,
                         vol0z, vol90z, vol180z, vol270z, vol0fz, vol90fz, vol180fz, vol270fz]

    return augmented_volumes

def combine_augmented(outputs):
    assert len(outputs) == 16
    for i in range(8, 16):
        outputs[i] = torch.flip(outputs[i], [2])
    for i in range(4, 8):
        outputs[i] = torch.flip(outputs[i], [3])
    for i in range(12, 16):
        outputs[i] = torch.flip(outputs[i], [3])
    for i in range(1, 16, 4):
        outputs[i] = torch.rot90(outputs[i], -1, [3, 4])
    for i in range(2, 16, 4):
        outputs[i] = torch.rot90(outputs[i], -1, [3, 4])
        outputs[i] = torch.rot90(outputs[i], -1, [3, 4])
    for i in range(3, 16, 4):
        outputs[i] = torch.rot90(outputs[i], 1, [3, 4])

    # output = torch.zeros_like(outputs[0], dtype=torch.float64)
    # for i in range(len(outputs)):
    #     output += outputs[i].double()
    # output = output / 16.0
    for i in range(len(outputs)):
        outputs[i] = outputs[i].unsqueeze(0)
    output = torch.min(torch.cat(outputs, 0), 0)[0]

    return output, outputs

def main():
    args = get_args(mode='test')

    print('0. initial setup')
    model_io_size, device = init(args)
    print('model I/O size:', model_io_size) 

    print('1. setup data')
    test_loader, volume_shape, pad_size = get_input(args, model_io_size, 'test')

    print('2. setup model')
    model, _ = setup_model(args, device, model_io_size=model_io_size, exact=True, non_linearity=torch.tanh)

    if not args.only_gradients:
        class ModelArgs(object):
            pass
        args2 = ModelArgs()
        args2.task = args.task
        args2.architecture = 'unetv3'
        args2.pre_model = args.pre_model_second
        args2.load_model = args.load_model_second
        args2.in_channel = 3
        args2.out_channel = 1
        args2.batch_size = args.batch_size
        model_2 = setup_model(args2, device, model_io_size=model_io_size, exact=True, non_linearity=torch.sigmoid)
    else:
        model_2 = None

    print('3. start testing')
    test(args, test_loader, model, model_2, device, model_io_size, volume_shape, pad_size)
  
    print('4. finish testing')

if __name__ == "__main__":
    main()