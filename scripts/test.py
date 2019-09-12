import os,sys
import numpy as np
import torch
import h5py, time, itertools, datetime
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize_aff

def test(args, test_loader, model, device, model_io_size, volume_shape, pad_size):
    # switch to eval mode
    model.eval()
    volume_id = 0
    ww = blend(model_io_size)
    NUM_OUT = args.out_channel

    result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in volume_shape]
    prediction_points = np.empty((0, 4), dtype=np.uint32)
    # std_result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in volume_shape]
    weight = [np.zeros(x, dtype=np.float32) for x in volume_shape]
    print(result[0].shape, weight[0].shape)

    if args.test_augmentation:
        print("Will augment (Rotate and Flip) data during inference.")

    start = time.time()
    with torch.no_grad():
        for i, (pos, volume) in enumerate(test_loader):
            volume_id += args.batch_size
            print('volume_id:', volume_id)

            # for gpu computing
            volume = volume.to(device)

            if args.test_augmentation:
                augmented_volumes = get_augmented(volume)
                augmented_outputs = []
                for aug_vol in augmented_volumes:
                    augmented_outputs.append(model(aug_vol))
                output, _ = combine_augmented(augmented_outputs)
            else:
                output = torch.Tensor().to(device)
                for itr in range(1):
                    output = torch.cat((output, model(volume).unsqueeze(dim=0)))
                #std = output.std(dim=0) ## calculate standard deviation
                output = output[0]

            prediction_points = np.append(prediction_points, pos, axis=0)

            sz = tuple([NUM_OUT]+list(model_io_size))
            for idx in range(output.size()[0]):
                st = pos[idx]
                if args.task == 3:
                    result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], st[3]:st[3]+sz[3]] \
                        = np.maximum(result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], st[3]:st[3]+sz[3]], \
                                     output[idx].cpu().detach().numpy().reshape(sz))
                else:
                    result[st[0]][:, st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]] \
                        += output[idx].cpu().detach().numpy().reshape(sz) * np.expand_dims(ww, axis=0)

                    weight[st[0]][st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]]\
                        += ww

                    # std_result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                    # st[3]:st[3]+sz[3]] += std[idx].cpu().detach().numpy().reshape(sz) * np.expand_dims(ww, axis=0)

                # weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], st[3]:st[3]+sz[3]] += ww

    end = time.time()
    print("prediction time:", (end-start))
    for vol_id in range(len(result)):
        if args.task != 3:
            result[vol_id] = result[vol_id] / (weight[vol_id] + np.finfo(np.float32).eps)

        data = (result[vol_id]*255).astype(np.uint8)
        data = data[:,
                    pad_size[0]:-pad_size[0],
                    pad_size[1]:-pad_size[1],
                    pad_size[2]:-pad_size[2]]
        print('Output shape: ', data.shape)
        hf = h5py.File(args.output + '/mask_' + str(vol_id) + '.h5', 'w')
        hf.create_dataset('main', data=data, compression='gzip')
        hf.close()

        hf = h5py.File(args.output + '/prediction_points' + str(vol_id) + '.h5', 'w')
        hf.create_dataset('main', data=prediction_points[:, 1:], compression='gzip')
        hf.close()

        # std_result[vol_id] = std_result[vol_id] / weight[vol_id]
        # std_result = std_result[0]
        # std_result = std_result[:,
        #        pad_size[0]:-pad_size[0],
        #        pad_size[1]:-pad_size[1],
        #        pad_size[2]:-pad_size[2]]
        # hf = h5py.File(args.output + '/std_' + str(vol_id) + '.h5', 'w')
        # hf.create_dataset('main', data=std_result, compression='gzip')
        # hf.close()


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
    test_loader, volume_shape, pad_size, _ = get_input(args, model_io_size, 'test')

    print('2. setup model')
    model = setup_model(args, device, exact=True)

    print('3. start testing')
    test(args, test_loader, model, device, model_io_size, volume_shape, pad_size)
  
    print('4. finish testing')

if __name__ == "__main__":
    main()