import os,sys
import numpy as np
import torch
import h5py, time, itertools, datetime
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize_aff, save_data

def test(args, test_loader, model, device, result_path):
    # switch to eval mode
    model.eval()
    volume_id = 0

    start = time.time()
    results = []
    with torch.no_grad():
        for i, (key, volume, out_skeleton_1, out_skeleton_2, out_skeleton_p, out_flux) in enumerate(test_loader):
            volume_id += args.batch_size
            print('processing:', volume_id)

            # for gpu computing
            volume = volume.to(device)
            out_skeleton_1, out_skeleton_2 = out_skeleton_1.to(device), out_skeleton_2.to(device)
            out_skeleton_p, out_flux = out_skeleton_p.to(device), out_flux.to(device)

            output = model(torch.cat((volume, out_skeleton_1, out_skeleton_2, out_skeleton_p, out_flux), dim=1))

            for idx in range(output.shape[0]):
                results.append((key[idx][0], key[idx][1], key[idx][2], key[idx][3], key[idx][4], output[idx].detach().cpu().item()))

    end = time.time()
    print("prediction time:", (end-start))

    result_path = result_path + 'matching'
    np.save(file=result_path, arr=results)
    print('matching results stored at: \n' + result_path)

def get_augmented(volume):
    # perform 16 Augmentations as mentioned in Kisuks thesis
    vol0    = volume
    vol90   = torch.rot90(vol0, 1, [3, 4])
    vol180  = torch.rot90(vol90,  1, [3, 4])
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
    torch.backends.cudnn.enabled = False
    model_io_size, device = init(args)
    print('model I/O size:', model_io_size)

    print('1. setup data')
    test_loader, volume_shape, pad_size = get_input(args, model_io_size, 'test')

    print('2. setup model')
    model = setup_model(args, device, model_io_size=model_io_size, exact=True, non_linearity=(torch.sigmoid, ))

    result_path = args.output + '/' + os.path.basename(os.path.dirname(args.pre_model)) + '/'
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    save_cmd_line(args, result_path + 'commandArgs.txt')  # Saving the command line args with machine name and time for later reference

    print('3. start testing')
    test(args, test_loader, model, device, result_path)

    print('4. finish testing')

if __name__ == "__main__":
    main()