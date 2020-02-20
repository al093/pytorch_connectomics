import os,sys
import numpy as np
import torch
import h5py, time, itertools, datetime
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize_aff
from tqdm import tqdm
import re

def test(args, test_loader, model, device, model_io_size, volume_shape, pad_size, result_path, result_file_pf, input_file_name, save_output):
    # switch to eval mode
    model.eval()
    ww = blend(model_io_size)
    result_skel = [np.zeros([1] + list(x), dtype=np.float32) for x in volume_shape]
    cropped_result_skel = []
    weight = [np.zeros(x, dtype=np.float32) for x in volume_shape]
    sz = tuple([1] + list(model_io_size))

    start = time.time()
    with torch.no_grad():
        for i, (pos, volume) in enumerate(test_loader):
            volume = volume.to(device)
            output = model(volume)

            for idx in range(output.size()[0]):
                st = pos[idx]
                result_skel[st[0]][:, st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]] \
                    += output[idx].cpu().detach().numpy().reshape(sz) * np.expand_dims(ww, axis=0)

                weight[st[0]][st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]]\
                    += ww

    end = time.time()
    print("prediction time:", (end-start))

    for vol_id in range(len(result_skel)):
        result_skel[vol_id] = result_skel[vol_id] / (weight[vol_id] + np.finfo(np.float32).eps)
        data_skel = result_skel[vol_id]
        data_skel = data_skel[0,
                pad_size[0]:-pad_size[0],
                pad_size[1]:-pad_size[1],
                pad_size[2]:-pad_size[2]]
        cropped_result_skel.append(data_skel)
        if save_output == True:
            skeleton_path = result_path + 'skeleton_' + input_file_names[vol_id] + '_' + result_file_pf + '.h5'
            with h5py.File(skeleton_path, 'w') as hf:
                hf.create_dataset('main', data=data_skel, compression='gzip')
            print('Skeleton stored at: \n' + gradient_path)
    return cropped_result_skel

def _run(args, save_output=True):
    print('0. initial setup')
    model_io_size, device = init(args)
    print('model I/O size:', model_io_size)

    print('1. setup data')
    test_loader, volume_shape, pad_size = get_input(args, model_io_size, 'test')

    print('2. setup model')
    model = setup_model(args, device, model_io_size=model_io_size, exact=True, non_linearity=(torch.sigmoid,))

    result_path = args.output + '/' + args.exp_name + '/'
    result_file_pf = re.split('_|.pth', os.path.basename(args.pre_model))[-2]
    print(result_file_pf)
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    save_cmd_line(args,
                  result_path + 'commandArgs.txt')  # Saving the command line args with machine name and time for later reference

    input_file_name = [os.path.basename(input_image)[:-3] for input_image in args.img_name.split('@')]

    print('3. start testing')
    result = test(args, test_loader, model, device, model_io_size, volume_shape, pad_size, result_path, result_file_pf,
                  input_file_name, save_output)

    print('4. finish testing')
    return result

def run(input_args_string, save_output):
    return _run(get_args(mode='test', input_args=input_args_string), save_output)

if __name__ == "__main__":
    _run(get_args(mode='test'))