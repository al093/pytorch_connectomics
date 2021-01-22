import os,sys
import numpy as np
import torch
import h5py, time, itertools, datetime
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize_aff
from tqdm import tqdm
import re, copy

def test(args, test_loader, models, device, model_io_size, volume_shape, pad_size, result_path, result_file_pf, input_file_names, save_output):
    for m in models: m.eval()

    ww = blend(model_io_size)
    weight = [np.zeros(x, dtype=np.float32) for x in volume_shape]
    sz = tuple(list(model_io_size))
    results = dict()
    cropped_results = dict()
    if args.use_skeleton_head:
        results["skeleton"] = [np.zeros([1] + list(x), dtype=np.float32) for x in volume_shape]
        cropped_results["skeleton"] = []
    if args.use_flux_head:
        results["flux"] = [np.zeros([3] + list(x), dtype=np.float32) for x in volume_shape]
        cropped_results["flux"] = []

    start = time.time()
    with torch.no_grad():
        for i, (pos, volume) in tqdm(enumerate(test_loader)):
            volume = volume.to(device)
            output_dict = models[0](volume)
            for output_type, output in output_dict.items():
                for idx in range(output.shape[0]):
                    st = pos[idx]
                    results[output_type][st[0]][..., st[1]:st[1] + sz[0], st[2]:st[2] + sz[1], st[3]:st[3] + sz[2]] \
                        += output[idx].cpu().detach().numpy() * np.expand_dims(ww, axis=0)

            for _, output in output_dict.items():
                for idx in range(output.shape[0]):
                    st = pos[idx]
                    weight[st[0]][st[1]:st[1] + sz[0], st[2]:st[2] + sz[1], st[3]:st[3] + sz[2]] += ww
                break

    end = time.time()
    print("Model prediction time:", (end-start))

    for output_type, output in results.items():
        for vol_id in range(len(output)):
            output[vol_id] = output[vol_id] / (weight[vol_id] + np.finfo(np.float32).eps)
            cropped_results[output_type].append(np.squeeze(output[vol_id][..., pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1], pad_size[2]:-pad_size[2]]))
            if save_output:
                result_file = result_path + '/' + output_type + '_' + input_file_names[vol_id] + '_' + result_file_pf + f'_{vol_id}.h5'
                with h5py.File(result_file, 'w') as hf:
                    hf.create_dataset('main', data=cropped_results[output_type][-1], compression='gzip')
                    print(f'Model output stored at: \n {result_file}')

    return cropped_results

def _run(args, save_output = True):
    print('0. initial setup')
    model_io_size, device = init(args)
    print('model I/O size:', model_io_size) 

    print('1. setup data')
    test_loader, volume_shape, pad_size = get_input(args, model_io_size, 'test')

    print('2. setup model')
    models = [setup_model(args, device, model_io_size=model_io_size, exact=True)]

    result_path = args.output + '/' + args.exp_name + '/'
    result_file_pf = re.split('_|.pth', os.path.basename(args.pre_model))[-2]
    print(result_file_pf)
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    save_cmd_line(args, result_path + 'commandArgs.txt')  # Saving the command line args with machine name and time for later reference

    input_file_name = [os.path.basename(input_image)[:-3] for input_image in args.img_name.split('@')]

    print('3. start testing')
    result = test(args, test_loader, models, device, model_io_size, volume_shape, pad_size, result_path, result_file_pf, input_file_name, save_output)
  
    print('4. finish testing')
    return result

def run(input_args_string, save_output):
    return _run(get_args(mode='test', input_args=input_args_string), save_output)

if __name__ == "__main__":
    _run(get_args(mode='test'))