import h5py, pickle
import os, sys, subprocess
import numpy as np
from torch_connectomics.model.loss import *
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torch_connectomics.utils.skeleton import *

arguments = [sys.executable, '/n/home11/averma/pytorch_connectomics/scripts/testFlux.py',
'-o', '/n/pfister_lab2/Lab/alok/results/snemi/',
'-mi', '64,192,192',
'-g', '1',
'-c', '12',
'-b', '24',
'-ac', 'fluxNet',
'-lm', 'True',
'--task', '4',
'--in-channel', '1',
'--out-channel', '3']

exp_name = 'snemi_abStudy_interpolated+gradient'
model_bp = '/n/home11/averma/pytorch_connectomics/outputs/snemi/'
model_bp2 = exp_name + '/' + exp_name
arguments.extend(['-en', exp_name + '_multiple_inferences'])
data_path = '/n/pfister_lab2/Lab/alok/snemi/skeleton/val/val_image_half.h5'
data_filename = os.path.basename(data_path)
arguments.extend(['-dn', data_path])
result_bp = '/n/pfister_lab2/Lab/alok/results/snemi/'
temp_folder = result_bp + exp_name
resolution = (30.0, 6.0, 6.0)
gt_flux= np.asarray(h5py.File('/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/grad_distance_context.h5', 'r')['main'])
gt_context = np.asarray(h5py.File('/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton_context.h5', 'r')['main'])
gt_skel = np.asarray(h5py.File('/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton.h5', 'r')['main'])
gt_skel_graphs_path = '/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/graph.h5'
with open(gt_skel_graphs_path, 'rb') as phandle:
    gt_skel_graphs = pickle.load(phandle)

eval = False
if eval == True:
    itrs = np.arange(110000, 180000, 2000)
    for itr in itrs:
        model_path = ['-pm', model_bp + model_bp2 + '_' + str(itr) + '.pth']
        print(arguments + model_path)
        p = subprocess.Popen(arguments + model_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while p.poll() is None:
            l = p.stdout.readline()
            print(l.decode('ascii').replace('\n', '####'))
            l = p.stderr.readline()
            print(l.decode('ascii').replace('\n', '####'))

        l = p.stdout.read()
        print(l.decode('ascii').replace('\n', '####'))
        l = p.stderr.read()
        print(l.decode('ascii').replace('\n', '####'))

        if p.returncode != 0:
            print('Error encountered while processing: ', itr)
            break

eval = False
if eval == True:
    itrs = np.arange(110000, 180000, 2000)
    writer = SummaryWriter(result_bp + exp_name + '_multiple_inferences')
    with torch.no_grad():
        errorCriteria = AngularAndScaleLoss(alpha=0.08, dim=0)
        gt_flux = torch.from_numpy(gt_flux_path)
        gt_context = torch.from_numpy((gt_context_path > 0).astype(np.float32))
        for itr in tqdm(itrs):
            result_path = result_bp + exp_name + '_multiple_inferences' + '/gradient_' + data_filename[:-3] + '_' + str(itr) + '.h5'
            pred_flux = torch.from_numpy(np.asarray(h5py.File(result_path, 'r')['main']))
            flux_loss, angular_l, scale_l = errorCriteria(pred_flux, gt_flux, weight=gt_context)
            print(itr, flux_loss.item(), angular_l.item(), scale_l.item())
            writer.add_scalars('Part-wise Losses',
                               {'Angular': angular_l.item(),
                                'Scale': scale_l.item()}, itr)
            writer.add_scalars('Loss', {'Overall Loss': flux_loss.item()}, itr)
    writer.close()

eval = True
if eval == True:
    #Precision, Recall, F-score, Connectivity calculator
    skel_params = {}
    skel_params['adaptive_threshold'] = 50
    skel_params['filter_size'] = 3
    skel_params['absolute_threshold'] = 0.25
    skel_params['min_skel_threshold'] = 400
    skel_params['block_size'] = [32, 100, 100] # Z, Y, X

    itrs = np.arange(100000, 270000, 2000)
    result_bp = '/n/pfister_lab2/Lab/alok/results/snemi/'
    writer = SummaryWriter(result_bp + exp_name + '_multiple_inferences')

    gt_skel_ids = np.unique(gt_skel)
    gt_skel_ids = gt_skel_ids[gt_skel_ids>0]
    # TODO remove that
    temp_folder = temp_folder + '\splittemp'

    for itr in tqdm(itrs):
        result_path = result_bp + exp_name + '_multiple_inferences' + '/gradient_' + data_filename[:-3] + '_' + str(itr) + '.h5'
        pred_flux = torch.from_numpy(np.asarray(h5py.File(result_path, 'r')['main']))
        skeleton, _ = compute_skeleton_from_gradient(pred_flux, skel_params)

        downsample_factor = (1, 1, 1)
        # split_skeleton = split_skeletons(skeleton, skel_params['min_skel_threshold'], resolution, downsample_factor, temp_folder, num_cpu=8)
        # split_skeleton = split_skeletons(split_skeleton, skel_params['min_skel_threshold'], resolution, downsample_factor, temp_folder, num_cpu=8)
        # split_skeleton = split_skeletons(split_skeleton, skel_params['min_skel_threshold'], resolution, downsample_factor, temp_folder, num_cpu=8)

        p, r, f, c = calculate_error_metric_2(skeleton, gt_skel_graphs=gt_skel_graphs, gt_context=gt_context,
                                              gt_skel_ids=gt_skel_ids, resolution=resolution, temp_folder=temp_folder,
                                              num_cpu=12)
        writer.add_scalars('Precision/Recall/F-Score/Connectivity',
                           {'Precision': p, 'Recall': r, 'F-Score':f, 'Connectivity':c}, itr)
    writer.close()