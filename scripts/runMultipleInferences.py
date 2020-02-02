import h5py
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
'-b', '16',
'-ac', 'fluxNet',
'-lm', 'True',
'--task', '4',
'--in-channel', '1',
'--out-channel', '3']

exp_name = 'snemi_abStudy_interpolated+gradient'
model_bp = '/n/home11/averma/pytorch_connectomics/outputs/snemi/'
model_bp2 = exp_name + '/' + exp_name
arguments.extend(['-en', exp_name + '_multiple_inferences_2'])
data_path = '/n/pfister_lab2/Lab/alok/snemi/skeleton/val/val_image_half.h5'
data_filename = os.path.basename(data_path)
arguments.extend(['-dn', data_path])

eval = True
if eval == True:
    itrs = np.arange(30000, 100000, 10000)
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

eval = True
if eval == True:
    itrs = np.arange(30000, 100000, 10000)
    result_bp = '/n/pfister_lab2/Lab/alok/results/snemi/'
    writer = SummaryWriter(result_bp + exp_name + '_multiple_inferences_2')
    with torch.no_grad():
        gt_flux = np.asarray(h5py.File('/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/grad_distance_context.h5', 'r')['main'])
        gt_flux = torch.from_numpy(gt_flux)
        gt_context = np.asarray(h5py.File('/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton_context.h5', 'r')['main'])
        gt_context = torch.from_numpy((gt_context>0).astype(np.float32))
        errorCriteria = AngularAndScaleLoss(alpha=0.08, dim=0)
        for itr in tqdm(itrs):
            result_path = result_bp + exp_name + '_multiple_inferences_2' + '/gradient_' + data_filename[:-3] + '_' + str(itr) + '.h5'
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
    resolution = (30.0, 6.0, 6.0)

    itrs = np.arange(30000, 100000, 10000)
    result_bp = '/n/pfister_lab2/Lab/alok/results/snemi/'
    writer = SummaryWriter(result_bp + exp_name + '_multiple_inferences_2')

    gt_skel = np.asarray(h5py.File('/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton.h5', 'r')['main'])
    gt_context = np.asarray(h5py.File('/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton_context.h5', 'r')['main'])
    gt_skel_ids = np.unique(gt_skel)
    gt_skel_ids = gt_skel_ids[gt_skel_ids>0]

    for itr in tqdm(itrs):
        result_path = result_bp + exp_name + '_multiple_inferences_2' + '/gradient_' + data_filename[:-3] + '_' + str(itr) + '.h5'
        pred_flux = torch.from_numpy(np.asarray(h5py.File(result_path, 'r')['main']))
        skeleton = compute_skeleton_from_gradient(pred_flux, skel_params)
        p, r, f, c = calculate_error_metric(skeleton, gt_skel=gt_skel, gt_context=gt_context, gt_skel_ids=gt_skel_ids, anisotropy=resolution)
        writer.add_scalars('Precision/Recall/F-Score/Connectivity',
                           {'Precision': p, 'Recall': r, 'F-Score':f, 'Connectivity':c}, itr)
    writer.close()

