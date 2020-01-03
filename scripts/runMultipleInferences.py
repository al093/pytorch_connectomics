import h5py
import os, sys, subprocess
import numpy as np
from torch_connectomics.model.loss import *
import torch
from tensorboardX import SummaryWriter

arguments = [sys.executable, '/n/home11/averma/pytorch_connectomics/scripts/testFlux.py',
'-o', '/n/pfister_lab2/Lab/alok/results/snemi/',
'-mi', '64,192,192',
'-g', '2',
'-c', '12',
'-b', '50',
'-ac', 'fluxNet',
'-lm', 'True',
'-dn', '/n/pfister_lab2/Lab/alok/snemi/skeleton/val/val_image_half.h5',
'--task', '4',
'--in-channel', '1',
'--out-channel', '3']

exp_name = 'snemi_onlyFlux_moreFilters_highDR_moreElastic'
model_bp = '/n/home11/averma/pytorch_connectomics/outputs/snemi/'
model_bp2 = exp_name + '/' + exp_name
#
itrs = np.arange(50000, 84000+1, 2000)
for itr in itrs:
    model_path = ['-pm', model_bp + model_bp2 + '_' + str(itr) + '.pth']
    print(arguments + model_path)
    p = subprocess.Popen(arguments + model_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while p.poll() is None:
        l = p.stdout.readline()
        print(l.decode('ascii'))
        l = p.stderr.readline()
        print(l.decode('ascii'))
    print(p.stdout.read())
    print(p.stderr.read())
    if p.returncode != 0:
        print('Error encountered while processing: ', itr)
        break

result_bp = '/n/pfister_lab2/Lab/alok/results/snemi/'
writer = SummaryWriter(result_bp + exp_name + '_validation')
with torch.no_grad():
    gt_flux = np.asarray(h5py.File('/n/pfister_lab2/Lab/alok/snemi/skeleton/val/1x/grad_distance_context.h5', 'r')['main'][:, 0:75])
    gt_flux = torch.from_numpy(gt_flux)
    gt_context = np.asarray(h5py.File('/n/pfister_lab2/Lab/alok/snemi/skeleton/val/1x/skeleton_context.h5', 'r')['main'][0:75])
    gt_context = torch.from_numpy((gt_context>0).astype(np.float32))
    errorCriteria = AngularAndScaleLoss(alpha=0.08, dim=0)
    itrs = np.arange(0, 84000 + 1, 2000)
    print('itr, flux_loss, angular_l, scale_l')
    for itr in itrs:
        result_path = result_bp + exp_name + '/gradient_0_' + str(itr) + '.h5'
        pred_flux = torch.from_numpy(np.asarray(h5py.File(result_path, 'r')['main']))
        flux_loss, angular_l, scale_l = errorCriteria(pred_flux, gt_flux, weight=gt_context)
        print(itr, flux_loss.item(), angular_l.item(), scale_l.item())
        writer.add_scalars('Part-wise Losses',
                           {'Angular': angular_l.item(),
                            'Scale': scale_l.item()}, itr)
        writer.add_scalars('Loss', {'Overall Loss': flux_loss.item()}, itr)

writer.close()