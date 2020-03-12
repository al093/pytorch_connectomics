import os, sys, argparse, h5py, time, pickle, nibabel, skimage
import numpy as np
import torch
from tqdm import tqdm

from torch_connectomics.utils.skeleton import *
from torch_connectomics.utils.vis import save_data, read_data, load_obj, save_obj
import testFlux, testSkeletonGrowing, testSkeleton

def my_bool(s):
    return s == 'True' or s == 'true' or s == '1'

def get_cmdline_args():
    parser = argparse.ArgumentParser(description='Specify if Flux Prediction/Splitting is to be skipped')
    parser.add_argument('--method', type=str, default='deepflux', help='Which method to run')
    parser.add_argument('--skip-inf', action='store_true', help='Skip running the deep net inference')
    parser.add_argument('--dataset', type=str, help='snemi | syntheticVessel')
    parser.add_argument('--tune', action='store_true', help='Will run inference on a range of threshold values and report the highest')
    return parser.parse_args()

########################################################################################################################
methods = ['deepflux', 'distanceTx', 'dilated', 'dvn']
args = get_cmdline_args()
if args.method not in methods:
    raise Exception('Specified method not in :', methods)

# generic arguments for running deepnet which are same for all methods
num_cpu=12
args_inference = ['-mi', '64,192,192', '-g', '1', '-c', '12', '-b', '12', '-ac', 'fluxNet', '-lm', 'True', '--task', '4',
                  '--in-channel', '1']
if args.method == 'deepflux':
    args_inference.extend(['--out-channel', '3'])
else:
    args_inference.extend(['--out-channel', '1'])

if args.dataset == 'snemi':
    matching_radius = 60.0
    resolution = (30.0, 6.0, 6.0)
    # SNEMI VAL VOLUME
    data_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/val/val_image_half.h5']
    gt_skel_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton.h5']
    gt_context_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton_context.h5']
    gt_skel_graphs_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/graph.h5']

    # SNEMI TRAIN VOLUME
    # data_path = ['/n/pfister_lab2/Lab/alok/snemi/train_image.h5']
    # gt_skel_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/skeleton.h5']
    # gt_context_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/skeleton_context.h5']
    # gt_skel_graphs_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/graph.h5']

    data_idxs = [0]
    output_base_path = '/n/pfister_lab2/Lab/alok/results/snemi/'
    if args.method == 'deepflux':
        exp_name = 'snemi_abStudy_deepflux_valVol'
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_DeepFlux_2/snemi_abStudy_DeepFlux_2_78000.pth'
    elif args.method == 'dilated':
        exp_name = 'snemi_abStudy_dilated_valVol'
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_binary_skeleton/snemi_abStudy_binary_skeleton_120000.pth'
    elif args.method == 'distanceTx':
        exp_name = 'snemi_abStudy_distanceTx_valVol'
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_skeleton/snemi_abStudy_skeleton_120000.pth'

    if args.method == 'deepflux':
        var_param_list = [0.15] if args.tune is False else np.arange(0.10, 0.95, 0.05) #tuned
    elif args.method == 'distanceTx':
        var_param_list = [0.45] if args.tune is False else np.arange(0.30, 0.60, 0.05) #tuned
    elif args.method == 'dilated':
        var_param_list = [0.90] if args.tune is False else np.arange(0.50, 0.95, 0.05) #tuned
    else:
        raise Exception('Method {:%s} is not defined'.format(args.method))

elif args.dataset == 'syntheticVessel':
    matching_radius = 1.0
    resolution = (1.0, 1.0, 1.0)
    with open('/n/home11/averma/pytorch_connectomics/cmdArgs/synVesselPaths.pkl', 'rb') as phandle:
        syn_paths = pickle.load(phandle)
    data_path, gt_skel_path, gt_context_path, gt_skel_graphs_path = [], [], [], []
    data_idxs = list(range(21, 26))
    for i, vol_data in syn_paths.items():
        if i in data_idxs:
            data_path.append(vol_data['dn'])
            gt_skel_path.append(vol_data['skn'])
            gt_context_path.append(vol_data['ln'])
            gt_skel_graphs_path.append(vol_data['gn'])

    output_base_path = '/n/pfister_lab2/Lab/alok/results/syntheticVessel/'
    if args.method == 'deepflux':
        exp_name = 'synVessel_abStudy_deepflux_valVol'
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/syntheticVessel/synVessel_abStudy_deepflux/synVessel_abStudy_deepflux_120000.pth'
    elif args.method == 'dilated':
        exp_name = 'synVessel_abStudy_dilated_valVol'
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/syntheticVessel/synVessel_abStudy_skeleton/synVessel_abStudy_skeleton_120000.pth'
    elif args.method == 'distanceTx':
        exp_name = 'synVessel_abStudy_distanceTx_valVol'
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/syntheticVessel/synVessel_abStudy_distanceTx/synVessel_abStudy_distanceTx_120000.pth'
    elif args.method == 'dvn':
        exp_name = 'synVessel_abStudy_dvn_valVol'
        basename = '/n/pfister_lab2/Lab/alok/results/syntheticVessel/dvn/sample/pred/cen_on_synthetic_test_prob_'
        dvn_files = [basename + str(i) + '.nii.gz' for i in data_idxs]
    if args.method == 'deepflux':
        var_param_list = [0.80] if args.tune is False else np.arange(0.10, 0.95, 0.05) #tuned
    elif args.method == 'distanceTx':
        var_param_list = [0.05] if args.tune is False else np.arange(0.05, 0.95, 0.05) #tuned
    elif args.method == 'dilated':
        var_param_list = [0.90] if args.tune is False else np.arange(0.05, 0.95, 0.05) #tuned
    elif args.method == 'dvn':
        var_param_list = [0.05] if args.tune is False else np.arange(0.05, 0.50, 0.05)
    else:
        raise Exception('Method {:%s} is not defined'.format(args.method))

temp_folder = output_base_path + exp_name + '/temp'
if not os.path.isdir(temp_folder):
    os.makedirs(temp_folder)

if args.method == 'dvn':
    all_errors = {}
    for var_param in var_param_list:
        lmd = var_param
        print('Threshold is :', lmd)
        skeletons = []
        for i, data_idx in enumerate(data_idxs):
            pred = np.array(nibabel.load(dvn_files[i]).dataobj)
            pred = (pred > lmd)
            skeleton = skimage.measure.label(pred, return_num=False).astype(np.uint16)
            min_skel_threshold = 400
            skeleton = remove_small_skeletons(skeleton, min_skel_threshold)
            skeletons.append(skeleton)
            save_data(skeleton, output_base_path + exp_name + '/' + str(data_idx) + '_skeletons_' +
                      '{:.2f}'.format(lmd) + '.h5')
        if args.tune is True:
            print('Computing Errors.')
            errors = calculate_errors_batch(list(zip(skeletons)), gt_skel_path, gt_context_path, resolution,
                                            temp_folder, num_cpu, matching_radius)
            all_errors[lmd] = errors[0]
else:
    # Run Model
    if args.skip_inf == False:
        args_inference.extend(['-o', output_base_path])
        args_inference.extend(['-pm', model_path])
        args_inference.extend(['-en', exp_name])
        args_inference.extend(['-dn', '@'.join(data_path)])
        print('Running Model')
        if args.method == methods[0]:
            prediction = testFlux.run(args_inference, save_output=False)
            for i, pred_i in enumerate(prediction):
                save_data(pred_i, output_base_path + exp_name + '/' + str(data_idxs[i]) + '_flux.h5')
        else:
            prediction = testSkeleton.run(args_inference, save_output=False)
            for i, pred_i in enumerate(prediction):
                save_data(pred_i, output_base_path + exp_name + '/' + str(data_idxs[i]) + '_skeleton_prob.h5')
    else:
        prediction = []
        for i in data_idxs:
            if args.method == methods[0]:
                prediction.append(read_data(output_base_path + exp_name + '/' + str(i) + '_flux.h5'))
            else:
                prediction.append(read_data(output_base_path + exp_name + '/' + str(i) + '_skeleton_prob.h5'))

if args.method == 'deepflux':
    if args.dataset == 'snemi':
        dilation_filter_sz = 3
        erosion_filter_sz = 4
    elif args.dataset == 'syntheticVessel':
        dilation_filter_sz = 3
        erosion_filter_sz = 4
    else:
        raise Exception('Not defined')
    all_errors = {}
    binned_directions = [None]*len(prediction)
    for var_param in var_param_list:
        lmd = var_param
        print('Threshold is :', lmd)
        skeletons = []
        for i, pred_i in enumerate(prediction):
            skeleton, bd = compute_skeleton_like_deepflux(pred_i, lmd=lmd, k1=dilation_filter_sz,
                                                                         k2=erosion_filter_sz,
                                                                         binned_directions=binned_directions[i])
            binned_directions[i] = bd
            min_skel_threshold = 400
            skeleton = remove_small_skeletons(skeleton, min_skel_threshold)
            skeletons.append(skeleton)
            # save_data(skeleton, output_base_path + exp_name + '/' + str(data_idxs[i]) + '_skeletons_' +  '{:.2f}'.format(lmd) + '.h5')
        if args.tune is True:
            print('Computing Errors.')
            errors = calculate_errors_batch(list(zip(skeletons)), gt_skel_path, gt_context_path,resolution,
                                            temp_folder, num_cpu, matching_radius)
            all_errors[lmd] = errors[0]
elif args.method == 'distanceTx' or args.method == 'dilated':
    dilation_filter_sz = 3
    erosion_filter_sz = 4
    all_errors = {}
    for var_param in var_param_list:
        lmd = var_param
        print('Threshold is :', lmd)
        skeletons = []
        for i, pred_i in enumerate(prediction):
            skeleton = compute_skeleton_from_scalar_field(pred_i, method=args.method, threshold=lmd,
                                                          k1=dilation_filter_sz, k2=erosion_filter_sz)
            min_skel_threshold = 400
            skeleton = remove_small_skeletons(skeleton, min_skel_threshold)
            skeletons.append(skeleton)
            # save_data(skeleton, output_base_path + exp_name + '/' + str(data_idxs[i]) + '_skeletons_' + '{:.2f}'.format(lmd) + '.h5')
        if args.tune is True:
            print('Computing Errors.')
            errors = calculate_errors_batch(list(zip(skeletons)), gt_skel_path, gt_context_path, resolution,
                                            temp_folder, num_cpu, matching_radius)
            all_errors[lmd] = errors[0]

if args.tune:
    max_score = -1
    max_key = None
    for key, error in all_errors.items():
        if error['hm'] > max_score:
            max_score = error['hm']
            max_key = key

    print('Method: {}, Dataset: {}'.format(args.method, args.dataset))
    print('Best score with threshold: ', max_key)
    print('P:    ' + '{:3.4f}'.format(all_errors[max_key]['p']))
    print('R:    ' + '{:3.4f}'.format(all_errors[max_key]['r']))
    print('PR:   ' + '{:3.4f}'.format(all_errors[max_key]['pr']))
    print('C:    ' + '{:3.4f}'.format(all_errors[max_key]['c']))
    print('PRC:  ' + '{:3.4f}'.format(all_errors[max_key]['hm']))
else:
    print('Computing Errors.')
    errors = calculate_errors_batch(list(zip(skeletons)), gt_skel_path, gt_context_path, resolution, temp_folder, num_cpu, matching_radius)