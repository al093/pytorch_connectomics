import os, sys, argparse, h5py, time, pickle
import numpy as np
from tqdm import tqdm
import torch

from torch_connectomics.utils.skeleton import *
from torch_connectomics.utils.vis import save_data, read_data
import testFlux, testSkeletonGrowing

def my_bool(s):
    return s == 'True' or s == 'true' or s == '1'

def get_cmdline_args():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--skip-flux',          action='store_true',  help='Skip Flux Computation')
    parser.add_argument('--skip-initial-skel',  action='store_true',  help='Skip Flux to Skel step')
    parser.add_argument('--skip-splitting',     action='store_true',  help='Skip Splitting')
    parser.add_argument('--skip-tracking',      action='store_true',  help='Skip tracking')
    parser.add_argument('--stop-after-split',   action='store_true',  help='Stops script after splittting')
    parser.add_argument('--stop-after-initial', action='store_true',  help='Stops After flux and skeleton eval')
    parser.add_argument('--tune',               action='store_true',  help='Tune threshold')
    parser.add_argument('--dataset',            type=str,             help='snemi | syntheticVessel')
    return parser.parse_args()

origin_time = time.time()
args = get_cmdline_args()

args_flux = ['-mi', '64,192,192', '-g', '1', '-c', '12', '-b', '12',
             '-ac', 'fluxNet', '-lm', 'True', '--task', '4', '--in-channel', '1', '--out-channel', '3']

if args.dataset == 'snemi':
    #-------------TRAIN--------------#
    # data_path = ['/n/pfister_lab2/Lab/alok/snemi/train_image.h5']
    # gt_skel_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/skeleton.h5']
    # gt_context_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/skeleton_context.h5']
    # gt_skel_graphs_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/graph.h5']
    # exp_name = 'snemi_abStudy_ours_train'
    # ---------------------------------

    #-----------VALIDATION------------#
    data_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/val/val_image_half.h5']
    gt_skel_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton.h5']
    gt_context_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton_context.h5']
    gt_skel_graphs_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/graph.h5']
    exp_name = 'snemi_abStudy_ours_val'
    # ---------------------------------

    data_idxs = [0]
    resolution = (30.0, 6.0, 6.0)
    matching_radius = 60.0
    flux_model_path =          '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_interpolated+gradient/snemi_abStudy_interpolated+gradient_120000.pth'
    flux_model_path_tracking = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_interpolated+gradient/snemi_abStudy_interpolated+gradient_120000.pth'
    # L2 Ablation model
    # flux_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_ours_onlyL2/snemi_abStudy_ours_onlyL2_120000.pth'

    tracking_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemiGrowing_32_steps_6/snemiGrowing_32_steps_6_20000.pth'
    output_base_path = '/n/pfister_lab2/Lab/alok/results/snemi/'
elif args.dataset == 'syntheticVessel':
    exp_name = 'synVessel_abStudy_ours_valVols'
    flux_model_path = '/n/home11/averma/pytorch_connectomics/outputs/syntheticVessel/synVessel_flux_ours_2/synVessel_flux_ours_2_110000.pth'
    # tracking_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemiGrowing_32_steps_3/snemiGrowing_32_steps_3_12000.pth'
    output_base_path = '/n/pfister_lab2/Lab/alok/results/syntheticVessel/'
    resolution = (1.0, 1.0, 1.0)
    matching_radius = 2.0

    with open('/n/home11/averma/pytorch_connectomics/cmdArgs/synVesselPaths.pkl', 'rb') as phandle:
        syn_paths = pickle.load(phandle)
    data_path, gt_skel_path, gt_context_path, gt_skel_graphs_path = [], [], [], []
    data_idxs = list(range(16, 21))
    for i, vol_data in syn_paths.items():
        if i in data_idxs:
            data_path.append(vol_data['dn'])
            gt_skel_path.append(vol_data['skn'])
            gt_context_path.append(vol_data['ln'])
            gt_skel_graphs_path.append(vol_data['gn'])
elif args.dataset == 'mri':
    exp_name = 'mri_abStudy_ours_valVols'
    flux_model_path = '/n/home11/averma/pytorch_connectomics/outputs/mri/mri_ours_3/mri_ours_3_12000.pth'
    # tracking_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemiGrowing_32_steps_3/snemiGrowing_32_steps_3_12000.pth'
    output_base_path = '/n/pfister_lab2/Lab/alok/results/mri/'
    resolution = (1.0, 1.0, 1.0)

    with open('/n/home11/averma/pytorch_connectomics/cmdArgs/mriPaths.pkl', 'rb') as phandle:
        syn_paths = pickle.load(phandle)

    data_path, gt_skel_path, gt_context_path, gt_skel_graphs_path = [], [], [], []
    keys = list(syn_paths.keys()); keys.sort()
    data_idxs = keys[26:31] # TODO|DEBUG change this to 26:31 for validation
    for i, vol_data in syn_paths.items():
        if i in data_idxs:
            data_path.append(vol_data['dn'])
            gt_skel_path.append(vol_data['skn'])
            gt_context_path.append(vol_data['ln'])
            gt_skel_graphs_path.append(vol_data['gn'])

if args.dataset == 'snemi':
    var_params = [0.60] if args.tune is False else np.arange(.50, .75, .05) # tuned
elif args.dataset == 'syntheticVessel':
    var_params = [0.35] if args.tune is False else np.arange(.20, .75, .05) # tuned
elif args.dataset == 'mri':
    var_params = [0.80] if args.tune is False else np.arange(.10, .95, .05)
else:
    raise Exception('dataset not defined')

num_cpu=12
args_flux.extend(['-o', output_base_path])
args_flux.extend(['-pm', flux_model_path])
args_flux.extend(['-en', exp_name])
args_flux.extend(['-dn', '@'.join(data_path)])

temp_folder = output_base_path + exp_name + '/temp'

# Run Flux Model
if args.skip_flux == False:
    print('Running Flux Net.')
    pred_flux = testFlux.run(args_flux, save_output=False)
    for i, pred_flux_i in enumerate(pred_flux):
        save_data(pred_flux_i, output_base_path + exp_name + '/' + str(data_idxs[i]) + '_flux.h5')
else:
    pred_flux = []
    for i in data_idxs:
        pred_flux.append(read_data(output_base_path + exp_name + '/' + str(i) + '_flux.h5'))

# Flux to skeleton
initial_skeletons = []
if args.skip_initial_skel == False:
    print('Computing skeleton from flux.')
    skel_params = {}
    skel_params['filter_size'] = 3
    skel_params['absolute_threshold'] = 0.25
    skel_params['block_size'] = [32, 100, 100]  # Z, Y, X
    all_errors = {}
    for var_param in var_params: #[0.60] for snemi
        print('Threshold value: ', var_param)
        skel_params['adaptive_threshold'] = 100 * var_param
        initial_skeletons = []
        for i, pred_flux_i in enumerate(pred_flux):
            skeleton, skel_divergence = compute_skeleton_from_gradient(pred_flux_i, skel_params)
            initial_skeletons.append(skeleton)
            save_data(skel_divergence, output_base_path + exp_name + '/' + str(data_idxs[i]) + '_skeleton_divergence_' + '{:.2f}'.format(var_param) + '.h5')
            save_data(skeleton, output_base_path + exp_name + '/' + str(data_idxs[i]) + '_initial_skeletons_' + '{:.2f}'.format(var_param) + '.h5')
        if args.tune is True:
            if args.dataset == 'mri':
                errors = calculate_binary_errors_batch(list(zip(initial_skeletons)), gt_skel_path, resolution,
                                                temp_folder, 0)
            else:
                errors = calculate_errors_batch(list(zip(initial_skeletons)), gt_skel_path, gt_context_path, resolution, temp_folder, 0, matching_radius)
            all_errors[var_param] = errors[0]

    #print the best score when tunining
    if args.tune:
        max_score = -1
        max_key = None
        for key, error in all_errors.items():
            metric = 'pr' if args.dataset == 'mri' else 'hm'
            if error[metric] > max_score:
                max_score = error[metric]
                max_key = key

        print('Method: {}, Dataset: {}'.format('Ours', args.dataset))
        print('Best score for initial skeletons is: ')
        print('Threshold was: {:.3f}'.format(max_key))
        print('P:    ' + '{:3.4f}'.format(all_errors[max_key]['p']))
        print('R:    ' + '{:3.4f}'.format(all_errors[max_key]['r']))
        print('PR:   ' + '{:3.4f}'.format(all_errors[max_key]['pr']))
        if args.dataset != 'mri':
            print('C:    ' + '{:3.4f}'.format(all_errors[max_key]['c']))
            print('PRC:  ' + '{:3.4f}'.format(all_errors[max_key]['hm']))
else:
    for i in data_idxs:
        initial_skeletons.append(read_data(output_base_path + exp_name + '/' + str(i) + '_initial_skeletons_' + '{:.2f}'.format(var_params[0]) + '.h5'))

if args.stop_after_initial is True:
    if args.tune is False:
        print('Computing Error.')
        if args.dataset == 'mri':
            errors = calculate_binary_errors_batch(list(zip(initial_skeletons)), gt_skel_path, resolution, temp_folder, 0)
        else:
            errors = calculate_errors_batch(list(zip(initial_skeletons)), gt_skel_path, gt_context_path, resolution, temp_folder, 0, matching_radius)
        sys.exit()
    else:
        sys.exit()

# Split skeletons
print('Splitting Skeletons.')
split_skeletons = []
if args.skip_splitting == False:
    for i, initial_skeleton in enumerate(tqdm(initial_skeletons)):
        min_skel_threshold = 400
        split_skeleton = remove_small_skeletons(initial_skeleton, min_skel_threshold)
        downsample_factor = (1, 1, 1)
        split_skeleton = split(split_skeleton, min_skel_threshold, resolution, downsample_factor, temp_folder, num_cpu=num_cpu)
        split_skeleton = split(split_skeleton, min_skel_threshold, resolution, downsample_factor, temp_folder, num_cpu=num_cpu)
        split_skeletons.append(split_skeleton)
        save_data(split_skeleton, output_base_path + exp_name + '/' + str(data_idxs[i]) + '_split_skeletons.h5')
else:
    for i in data_idxs:
        split_skeletons.append(read_data(output_base_path + exp_name + '/' + str(i) + '_split_skeletons.h5'))

if args.stop_after_split is True:
    print('Computing Error.')
    if args.dataset == 'mri':
        errors = calculate_binary_errors_batch(list(zip(initial_skeletons)), gt_skel_path, resolution, temp_folder, 0)
    else:
        errors = calculate_binary_errors_batch(list(zip(initial_skeletons, split_skeletons)), gt_skel_path, resolution, temp_folder, 0, matching_radius)
    sys.exit()

predicted_paths = []
if args.skip_tracking == False:
    # calulate and save the end points of split skeleton for testing
    print('Generating skeleton growing data for Deep Tracking network.')
    for i, split_skeleton in enumerate(tqdm(split_skeletons)):
        growing_data_file = output_base_path + exp_name + '/' + str(data_idxs[i]) + '_growing_data.h5'
        downsample_factor = (1, 1, 1)
        generate_skeleton_growing_data(split_skeleton, growing_data_file, resolution, downsample_factor, temp_folder, num_cpu=num_cpu)

        # Run Deep Tracking Network
        print('Running Deep Tracking Network')
        args_tracking = ['-mi', '16,96,96', '-g', '1', '-c', '0', '-b', '32', '-ac', 'directionNet',
                         '--task', '6', '--in-channel', '14', '--out-channel', '3', '-lm', 'True']
        args_tracking.extend(['-pm', tracking_model_path, '-o', output_base_path, '-en', exp_name, '-dn', data_path[i]])
        args_tracking.extend(['-pm_2', flux_model_path_tracking, '-sp', growing_data_file])
        args_tracking.extend(['-skn', output_base_path + exp_name + '/' + str(data_idxs[i]) + '_split_skeletons.h5'])
        args_tracking.extend(['-fn', output_base_path + exp_name + '/' + str(data_idxs[i]) + '_flux.h5'])

        tracking_result = testSkeletonGrowing.run(args_tracking, save_output=False)
        predicted_paths.append(tracking_result)
        with open(output_base_path + exp_name + '/' + str(data_idxs[i]) + '_predicted_paths.pkl', 'wb') as pfile:
            pickle.dump(tracking_result, pfile, protocol=pickle.HIGHEST_PROTOCOL)
else:
    for i in data_idxs:
        with open(output_base_path + exp_name + '/' + str(i) + '_predicted_paths.pkl', 'rb') as phandle:
             predicted_paths.append(pickle.load(phandle))

# Merge skeletons based on predicted paths
print('Merging skeletons.')
merged_skeletons = []
for i, di in enumerate(tqdm(data_idxs)):
    merged_skeleton = merge(split_skeletons[i], predicted_paths[i])
    save_data(merged_skeleton, output_base_path + exp_name + '/' + str(di) + '_merged_skeletons.h5')
    merged_skeletons.append(merged_skeleton)

print('Computing Errors.')
if args.dataset == 'mri':
    errors = calculate_binary_errors_batch(list(zip(initial_skeletons)), gt_skel_path, resolution, temp_folder, 0)
else:
    errors = calculate_errors_batch(list(zip(initial_skeletons, split_skeletons, merged_skeletons)), gt_skel_path, gt_context_path,
                                    resolution, temp_folder, num_cpu, matching_radius)