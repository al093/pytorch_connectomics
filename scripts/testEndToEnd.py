import os, sys, argparse, h5py, time, pickle
import numpy as np
from tqdm import tqdm
import torch

from torch_connectomics.utils.skeleton import *
from torch_connectomics.utils.skeleton.computeParallel import compute_ibex_skeleton_graphs
from torch_connectomics.utils.vis import save_data, read_data
import testFlux, testSkeletonGrowing

def my_bool(s):
    return s == 'True' or s == 'true' or s == '1'

def get_cmdline_args():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--skip-flux',          action='store_true',  help='Skip Flux Computation')
    parser.add_argument('--skip-initial',  action='store_true',  help='Skip Flux to Skel step')
    parser.add_argument('--skip-splitting',     action='store_true',  help='Skip Splitting')
    parser.add_argument('--skip-tracking',      action='store_true',  help='Skip tracking')
    parser.add_argument('--stop-after-split',   action='store_true',  help='Stops script after splittting')
    parser.add_argument('--stop-after-initial', action='store_true',  help='Stops After flux and skeleton eval')
    parser.add_argument('--tune',               action='store_true',  help='Tune threshold')
    parser.add_argument('--dataset',            type=str,             help='snemi | syntheticVessel')
    parser.add_argument('--set',                type=str,             help='val | test')
    return parser.parse_args()

origin_time = time.time()
args = get_cmdline_args()

args_flux = ['-mi', '64,192,192', '-g', '1', '-c', '12', '-b', '32',
             '-ac', 'fluxNet', '-lm', 'True', '--task', '4', '--in-channel', '1', '--out-channel', '3']

if args.dataset == 'snemi':
    data_idxs = [0]
    resolution = np.uint16((30, 6, 6))
    matching_radius = 60.0
    ibex_downsample_fac = np.uint16((1, 5, 5))
    erl_overlap_allowance = np.int32((2, 5, 5))
    output_base_path = '/n/pfister_lab2/Lab/alok/results/snemi/'

    if args.set == 'val':
        data_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/val/val_image_half.h5']
        gt_skel_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton.h5']
        gt_context_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton_context.h5']
        gt_skel_graphs_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/graph.h5']
        exp_name = 'snemi_abStudy_ours_valVol_junctionsFocus'
    elif args.set == 'test':
        data_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/test/image.h5']
        gt_skel_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/test/1_0x/skeleton.h5']
        gt_context_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/test/1_0x/skeleton_context.h5']
        gt_skel_graphs_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/test/1_0x/graph.h5']
        exp_name = 'snemi_abStudy_ours_testVol_old'
    elif args.set == 'train':
        data_path = ['/n/pfister_lab2/Lab/alok/snemi/train_image.h5']
        gt_skel_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/skeleton.h5']
        gt_context_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/skeleton_context.h5']
        gt_skel_graphs_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/graph.h5']
        exp_name = 'snemi_abStudy_ours_trainVol_junctionsFocus'
    else:
        raise Exception('Set not defined')

    flux_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_interpolated+gradient/snemi_abStudy_interpolated+gradient_120000.pth'
    # flux_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_junctions_focus/snemi_junctions_focus_74000.pth'
    # flux_model_path_tracking = flux_model_path

    # flux_model_path_tracking = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_interpolated+gradient/snemi_abStudy_interpolated+gradient_120000.pth'
    # tracking_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemiGrowing_32_steps_6/snemiGrowing_32_steps_6_20000.pth'

    # flux_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_junctions_focus/snemi_junctions_focus_74000.pth'
    # flux_model_path_tracking = flux_model_path
    # tracking_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_real_24_div_flux/snemi_real_24_div_flux_27000.pth'

    flux_model_path_tracking = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemiGrowing_32_steps_dropout_e2e_3/snemiGrowing_32_steps_dropout_e2e_3_fluxNet_30000.pth'
    tracking_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemiGrowing_32_steps_dropout_e2e_3/snemiGrowing_32_steps_dropout_e2e_3_30000.pth'

elif args.dataset == 'syntheticVessel':
    flux_model_path = '/n/home11/averma/pytorch_connectomics/outputs/syntheticVessel/synVessel_flux_ours_2/synVessel_flux_ours_2_110000.pth'

    output_base_path = '/n/pfister_lab2/Lab/alok/results/syntheticVessel/'
    resolution = np.uint16((1, 1, 1))
    matching_radius = 2.0
    erl_overlap_allowance = np.int32((2, 2, 2))
    ibex_downsample_fac = np.uint16((1, 5, 5))

    with open('/n/home11/averma/pytorch_connectomics/cmdArgs/synVesselPaths.pkl', 'rb') as phandle:
        syn_paths = pickle.load(phandle)
    data_path, gt_skel_path, gt_context_path, gt_skel_graphs_path = [], [], [], []
    if args.set == 'val':
        data_idxs = list(range(21, 26))
        exp_name = 'synVessel_abStudy_ours_valVols'
    elif args.set == 'test':
        data_idxs = list(range(26, 41))
        exp_name = 'synVessel_abStudy_ours_testVols'
    else:
        raise Exception('Set not defined')

    for i, vol_data in syn_paths.items():
        if i in data_idxs:
            data_path.append(vol_data['dn'])
            gt_skel_path.append(vol_data['skn'])
            gt_context_path.append(vol_data['ln'])
            gt_skel_graphs_path.append(vol_data['gn'])


elif args.dataset == 'mri':
    flux_model_path = '/n/home11/averma/pytorch_connectomics/outputs/mri/mri_ours_3/mri_ours_3_40000.pth'
    output_base_path = '/n/pfister_lab2/Lab/alok/results/mri/'
    resolution = np.uint16((1, 1, 1))
    matching_radius = 2.0

    with open('/n/home11/averma/pytorch_connectomics/cmdArgs/mriPaths.pkl', 'rb') as phandle:
        syn_paths = pickle.load(phandle)

    data_path, gt_skel_path, gt_context_path, gt_skel_graphs_path = [], [], [], []
    keys = list(syn_paths.keys()); keys.sort()
    if args.set == 'val':
        data_idxs = keys[28:29]
        exp_name = 'mri_abStudy_ours_valVols'
    elif args.set == 'test':
        data_idxs = keys[31:42]
        exp_name = 'mri_abStudy_ours_testVols'
    elif args.set == 'train':
        data_idxs = keys[10:26]
        exp_name = 'mri_abStudy_ours_trainVols'
    else:
        raise Exception('Set not defined')
    for i, vol_data in syn_paths.items():
        if i in data_idxs:
            data_path.append(vol_data['dn'])
            gt_skel_path.append(vol_data['skn'])
            gt_context_path.append(vol_data['ln'])
            gt_skel_graphs_path.append(vol_data['gn'])


elif args.dataset == 'liver':
    flux_model_path = '/n/pfister_lab2/Lab/meng/pytorch_connectomics/flux_output_1flux/flux_60000.pth'
    # tracking_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemiGrowing_32_steps_3/snemiGrowing_32_steps_3_12000.pth'
    output_base_path = '/n/pfister_lab2/Lab/alok/results/liver/'
    resolution = (1.0, 1.0, 1.0)
    matching_radius = 2.0

    with open('/n/home11/averma/pytorch_connectomics/cmdArgs/liverPaths.pkl', 'rb') as phandle:
        syn_paths = pickle.load(phandle)

    data_path, gt_skel_path, gt_context_path, gt_skel_graphs_path = [], [], [], []
    keys = list(syn_paths.keys()); keys.sort()
    if args.set == 'val':
        data_idxs = keys[9:13]
        exp_name = 'liver_abStudy_ours_valVols'
    elif args.set == 'test':
        data_idxs = keys[13:19]
        exp_name = 'liver_abStudy_ours_testVols'
    elif args.set == 'train':
        data_idxs = keys[0:9]
        exp_name = 'liver_abStudy_ours_trainVols'
    else:
        raise Exception('Set not defined')

    for i, vol_data in syn_paths.items():
        if i in data_idxs:
            data_path.append(vol_data['dn'])
            gt_skel_path.append(vol_data['skn'])
            gt_context_path.append(vol_data['ln'])
            gt_skel_graphs_path.append(vol_data['gn'])

if args.dataset == 'snemi':
    var_params = [0.55] if args.tune is False else np.arange(.50, .75, .05) # tuned for ERL
elif args.dataset == 'syntheticVessel':
    var_params = [0.80] if args.tune is False else np.arange(.10, .90, .05) # tuned for F1+ERL
elif args.dataset == 'mri':
    var_params = [0.75] if args.tune is False else np.arange(.10, .90, .05)
elif args.dataset == 'liver':
    var_params = [0.60] if args.tune is False else np.arange(.10, .90, .05)
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
if args.skip_initial == False:
    print('Computing skeleton from flux.')
    skel_params = {}
    skel_params['filter_size'] = 3
    skel_params['absolute_threshold'] = 0.25
    skel_params['block_size'] = [32, 100, 100]  # Z, Y, X
    all_errors = {}
    for var_param in var_params:
        print('Threshold value: ', var_param)
        skel_params['adaptive_threshold'] = 100 * var_param
        initial_skeletons = []
        for i, pred_flux_i in enumerate(pred_flux):
            skeleton, skel_divergence = compute_skeleton_from_gradient(pred_flux_i, skel_params)
            initial_skeletons.append(skeleton)
            save_data(skeleton, output_base_path + exp_name + '/' + str(data_idxs[i]) + '_initial_skeletons_' + '{:.2f}'.format(var_param) + '.h5')
            if var_param == var_params[0]:
                save_data(skel_divergence, output_base_path + exp_name + '/' + str(data_idxs[i]) + '_skeleton_divergence.h5')
        if args.tune is True:
            if args.dataset == 'mri':
                errors = calculate_binary_errors_batch(list(zip(initial_skeletons)), gt_skel_path, resolution, temp_folder, 0, matching_radius)
            else:
                errors = calculate_errors_batch(list(zip(initial_skeletons)), gt_skel_path, gt_context_path, resolution,
                                                temp_folder, 0, matching_radius, ibex_downsample_fac, erl_overlap_allowance)
            all_errors[var_param] = errors[0]

    #print the best score when tunining
    if args.tune:
        max_score = -1
        max_key = None
        metric = 'erl' if args.dataset != 'syntheticVessel' else 'pr'
        print('Maximizing metric: ', metric)
        for key, error in all_errors.items():
            if error[metric] > max_score:
                max_score = error[metric]
                max_key = key

        with open(output_base_path + exp_name + '/all_errors.pkl', 'wb') as pfile:
            pickle.dump(all_errors, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        print('Method: {}, Dataset: {}, Set: {}'.format('Ours', args.dataset, args.set))
        print('Best score with threshold: {:.3f}'.format(max_key))
        print('P:    ' + '{:3.4f}'.format(all_errors[max_key]['p']))
        print('R:    ' + '{:3.4f}'.format(all_errors[max_key]['r']))
        print('PR:   ' + '{:3.4f}'.format(all_errors[max_key]['pr']))
        if 'c' in all_errors[max_key].keys(): print('C:    ' + '{:3.4f}'.format(all_errors[max_key]['c']))
        if 'hm' in all_errors[max_key].keys(): print('PRC:  ' + '{:3.4f}'.format(all_errors[max_key]['hm']))
        if 'erl' in all_errors[max_key].keys(): print('ERL:   ' + '{:3.4f}'.format(all_errors[max_key]['erl']))
else:
    for i in data_idxs:
        initial_skeletons.append(read_data(output_base_path + exp_name + '/' + str(i) + '_initial_skeletons_' + '{:.2f}'.format(var_params[0]) + '.h5'))

if args.stop_after_initial is True:
    if args.tune is False:
        print('Computing Error.')
        print('Method: {}, Dataset: {}, Set: {}, Threshold: {:.3f}'.format('Ours', args.dataset, args.set, var_params[0]))
        if args.dataset == 'mri':
            errors_binary = calculate_binary_errors_batch(list(zip(skeletons)), gt_skel_path, resolution, temp_folder, 0, matching_radius)
        else:
            errors = calculate_errors_batch(list(zip(initial_skeletons)), gt_skel_path, gt_context_path, resolution,
                                            temp_folder, 0, matching_radius, ibex_downsample_fac, erl_overlap_allowance)
            errors_binary = calculate_binary_errors_batch(list(zip(initial_skeletons)), gt_skel_path, resolution, temp_folder,
                                                          0, matching_radius)
        sys.exit()
    else:
        sys.exit()

# Split skeletons
print('Splitting Skeletons.')
split_skeletons = []
if args.skip_splitting == False:
    for i, initial_skeleton in enumerate(tqdm(initial_skeletons)):
        min_skel_threshold = 400
        split_skeleton = initial_skeleton
        # split_skeleton = remove_small_skeletons(initial_skeleton, min_skel_threshold)
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
    print('Method: {}, Dataset: {}, Set: {}, Threshold: {:.3f}'.format('Ours', args.dataset, args.set, var_params[0]))
    if args.dataset == 'mri' and False:
        errors = calculate_binary_errors_batch(list(zip(initial_skeletons)), gt_skel_path, resolution, temp_folder, 0)
    else:
        errors = calculate_errors_batch(list(zip(initial_skeletons, split_skeletons)), gt_skel_path, gt_context_path,
                                        resolution, temp_folder, 0, matching_radius, ibex_downsample_fac, erl_overlap_allowance)
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
                         '--task', '6', '--in-channel', '14', '--out-channel', '3', '-lm', 'True', '-tsteps', '32', '-upc', 'True']
        args_tracking.extend(['-pm', tracking_model_path, '-o', output_base_path, '-en', exp_name, '-dn', data_path[i]])
        args_tracking.extend(['-pm_2', flux_model_path_tracking, '-sp', growing_data_file])
        args_tracking.extend(['-skn', output_base_path + exp_name + '/' + str(data_idxs[i]) + '_split_skeletons.h5'])
        args_tracking.extend(['-fn', output_base_path + exp_name + '/' + str(data_idxs[i]) + '_flux.h5'])
        args_tracking.extend(['-divn', output_base_path + exp_name + '/' + str(data_idxs[i]) + '_skeleton_divergence.h5'])

        tracking_result = testSkeletonGrowing.run(args_tracking, save_output=False)[0]
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
print('Method: {}, Dataset: {}, Set: {}, Threshold: {:.3f}'.format('Ours', args.dataset, args.set, var_params[0]))
if args.dataset == 'mri':
    errors = calculate_binary_errors_batch(list(zip(initial_skeletons)), gt_skel_path, resolution, temp_folder, 0)
else:
    errors = calculate_errors_batch(list(zip(initial_skeletons, split_skeletons, merged_skeletons)), gt_skel_path, gt_context_path,
                                    resolution, temp_folder, num_cpu, matching_radius, ibex_downsample_fac, erl_overlap_allowance)
    errors_binary = calculate_binary_errors_batch(list(zip(initial_skeletons, split_skeletons, merged_skeletons)), gt_skel_path, resolution, temp_folder,
                                                  0, matching_radius)