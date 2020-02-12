'''
1. Take a model paths: flux/other baseline model and growing model

2. Run flux/other baseline model and save output

3. run skeletonization code
    also run error eval if needed

4. Run splitting code
    also run error evaluation code if needed

4. save end points seed file needed for growing model

5. Run growing model and save results

6. Merge skeletong
    run error evaluation code
    save results

save error and other run parameters in a dict
'''

import os, sys, argparse, h5py, time, pickle
import numpy as np
import torch

from torch_connectomics.utils.skeleton import *
from torch_connectomics.utils.vis import save_data, read_data
import testFlux, testSkeletonGrowing

def my_bool(s):
    return s == 'True' or s == 'true' or s == '1'

def get_cmdline_args():
    parser = argparse.ArgumentParser(description='Specify if Flux Prediction/Splitting is to be skipped')
    parser.add_argument('--skip-flux', type=my_bool, default=False,
                        help='Skip Flux Computation')
    parser.add_argument('--skip-splitting', type=my_bool, default=False,
                        help='Skip Flux and splitting')
    parser.add_argument('--skip-tracking', type=my_bool, default=False,
                        help='Skip tracking, only merge' )
    return parser.parse_args()


origin_time = time.time()
args = get_cmdline_args()

args_flux = ['-mi', '64,192,192', '-g', '1', '-c', '12', '-b', '32',
             '-ac', 'fluxNet', '-lm', 'True', '--task', '4', '--in-channel', '1', '--out-channel', '3']

# data_path = '/n/pfister_lab2/Lab/alok/snemi/skeleton/val/val_image_half.h5'
# gt_skel_path = '/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton.h5'
# gt_context_path = '/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton_context.h5'
# gt_skel_graphs_path = '/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/graph.h5'
# exp_name = 'snemi_abStudy_fluxOnly_valVol'

data_path = '/n/pfister_lab2/Lab/alok/snemi/train_image.h5'
gt_skel_path = '/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/skeleton.h5'
gt_context_path = '/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/skeleton_context.h5'
gt_skel_graphs_path = '/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/graph.h5'
exp_name = 'snemi_abStudy_fluxOnly_trainVol'

args_flux.extend(['-en', exp_name])
output_base_path = '/n/pfister_lab2/Lab/alok/results/snemi/'
args_flux.extend(['-o', output_base_path])
args_flux.extend(['-dn', data_path])
flux_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_interpolated+gradient/snemi_abStudy_interpolated+gradient_120000.pth'
args_flux.extend(['-pm', flux_model_path])
num_cpu=10

# Run Flux Model
if args.skip_flux == False:
    print('Running Flux Net.')
    pred_flux = testFlux.run(args_flux, save_output=False)[0]
    save_data(pred_flux, output_base_path + exp_name + '/flux.h5')
else:
    pred_flux = read_data(output_base_path + exp_name + '/flux.h5')

temp_folder = output_base_path + exp_name + '/temp'
gt_skel = np.asarray(h5py.File(gt_skel_path, 'r')['main'])
gt_context = np.asarray(h5py.File(gt_context_path, 'r')['main'])
with open(gt_skel_graphs_path, 'rb') as phandle:
    gt_skel_graphs = pickle.load(phandle)

gt_skel_ids = np.unique(gt_skel)
gt_skel_ids = gt_skel_ids[gt_skel_ids > 0]
resolution = (30.0, 6.0, 6.0)
downsample_factor = (1, 1, 1)

if args.skip_splitting == False:
    # Flux to skeleton
    skel_params = {}
    skel_params['adaptive_threshold'] = 50
    skel_params['filter_size'] = 3
    skel_params['absolute_threshold'] = 0.25
    skel_params['min_skel_threshold'] = 400
    skel_params['block_size'] = [32, 100, 100]  # Z, Y, X
    print('Computing skeleton from flux.')
    start_time = time.time()
    skeleton, flux_div = compute_skeleton_from_gradient(pred_flux, skel_params)
    print('Time taken: ', time.time() - start_time)
    save_data(flux_div, output_base_path + exp_name + '/flux_divergence.h5')
    save_data(skeleton, output_base_path + exp_name + '/initial_skeletons.h5')
    # Calculate error metric
    print('Calculating error for initial skeletons')
    start_time = time.time()
    p, r, f, c = calculate_error_metric_2(skeleton, gt_skel_graphs=gt_skel_graphs, gt_skel_ids=gt_skel_ids,
                                          gt_context=gt_context, resolution=resolution, temp_folder=temp_folder, num_cpu=num_cpu)

    print('Time taken: ', time.time() - start_time)

    # Split skeletons
    print('Splitting skeletons.')
    start_time = time.time()
    split_skeleton = split_skeletons(skeleton, skel_params['min_skel_threshold'], resolution, downsample_factor, temp_folder, num_cpu=num_cpu)
    split_skeleton = split_skeletons(split_skeleton, skel_params['min_skel_threshold'], resolution, downsample_factor, temp_folder, num_cpu=num_cpu)
    save_data(split_skeleton, output_base_path + exp_name + '/split_skeletons.h5')
    print('Time taken: ', time.time() - start_time)
else:
    split_skeleton = read_data(output_base_path + exp_name + '/split_skeletons.h5')

start_time = time.time()
print('Calculating error for split skeletons.')

p_s, r_s, f_s, c_s = calculate_error_metric_2(split_skeleton, gt_skel_graphs=gt_skel_graphs, gt_skel_ids=gt_skel_ids,
                                              gt_context=gt_context, resolution=resolution, temp_folder=temp_folder, num_cpu=num_cpu)
print('Time taken: ', time.time() - start_time)

# save_data(vol_1, output_base_path + exp_name + '/vol_1.h5')
# save_data(vol_2, output_base_path + exp_name + '/vol_2.h5')
# save_data(vol_3, output_base_path + exp_name + '/vol_3.h5')
# import pdb; pdb.set_trace()

if args.skip_tracking is False:
    # calulate and save the end points of split skeleton for testing
    print('Generating skeleton growing data for Deep Tracking network.')
    start_time = time.time()
    growing_data_file = output_base_path + exp_name + '/growing_data.h5'
    generate_skeleton_growing_data(split_skeleton, growing_data_file, resolution, downsample_factor, temp_folder, num_cpu=num_cpu)
    print('Time taken: ', time.time() - start_time)

    # Run Deep Tracking Network
    print('Running Deep Tracking Network.')
    args_tracking = ['-mi', '16,96,96',
                        '-g', '1',
                        '-c', '0',
                        '-b', '32',
                        '-ac', 'directionNet',
                        '--task', '6',
                        '--in-channel', '14',
                        '--out-channel', '3',
                        '-lm', 'True']

    tracking_model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemiGrowing_32_steps_3/snemiGrowing_32_steps_3_2500.pth'
    args_tracking.extend(['-pm', tracking_model_path])
    args_tracking.extend(['-o', output_base_path])
    args_tracking.extend(['-en', exp_name])
    args_tracking.extend(['-dn', data_path])
    args_tracking.extend(['-pm_2', flux_model_path])
    args_tracking.extend(['-sp', growing_data_file])
    args_tracking.extend(['-skn', output_base_path + exp_name + '/split_skeletons.h5'])
    args_tracking.extend(['-fn', output_base_path + exp_name + '/flux.h5'])
    start_time = time.time()
    predicted_paths = testSkeletonGrowing.run(args_tracking, save_output=True)
    print('Time taken: ', time.time() - start_time)
else:
    predicted_paths = testSkeletonGrowing.load_tracking_results(output_base_path + exp_name + '/predicted_paths.h5')

# Merge skeletons based on predicted paths
print('Merging skeletons.')
start_time = time.time()
merged_skeleton = merge_skeletons(split_skeleton, predicted_paths)
print('Time taken: ', time.time() - start_time)
save_data(merged_skeleton, output_base_path + exp_name + '/merged_skeletons.h5')
print('Calculating final skeleton error')
start_time = time.time()
p_m, r_m, f_m, c_m = calculate_error_metric_2(merged_skeleton, gt_skel_graphs=gt_skel_graphs, gt_skel_ids=gt_skel_ids,
                                              gt_context=gt_context, resolution=resolution, temp_folder=temp_folder, num_cpu=num_cpu)
print('Time taken: ', time.time() - start_time)

if args.skip_splitting is False:
    # print all error metrics
    print('Precision:    {:3.4f}  {:3.4f}  {:3.4f}'.format(p, p_s, p_m))
    print('Recall:       {:3.4f}  {:3.4f}  {:3.4f}'.format(r, r_s, r_m))
    print('F Score:      {:3.4f}  {:3.4f}  {:3.4f}'.format(f, f_s, f_m))
    print('Connectivity: {:3.4f}  {:3.4f}  {:3.4f}'.format(c, c_s, c_m))
else:
    # print all error metrics
    print('Precision:    {:3.4f}  {:3.4f}'.format(p_s, p_m))
    print('Recall:       {:3.4f}  {:3.4f}'.format(r_s, r_m))
    print('F Score:      {:3.4f}  {:3.4f}'.format(f_s, f_m))
    print('Connectivity: {:3.4f}  {:3.4f}'.format(c_s, c_m))

print('Total time taken: ', time.time() - origin_time)