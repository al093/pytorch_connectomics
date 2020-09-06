import os, sys, argparse, glob
import pickle, nibabel, re, json
import numpy as np

from torch_connectomics.data.dataset.dataset_eval_params import *
from torch_connectomics.utils.skeleton import *
from torch_connectomics.utils.vis import save_data, read_data

import testFlux, testSkeletonGrowing, testSkeleton


def my_bool(s):
    return s == 'True' or s == 'true' or s == '1'

def get_cmdline_args(args):
    parser = argparse.ArgumentParser(description='Specify model dir etc.')
    parser.add_argument('--method',                 type=str,           default='deepflux', help='Which method to run')
    parser.add_argument('--tune',                   type=my_bool,       default=False, help='Tune for parameters')
    parser.add_argument('--force-run-inference',    type=my_bool,       default=False, help='Force run inference even if files exist')
    parser.add_argument('--force-run-skeleton',     type=my_bool,       default=False, help='Force run flux to skeleton even if files exist')
    parser.add_argument('--dataset',                type=str,               help='snemi | syntheticVessel | segEM')
    parser.add_argument('--set',                    type=str,               help='train | val | test')
    parser.add_argument('--exp-name',               type=str,               help='Experiment name')
    parser.add_argument('--model-dir',              type=str,               help='directory with one or more models (.pth) files')
    parser.add_argument('--div-threshold',          type=float,             default=None, help='Threshold divergence at this value to get skeletons')
    parser.add_argument('--min-skeleton-vol-threshold',          type=int,               default=400, help='Remove all skeletons smaller than this size')
    return parser.parse_known_args(args)[0]

if __name__ == "__main__":
    ########################################################################################################################
    methods = ['deepflux', 'distanceTx', 'dilated', 'dvn', 'ours']
    args = get_cmdline_args(sys.argv)
    if args.method not in methods:
        raise Exception('Specified method not in :', methods)

    # generic arguments for running deepnet which are same for all methods
    exp_name = args.exp_name
    num_cpu = 12

    if args.dataset == 'snemi':
        get_dataset_fn = get_snemi
    elif args.dataset == 'syntheticVessel':
        get_dataset_fn = get_synthetic
    elif args.dataset == 'mri':
        get_dataset_fn = get_mri
    elif args.dataset == 'segEM':
        get_dataset_fn = get_segEM
    elif args.dataset == 'VISOR40':
        get_dataset_fn = get_visor40
    else:
        print("Dataset unknown.")
        raise NotImplementedError()

    # get dataset paths
    (paths, erl_overlap_allowance, ibex_downsample_fac, matching_radius,
    resolution, output_base_path, var_params, data_path, gt_skel_path, \
    gt_context_path, gt_skel_graphs_path) = get_dataset_fn(args.set, args.method, args.tune)

    # if div threshold is defined override that
    if args.div_threshold:
        var_params = [args.div_threshold]

    # run method
    if args.method in ['ours']:
        # Read the model files and run them one by one
        model_files = glob.glob(args.model_dir + "/*.pth")
        error_dict = dict()
        output_results_file = output_base_path + exp_name + '/results.json'
        for model_file in model_files:
            itr = int(re.split(r'_|\.', model_file)[-2])
            model_run_args = sys.argv + ['-lm', 'True', '-pm', model_file, '-o', output_base_path, '-dn', '@'.join(data_path)]
            output_path_itr = output_base_path + exp_name + '/' + str(itr)
            temp_folder = output_path_itr + '/temp'

            if not os.path.isdir(output_path_itr):
                os.makedirs(output_path_itr)
            if not os.path.isdir(temp_folder):
                os.makedirs(temp_folder)

            # Run Model
            print('Running Model')
            if args.method in [methods[0], methods[4]]:
                # check if all output flux files are present, if not run the model on all of the input files
                if args.force_run_inference:
                    run_model = True
                else:
                    run_model = False
                    for i, vol_data in enumerate(data_path):
                        flux_file_name = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + '_flux.h5'
                        if not os.path.isfile(flux_file_name):
                            print(f'{flux_file_name} was not present. So will run the model.')
                            run_model = True
                            break

                if run_model:
                    prediction = testFlux.run(model_run_args, save_output=False)
                    for i, vol_data in enumerate(data_path):
                        flux_file_name = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + '_flux.h5'
                        save_data(prediction[i], flux_file_name)
                else:
                    print("Not running model because out files are present in output folder.")
            else:
                # TODO(alok) not implemented correctly for skeletons
                raise NotImplementedError()
                prediction = testSkeleton.run(model_run_args, save_output=False)
                for i, pred_i in enumerate(prediction):
                    save_data(pred_i, output_path_itr + '/' + str(data_idxs[i]) + '_skeleton_prob.h5')

            # Compute skeletons from flux
            if args.method == "ours":
                skel_params = dict(filter_size=3, absolute_threshold=0.25, block_size=[32, 100, 100])  # Z, Y, X
                all_errors = dict()

                gt_skeletons = []
                gt_contexts = []
                for s, c, in zip(gt_skel_path, gt_context_path):
                    gt_skeletons.append(read_data(s))
                    gt_contexts.append(read_data(c))

                for var_param in var_params:
                    print('Threshold value: ', var_param)
                    skel_params['adaptive_threshold'] = 100 * var_param
                    initial_skeletons = []

                    for i, vol_data in enumerate(data_path):
                        flux_file_name = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + '_flux.h5'
                        initial_skeletons_filename = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + '_initial_skeletons_' + '{:.2f}'.format(var_param) + '.h5'
                        div_filename = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + '_divergence.h5'

                        if (not os.path.isfile(initial_skeletons_filename)) or args.force_run_skeleton:
                            print('Computing skeletons from flux')
                            pred_flux_i = read_data(flux_file_name)
                            skeleton, skel_divergence = compute_skeleton_from_gradient(pred_flux_i, skel_params)
                            skeleton = remove_small_skeletons(skeleton, args.min_skeleton_vol_threshold)
                            save_data(skeleton, initial_skeletons_filename)
                            if var_param == var_params[0]:
                                save_data(skel_divergence, div_filename)

                            initial_skeletons.append(skeleton)
                        else:
                            initial_skeletons.append(read_data(initial_skeletons_filename))

                    errors = calculate_errors_batch([[skel] for skel in initial_skeletons], gt_skeletons, gt_contexts, resolution,
                                                    temp_folder, 0, matching_radius, ibex_downsample_fac,
                                                    erl_overlap_allowance)

                    all_errors[var_param] = errors[0]

            error_dict[itr] = all_errors

            prev_error_dict = None
            if os.path.isfile(output_results_file):
                try:
                    with open(output_results_file, 'r') as handle:
                        prev_error_dict = json.load(handle)
                except:
                    prev_error_dict = None
            if not prev_error_dict:
                prev_error_dict = dict()

            with open(output_results_file, 'w') as handle:
                json.dump(prev_error_dict.update(error_dict), handle)
                print("Results stored at: ", output_results_file)
