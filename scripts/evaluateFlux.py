import skimage
import timeit

from torch_connectomics.data.dataset.dataset_eval_params import *
from torch_connectomics.utils.skeleton import *
from torch_connectomics.utils.skeleton.gradientProcessing import split_skeletons_using_graph
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
    parser.add_argument('--dataset',                type=str,               help='snemi | VISOR40 | segEM | coronary')
    parser.add_argument('--set',                    type=str,               help='train | val | test')
    parser.add_argument('--exp-name',               type=str,               help='Experiment name')
    parser.add_argument('--model-dir',              type=str,               help='directory with one or more models (.pth) files')
    parser.add_argument('--checkpoint-regex',       type=str,               default=None,   help='To limit model checkpoint files to be evaluated')
    parser.add_argument('--div-threshold',          type=float,             default=None,   help='Threshold divergence at this value to get skeletons')
    parser.add_argument('--dataset-scale',          type=float,             default=1.0,    help='Get different input scale.')
    parser.add_argument('--min-skeleton-vol-threshold',          type=int,  default=400,    help='Remove all skeletons smaller than this size')
    parser.add_argument('-c', '--num-cpu',          type=int,               default=12,     help='Number of parallel threads to use.')
    parser.add_argument('--use-skeleton-head',      type=my_bool,           default=False,  help='Use Skeleton head.')
    parser.add_argument('--use-flux-head',          type=my_bool,           default=False,  help='Use Flux head after the flux model.')
    parser.add_argument('--split-skeletons',        type=my_bool,           default=False,  help='Split skeletons based on topology.')
    parser.add_argument('--calculate-erl',          type=my_bool,           default=True,   help='Does not calculate ERL metric if False.')

    return parser.parse_known_args(args)[0]

def print_best_params(results_dict, metric):
    best_checkpoint, best_th, best_metric = -1, -1, -1
    for checkpoint, results in results_dict.items():
        for th, result in results.items():
            if best_metric <= result[metric]:
                best_metric = result[metric]
                best_checkpoint = checkpoint
                best_th = th
    print(f'Best {metric} score is {best_metric} for checkpoint {best_checkpoint} and threshold {best_th}')
    return best_checkpoint, best_th

if __name__ == "__main__":
    ########################################################################################################################
    methods = ['deepflux', 'distanceTx', 'dilated', 'dvn', 'ours']
    args = get_cmdline_args(sys.argv)
    if args.method not in methods:
        raise Exception('Specified method not in :', methods)

    # generic arguments for running deepnet which are same for all methods
    exp_name = args.exp_name
    num_cpu = args

    if args.dataset == 'snemi':
        get_dataset_fn = get_snemi
    elif args.dataset == 'segEM':
        get_dataset_fn = get_segEM
    elif args.dataset == 'VISOR40':
        get_dataset_fn = get_visor40
    elif args.dataset == 'coronary':
        get_dataset_fn = get_coronary
    else:
        print("Dataset unknown.")
        raise NotImplementedError()

    # get dataset paths
    (paths, erl_overlap_allowance, ibex_downsample_fac, matching_radius,
    resolution, output_base_path, var_params, data_path, gt_skel_path,
    gt_context_path, gt_skel_graphs_path, remove_borders) = get_dataset_fn(args)

    # if div threshold is defined override that
    if args.div_threshold:
        var_params = [args.div_threshold]

    # run method
    if args.method in ['ours']:
        # Read model files and run them one by one
        model_files = glob.glob(args.model_dir + "/*.pth")
        if args.checkpoint_regex:
            model_files = [model_file for model_file in model_files if re.search(args.checkpoint_regex, model_file)]

        error_dict = dict()
        output_results_file = output_base_path + exp_name + '/results.json'
        for model_file in model_files:
            model_predictions_buffer = dict()
            itr = int(re.split(r'_|\.|/', model_file)[-2])
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
                        if args.use_skeleton_head:
                            prediction_type = 'skeleton'
                        elif args.use_flux_head:
                            prediction_type = 'flux'
                        else:
                            raise ValueError('Specify if flux or skeleton head is to be used?')
                        file_name = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + f'_{prediction_type}_prediction.h5'
                        if not os.path.isfile(file_name):
                            run_model = True
                            print(f"Will run inference as file {file_name} is not found.")
                            break

                if run_model:
                    predictions_dict = testFlux.run(model_run_args, save_output=False)
                    for prediction_type, prediction in predictions_dict.items():
                        for i, vol_data in enumerate(data_path):
                            file_name = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + f'_{prediction_type}_prediction.h5'
                            save_data(prediction[i], file_name)
                            model_predictions_buffer[file_name] = prediction[i]
                else:
                    print("Not running model because prediction files are present in output folder.")
            else:
                # TODO(alok) not implemented correctly for skeletons
                raise NotImplementedError("Not implemented any other method.")

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
                        initial_skeletons_filename = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + '_initial_skeletons_' + '{:.2f}'.format(var_param) + '.h5'

                        if (not os.path.isfile(initial_skeletons_filename)) or args.force_run_skeleton:
                            print('Computing skeletons.')
                            if args.use_skeleton_head:
                                skeleton_file = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + f'_skeleton_prediction.h5'
                                pred_skeleton_i = model_predictions_buffer.get(skeleton_file) if \
                                    skeleton_file in model_predictions_buffer else read_data(skeleton_file)
                                skeleton = compute_skeleton_from_probability(pred_skeleton_i, skel_params, remove_borders)
                            elif args.use_flux_head:
                                flux_file = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + f'_flux_prediction.h5'
                                pred_flux_i = model_predictions_buffer.get(flux_file) if \
                                    flux_file in model_predictions_buffer.keys() else read_data(flux_file)
                                start = timeit.default_timer()
                                skel_divergence = divergence_3d(pred_flux_i)
                                print(f"divergence_3d call took : {(timeit.default_timer() - start):.4f}")

                                start = timeit.default_timer()
                                skeleton = compute_skeleton_from_probability(skel_divergence, skel_params, remove_borders)
                                print(f"compute_skeleton_from_probability call took : {(timeit.default_timer() - start):.4f}")

                                if var_param == var_params[0]:
                                    div_filename = output_path_itr + '/' + os.path.basename(vol_data).split('.h5')[0] + '_divergence.h5'
                                    save_data(skel_divergence, div_filename)

                            if args.split_skeletons:
                                print('Splitting skeletons.')
                                skeleton = split_skeletons_using_graph(skeleton, temp_folder + '/ibex_graphs/', resolution)

                            skeleton = remove_small_skeletons(skeleton, args.min_skeleton_vol_threshold)

                            if skeleton.shape != gt_contexts[i].shape:
                                skeleton = skimage.transform.resize(skeleton, gt_contexts[i].shape, order=0, mode='edge',
                                                                    clip=True, preserve_range=True, anti_aliasing=False).astype(np.uint16)

                            save_data(skeleton, initial_skeletons_filename)
                            initial_skeletons.append(skeleton)
                        else:
                            initial_skeletons.append(read_data(initial_skeletons_filename))

                    errors = calculate_errors_batch([[skel] for skel in initial_skeletons], gt_skeletons, gt_contexts, resolution,
                                                    temp_folder, 8, matching_radius, ibex_downsample_fac,
                                                    erl_overlap_allowance, args.calculate_erl)

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
                prev_error_dict.update(error_dict)
                json.dump(prev_error_dict, handle)
                print("Results stored at: ", output_results_file)

        print_best_params(prev_error_dict, 'erl')
        print_best_params(prev_error_dict, 'pr')
