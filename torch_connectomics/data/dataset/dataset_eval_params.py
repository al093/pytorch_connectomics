import os, sys, argparse, glob
import pickle, nibabel, re, json
import numpy as np


def get_segEM(set, method, tune):
    '''
    Parameters
    ----------
    set: str
        val or test or train
    method: str
        ours, ...
    tune: bool
        if 'True' returns multiple variable parameters

    Returns
    -------
    paths: str
        dataset paths
    erl_overlap_allowance: ndarray
        numpy int32 array of shape (3,)
    ibex_downsample_fac: ndarray
        numpy int32 array of shape (3,)
    matching_radius: float
    resolution: ndarray
    output_base_path: str
    var_params: list
        list of float which are threshold parameters
    data_path: list[str]
        image paths
    gt_skel_path: list[str]
    gt_context_path: list[str]
    gt_skel_graphs_path: list[str]
    '''

    erl_overlap_allowance = np.int32((2, 2, 2))
    ibex_downsample_fac = np.uint16((2, 2, 2))
    matching_radius = 2.0
    resolution = np.uint16((28, 11, 11))
    output_base_path = '/n/pfister_lab2/Lab/alok/results/segEM/'

    with open('/n/home11/averma/pytorch_connectomics/cmdArgs/segEMPaths.json', 'r') as phandle:
        paths = json.load(phandle)[set]

    if args.div_threshold:
        var_params = [args.div_threshold]
    else:
        if method == 'ours':
            var_params = np.arange(0.30, 0.90, 0.05) if tune else [0.65]
        elif method == 'deepflux':
            var_params = [0.65] if tune is False else np.arange(0.05, 0.90, 0.05)
        elif method == 'distanceTx':
            var_params = [0.45] if tune is False else np.arange(0.10, 0.90, 0.05)
        elif method == 'dilated':
            var_params = [0.85] if tune is False else np.arange(0.10, 0.90, 0.05)
        elif method == 'dvn':
            var_params = [0.30] if tune is False else np.arange(0.10, 0.90, 0.05)
        else:
            raise Exception(f'Method {method} is not defined')

    return paths, erl_overlap_allowance, ibex_downsample_fac, matching_radius, \
           resolution, output_base_path, var_params, data_path, gt_skel_path, \
           gt_context_path, gt_skel_graphs_path


def get_visor40(set, method, tune):
    erl_overlap_allowance = np.int32((2, 2, 2))
    ibex_downsample_fac = np.uint16((2, 2, 2))
    matching_radius = 2.0
    resolution = np.uint16((1, 1, 1))
    output_base_path = '/n/pfister_lab2/Lab/alok/results/VISOR40/'

    with open('/n/home11/averma/pytorch_connectomics/cmdArgs/VISOR40Paths.json', 'r') as phandle:
        paths = json.load(phandle)[set]

    data_path = paths['dn']
    gt_skel_path = paths['skn']
    gt_context_path = paths['ln']
    gt_skel_graphs_path = paths['gn']

    if method == 'ours':
        var_params = np.arange(0.30, 0.90, 0.05) if tune else [0.65]
    elif method == 'deepflux':
        var_params = [0.65] if tune is False else np.arange(0.05, 0.90, 0.05)
    elif method == 'distanceTx':
        var_params = [0.45] if tune is False else np.arange(0.10, 0.90, 0.05)
    elif method == 'dilated':
        var_params = [0.85] if tune is False else np.arange(0.10, 0.90, 0.05)
    elif method == 'dvn':
        var_params = [0.30] if tune is False else np.arange(0.10, 0.90, 0.05)
    else:
        raise Exception(f'Method {method} is not defined')

    return paths, erl_overlap_allowance, ibex_downsample_fac, matching_radius, \
           resolution, output_base_path, var_params, data_path, gt_skel_path, \
           gt_context_path, gt_skel_graphs_path


def get_mri(set, method, tune):
    raise NotImplementedError('Not implemented corectly')

    matching_radius = 2.0
    resolution = (8.0, 5.0, 5.0)
    output_base_path = '/n/pfister_lab2/Lab/alok/results/mri/'

    with open('/n/home11/averma/pytorch_connectomics/cmdArgs/mriPaths.pkl', 'rb') as phandle:
        syn_paths = pickle.load(phandle)

    data_path, gt_skel_path, gt_context_path, gt_skel_graphs_path = [], [], [], []
    keys = list(syn_paths.keys());
    keys.sort()
    if args.set == 'val':
        data_idxs = keys[26:31]
        exp_name_pf = 'valVols'
    elif args.set == 'test':
        data_idxs = keys[31:42]
        exp_name_pf = 'testVols'
    elif args.set == 'train':
        data_idxs = keys[10:26]
        exp_name_pf = 'trainVols'
    else:
        raise Exception('Set not defined')

    if args.method == 'deepflux':
        exp_name = 'mri_abStudy_deepflux_' + exp_name_pf
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/mri/mri_deepflux/mri_deepflux_120000.pth'
    elif args.method == 'dilated':
        exp_name = 'mri_abStudy_dilated_' + exp_name_pf
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/mri/mri_dilated/mri_dilated_120000.pth'
    elif args.method == 'distanceTx':
        exp_name = 'mri_abStudy_distanceTx_' + exp_name_pf
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/mri/mri_distanceTx/mri_distanceTx_148000.pth'

    for i in data_idxs:
        vol_data = syn_paths[i]
        data_path.append(vol_data['dn'])
        gt_skel_path.append(vol_data['skn'])
        gt_context_path.append(vol_data['ln'])
        gt_skel_graphs_path.append(vol_data['gn'])

    if args.method == 'deepflux':
        var_param_list = [0.30] if args.tune is False else np.arange(0.10, 0.90, 0.05)  # Tuned for PR
    elif args.method == 'distanceTx':
        var_param_list = [0.40] if args.tune is False else np.arange(0.10, 0.90, 0.05)  # X
    elif args.method == 'dilated':
        var_param_list = [0.75] if args.tune is False else np.arange(0.10, 0.90, 0.05)  # tuned for F1
    else:
        raise Exception('Method {:%s} is not defined'.format(args.method))

    return paths, erl_overlap_allowance, ibex_downsample_fac, matching_radius, \
           resolution, output_base_path, var_params


def get_snemi(set, method, tune):
    raise NotImplementedError('Not implemented Correctly')

    matching_radius = 60.0
    resolution = np.uint16((30, 6, 6))
    matching_radius = 60.0
    ibex_downsample_fac = np.uint16((1, 5, 5))
    erl_overlap_allowance = np.int32((2, 5, 5))

    if args.set == 'val':
        data_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/val/val_image_half.h5']
        gt_skel_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton.h5']
        gt_context_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/skeleton_context.h5']
        gt_skel_graphs_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/val/1_0x/graph.h5']
        exp_name_pf = 'valVol'
    elif args.set == 'test':
        data_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/test/image.h5']
        gt_skel_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/test/1_0x/skeleton.h5']
        gt_context_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/test/1_0x/skeleton_context.h5']
        gt_skel_graphs_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/test/1_0x/graph.h5']
        exp_name_pf = 'testVol'
    elif args.set == 'train':
        data_path = ['/n/pfister_lab2/Lab/alok/snemi/train_image.h5']
        gt_skel_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/skeleton.h5']
        gt_context_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/skeleton_context.h5']
        gt_skel_graphs_path = ['/n/pfister_lab2/Lab/alok/snemi/skeleton/splineInterp/1_0x/graph.h5']
        exp_name_pf = 'trainVol'

    data_idxs = [0]
    output_base_path = '/n/pfister_lab2/Lab/alok/results/snemi/'
    if args.method == 'deepflux':
        exp_name = 'snemi_abStudy_deepflux_' + exp_name_pf
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_DeepFlux_2/snemi_abStudy_DeepFlux_2_78000.pth'
    elif args.method == 'dilated':
        exp_name = 'snemi_abStudy_dilated_' + exp_name_pf
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_binary_skeleton/snemi_abStudy_binary_skeleton_120000.pth'
    elif args.method == 'distanceTx':
        exp_name = 'snemi_abStudy_distanceTx_' + exp_name_pf
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/snemi/snemi_abStudy_skeleton/snemi_abStudy_skeleton_120000.pth'

    if args.method == 'deepflux':
        var_param_list = [0.15] if args.tune is False else np.arange(0.10, 0.95, 0.05)  # tuned
    elif args.method == 'distanceTx':
        var_param_list = [0.45] if args.tune is False else np.arange(0.30, 0.60, 0.05)  # tuned
    elif args.method == 'dilated':
        var_param_list = [0.90] if args.tune is False else np.arange(0.50, 0.95, 0.05)  # tuned
    else:
        raise Exception('Method {:%s} is not defined'.format(args.method))

    return paths, erl_overlap_allowance, ibex_downsample_fac, matching_radius, \
           resolution, output_base_path, var_params


def get_synthetic(set, method, tune):
    raise NotImplementedError()

    erl_overlap_allowance = np.int32((2, 2, 2))
    ibex_downsample_fac = np.uint16((1, 1, 1))
    matching_radius = 2.0
    resolution = np.uint16((1, 1, 1))
    output_base_path = '/n/pfister_lab2/Lab/alok/results/syntheticVessel/'

    if args.set == 'val':
        data_idxs = list(range(21, 23))
        exp_name_pf = 'valVols'
    elif args.set == 'test':
        data_idxs = list(range(26, 36))
        exp_name_pf = 'testVols'
    else:
        raise Exception('Set not defined')

    if args.method == 'deepflux':
        exp_name = 'synVessel_abStudy_deepflux_' + exp_name_pf
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/syntheticVessel/synVessel_abStudy_deepflux/synVessel_abStudy_deepflux_120000.pth'
    elif args.method == 'dilated':
        exp_name = 'synVessel_abStudy_dilated_' + exp_name_pf
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/syntheticVessel/synVessel_abStudy_skeleton/synVessel_abStudy_skeleton_120000.pth'
    elif args.method == 'distanceTx':
        exp_name = 'synVessel_abStudy_distanceTx_' + exp_name_pf
        model_path = '/n/home11/averma/pytorch_connectomics/outputs/syntheticVessel/synVessel_abStudy_distanceTx/synVessel_abStudy_distanceTx_120000.pth'
    elif args.method == 'dvn':
        exp_name = 'synVessel_abStudy_dvn_' + exp_name_pf
        basename = '/n/pfister_lab2/Lab/alok/results/syntheticVessel/dvn/sample/pred/cen_on_synthetic_test_prob_'
        dvn_files = [basename + str(i) + '.nii.gz' for i in data_idxs]

    with open('/n/home11/averma/pytorch_connectomics/cmdArgs/synVesselPaths.pkl', 'rb') as phandle:
        syn_paths = pickle.load(phandle)
    data_path, gt_skel_path, gt_context_path, gt_skel_graphs_path = [], [], [], []
    for i, vol_data in syn_paths.items():
        if i in data_idxs:
            data_path.append(vol_data['dn'])
            gt_skel_path.append(vol_data['skn'])
            gt_context_path.append(vol_data['ln'])
            gt_skel_graphs_path.append(vol_data['gn'])

    if args.method == 'deepflux':
        var_param_list = [0.65] if args.tune is False else np.arange(0.05, 0.90, 0.05)  # tuned for best pr+erl
    elif args.method == 'distanceTx':
        var_param_list = [0.45] if args.tune is False else np.arange(0.10, 0.90, 0.05)  # tuned for best pr+erl
    elif args.method == 'dilated':
        var_param_list = [0.85] if args.tune is False else np.arange(0.10, 0.90, 0.05)  # tuned for best pr+erl
    elif args.method == 'dvn':
        var_param_list = [0.30] if args.tune is False else np.arange(0.10, 0.90, 0.05)  # tuned for best pr+erl
    else:
        raise Exception('Method {:%s} is not defined'.format(args.method))

    return paths, erl_overlap_allowance, ibex_downsample_fac, matching_radius, \
           resolution, output_base_path, var_params