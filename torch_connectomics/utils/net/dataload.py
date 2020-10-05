import os,sys
import numpy as np
import h5py
import scipy

import torch
import torch.utils.data

from torch_connectomics.data.dataset import AffinityDataset, SynapseDataset, MitoDataset, \
    MaskDataset, MaskDatasetDualInput, FluxAndSkeletonDataset, MatchSkeletonDataset, SkeletonGrowingDataset
from torch_connectomics.data.utils import collate_fn_growing, collate_fn_test, collate_fn_var
from torch_connectomics.data.augmentation import *
from torch_connectomics.utils.net.serialSampler import SerialSampler

TASK_MAP = {0: 'neuron segmentation',
            1: 'synapse detection',
            2: 'mitochondria segmentation',
            3: 'mask prediction',
            4: 'skeleton prediction',
            5: 'skeleton matching',
            6: 'skeleton growing'}

def get_input(args, model_io_size, mode='train', model=None):
    """Prepare dataloader for training and inference.
    """
    print('Task: ', TASK_MAP[args.task])
    assert mode in ['train', 'test', 'validation']

    volume_shape = []

    if mode=='validation':
        img_name = args.val_img_name.split('@')
    else:
        img_name = args.img_name.split('@')
        s_points = [None] * len(img_name)
        skeleton = [None] * len(img_name)
        flux = [None] * len(img_name)
        flux_2 = [None] * len(img_name)
        weight = [None] * len(img_name)
        divergence = [None] * len(img_name)

    if args.task != 5 and args.task != 6:
        if mode=='validation':
            seg_name = args.val_seg_name.split('@')
        elif mode=='train':
            seg_name = args.seg_name.split('@')

    if args.task in [3, 4, 5, 6]:
        if args.seed_points is not None:
            seed_points_files = args.seed_points.split('@')

    if args.task in [4, 5, 6]:
        skeleton_files = None
        if args.skeleton_name is not None:
            skeleton_files = args.skeleton_name.split('@')

        flux_files = None
        if args.flux_name is not None:
            flux_files = args.flux_name.split('@')

        weight_files = None
        if args.weight_name is not None:
            weight_files = args.weight_name.split('@')

        flux_files_2 = None
        if args.flux_name_gt is not None:
            flux_files_2 = args.flux_name_gt.split('@')

    if args.task in [6]:
        divergence_files = None
        if args.div_name is not None:
            divergence_files = args.div_name.split('@')

    # 1. load data
    model_input = [None]*len(img_name)

    if args.task != 5 and args.task != 6:
        if mode=='train' or mode=='validation':
            assert len(img_name)==len(seg_name)
            model_label = [None]*len(seg_name)

    if args.task != 6 and (mode=='train' or mode=='validation') and args.data_aug is True:
        # setup augmentor
        elastic_augmentor = Elastic(alpha=6.0, p=0.75)
        augmentation_methods = [
            Rotate(p=1.0),
            Flip(p=1.0),
            elastic_augmentor,
            Grayscale(p=0.75),
            Blur(min_sigma=1, max_sigma=2, min_slices=model_io_size[0]//6, max_slices=model_io_size[0]//4, p=0.4),
            MissingParts(p=0.5)
            # MissingSection(p=0.5),
            # MisAlignment2(p=1.0, displacement=16)
            ]

        # if the input is symmetric, and more importantly if the resolution is isometric
        # we can perform swapping of z with (y or x) axis
        if args.symmetric:
            augmentation_methods.append(SwapZ(0.5))

        augmentor = Compose(augmentation_methods,
                            input_size=model_io_size)
        elastic_augmentor.set_input_sz(augmentor.sample_size)
        sample_input_size = augmentor.sample_size
    else:
        augmentor = None
        sample_input_size = model_io_size

    print('Data augmentation: ', augmentor is not None)

    SHUFFLE = (mode=='train' or mode=='validation')
    print('Batch size: ', args.batch_size)

    if mode == 'test' and args.task not in [5]:
        pad_size = np.array(model_io_size//2, dtype=np.int64)
    elif args.task == 6:
        pad_size = np.array(model_io_size//2, dtype=np.int64)
        pad_size = np.array((0, 0, 0), dtype=np.int64)
    else:
        pad_size = np.array((0, 0, 0), dtype=np.int64)
        # pad_size = augmentor.sample_size//2
    pad_size_tuple = ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2]))

    for i in range(len(img_name)):
        if not is_ddp(args):
            image = np.array((h5py.File(img_name[i], 'r')['main']))
            model_input[i] = image
            if image.dtype in [np.float16, np.float32, np.float64]:
                model_input[i] = np.array(image, copy=False, dtype=np.float32)
            elif image.dtype == np.uint8:
                model_input[i] = np.array(image / np.float32(255.0), copy=False, dtype=np.float32)
            else:
                raise Exception('Image datatype was not uint8 or float, not sure how to normalize.')
            model_input[i] = np.pad(model_input[i], pad_size_tuple, 'reflect')
            model_input[i] = model_input[i].astype(np.float32)
        else:
            model_input[i] = h5py.File(img_name[i], 'r')['main']
            if not all([j == (0, 0) for j in pad_size_tuple]):
                raise NotImplementedError("In Distributed Data parallel mode padding the input volumes"
                                          "is not supported yet")

        print(f"Image name: {img_name[i]}")
        print("Shape: ", model_input[i].shape)
        volume_shape.append(model_input[i].shape)

        if mode == 'test':
            if args.task == 5:
                # It must be ensured that all centroid points have enough crop area around them
                # These Points are the origin.
                npf = np.load(seed_points_files[i], allow_pickle=True)
                s_points[i] = [np.vstack([npf.item().get('match'), npf.item().get('no_match')])]
                skeleton[i] = np.array((h5py.File(skeleton_files[i], 'r')['main']))
                flux[i] = np.array((h5py.File(flux_files[i], 'r')['main']))
            elif args.task == 6:
                data = {}
                with h5py.File(seed_points_files[i], 'r') as hf:
                    for g in hf.keys():
                        d = {}
                        d['path'] = np.asarray(hf.get(g)['vertices']) + pad_size.astype(np.float32)
                        d['sids'] = np.asarray(hf.get(g)['sids'])
                        data[int(g)] = d
                s_points[i] = data

                # load skeletons
                if skeleton_files is not None:
                    skeleton[i] = np.array((h5py.File(skeleton_files[i], 'r')['main']))
                    skeleton[i] = np.pad(skeleton[i], pad_size_tuple)

                if flux_files is not None:
                    flux[i] = np.array((h5py.File(flux_files[i], 'r')['main']))
                    flux[i] = np.pad(flux[i], ((0, 0),) + pad_size_tuple)

                if divergence_files is not None:
                    divergence[i] = np.array((h5py.File(divergence_files[i], 'r')['main']))
                    divergence[i] = np.pad(divergence[i], pad_size_tuple)

        elif mode == 'train' or mode == 'validation':
            if args.task != 5 and args.task != 6:
                if is_ddp(args):
                    model_label[i] = h5py.File(seg_name[i], 'r')['main']
                else:
                    model_label[i] = np.array((h5py.File(seg_name[i], 'r')['main']))

            if args.task == 3 or args.task == 4:
                s_points[i] = load_list_from_h5(seed_points_files[i])
                half_aug_sz = sample_input_size//2
                vol_size = model_input[i].shape
                new_list = []
                for idx in range(len(s_points[i])):
                    b = s_points[i][idx]
                    b = b[np.all(b >= half_aug_sz, axis=1), :]
                    b = b[np.all((vol_size - b) > half_aug_sz, axis=1), :]
                    if b.shape[0] > 0:
                        new_list.append(b)
                s_points[i] = new_list
            elif args.task == 5:
                #  it must be ensured externally that all centroid points have enough crop area around them,
                #  no check is done here. Rotation Augmentation is not supported yet
                #  These Points are the origin.
                npf = np.load(seed_points_files[i], allow_pickle=True)
                s_points[i] = [npf.item().get('match'), npf.item().get('no_match')]

            elif args.task == 6:
                # load the skeleton growing datasets
                data = {}
                with h5py.File(seed_points_files[i], 'r') as hf:
                    for g in hf.keys():
                        d = {}
                        d['path'] = np.asarray(hf.get(g)['vertices']) + pad_size.astype(np.float32)
                        if d['path'].shape[0] <= 2:
                            continue
                        d['sids'] = np.asarray(hf.get(g)['sids'])
                        if 'first_split_node' in hf.get(g).keys():
                            d['first_split_node'] = np.asarray(hf.get(g)['first_split_node'])[0]
                        data[int(g)] = d
                s_points[i] = data

                if divergence_files is not None:
                    divergence[i] = np.array((h5py.File(divergence_files[i], 'r')['main']))
                    divergence[i] = np.pad(divergence[i], pad_size_tuple)

            # load skeletons
            if skeleton_files is not None:
                if is_ddp(args):
                    skeleton[i] = h5py.File(skeleton_files[i], 'r')['main']
                else:
                    skeleton[i] = np.array((h5py.File(skeleton_files[i], 'r')['main']))
                    skeleton[i] = np.pad(skeleton[i], pad_size_tuple)

            if flux_files is not None:
                if is_ddp(args):
                    flux[i] = h5py.File(flux_files[i], 'r')['main']
                else:
                    flux[i] = np.array((h5py.File(flux_files[i], 'r')['main']))
                    flux[i] = np.pad(flux[i], ((0,0),) + pad_size_tuple)

            if args.train_end_to_end is True:
                assert (flux_files_2 is not None)
                flux_2[i] = np.array((h5py.File(flux_files_2[i], 'r')['main']))
                flux_2[i] = np.pad(flux_2[i], ((0,0),) + pad_size_tuple)
            else:
                flux_2 = None

            #load weight files:
            if weight_files is not None:
                if is_ddp(args):
                    weight[i] = h5py.File(weight_files[i], 'r')['main']
                else:
                    weight[i] = np.array((h5py.File(weight_files[i], 'r')['main']))
                    weight[i] = np.pad(weight[i], pad_size_tuple)

        if args.task not in [5, 6]:
            if mode=='train' or mode=='validation':
                model_label[i] = np.pad(model_label[i], pad_size_tuple, 'reflect')
                assert model_input[i].shape == model_label[i].shape

    if mode=='test' and args.task == 3:
        b = np.array(h5py.File(seed_points_files[0], 'r')[str(args.segment_id)])
        if len(b.shape) == 1: #  only one point was read, make it into 2D
            b = b.reshape((1, 3))
        s_points = [[b.astype(np.uint32)]]

        print('Num of initial seed points: ', s_points[0][0].shape[0])
        # read the initial segmentation volume and choose the neuron which needs to be run
        # read the seed points from another h5 file
        if args.initial_seg is not None:
            initial_seg = np.array((h5py.File(args.initial_seg, 'r')['main']))
            # initial_seg = np.array((h5py.File(args.initial_seg, 'r')['main'])[bs[0]:be[0], bs[1]:be[1], bs[2]:be[2]])
            initial_seg = (initial_seg == args.segment_id)
            initial_seg = np.pad(initial_seg, pad_size_tuple, 'reflect')

    if mode=='train' or mode=='validation':
        if args.task == 0: # affininty prediction
            dataset = AffinityDataset(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                      sample_label_size=sample_input_size, augmentor=augmentor, mode = 'train')
        elif args.task == 1: # synapse detection
            dataset = SynapseDataset(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                     sample_label_size=sample_input_size, augmentor=augmentor, mode = 'train')
        elif args.task == 2: # mitochondira segmentation
            dataset = MitoDataset(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                  sample_label_size=sample_input_size, augmentor=augmentor, mode = 'train')
        elif args.task == 3: # mask prediction
            if args.in_channel == 2:
                augmentor_1 = Compose([Grayscale(p=0.75),
                                       MissingParts(p=0.9)],
                                       input_size=model_io_size)
                dataset = MaskDatasetDualInput(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                      sample_label_size=sample_input_size, augmentor_pre=augmentor_1, augmentor=augmentor,
                                      mode='train', seed_points=s_points, pad_size=pad_size.astype(np.uint32), model=model)
        elif args.task == 4:  # skeleton/flux prediction
            if is_ddp(args):
                # if ddp is used call the dataset with paths of the input, label, skeleton, flux and weight
                # The dataset will read small chunks of data as needed iso of loading the entire volume in memory.
                model_input_paths = img_name
                model_label_paths = seg_name
                skeleton_paths = skeleton_files
                flux_paths = flux_files
                weight_paths = weight_files
                dataset = FluxAndSkeletonDataset(volume=model_input_paths, label=model_label_paths,
                                                 skeleton=skeleton_paths, flux=flux_paths, weight=weight_paths,
                                                 sample_input_size=sample_input_size,
                                                 sample_label_size=sample_input_size,
                                                 augmentor=augmentor, mode='train', seed_points=s_points,
                                                 pad_size=pad_size.astype(np.uint32))
            else:
                dataset = FluxAndSkeletonDataset(volume=model_input, label=model_label, skeleton=skeleton, flux=flux,
                                                 weight=weight, sample_input_size=sample_input_size,
                                                 sample_label_size=sample_input_size,
                                                 augmentor=augmentor, mode='train', seed_points=s_points,
                                                 pad_size=pad_size.astype(np.uint32))

        elif args.task == 5:  # skeleton match prediction
            dataset = MatchSkeletonDataset(image=model_input, skeleton=skeleton, flux=flux,
                                             sample_input_size=sample_input_size, sample_label_size=sample_input_size,
                                             augmentor=augmentor, mode='train', seed_points=s_points,
                                             pad_size=pad_size.astype(np.uint32))

        elif args.task == 6:  # skeleton match prediction
            dataset = SkeletonGrowingDataset(image=model_input, skeleton=skeleton, flux=flux, divergence=divergence,
                                             growing_data=s_points, flux_gt=flux_2, mode='train')

        if args.task == 6:
            c_fn = collate_fn_growing
            pin_memory = False
        else:
            c_fn = collate_fn_var
            pin_memory = True

        img_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=SHUFFLE,
                                                 collate_fn=c_fn, num_workers=args.num_cpu, pin_memory=pin_memory,
                                                 worker_init_fn=get_worker_init_fn(args.local_rank))

        return img_loader

    else:
        if args.task == 0 or args.task == 4:
            dataset = AffinityDataset(volume=model_input, label=None, sample_input_size=model_io_size, \
                                      sample_label_size=None, sample_stride=model_io_size // 2, \
                                      augmentor=None, mode='test')
        elif args.task == 1:
            dataset = SynapseDataset(volume=model_input, label=None, sample_input_size=model_io_size, \
                                     sample_label_size=None, sample_stride=model_io_size // 2, \
                                     augmentor=None, mode='test')
        elif args.task == 2:
            dataset = MitoDataset(volume=model_input, label=None, sample_input_size=model_io_size, \
                                  sample_label_size=None, sample_stride=model_io_size // 2, \
                                  augmentor=None, mode='test')
        elif args.task == 3:
            dataset = MaskDataset(volume=model_input, label=None, sample_input_size=model_io_size, \
                                  sample_label_size=None, sample_stride=model_io_size // 2, \
                                  augmentor=None, mode='test', seed_points=s_points,
                                  pad_size=pad_size.astype(np.uint32))
        elif args.task == 5:
            dataset = MatchSkeletonDataset(image=model_input, skeleton=skeleton, flux=flux,
                                           sample_input_size=model_io_size, sample_label_size=model_io_size,
                                           augmentor=None, mode='test', seed_points=s_points,
                                           pad_size=pad_size.astype(np.uint32))
        elif args.task == 6:
            dataset = SkeletonGrowingDataset(image=model_input, skeleton=skeleton, flux=flux, divergence=divergence,
                                             growing_data=s_points, mode='test')

        if args.task == 6:
            c_fn = collate_fn_growing
            pin_memory = False
        else:
            c_fn = collate_fn_var
            pin_memory = True

        if args.task != 3:

            if is_ddp(args):
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=SHUFFLE)
            else:
                train_sampler = None

            img_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                     shuffle=(SHUFFLE and (not train_sampler)), collate_fn=c_fn,
                                                     num_workers=args.num_cpu, pin_memory=pin_memory,
                                                     sampler=train_sampler)

            return img_loader, volume_shape, pad_size
        else:
            assert len(img_name) == 1
            img_loader = SerialSampler(dataset, args.batch_size, pad_size, s_points[0][0] + pad_size, args.in_channel)
            if args.initial_seg is not None:
                return img_loader, volume_shape, pad_size, initial_seg
            else:
                return img_loader, volume_shape, pad_size, None

#### Helper Function ###
def crop_cremi(image, label, path):
    filename = os.path.basename(path)
    basepath = os.path.dirname(path)
    filename = filename.replace('im_', 'crop_')
    filename = filename.replace('_200.h5', '.txt')
    crop_filepath = basepath + '/../align/' + filename
    try:
        with open(crop_filepath, 'r') as file:
            coords = file.read()
            coords = coords.split(',')
            coords = [int(i) for i in coords]
            image = image[coords[4]:coords[5], coords[2]:coords[3], coords[0]:coords[1]]
            label = label[coords[4]:coords[5], coords[2]:coords[3], coords[0]:coords[1]]
            return image, label

    except IOError:
        print('Could not read file: ' + crop_filepath)
        return None, None

def shrink_range_uint32(seg):
    maxid = np.max(seg)
    minid = np.min(seg)

    assert minid > -1, 'Labels have value less than zero, not supported yet'
    assert maxid < np.iinfo(np.uint32).max - 1, 'The max input cannot be represented by uint32'
    labels = np.zeros(maxid.astype(np.uint32) + 1, dtype=np.uint32)

    ids = (np.unique(seg)).astype(np.uint32)
    labels[ids] = np.arange(ids.shape[0], dtype=np.uint32)
    print(labels)
    print(seg.shape)
    return labels[seg]

def load_list_from_h5(h5_path):
    print()
    h_list = []
    h_file = h5py.File(h5_path, 'r')
    for key in list(h_file):
        h_list.append(np.asarray(h_file[key]))
    h_list = [x for x in h_list if x.shape[0] > 0]
    return h_list

def load_seeds_from_txt(txt_path):
    seeds = np.loadtxt(open(txt_path, 'rb'))
    if len(seeds.shape) == 1:
        seeds = seeds.reshape((1, 3))
    seeds = seeds.astype(np.uint32)
    return [seeds]

def is_ddp(args):
    return args.local_rank is not None

def get_worker_init_fn(rank):
    rank = 0 if rank is None else rank
    def worker_init_fn(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(initial_seed + worker_id + 123*rank)

    return worker_init_fn