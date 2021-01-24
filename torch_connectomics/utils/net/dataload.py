import numpy as np
import h5py
from functools import partial

import torch
import torch.utils.data

from torch_connectomics.data.dataset import AffinityDataset, FluxAndSkeletonDataset, MatchSkeletonDataset
from torch_connectomics.data.utils import collate_fn_var
from torch_connectomics.data.augmentation import *

TASK_MAP = {4: 'flux/skeleton prediction',
            5: 'skeleton matching'}

def get_input(args, model_io_size, mode='train', model=None):
    """Prepare dataloader for training and inference. """

    print('Task: ', TASK_MAP[args.task])
    assert mode in ['train', 'test']

    img_files =                     args.img_name.split('@')
    label_files =                   args.label_name.split('@')          if args.label_name else None
    seed_points_files =             args.seed_points.split('@')         if args.seed_points else None
    skeleton_files =                args.skeleton_name.split('@')       if args.skeleton_name else None
    flux_files =                    args.flux_name.split('@')           if args.flux_name else None
    weight_files =                  args.weight_name.split('@')         if args.weight_name else None
    skeleton_probability_files =    args.skel_prob_name.split('@')      if args.skel_prob_name else None

    img =                   [None] * len(img_files)
    img_shape =             [None] * len(img_files)
    label =                 [None] * len(label_files)                   if label_files else None
    seed_points =           [None] * len(seed_points_files)             if seed_points_files else None
    skeleton =              [None] * len(skeleton_files)                if skeleton_files else None
    flux =                  [None] * len(flux_files)                    if flux_files else None
    weight =                [None] * len(weight_files)                  if weight_files else None
    skeleton_probability =  [None] * len(skeleton_probability_files)    if skeleton_probability_files else None


    if mode is 'train' and args.data_aug:
        elastic_augmentor = Elastic(alpha=6.0, p=0.75)
        augmentation_methods = [Rotate(p=0.5), Flip(p=0.5), elastic_augmentor, Grayscale(p=0.75),
                                Blur(min_sigma=1, max_sigma=2, min_slices=model_io_size[0]//6,
                                     max_slices=model_io_size[0]//4, p=0.4),
                                CutNoise(), CutBlur(), MotionBlur(), MissingParts(p=0.5)]

        # if the input is symmetric, and more importantly if the resolution is isometric
        # we can perform swapping of z with (y or x) axis
        if args.symmetric:
            augmentation_methods.append(SwapZ(0.5))

        augmentor = Compose(augmentation_methods,
                            input_size=model_io_size)
        elastic_augmentor.set_input_sz(augmentor.sample_size)
        sample_input_size = augmentor.sample_size
        if args.pad_input:
            pad_size = np.array(sample_input_size//2, dtype=np.int64)
        else:
            pad_size = np.array((0, 0, 0), dtype=np.int64)
    else:
        augmentor = None
        sample_input_size = model_io_size
        pad_size = np.array(model_io_size // 2, dtype=np.int64)

    pad_size_tuple = ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2]))
    print('Data augmentation: ', augmentor is not None)
    print('Padding inputs by: ', pad_size)

    for i in range(len(img_files)):
        if not is_ddp(args):
            image = np.array((h5py.File(img_files[i], 'r')['main']))
            img[i] = image
            if image.dtype in [np.float16, np.float32, np.float64]:
                img[i] = np.array(image, copy=False, dtype=np.float32)
            elif image.dtype == np.uint8:
                img[i] = np.array(image / np.float32(255.0), copy=False, dtype=np.float32)
            else:
                raise Exception('Image datatype was not uint8 or float, not sure how to normalize.')
            img[i] = np.pad(img[i], pad_size_tuple, 'reflect')
            img[i] = img[i].astype(np.float32)
        else:
            img[i] = h5py.File(img_files[i], 'r')['main']
            if not np.all(pad_size == 0):
                raise NotImplementedError("In Distributed Data parallel mode padding the input volumes"
                                          "is not supported yet")

        print(f"Image name: {img_files[i]}")
        print(f"Shape: {img[i].shape}")
        img_shape[i] = img[i].shape

        if label_files:
            if is_ddp(args):
                label[i] = h5py.File(label_files[i], 'r')['main']
            else:
                label[i] = np.array((h5py.File(label_files[i], 'r')['main']))
                label[i] = np.pad(label[i], pad_size_tuple, 'reflect')

        if seed_points_files:
            if args.task == 4:
                seed_points[i] = load_list_from_h5(seed_points_files[i])
                if not args.pad_input:
                    half_aug_sz = sample_input_size // 2
                    vol_size = img[i].shape
                    new_list = []
                    for idx in range(len(seed_points[i])):
                        b = seed_points[i][idx]
                        b = b[np.all(b >= half_aug_sz, axis=1), :]
                        b = b[np.all((vol_size - b) > half_aug_sz, axis=1), :]
                        if b.shape[0] > 0:
                            new_list.append(b)
                    seed_points[i] = new_list
            else:
                seed_points[i] = np.load(seed_points_files[i], allow_pickle=True).item()

        if skeleton_probability_files:
            skeleton_probability[i] = np.array((h5py.File(skeleton_probability_files[i], 'r')['main']))
            skeleton_probability[i] = skeleton_probability[i].astype(np.float32, copy=False)
            skeleton_probability[i] = np.pad(skeleton_probability[i], pad_size_tuple)

        if skeleton_files:
            if is_ddp(args):
                skeleton[i] = h5py.File(skeleton_files[i], 'r')['main']
            else:
                skeleton[i] = np.array((h5py.File(skeleton_files[i], 'r')['main']))
                skeleton[i] = np.pad(skeleton[i], pad_size_tuple, 'reflect')

        if flux_files:
            if is_ddp(args):
                flux[i] = h5py.File(flux_files[i], 'r')['main']
            else:
                flux[i] = np.array((h5py.File(flux_files[i], 'r')['main']), dtype=np.float32)
                flux[i] = np.pad(flux[i], ((0,0),) + pad_size_tuple, 'reflect')

        if weight_files:
            if is_ddp(args):
                weight[i] = h5py.File(weight_files[i], 'r')['main']
            else:
                weight[i] = np.array((h5py.File(weight_files[i], 'r')['main']))
                weight[i] = np.pad(weight[i], pad_size_tuple, 'reflect')


        if mode=='train': assert img[i].shape == label[i].shape, 'Image size and label size are different'

    if mode=='train':
        if args.task == 4:  # skeleton/flux prediction
            dataset = partial(FluxAndSkeletonDataset, sample_input_size=sample_input_size,
                                             sample_label_size=sample_input_size,
                                             augmentor=augmentor, mode='train', seed_points=seed_points,
                                             pad_size=pad_size.astype(np.int32), dataset_resolution=args.resolution,
                                             sample_whole_vol=args.sample_whole_volume)
            if is_ddp(args):
                # if ddp is used call the dataset with paths of the input, label, skeleton, flux and weight
                # The dataset will read small chunks of data as needed iso of loading the entire volume in memory.
                model_input_paths = img_files
                model_label_paths = label_files
                skeleton_paths = skeleton_files
                flux_paths = flux_files
                weight_paths = weight_files
                dataset = dataset(volume=model_input_paths, label=model_label_paths,
                                  skeleton=skeleton_paths, flux=flux_paths, weight=weight_paths)
            else:
                dataset = dataset(volume=img, label=label, skeleton=skeleton, flux=flux, weight=weight)

        elif args.task == 5:  # skeleton match prediction
            dataset = MatchSkeletonDataset(image=img, skeleton=skeleton, flux=flux,
                                           skeleton_probability=skeleton_probability, sample_input_size=sample_input_size,
                                           sample_label_size=sample_input_size, augmentor=augmentor, mode='train',
                                           seed_points=seed_points, pad_size=pad_size.astype(np.int32),
                                           dataset_resolution=args.resolution)
        else:
            raise NotImplementedError('Task not found.')

        if is_ddp(args):
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        else:
            train_sampler = None

        img_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 collate_fn=collate_fn_var, num_workers=args.num_cpu,
                                                 pin_memory=True, shuffle=not train_sampler,
                                                 worker_init_fn=get_worker_init_fn(args.local_rank),
                                                 sampler=train_sampler)
        return img_loader
    else:
        if args.task == 4:
            dataset = AffinityDataset(volume=img, label=None, sample_input_size=model_io_size,
                                      sample_label_size=None, sample_stride=model_io_size // 2,
                                      augmentor=None, mode='test')
        elif args.task == 5:
            dataset = MatchSkeletonDataset(image=img, skeleton=skeleton, flux=flux, skeleton_probability=skeleton_probability,
                                           sample_input_size=model_io_size, sample_label_size=model_io_size,
                                           augmentor=None, mode='test', seed_points=seed_points,
                                           pad_size=pad_size.astype(np.int32))
        else:
            raise NotImplementedError('Task not found.')

        img_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=False, collate_fn=collate_fn_var,
                                                 num_workers=args.num_cpu, pin_memory=True)

        return img_loader, img_shape, pad_size


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


def is_ddp(args):
    return args.local_rank is not None


def get_worker_init_fn(rank):
    rank = 0 if rank is None else rank
    def worker_init_fn(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(initial_seed + worker_id + 123*rank)

    return worker_init_fn