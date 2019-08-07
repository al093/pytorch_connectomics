import os,sys
import numpy as np
import h5py
import scipy

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from torch_connectomics.data.dataset import AffinityDataset, SynapseDataset, MitoDataset, MaskDataset, MaskDatasetDualInput
from torch_connectomics.data.utils import collate_fn, collate_fn_2, collate_fn_test
from torch_connectomics.data.augmentation import *
from torch_connectomics.utils.net.serialSampler import SerialSampler

TASK_MAP = {0: 'neuron segmentation',
            1: 'synapse detection',
            2: 'mitochondria segmentation',
            3: 'mask prediction'}
 

def get_input(args, model_io_size, mode='train'):
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

    if mode=='validation':
        seg_name = args.val_seg_name.split('@')
    elif mode=='train':
        seg_name = args.seg_name.split('@')

    if args.task == 3:
        assert args.seed_points is not None
        seed_points_files = args.seed_points.split('@')

    # 1. load data
    model_input = [None]*len(img_name)

    if mode=='train' or mode=='validation':
        assert len(img_name)==len(seg_name)
        model_label = [None]*len(seg_name)

    if mode=='train' or mode=='validation':
        # setup augmentor
        augmentor = Compose([Rotate(p=1.0),
                             Rescale(p=0.5),
                             Flip(p=1.0),
                             Elastic(alpha=12.0, p=0.75),
                             Grayscale(p=0.75),
                             MissingParts(p=0.9),
                             MissingSection(p=0.5),
                             MisAlignment(p=1.0, displacement=16)
                             # ,SwapZ(p=0.5)
                             ],
                             input_size = model_io_size)
        # augmentor = None # debug
    else:
        augmentor = None

    print('data augmentation: ', augmentor is not None)
    SHUFFLE = (mode=='train' or mode=='validation')
    print('batch size: ', args.batch_size)

    if mode == 'test':
        pad_size = model_io_size//2
    else:
        # pad_size = (0,0,0)
        pad_size = augmentor.sample_size//2
    pad_size = pad_size.astype(np.int64)

    for i in range(len(img_name)):

        # bs = [0, 0, 0]
        # be = [800, 500, 800]

        # Train Bounds
        bs = [374, 1242, 0]
        be = [1053, 3025, 1059]

        model_input[i] = np.array((h5py.File(img_name[i], 'r')['main'])[bs[0]:be[0], bs[1]:be[1], bs[2]:be[2]])/255.0
        print('Input Data size: ', model_input[i].shape)
        print(mode)
        if mode == 'test' and args.scale_input != 1:
            print('Original volume size: ', model_input[i].shape)
            model_input[i] = scipy.ndimage.zoom(model_input[i], float(args.scale_input))
            print('Final volume size: ', model_input[i].shape)

        if mode == 'train' or mode == 'validation':
            model_label[i] = np.array((h5py.File(seg_name[i], 'r')['main'])[bs[0]:be[0], bs[1]:be[1], bs[2]:be[2]])
            # model_label[i] = shrink_range_uint32(model_label[i])
            if args.task == 3:
                s_points[i] = load_list_from_h5(seed_points_files[i])
                for idx in range(len(s_points[i])):
                    b = s_points[i][idx]
                    b = b[b[:, 0] < be[0], :]
                    b = b[b[:, 1] < be[1], :]
                    b = b[b[:, 2] < be[2], :]
                    b = b - np.array([bs[0], bs[1], bs[2]], dtype=np.uint32)
                    b = b[b[:, 0] >= 0, :]
                    b = b[b[:, 1] >= 0, :]
                    b = b[b[:, 2] >= 0, :]
                    s_points[i][idx] = b

                #concatenate all the bins together and shuffle them
                # s_points[i] = [np.concatenate(s_points[i], axis=0)]
                # np.random.shuffle(s_points[i][0])

            print(img_name[i])
            print(seg_name[i])

            # crop input to match with labels
            label_sz = np.array(model_label[i].shape)
            input_sz = np.array(model_input[i].shape)
            diff = input_sz - label_sz
            if (np.any(diff > 0)):
                diff = diff // 2
                model_input[i] = model_input[i][diff[0]:diff[0] + label_sz[0], diff[1]:diff[1] + label_sz[1],
                                 diff[2]:diff[2] + label_sz[2]]

        model_input[i] = np.pad(model_input[i], ((pad_size[0],pad_size[0]),
                                                 (pad_size[1],pad_size[1]),
                                                 (pad_size[2],pad_size[2])), 'reflect')
        print("volume shape: ", model_input[i].shape)
        volume_shape.append(model_input[i].shape)
        model_input[i] = model_input[i].astype(np.float32)

        if mode=='train' or mode=='validation':
            model_label[i] = np.pad(model_label[i], ((pad_size[0], pad_size[0]),
                                                           (pad_size[1], pad_size[1]),
                                                           (pad_size[2], pad_size[2])), 'reflect')
            model_label[i] = model_label[i].astype(np.float32)

            print("label shape: ", model_label[i].shape)
            assert model_input[i].shape == model_label[i].shape

        if mode=='test' and args.task == 3:
            s_points[i] = load_seeds_from_txt(seed_points_files[i])
            b = s_points[i][0]
            b = b[b[:, 0] < be[0], :]
            b = b[b[:, 1] < be[1], :]
            b = b[b[:, 2] < be[2], :]
            b = b - np.array([bs[0], bs[1], bs[2]], dtype=np.uint32)
            b = b[b[:, 0] >= 0, :]
            b = b[b[:, 1] >= 0, :]
            b = b[b[:, 2] >= 0, :]
            s_points[i][0] = b

    if mode=='train' or mode=='validation':
        if augmentor is None:
            sample_input_size = model_io_size
        else:
            sample_input_size = augmentor.sample_size

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
            dataset = MaskDatasetDualInput(volume=model_input, label=model_label, sample_input_size=sample_input_size,
                                  sample_label_size=sample_input_size, augmentor=augmentor, mode = 'train',
                                  seed_points=s_points, pad_size=pad_size.astype(np.uint32))

        if args.task == 3:
            c_fn = collate_fn_2
        else:
            c_fn = collate_fn

        img_loader =  torch.utils.data.DataLoader(
              dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn=c_fn,
              num_workers=args.num_cpu, pin_memory=True)
        return img_loader

    else:
        if args.task == 0:
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

        if args.task != 3:
            img_loader =  torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn_test,
                    num_workers=args.num_cpu, pin_memory=True)
        else:
            assert len(img_name) == 1
            img_loader = SerialSampler(dataset, args.batch_size, pad_size, s_points[0][0] + pad_size)

        return img_loader, volume_shape, pad_size

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
