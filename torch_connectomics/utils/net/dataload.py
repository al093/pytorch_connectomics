import os,sys
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from torch_connectomics.data.dataset import AffinityDataset, SynapseDataset, MitoDataset
from torch_connectomics.data.utils import collate_fn, collate_fn_test
from torch_connectomics.data.augmentation import *

TASK_MAP = {0: 'neuron segmentation',
            1: 'synapse detection',
            2: 'mitochondria segmentation'}
 

def get_input(args, model_io_size, mode='train'):
    """Prepare dataloader for training and inference.
    """
    print('Task: ', TASK_MAP[args.task])
    assert mode in ['train', 'test', 'validation']

    if mode=='test':
        pad_size = model_io_size // 2
    else:
        #pad_size = (0,0,0)
        pad_size = model_io_size // 2

    volume_shape = []


    if mode=='validation':
        img_name = args.val_img_name.split('@')
    else:
        img_name = args.img_name.split('@')


    if mode=='validation':
        seg_name = args.val_seg_name.split('@')
    elif mode=='train':
        seg_name = args.seg_name.split('@')
    
    # 1. load data
    model_input = [None]*len(img_name)
    if mode=='train' or mode=='validation':
        assert len(img_name)==len(seg_name)
        model_label = [None]*len(seg_name)

    for i in range(len(img_name)):
        model_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0

        if mode=='train' or mode=='validation':
            model_label[i] = np.array(h5py.File(seg_name[i], 'r')['main'])

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
            model_label[i] = model_label[i].astype(np.float32)
            print("label shape: ", model_label[i].shape)
            model_label[i] = np.pad(model_label[i], ((pad_size[0],pad_size[0]),
                                                     (pad_size[1],pad_size[1]),
                                                     (pad_size[2],pad_size[2])), 'reflect')
            assert model_input[i].shape == model_label[i].shape

    if mode=='train' or mode=='validation':
        # setup augmentor
        augmentor = Compose([Rotate(p=1.0),
                             Rescale(p=0.5),
                             Flip(p=1.0),
                             Elastic(alpha=12.0, p=0.75),
                             Grayscale(p=0.75),
                             MissingParts(p=0.9),
                             MissingSection(p=0.5),
                             MisAlignment(p=1.0, displacement=16)], 
                             input_size = model_io_size)
        # augmentor = None # debug
    else:
        augmentor = None

    print('data augmentation: ', augmentor is not None)
    SHUFFLE = (mode=='train' or mode=='validation')
    print('batch size: ', args.batch_size)

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

        img_loader =  torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn,
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

        img_loader =  torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn_test,
                num_workers=args.num_cpu, pin_memory=True)                  
        return img_loader, volume_shape, pad_size