import os, glob, re
import numpy as np
import datetime

import torch
import torch.utils.data

from torch_connectomics.model.model_zoo import *
from torch_connectomics.libs.sync import DataParallelWithCallback

# tensorboardX
from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# functions
def init(args):
    sn = args.output+'/'

    if args.local_rank in [None, 0]:
        if not os.path.isdir(sn):
            os.makedirs(sn)

    # I/O size in (z,y,x), no specified channel number
    model_io_size = np.array([x for x in args.model_input.split(',')], dtype=np.uint32)

    # select training machine
    if args.local_rank is not None:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.local_rank in [None, 0]:
        print("output path: ", sn)
        print("device: ", device)

    return model_io_size, device

def get_logger(args):
    log_name = args.output+'/log'
    date = str(datetime.datetime.now()).split(' ')[0]
    time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
    log_name += '_approx_' + date + '_' + time + '_' + str(args.local_rank)
    logger = open(log_name+'.txt','w') # unbuffered, write instantly

    # tensorboardX
    writer = SummaryWriter(args.output + '/runs/' + date + '_' + time + '_' + str(args.local_rank))
    return logger, writer

def setup_model(args, device, model_io_size):

    MODEL_MAP = {'unetv0': unetv0,
                 'unetv1': unetv1,
                 'unetv2': unetv2,
                 'unetv3': unetv3,
                 'unetLite': unetLite,
                 'unetv3DualHead': unetv3DualHead,
                 'cNet': ClassificationNet,
                 'fluxNet': FluxNet,
                 'directionNet': DirectionNet,
                 'fpn': fpn,
                 'fluxToSkeletonHead': FluxToSkeletonHead}

    assert args.architecture in MODEL_MAP.keys()
    if args.task == 2:
        model = MODEL_MAP[args.architecture](in_channel=1, out_channel=args.out_channel, act='tanh')
    elif args.task == 6:
        assert args.architecture == 'directionNet'
        model = MODEL_MAP[args.architecture](in_channel=args.in_channel, input_sz=model_io_size)
    elif args.task == 5:
        assert args.architecture == 'cNet'
        model = MODEL_MAP[args.architecture](in_channel=args.in_channel)
    else:
        if args.architecture == 'fluxNet':
            model = MODEL_MAP[args.architecture](in_channel=args.in_channel,
                                                 aspp_dilation_ratio=args.aspp_dilation_ratio, symmetric=args.symmetric,
                                                 use_skeleton_head=args.use_skeleton_head, use_flux_head=args.use_flux_head)

            if hasattr(args, 'use_dropblock') and args.use_dropblock:
                model.init_dropblock(0.01, 0.15, 10, 24)
    print('model: ', model.__class__.__name__)

    if args.local_rank is not None:
        model = model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)
    else:
        model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
        model = model.to(device)

    latest_checkpoint_file = get_latest_checkpoint_path(args.output)

    if latest_checkpoint_file or args.load_model:
        print(f'Loading pretrained model: {args.pre_model}')
        checkpoint = torch.load(latest_checkpoint_file or args.pre_model, map_location=device)
        if checkpoint.get(model.module.__class__.__name__ + '_state_dict'):
            model.load_state_dict(checkpoint[model.module.__class__.__name__ + '_state_dict'], strict=True)
        else:
            print(f"Did not find {model.module.__class__.__name__ + '_state_dict'} model dict in the checkpoint file.")
    return model

def restore_state(optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, args, device):
    latest_checkpoint_file = get_latest_checkpoint_path(args.output)

    if latest_checkpoint_file or args.warm_start:
        if args.local_rank in [None, 0]:
            print('Trying to load optimizer and scheduler state from checkpoint.')

        checkpoint = torch.load(latest_checkpoint_file or args.pre_model, map_location=device)
        iteration = checkpoint.get('iteration', 0)

        if checkpoint.get('scheduler_state_dict', None):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if args.local_rank in [None, 0]:
                print("Scheduler restored.")

        if checkpoint.get('optimizer_state_dict', None):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.local_rank in [None, 0]:
                print("Optimizer restored.")
        loss = checkpoint.get('loss', 0)
        return iteration, loss
    else:
        return 0, 0.0

def get_latest_checkpoint_path(exp_dir):
    latest_checkpoint_file = None
    training_files = glob.glob(exp_dir + '*.pth')
    latest_itr = -1
    for f in training_files:
        itr = int(re.split(r'_|\.|/', os.path.basename(f))[-2])
        if itr > latest_itr:
            latest_checkpoint_file, latest_itr = f, itr
    return latest_checkpoint_file

def setup_lstm_model(args, device, model_io_size):
    model = LSTMHead(input_sz=model_io_size)
    model = model.to(device)

    if bool(args.load_model_lstm):
        print('Load pretrained LSTM model:')
        print(args.pre_model_lstm)
        model.load_state_dict(torch.load(args.pre_model_lstm, map_location=device))
    return model

def blend(sz, sigma=0.5, mu=0.0):  
    """
    Gaussian blending
    """
    zz, yy, xx = np.meshgrid(np.linspace(-1,1,sz[0], dtype=np.float32), 
                                np.linspace(-1,1,sz[1], dtype=np.float32),
                                np.linspace(-1,1,sz[2], dtype=np.float32), indexing='ij')

    dd = np.sqrt(zz*zz + yy*yy + xx*xx)
    ww = 1e-4 + np.exp(-( (dd-mu)**2 / ( 2.0 * sigma**2 )))
    print('weight shape:', ww.shape)

    return ww
