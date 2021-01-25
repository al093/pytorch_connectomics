import argparse
import sys
import datetime
import os

import numpy as np

class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        with values as f:
            print(f)
            parser.parse_known_args(f.read().split(), namespace)

def get_args(mode='train', input_args=None):

    assert mode in ['train', 'test']
    parser = argparse.ArgumentParser(description=f'Specify model {mode} arguments.')

    parser.add_argument('--task', type=int, default=4, help='Specify the task')
    parser.add_argument('-en', '--exp-name', type=str, default=None,
                        help='Name of the model files to be saved while training, '
                             'for testing name for the folder to save the output files')

    # I/O
    parser.add_argument('-o','--output', default=None, help='Output path')
    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204', help='I/O size of deep network')
    parser.add_argument('-dl', '--disable-logging', type=my_bool, default=False, help='If True, Tensorflow and other log files are NOT created.')

    # model option
    parser.add_argument('-ac', '--architecture', help='model architecture')
    parser.add_argument('-lm', '--load-model', type=my_bool, default=False, help='Use pre-trained model')
    parser.add_argument('-pm', '--pre-model', type=str, default='', help='Pre-trained model path')
    parser.add_argument('--out-channel', type=int, default=3, help='Number of output channel(s).')
    parser.add_argument('--in-channel', type=int, default=1, help='Number of input channel(s).')
    parser.add_argument('--aspp-dilation-ratio', type=int, default=1,
                        help='Convolutional Dilation along Z slices can be different if ratio is other than 1.'
                             'Should be ideally equal to the resolution anisotropy. Only defined for FluxNet.')
    parser.add_argument('--symmetric', type=my_bool, default=False, help='Pass true if the resolution is isotropic.')
    parser.add_argument('--use-skeleton-head', type=my_bool, default=False, help='Use Skeleton head at the end of decoder')
    parser.add_argument('--use-flux-head', type=my_bool, default=False, help='Use flux head at the end of decoder.')
    parser.add_argument('--use-penultimate', type=my_bool, default=False,
                        help='Will use penultimate layer also for skeleton matching classifier.')

    # machine option
    parser.add_argument('--num-gpu', type=int,  default=-1, help='Number of gpu')
    parser.add_argument('-c', '--num-cpu', type=int,  default=1, help='Number of cpu')
    parser.add_argument('-b', '--batch-size', type=int,  default=1, help='Batch size')
    parser.add_argument("--local_rank", type=int, default=None, help='Specified when using DDP')

    if mode == 'train':

        parser.add_argument('-ft','--finetune', type=bool, default=False, help='Fine-tune on previous model [Default: False]')
        parser.add_argument('--data-aug', type=my_bool, default=False, help='Augment data')
        parser.add_argument('--pad-input', type=my_bool, default=False,
                            help='Pad all input volumes with half of the augmentor sample size.')
        parser.add_argument('--use-dropblock', type=my_bool, default=False, help='Use Dropblock for augmentation.')
        parser.add_argument('--sample-whole-volume', type=my_bool, default=False,
                            help='If True, it ignore the seed points files and samples '
                                 'training blocks from the whole volume instead.')

        # optimization option
        parser.add_argument('-lt', '--loss', type=int, default=1, help='Loss function')
        parser.add_argument('-lr', type=float, default=0.0001, help='Initial learning rate')
        parser.add_argument('--iteration-total', type=int, default=1000, help='Total number of iteration')
        parser.add_argument('--iteration-save', type=int, default=100, help='Number of iteration to save')
        parser.add_argument('--lr-scheduler', type=str, default='step', help='Learning rate schedule. Possible values: step, linear')
        parser.add_argument('--warm-start', type=my_bool, default=False,
                            help='If --load-model and --pm is set, and if '
                                 'warm starting the training is needed, set this to true.')
        parser.add_argument('--lr-final', type=float, default=1e-5,
                          help='Final Learning rate for linearDecay schedule')
        parser.add_argument('--decay-till-step', type=int, default=50000,
                          help='Constant (--lr-final) learning rate after this step for linearDecay schedule.')


    if mode == 'test':
        parser.add_argument('-ta', '--test-augmentation',  type=my_bool, default=False,
                            help='Perform Data Augmentation during inference')
        parser.add_argument('-si', '--scale-input', type=float, default='1.0',
                            help='Scale factor for entire input volume')
        parser.add_argument('-is', '--initial-seg', type=str, default=None,
                            help='Initial segmentation volume (e.g. neurons)')

    # Data files needed for specific tasks, not all need to provided.
    parser.add_argument('-dn', '--img-name', default=None, help='Image data path')
    parser.add_argument('-sp', '--seed-points', type=str, default=None,
                        help='File path for seed points which need to be trained or tested on')

    parser.add_argument('-ln', '--label-name', default=None, help='Ground-truth label|segmentation|skeleton-ctx path')
    parser.add_argument('-skn', '--skeleton-name', type=str, default=None, help='Ground-truth skeleton path')
    parser.add_argument('-fn', '--flux-name', type=str, default=None, help='Ground-truth Flux path')
    parser.add_argument('-fngt', '--flux-name-gt', type=str, default=None,
                        help='Ground-truth Flux path for end-to-end training of skeleton growing.')
    parser.add_argument('--skel-prob-name', type=str, default=None, help='Divergence volumes path for skeleton matching.')
    parser.add_argument('--train-end-to-end', type=my_bool, default=False, help='Run all models and train them simultaneously.')
    parser.add_argument('-res', '--resolution', type=str, default='30.0,6.0,6.0', help='Resolution of input volumes (Z, Y, X)')
    parser.add_argument('-wn', '--weight-name', type=str, default=None, help='Weight array to be used for loss calculations.')

    parser.add_argument('--argsFile', type=open, action=LoadFromFile)

    args, unknown = parser.parse_known_args(args=input_args or None)

    if args.resolution:
        args.resolution = split_resolution_string(args.resolution)

    return args

def save_cmd_line(args, filepath=None):
    # saving all command line args to file
    if not filepath:
        filepath = '../experimentDetails/' + args.exp_name

    with open(filepath + '.txt', 'w') as f:
        f.write(str(datetime.datetime.now()))
        f.write(os.uname()[1])
        f.write('\n'.join(sys.argv[1:]))

def my_bool(s):
    return s == 'True' or s == 'true' or s == '1'

def split_resolution_string(resolution:str)->np.array:
    return np.array([x for x in resolution.split(',')], dtype=np.float32)