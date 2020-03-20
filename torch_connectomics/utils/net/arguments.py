import argparse
import sys
import datetime
import os


class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            print(f)
            parser.parse_args(f.read().split(), namespace)

def get_args(mode='train', input_args=None):

    assert mode in ['train', 'test']
    if mode == 'train':
        parser = argparse.ArgumentParser(description='Specify model training arguments.')
    else:
        parser = argparse.ArgumentParser(description='Specify model inference arguments.')
    
    # define tasks
    # {0: neuron segmentationn, 1: synapse detection, 2: mitochondira segmentation}
    parser.add_argument('--task', type=int, default=0,
                        help='specify the task')

    # I/O
    parser.add_argument('-t','--train',  default='/n/coxfs01/',
                        help='Input folder (train)')
    parser.add_argument('-dn','--img-name',  default='im_uint8.h5',
                        help='Image data path')
    parser.add_argument('-o','--output', default='result/train/',
                        help='Output path')
    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204',
                        help='I/O size of deep network')
    parser.add_argument('-dl', '--disable-logging', type=my_bool, default=False,
                        help='If True, Tensorflow and other log files are NOT created.')

    # model option
    parser.add_argument('-ac', '--architecture', help='model architecture')
    parser.add_argument('-lm', '--load-model', type=my_bool, default=False,
                        help='Use pre-trained model')                
    parser.add_argument('-pm', '--pre-model', type=str, default='',
                        help='Pre-trained model path')
    parser.add_argument('-ism', '--init-second-model', type=my_bool, default=False,
                        help='Train/Test second model together')
    parser.add_argument('-lm_2', '--load-model-second', type=my_bool, default=False,
                        help='Use pre-trained second model')
    parser.add_argument('-pm_2', '--pre-model-second', type=str, default='',
                        help='Pre-trained second model path')
    parser.add_argument('--out-channel', type=int, default=3,
                        help='Number of output channel(s).')
    parser.add_argument('--in-channel', type=int, default=1,
                        help='Number of input channel(s).')

    # machine option
    parser.add_argument('-g', '--num-gpu', type=int,  default=1,
                        help='Number of gpu')
    parser.add_argument('-c', '--num-cpu', type=int,  default=1,
                        help='Number of cpu')
    parser.add_argument('-b', '--batch-size', type=int,  default=1,
                        help='Batch size')

    #extra options(task specific)
    parser.add_argument('-sp', '--seed-points', type=str, default=None,
                        help='File path for seed points which need to be trained or tested on')

    if mode == 'train':
        parser.add_argument('-ln','--seg-name',  default='seg-groundtruth2-malis.h5',
                            help='Ground-truth label path')

        #Added input args for validation set
        parser.add_argument('-vln', '--val-seg-name', default='seg-groundtruth2-malis.h5',
                            help='Validation Ground-truth label path')

        parser.add_argument('-vdn', '--val-img-name', default='im_uint8.h5',
                            help='Validation Image data path')

        parser.add_argument('-ft','--finetune', type=bool, default=False,
                            help='Fine-tune on previous model [Default: False]')

        parser.add_argument('-tgdtx','--train-grad-dtx', type=my_bool, default=True,
                            help='If False, will train from gradient to skeleton predictions')

        # optimization option
        parser.add_argument('-lt', '--loss', type=int, default=1,
                            help='Loss function')
        parser.add_argument('-lr', type=float, default=0.0001,
                            help='Learning rate')
        parser.add_argument('--iteration-total', type=int, default=1000,
                            help='Total number of iteration')
        parser.add_argument('--iteration-save', type=int, default=100,
                            help='Number of iteration to save')

    if mode == 'test':
        parser.add_argument('-ta', '--test-augmentation',  type=my_bool, default=False,
                            help='Perform Data Augmentation during inference')
        parser.add_argument('-si', '--scale-input', type=float, default='1.0',
                            help='Scale factor for entire input volume')
        parser.add_argument('-is', '--initial-seg', type=str, default=None,
                            help='Initial segmentation volume (e.g. neurons)')
        parser.add_argument('-sid', '--segment-id', type=int, default=1,
                            help='segmentation id of the neuron to be segmented')

    parser.add_argument('-en', '--exp-name', type=str, default=None,
                        help='Name of the model files to be saved while training, for testing name for the folder to save the output files')

    parser.add_argument('-skn', '--skeleton-name', type=str, default=None,
                        help='Ground-truth skeleton path')

    parser.add_argument('-fn', '--flux-name', type=str, default=None,
                        help='Ground-truth Flux path')

    parser.add_argument('-fngt', '--flux-name-gt', type=str, default=None,
                        help='Ground-truth Flux path for end-to-end training of skeleton growing.')

    parser.add_argument('-divn', '--div-name', type=str, default=None,
                        help='Divergence volumes path for skeleton tracking.')

    parser.add_argument('-te2e', '--train-end-to-end', type=my_bool, default=False,
                        help='Train skeleton growing end to end, will need fn_2 to be defined.')

    parser.add_argument('-res', '--resolution', type=str, default='30.0,6.0,6.0',
                        help='Resolution of input volumes (Z, Y, X)')

    parser.add_argument('-tsteps', '--tracking-steps', type=int, default=32,
                        help='Total allowed steps for tracking network ')

    parser.add_argument('-upc', '--use-precomputed', type=my_bool, default=True,
                        help='Use precomputed flux, gradient and global features for training tracking')

    parser.add_argument('-wn', '--weight-name', type=str, default=None,
                        help='Weight array to be used for loss calculations.')

    parser.add_argument('--argsFile', type=open, action=LoadFromFile)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

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
