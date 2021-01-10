import os,sys
import h5py, time, itertools, datetime
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve
import torch
import tensorboardX as tfx

from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import *


class Accuracy():
    def __init__(self, threshold=0.50):
        self.threshold = threshold
        self.pred = list()
        self.gt = list()

    def append(self, pred, gt, *args, **kwargs):
        if type(gt) is torch.Tensor:
            gt = gt.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
        gt = gt.ravel()
        pred = pred.ravel()
        self.gt.extend(gt)
        self.pred.extend(pred)

    def compute_and_plot(self, tb_writer: tfx.SummaryWriter = None):
        fpr, tpr, th = roc_curve(np.array(self.gt), np.array(self.pred))
        if tb_writer:
            for y, x in zip(tpr, fpr):
                tb_writer.add_scalars('Graphs', {'ROC': y*100}, x*100)
            tb_writer.flush()

        p, r, th = precision_recall_curve(np.array(self.gt), np.array(self.pred))

        if tb_writer:
            tb_writer.add_pr_curve('pr_curve', np.array(self.gt), np.array(self.pred), 0)
            for y, x in zip(p, r):
                tb_writer.add_scalars('Graphs', {'PR-Curve': y*100}, x*100)
            tb_writer.flush()

        return p, r, th

def eval(args, val_loader, models, metrics, device, writer, save_output):
    for m in models: m.eval()
    results = []
    for iteration, data in enumerate(tqdm(val_loader), start=1):
        sys.stdout.flush()

        sample, volume, out_skeleton_1, out_skeleton_2, out_flux, match = data

        volume_gpu, match_gpu = volume.to(device), match.to(device)
        out_skeleton_1_gpu, out_skeleton_2_gpu = out_skeleton_1.to(device), out_skeleton_2.to(device)

        if not (args.train_end_to_end or args.use_penultimate):
            pred_flux = out_flux.to(device)
        else:
            with torch.no_grad():
                model_output = models[0](volume_gpu, get_penultimate_layer=True)
                pred_flux = model_output['flux']

        next_model_input = [volume_gpu, out_skeleton_1_gpu, out_skeleton_2_gpu, pred_flux]

        if args.use_penultimate:
            last_layer = model_output['penultimate_layer']
            next_model_input.append(last_layer)

        out_match = models[1](torch.cat(next_model_input, dim=1))
        out_match = torch.nn.functional.sigmoid(out_match)

        metrics[0].append(out_match, match_gpu)

        # append to results list
        results.extend(list(zip(sample,
                                match.detach().numpy(),
                                out_match.detach().cpu().numpy())))
        if save_output:
            np.save(args.output + 'cls_results.npy', results)
    return results, metrics[0].compute_and_plot(writer)

def _run(args, save_output):
    args.output = args.output + args.exp_name + '/'

    model_io_size, device = init(args)

    if args.disable_logging is not True:
        _, writer = get_logger(args)
    else:
        logger, writer = None, None
        print('No log file would be created.')

    classification_model = setup_model(args, device, model_io_size, non_linearity=(torch.sigmoid,))

    class ModelArgs(object):
        pass
    args2 = ModelArgs()
    args2.task = 4
    args2.architecture = 'fluxNet'
    args2.in_channel = 1
    args2.out_channel = 3
    args2.num_gpu = args.num_gpu
    args2.pre_model = args.pre_model
    args2.load_model = args.load_model
    args2.use_skeleton_head = args.use_skeleton_head
    args2.use_flux_head = args.use_flux_head
    args2.aspp_dilation_ratio = args.aspp_dilation_ratio
    args2.resolution = args.resolution
    args2.symmetric = args.symmetric
    args2.batch_size = args.batch_size
    args2.local_rank = args.local_rank
    flux_model = setup_model(args2, device, model_io_size, non_linearity=(torch.tanh,))
    models = [flux_model, classification_model]

    val_loader, _, _ = get_input(args, model_io_size, 'test', model=None)

    metrics = [Accuracy()]

    print('Start Evaluation.')
    out = eval(args, val_loader, models, metrics, device, writer, save_output)

    print('Evaluation finished.')
    if args.disable_logging is not True:
        writer.close()
    return out

def run(input_args_string, save_output):
    return _run(get_args(mode='test', input_args=input_args_string), save_output)

if __name__ == "__main__":
    _run(get_args(mode='test'), save_output=True)
