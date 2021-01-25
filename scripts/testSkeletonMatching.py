import copy
from sklearn.metrics import roc_curve, precision_recall_curve
from tqdm import tqdm
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

    def compute_and_plot(self, tb_writer: tfx.SummaryWriter = None, itr = 0):
        fpr, tpr, th = roc_curve(np.array(self.gt), np.array(self.pred))
        if tb_writer:
            for y, x in zip(tpr, fpr):
                tb_writer.add_scalars('Graphs', {'ROC': y*100}, x*100)
            tb_writer.flush()

        p, r, th = precision_recall_curve(np.array(self.gt), np.array(self.pred))

        if tb_writer:
            tb_writer.add_pr_curve('pr_curve', np.array(self.gt), np.array(self.pred), itr)
            for y, x in zip(p, r):
                tb_writer.add_scalars('Graphs', {'PR-Curve': y*100}, x*100)
            tb_writer.flush()

        return p, r, th

def eval(args, val_loader, models, metrics, device, writer, save_output, itr=0):
    for m in models: m.eval()
    results = []
    for iteration, data in enumerate(tqdm(val_loader), start=1):
        sys.stdout.flush()

        sample, volume, out_skeleton_1, out_skeleton_2, out_flux, out_skeleton_p, match = data

        volume_gpu, match_gpu = volume.to(device), match.to(device)
        out_skeleton_1_gpu, out_skeleton_2_gpu = out_skeleton_1.to(device), out_skeleton_2.to(device)

        with torch.no_grad():
            if not (args.train_end_to_end or args.use_penultimate):
                if args.use_skeleton_head:
                    pred = out_skeleton_p.to(device)
                elif args.use_flux_head:
                    pred = out_flux.to(device)
            else:
                model_output = models[0](volume_gpu, get_penultimate_layer=True)
                if args.use_skeleton_head and not args.use_flux_head:
                    output_key = 'skeleton'
                elif args.use_flux_head and not args.use_skeleton_head:
                    output_key = 'flux'
                else:
                    raise NotImplementedError("Matching implemented only with one head for now.")
                pred = model_output[output_key]

            next_model_input = [volume_gpu, out_skeleton_1_gpu, out_skeleton_2_gpu, pred]

            if args.use_penultimate:
                last_layer = model_output['penultimate_layer']
                next_model_input.append(last_layer)

            out_match = models[1](torch.cat(next_model_input, dim=1))
            out_match = torch.sigmoid(out_match)

        metrics[0].append(out_match, match_gpu)

        # append to results list
        results.extend(list(zip(sample,
                                match.detach().numpy(),
                                out_match.detach().cpu().numpy())))
        if save_output:
            np.save(args.output + 'cls_results.npy', results)
    return results, metrics[0].compute_and_plot(writer, itr)

def _run(args, save_output):
    args.output = args.output + args.exp_name + '/'

    model_io_size, device = init(args)

    if args.disable_logging is not True:
        _, writer = get_logger(args)
    else:
        logger, writer = None, None
        print('No log file would be created.')

    classification_model = setup_model(args, device, model_io_size)

    print('Setting up flux/skeleton model')
    args_2 = copy.deepcopy(args)
    args_2.task = 4
    args_2.architecture = 'fluxNet'
    args_2.in_channel = 1
    args_2.out_channel = 3
    flux_model = setup_model(args_2, device, model_io_size)

    models = [flux_model, classification_model]

    val_loader, _, _ = get_input(args, model_io_size, 'test', model=None)

    metrics = [Accuracy()]

    checkpoint = torch.load(args.pre_model, map_location=device)
    iteration = checkpoint.get('iteration', 0)

    print('Start Evaluation.')
    out = eval(args, val_loader, models, metrics, device, writer, save_output, itr=iteration)

    print('Evaluation finished.')
    if args.disable_logging is not True:
        writer.close()
    return out

def run(input_args_string, save_output):
    return _run(get_args(mode='test', input_args=input_args_string), save_output)

if __name__ == "__main__":
    _run(get_args(mode='test'), save_output=True)
