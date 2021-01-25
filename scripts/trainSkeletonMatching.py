import time, copy

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import *


def train(args, train_loader, models, device, loss_fns, optimizer, scheduler, logger, writer, regularization=None):
    models[0].train() if args.train_end_to_end else models[0].eval()
    models[1].train()

    last_iteration_num, _ = restore_state(optimizer, scheduler, args, device)

    start = time.time()
    training_completed = False
    while not training_completed:
        for iteration, data in enumerate(train_loader, start=last_iteration_num + 1):
            sys.stdout.flush()

            if iteration < 50:
                print('time taken for itr: ', time.time() - start)
                start = time.time()

            _, volume, out_skeleton_1, out_skeleton_2, out_flux, out_skeleton_p, out_weight, match = data
            volume_gpu = volume.to(device)
            out_skeleton_1_gpu, out_skeleton_2_gpu = out_skeleton_1.to(device), out_skeleton_2.to(device)
            match = match.to(device)

            if not args.train_end_to_end and not args.use_penultimate:
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
                out_flux_gpu = out_flux.to(device)
                out_weight_gpu = out_weight.to(device)

            next_model_input = [volume_gpu, out_skeleton_1_gpu, out_skeleton_2_gpu, pred]

            if args.use_penultimate:
                last_layer = model_output['penultimate_layer']
                next_model_input.append(last_layer)

            out_match = models[1](torch.cat(next_model_input, dim=1))

            if not isinstance(loss_fns[0], nn.BCEWithLogitsLoss):
                out_match = torch.sigmoid(out_match)

            cls_loss = loss_fns[0](out_match, match)
            if args.train_end_to_end and args.use_flux_head:
                flux_loss, _, _ = loss_fns[1](pred, out_flux_gpu, out_weight_gpu)
                cls_loss_weight = 0.80
                loss = cls_loss_weight*cls_loss + (1.0 - cls_loss_weight)*flux_loss
                loss_dict = {'Total loss': loss.item(),
                             'Flux loss': (1.0 - cls_loss_weight)*flux_loss.item(),
                             'Cls loss': cls_loss_weight*cls_loss.item()}
            else:
                loss = cls_loss
                loss_dict = {'Total loss': loss.item()}

            # compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            print('[Iteration %d] train_loss=%0.4f lr=%.6f' % (iteration, loss.item(),
                                                               optimizer.param_groups[0]['lr']))
            if logger and writer:
                logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration,
                                                                         loss.item(), optimizer.param_groups[0]['lr']))
                writer.add_scalars('Loss', loss_dict, iteration)
                writer.add_scalars('LR', {'lr': optimizer.param_groups[0]['lr']}, iteration)

            if iteration % 500 == 0:
                title = "Image/GT_Skeleton"
                if writer:
                    visualize(volume, out_skeleton_1, out_skeleton_2, iteration, writer, title=title)

            # Termination condition
            training_completed = iteration >= args.iteration_total

            # Save model
            if (iteration % args.iteration_save) == 0 or training_completed:
                save_dict = {models[0].module.__class__.__name__ + '_state_dict': models[0].state_dict(),
                             models[1].module.__class__.__name__ + '_state_dict': models[1].state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'scheduler_state_dict': scheduler.state_dict(),
                             'loss': loss,
                             'iteration': iteration}
                torch.save(save_dict, args.output + '%d.pth' % iteration)

        last_iteration_num = iteration


def main():
    args = get_args(mode='train')

    save_cmd_line(args)  # Saving the command line args with machine name and time for later reference
    args.output = args.output + args.exp_name + '/'

    print('Initial setup')
    model_io_size, device = init(args)

    if args.disable_logging is not True:
        logger, writer = get_logger(args)
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

    print('Setup data')
    train_loader = get_input(args, model_io_size, 'train', model=None)

    print('Setup loss function')
    loss_fns = [nn.BCEWithLogitsLoss()]
    loss_fns.append(AngularAndScaleLoss(alpha=0.25))
    # loss_fns = [kornia.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')]

    print('Setup optimizer')
    model_parameters = list()
    [model_parameters.extend(model.parameters()) for model in models[1:]]

    optimizer = torch.optim.Adam(model_parameters, lr=1, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=1e-6, amsgrad=True)

    if args.lr_scheduler == 'step':
        optimizer.defaults['lr'] = args.lr
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=round(args.iteration_total / 5), gamma=0.75)
    elif args.lr_scheduler == 'linear':
        initial_lr = args.lr
        final_lr = args.lr_final
        decay_till_step = args.decay_till_step

        def linear_decay_lambda(step):
            lr = final_lr if step >= decay_till_step else \
                initial_lr + (float(step) / decay_till_step) * (final_lr - initial_lr)
            return lr

        # linear_decay_lambda = lambda step:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_decay_lambda)
    else:
        raise ValueError("Learning rate scheduler is not defined to any known types.")

    print('Start training.')
    train(args, train_loader, models, device, loss_fns, optimizer, scheduler, logger, writer)

    print('Training finished.')
    if args.disable_logging is not True:
        logger.close()
        writer.close()


if __name__ == "__main__":
    main()
