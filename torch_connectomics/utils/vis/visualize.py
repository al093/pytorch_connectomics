import torch
import torchvision.utils as vutils
import h5py
import numpy as np

N = 15 # default maximum number of sections to show
min_batch = 3
def prepare_data(volume, label, output, input_label=None):

    if len(volume.size()) == 4:   # 2D Inputs
        if volume.size()[0] > N:
            return volume[:N], label[:N], output[:N]
        else:
            return volume, label, output
    elif len(volume.size()) == 5: # 3D Inputs
        if(volume.shape[0] >= min_batch): #show slices from different batches
            start_slice_number = volume.shape[2]//2 - int(N/min_batch)//2
            volume = volume[:min_batch, :, start_slice_number:start_slice_number+int(N/min_batch), :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, volume.shape[1], volume.shape[3], volume.shape[4])
            label = label[:min_batch,   :, start_slice_number:start_slice_number+int(N/min_batch), :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, label.shape[1], label.shape[3], label.shape[4])
            output = output[:min_batch, :, start_slice_number:start_slice_number+int(N/min_batch), :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, output.shape[1], output.shape[3], output.shape[4])
            if input_label is not None:
                input_label = input_label[:min_batch,   :, start_slice_number:start_slice_number+int(N/min_batch), :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, input_label.shape[1], input_label.shape[3], input_label.shape[4])
        else:
            volume, label, output = volume[0].permute(1,0,2,3), label[0].permute(1,0,2,3), output[0].permute(1,0,2,3)
            if input_label is not None:
                input_label = input_label[0].permute(1, 0, 2, 3)
        if volume.size()[0] > N:
            if input_label is not None:
                return volume[:N], label[:N], output[:N], input_label[:N]
            else:
                return volume[:N], label[:N], output[:N]
        else:
            if input_label is not None:
                return volume, label, output, input_label
            else:
                return volume, label, output

def visualize(volume, label, output, iteration, writer, mode='Train', input_label=None):

    if input_label is not None:
        volume, label, output, input_label = prepare_data(volume, label, output, input_label)
    else:
        volume, label, output = prepare_data(volume, label, output)

    sz = volume.size() # z,c,y,x
    volume_visual = volume.detach().cpu().expand(sz[0], 3, sz[2], sz[3])
    if output.shape[1] == 1:
        output_visual = output.detach().cpu().expand(sz[0], 3, sz[2], sz[3])
        label_visual = label.detach().cpu().expand(sz[0], 3, sz[2], sz[3])
    elif output.shape[1] > 1:
        output_visual = output.detach().cpu()
        label_visual = label.detach().cpu()

    if input_label is not None:
        if output.shape[1] == 1:
            input_label_visual = input_label.detach().cpu().expand(sz[0], 3, sz[2], sz[3])
        elif output.shape[1] > 1:
            input_label_visual = input_label.detach().cpu()

    canvas = []
    for idx in range(volume_visual.shape[0]):
        canvas.append(volume_visual[idx])

        if input_label is not None:
            if input_label.shape[1] == 1:
                canvas.append(input_label_visual[idx])
            elif input_label.shape[1] > 1:
                for i in range(input_label_visual.shape[1]):
                    canvas.append(input_label_visual[idx, i:i+1].expand(3, sz[2], sz[3]))

        if output.shape[1] == 1:
            canvas.append(output_visual[idx])
            canvas.append(label_visual[idx])
        elif output.shape[1] > 1:
            for i in range(output.shape[1]):
                canvas.append(output_visual[idx, i:i+1].expand(3, sz[2], sz[3]))
                canvas.append(label_visual[idx, i:i+1].expand(3, sz[2], sz[3]))

    nrow = volume.shape[1] + label.shape[1] + output.shape[1]
    if input_label is not None:
        nrow += input_label.shape[1]

    canvas_show = vutils.make_grid(canvas, nrow=nrow, normalize=False, scale_each=True)

    writer.add_image(mode + ' Mask', canvas_show, iteration)

    # volume_show = vutils.make_grid(volume_visual, nrow=N, normalize=False, scale_each=True)
    # output_show = vutils.make_grid(output_visual, nrow=N, normalize=False, scale_each=True)
    # label_show = vutils.make_grid(label_visual, nrow=N, normalize=False, scale_each=True)
    # writer.add_image('Input', volume_show, iteration)
    # writer.add_image('Label', label_show, iteration)
    # writer.add_image('Output', output_show, iteration)

def visualize_aff(volume, label, output, iteration, writer, mode='Train'):
    volume, label, output = prepare_data(volume, label, output)

    sz = volume.size() # z,c,y,x
    canvas = []
    volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    canvas.append(volume_visual)
    output_visual = [output[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(3)]
    label_visual = [label[:,i].detach().cpu().unsqueeze(1).expand(sz[0],3,sz[2],sz[3]) for i in range(3)]

    error_map = [torch.abs(output_visual[i] - label_visual[i]) for i in range(3)]
    for i in range(3):
        error_map[i][:, 1:3] = 0

    canvas = canvas + output_visual
    canvas = canvas + label_visual
    canvas = canvas + error_map
    reorder = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9]
    canvas = [canvas[i] for i in reorder]
    canvas_merge = torch.cat(canvas, 0)

    canvas_merge = canvas_merge[torch.tensor([c + N*r for c in range(N) for r in range(10)])]
    canvas_show = vutils.make_grid(canvas_merge, nrow=10, normalize=True, scale_each=True)

    writer.add_image(mode + ' Affinity', canvas_show, iteration)

def save_data(data, fileName):
    with h5py.File(fileName, 'w') as hfile:
        hfile.create_dataset('main', data=data, compression='gzip')

def save_all(input, gt_label, gt_flux, gt_skeleton, out_flux, out_mask, data_name_prefix, path):
    save_data((input.cpu().detach().numpy()*255).astype(np.uint8),  path + '/input_' + data_name_prefix + '.h5')
    save_data(gt_label.cpu().detach().numpy().astype(np.uint16), path + '/gt_label_' + data_name_prefix + '.h5')
    save_data(gt_flux.cpu().detach().numpy(), path + '/gt_flux_' + data_name_prefix + '.h5')
    save_data(gt_skeleton.cpu().detach().numpy().astype(np.uint16), path + '/gt_skeleton_' + data_name_prefix + '.h5')
    save_data(out_flux.cpu().detach().numpy(), path + '/out_flux_' + data_name_prefix + '.h5')
    save_data(out_mask.cpu().detach().numpy().astype(np.uint16), path + '/in_2d_mask' + data_name_prefix + '.h5')