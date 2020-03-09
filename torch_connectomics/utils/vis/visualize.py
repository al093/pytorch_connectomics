import torch
import torchvision.utils as vutils
import h5py
import numpy as np
import pickle

N = 15 # default maximum number of sections to show
min_batch = 3
def prepare_data(volume, label, output, input_label=None, color_data=None):

    if len(volume.size()) == 4:   # 2D Inputs
        if volume.size()[0] > N:
            return [volume[:N], label[:N], output[:N]]
        else:
            return [volume, label, output]

    elif len(volume.size()) == 5: # 3D Inputs
        if volume.shape[0] >= min_batch: #show slices from different batches
            start_slice_number = volume.shape[2]//2 - int(N/min_batch)//2
            volume = volume[:min_batch, :, start_slice_number:start_slice_number+int(N/min_batch), :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, volume.shape[1], volume.shape[3], volume.shape[4])
            label = label[:min_batch,   :, start_slice_number:start_slice_number+int(N/min_batch), :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, label.shape[1], label.shape[3], label.shape[4])

            if output is not None:
                output = output[:min_batch, :, start_slice_number:start_slice_number+int(N/min_batch), :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, output.shape[1], output.shape[3], output.shape[4])
            if input_label is not None:
                input_label = input_label[:min_batch,   :, start_slice_number:start_slice_number+int(N/min_batch), :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, input_label.shape[1], input_label.shape[3], input_label.shape[4])
            if color_data is not None:
                color_data = color_data[:min_batch, :, start_slice_number:start_slice_number+int(N/min_batch), :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, color_data.shape[1], color_data.shape[3], color_data.shape[4])
        else:
            volume, label = volume[0].permute(1,0,2,3), label[0].permute(1,0,2,3)
            if output is not None:
                output = output[0].permute(1, 0, 2, 3)
            if input_label is not None:
                input_label = input_label[0].permute(1, 0, 2, 3)
            if color_data is not None:
                color_data = color_data[0].permute(1, 0, 2, 3)

        if volume.size()[0] > N:
            ret_list = [volume[:N], label[:N]]
            ret_list.append(output[:N] if output is not None else None)
            if input_label is not None:
                ret_list.append(input_label[:N])
            if color_data is not None:
                ret_list.append(color_data[:N])
        else:
            ret_list = [volume, label, output]
            if input_label is not None:
                ret_list.append(input_label)
            if color_data is not None:
                ret_list.append(color_data)

    return ret_list

def visualize(volume, label, output, iteration, writer, mode='Train', input_label=None, color_data=None):

    prepared_data = prepare_data(volume, label, output, input_label, color_data)
    if len(prepared_data) == 3:
        volume, label, output = prepared_data
    elif len(prepared_data) == 4:
        if input_label is not None:
            volume, label, output, input_label = prepared_data
        else:
            volume, label, output, color_data = prepared_data
    else:
        volume, label, output, input_label, color_data = prepared_data

    sz = volume.size() # z,c,y,x
    volume_visual = volume.detach().cpu().expand(sz[0], 3, sz[2], sz[3])
    if volume.shape[1] == 1:
        output_visual = output.detach().cpu().expand(sz[0], 3, sz[2], sz[3]) if output is not None else None
        label_visual = label.detach().cpu().expand(sz[0], 3, sz[2], sz[3])
    elif volume.shape[1] > 1:
        output_visual = output.detach().cpu() if output is not None else None
        label_visual = label.detach().cpu()

    if input_label is not None:
        if input_label.shape[1] == 1:
            input_label_visual = input_label.detach().cpu().expand(sz[0], 3, sz[2], sz[3])
        elif input_label.shape[1] > 1:
            input_label_visual = input_label.detach().cpu()

    if color_data is not None:
        color_data_visual = color_data.detach().cpu()

    canvas = []
    for idx in range(volume_visual.shape[0]):
        canvas.append(volume_visual[idx])

        if input_label is not None:
            if input_label.shape[1] == 1:
                canvas.append(input_label_visual[idx])
            elif input_label.shape[1] > 1:
                for i in range(input_label_visual.shape[1]):
                    canvas.append(input_label_visual[idx, i:i+1].expand(3, sz[2], sz[3]))

        if volume.shape[1] == 1:
            canvas.append(label_visual[idx])
            if output is not None:
                canvas.append(output_visual[idx])
        elif volume.shape[1] > 1:
            for i in range(output.shape[1]):
                canvas.append(label_visual[idx, i:i+1].expand(3, sz[2], sz[3]))
                if output is not None:
                    canvas.append(output_visual[idx, i:i+1].expand(3, sz[2], sz[3]))
        if color_data is not None:
            for i in range(color_data_visual.shape[1] // 3):
                canvas.append(color_data_visual[idx, i*3:(i+1)*3])

    nrow = volume.shape[1] + label.shape[1] \
        + (output.shape[1] if output is not None else 0) \
        + (input_label.shape[1] if input_label is not None else 0) \
        + (color_data.shape[1] // 3 if color_data  is not None else 0)

    canvas_show = vutils.make_grid(canvas, nrow=nrow, normalize=False, scale_each=False)

    writer.add_image(mode + ' Mask', canvas_show, iteration)

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

def vec_to_angle(grad_field):
    alpha = torch.atan2(grad_field[2], grad_field[1])
    beta = torch.atan2(grad_field[0], np.sqrt(grad_field[1]**2 + grad_field[2]**2))
    return alpha, beta

def vec_to_RGB(grad_field):
    #shape of grad_field is N, 3, Z, Y, X
    norm = torch.sqrt(torch.sum(grad_field**2, dim=1, keepdim=True))
    norm[norm <= 1.0] = 1.0
    n_grad_field = grad_field / (2.0 * norm)
    n_grad_field += 0.5
    return n_grad_field

def save_data(data, fileName):
    if data is None:
        print('Data not saved because it was none')
    else:
        with h5py.File(fileName, 'w') as hfile:
            hfile.create_dataset('main', data=data, compression='gzip')

def save_all(input, gt_label, gt_flux, gt_skeleton, out_flux, out_mask, data_name_prefix, path):
    save_data((input.cpu().detach().numpy()*255).astype(np.uint8),  path + '/input_' + data_name_prefix + '.h5')
    save_data(gt_label.cpu().detach().numpy().astype(np.uint16), path + '/gt_label_' + data_name_prefix + '.h5')
    save_data(gt_flux.cpu().detach().numpy(), path + '/gt_flux_' + data_name_prefix + '.h5')
    save_data(gt_skeleton.cpu().detach().numpy().astype(np.uint16), path + '/gt_skeleton_' + data_name_prefix + '.h5')
    save_data(out_flux.cpu().detach().numpy(), path + '/out_flux_' + data_name_prefix + '.h5')
    save_data(out_mask.cpu().detach().numpy().astype(np.uint16), path + '/in_2d_mask' + data_name_prefix + '.h5')

def save_volumes_in_dict(volumes_dict, base_path):
    for key, vol in volumes_dict.items():
        save_data(vol, base_path + '/' + key)

def read_data(filename):
    with h5py.File(filename, 'r') as hfile:
        return np.asarray(hfile['main'])

def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)