import os,sys
import numpy as np
import torch
import torch.nn.functional as F
import h5py, time, itertools, datetime
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_erosion

from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize_aff

def test(args, test_loader, model, device, model_io_size, volume_shape, pad_size, initial_seg=None):
    # switch to eval mode
    model.eval()
    volume_id = 0
    ww = blend(model_io_size)
    NUM_OUT = args.out_channel
    sel_cpu = np.ones((3, 3, 3), dtype=bool)
    sel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=device)
    if initial_seg is not None:
        result = [np.expand_dims(initial_seg, axis=0)]
    else:
        result = [np.stack([np.zeros(x, dtype=bool) for _ in range(NUM_OUT)]) for x in volume_shape]

    result_raw = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in volume_shape]
    prediction_points = []

    weight = [np.zeros(x, dtype=np.float32) for x in volume_shape]
    print(result[0].shape, weight[0].shape)

    if args.test_augmentation:
        print("Will augment (Rotate and Flip) data during inference.")

    seed_points_files = args.seed_points.split('@')
    test_loader.set_out_array(result[0][0])
    sz = tuple([NUM_OUT] + list(model_io_size))
    start = time.time()
    with torch.no_grad():
        itr_max = 0
        while test_loader.remaining_pos() > 0 :
            itr_max += 1
            pos, volume, past_pred = test_loader.get_input_data()
            volume = volume.to(device)
            past_pred = past_pred.to(device)

            output_raw = model(torch.cat((volume, past_pred), 1))

            output = output_raw > 0.85
            output_raw = output_raw.cpu().detach().numpy()

            for idx in range(output.shape[0]):

                st = pos[idx]
                out_mask = output[idx][0].cpu().detach().numpy().astype(bool)

                if out_mask[tuple(test_loader.dataset.half_input_sz)]:
                    cc_out_mask, _ = label(out_mask)
                    out_mask = (cc_out_mask == cc_out_mask[tuple(test_loader.dataset.half_input_sz)])
                    result[st[0]][0, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], st[3]:st[3]+sz[3]] |= out_mask

                    result_raw[st[0]][:, st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]] \
                        = np.maximum(result[st[0]][:, st[1]:st[1] + sz[1], st[2]:st[2] + sz[2], st[3]:st[3] + sz[3]], \
                                     output_raw[idx].reshape(sz))
                    prediction_points.append(st[1:] - test_loader.dataset.seed_points_offset)  # appending center points wrt the unpadded volume

                    if True:
                        out_mask = torch.from_numpy(binary_erosion(out_mask, sel_cpu).astype(np.float32)).to(device)
                        out_mask = out_mask.unsqueeze(0).unsqueeze(0)
                        edge = (F.conv3d(out_mask, sel, padding=1))[0, 0]
                        edge = (edge > 0) * (edge < 9)
                        edge = F.interpolate(edge.unsqueeze(0).unsqueeze(0).float(), scale_factor=1 / 4, mode='trilinear')
                        edge = edge > .50
                        edge_pos = (torch.nonzero(edge[0, 0])*4).cpu().detach().numpy().astype(np.uint32)
                        test_loader.compute_new_pos(out_mask, edge_pos, st[1:])

    end = time.time()
    print("prediction time:", (end-start))
    for vol_id in range(len(result)):
        data = result[vol_id]
        data = data[:,
                    pad_size[0]:-pad_size[0],
                    pad_size[1]:-pad_size[1],
                    pad_size[2]:-pad_size[2]]
        print('Output shape: ', data.shape)
        hf = h5py.File(args.output + '/mask_' + str(vol_id) + '.h5', 'w')
        hf.create_dataset('main', data=data, compression='gzip')
        hf.close()

        hf = h5py.File(args.output + '/prediction_points' + str(vol_id) + '.h5', 'w')
        hf.create_dataset('main', data=np.array(prediction_points), compression='gzip')
        hf.close()

        data = result_raw[vol_id]
        data = data[:,
                    pad_size[0]:-pad_size[0],
                    pad_size[1]:-pad_size[1],
                    pad_size[2]:-pad_size[2]]
        data = (data*255).astype(np.uint8)
        hf = h5py.File(args.output + '/mask_raw' + str(vol_id) + '.h5', 'w')
        hf.create_dataset('main', data=data, compression='gzip')
        hf.close()

def get_augmented(volume):
    # perform 16 Augmentations as mentioned in Kisuks thesis
    vol0    = volume
    vol90   = torch.rot90(vol0, 1, [3, 4])
    vol180  = torch.rot90(vol90,  1,  [3, 4])
    vol270  = torch.rot90(vol180, 1, [3, 4])

    vol0f   = torch.flip(vol0,   [3])
    vol90f  = torch.flip(vol90,  [3])
    vol180f = torch.flip(vol180, [3])
    vol270f = torch.flip(vol270, [3])

    vol0z   = torch.flip(vol0,   [2])
    vol90z  = torch.flip(vol90,  [2])
    vol180z = torch.flip(vol180, [2])
    vol270z = torch.flip(vol270, [2])

    vol0fz   = torch.flip(vol0f,   [2])
    vol90fz  = torch.flip(vol90f,  [2])
    vol180fz = torch.flip(vol180f, [2])
    vol270fz = torch.flip(vol270f, [2])

    augmented_volumes = [vol0,  vol90,  vol180,  vol270,  vol0f,  vol90f,  vol180f,  vol270f,
                         vol0z, vol90z, vol180z, vol270z, vol0fz, vol90fz, vol180fz, vol270fz]

    return augmented_volumes

def combine_augmented(outputs):
    assert len(outputs) == 16
    for i in range(8, 16):
        outputs[i] = torch.flip(outputs[i], [2])
    for i in range(4, 8):
        outputs[i] = torch.flip(outputs[i], [3])
    for i in range(12, 16):
        outputs[i] = torch.flip(outputs[i], [3])
    for i in range(1, 16, 4):
        outputs[i] = torch.rot90(outputs[i], -1, [3, 4])
    for i in range(2, 16, 4):
        outputs[i] = torch.rot90(outputs[i], -1, [3, 4])
        outputs[i] = torch.rot90(outputs[i], -1, [3, 4])
    for i in range(3, 16, 4):
        outputs[i] = torch.rot90(outputs[i], 1, [3, 4])

    # output = torch.zeros_like(outputs[0], dtype=torch.float64)
    # for i in range(len(outputs)):
    #     output += outputs[i].double()
    # output = output / 16.0
    for i in range(len(outputs)):
        outputs[i] = outputs[i].unsqueeze(0)
    output = torch.min(torch.cat(outputs, 0), 0)[0]

    return output, outputs

def main():
    args = get_args(mode='test')

    print('0. initial setup')
    model_io_size, device = init(args)
    print('model I/O size:', model_io_size)

    print('1. setup data')
    test_loader, volume_shape, pad_size, initial_seg = get_input(args, model_io_size, 'test')

    print('2. setup model')
    model = setup_model(args, device, exact=True)

    print('3. start testing')
    test(args, test_loader, model, device, model_io_size, volume_shape, pad_size, initial_seg)

    print('4. finish testing')

if __name__ == "__main__":
    main()