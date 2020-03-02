import neuroglancer
import numpy as np
import h5py, pickle
from scipy import ndimage
import random

def write_hf(data, name):
    with h5py.File(name, 'w') as hf:
        hf.create_dataset('main', data=data, compression='gzip')

def read_hf(name):
    with h5py.File(name, 'r') as hf:
        data = np.asarray(hf['main'])
    return data

def load_dict(h5_path):
    f_dict = {}
    h_file = h5py.File(h5_path, 'r')
    for key in list(h_file):
        f_dict[int(key)] = np.asarray(h_file[key])
    return f_dict

def show_grad_as_color(path, name, bounds=None, resolution=None):
    global viewer
    global res

    if resolution is None:
        r = res
    else:
        r = resolution
    print('Loading: ' + path)
    print('as: ' + name)
    with h5py.File(path, 'r') as hf:
        hf_keys = hf.keys()
        print(list(hf_keys))
        for key in list(hf_keys):

            if bounds is not None:
                data = np.array(hf[key][bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]])
            else:
                data = np.array(hf[key])

            data = grad_to_RGB(data)

            with viewer.txn() as s:
                s.layers[name] = neuroglancer.ImageLayer(source=neuroglancer.LocalVolume(data, voxel_size=res),
                                                            shader = """void main()
                                                                        { 
                                                                            emitRGB(vec3(toNormalized(getDataValue(0)), 
                                                                            toNormalized(getDataValue(1)), 
                                                                            toNormalized(getDataValue(2)))); 
                                                                        }""",)

def grad_to_RGB(grad_field):
    #shape of grad_field is 3, Z, Y, X
    norm = np.sqrt(np.sum(grad_field**2, axis=1, keepdims=True))
    mask = (norm < 1e-5)
    loc_mask = np.nonzero(mask)

    norm[norm <= 1.0] = 1.0
    n_grad_field = grad_field / (2.0 * norm)
    n_grad_field += 0.5

    n_grad_field[:, loc_mask[0], loc_mask[1], loc_mask[2]] = 0.0
    return n_grad_field

def show(path, name, bounds=None, is_image=False, resolution=None, normalize=False, bb=False):
    global viewer
    global res
    if resolution is None:
        r = res
    else:
        r = resolution
    print('Loading: ' + path)
    print('as: ' + name)
    hf = h5py.File(path, 'r')
    hf_keys = hf.keys()
    print(list(hf_keys))
    for key in list(hf_keys):

        if bounds is not None:
            data = np.array(hf[key][bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]])
        else:
            data = np.array(hf[key])

        if not is_image:
            volume_type = 'segmentation'
        else:
            if normalize:
                data = (data - data.min()) / ((data.max() - data.min()) + np.finfo(np.float32).eps)
            volume_type = 'image'
            # if(isinstance(data, np.floating)):
            #     data = (data*255).astype(np.uint8)
            # else:
            #     data = data.astype(np.uint8)

        with viewer.txn() as s:
            s.layers.append(
                name=name,
                layer=neuroglancer.LocalVolume(
                    data=data,
                    voxel_size=r,
                    volume_type=volume_type
                ))

        if bb:
            show_bbox([0, 0, 0], data.shape, name)

def show_vec(direction_vecs_path, vec_lec, name):
    global viewer
    global res
    h5file = h5py.File(direction_vecs_path, 'r')
    for key in h5file.keys():
        line_id = 0
        data = h5file[key]
        with viewer.txn() as s:
            s.layers.append(name='dir_' + key,
                            layer=neuroglancer.AnnotationLayer(voxelSize=res))
            annotations = s.layers[-1].annotations
            for i in range(data.shape[0]):
                annotations.append(
                    neuroglancer.LineAnnotation(
                        id=line_id,
                        point_a=(data[i])[::-1],
                        point_b=(data[i] + vec_lec*data[i])[::-1]
                    ))
                annotations.append(
                    neuroglancer.PointAnnotation(
                        id=line_id+500000,
                        point=(data[i])[::-1],
                    ))
                line_id += 1


def show_vec_2(direction_vecs_path, vec_lec, name):
    global viewer
    global res
    h5file = h5py.File(direction_vecs_path, 'r')
    for key in h5file.keys():
        line_id = 0
        data = h5file[key]
        with viewer.txn() as s:
            s.layers.append(name='dir_' + key,
                            layer=neuroglancer.AnnotationLayer(voxelSize=res))
            annotations = s.layers[-1].annotations
            for i in range(data.shape[0]):
                annotations.append(
                    neuroglancer.LineAnnotation(
                        id=line_id,
                        point_a=data[i][2::-1],
                        point_b=data[i][2::-1] + vec_lec*data[i][5:2:-1]
                    ))
                annotations.append(
                    neuroglancer.PointAnnotation(
                        id=line_id+500000,
                        point=data[i][2::-1] + vec_lec*data[i][5:2:-1]
                    ))
                line_id += 1


def show_bbox(pointa, pointb, name):
    global viewer
    global res
    pointa = np.asarray(pointa)
    pointb = np.asarray(pointb)
    with viewer.txn() as s:
        s.layers.append(name='bb_' + name,
                        layer=neuroglancer.AnnotationLayer(voxelSize=res))
        s.layers[-1].annotations.append(
            neuroglancer.AxisAlignedBoundingBoxAnnotation(
                id=0,
                point_a=(pointa)[::-1],
                point_b=(pointb)[::-1]
            ))

def show_lines(filepath, name):
    global viewer
    global res
    h5file = h5py.File(filepath, 'r')
    with viewer.txn() as s:
        s.layers.append(name='start_' + name,
                        layer=neuroglancer.AnnotationLayer(voxelSize=res))
        start = s.layers[-1].annotations

        s.layers.append(name='end_' + name,
                        layer=neuroglancer.AnnotationLayer(voxelSize=res))
        end = s.layers[-1].annotations
        line_id = 0
        for key in h5file.keys():
            data = np.asarray(h5file[key])
            start.append(
                neuroglancer.LineAnnotation(
                    id=line_id,
                    point_a=data[2::-1],
                    point_b=data[5:2:-1]))
            end.append(
                neuroglancer.PointAnnotation(
                    id=line_id,
                    point=data[0:3]))
            line_id += 1

def show_matches_gt_npy(filepath, name):
    global viewer
    global res
    with viewer.txn() as s:
        npf = np.load(filepath)
        keys = npf.item().keys()
        for key in keys:
            l = np.asarray(npf.item().get(key))

            s.layers.append(name='start_' + name,
                            layer=neuroglancer.AnnotationLayer(voxelSize=res))
            start = s.layers[-1].annotations

            s.layers.append(name='end_' + name,
                            layer=neuroglancer.AnnotationLayer(voxelSize=res))
            end = s.layers[-1].annotations
            id = -1
            for data in l:
                id += 1
                start.append(
                    neuroglancer.LineAnnotation(
                        id=id,
                        point_a=data[2][::-1],
                        point_b=data[3][::-1]))
                end.append(
                    neuroglancer.PointAnnotation(
                        id=id,
                        point=data[2][::-1]))


def show_matches_pred_npy(filepath, name, weight_th=0.80):
    global viewer
    global res
    with viewer.txn() as s:
        results = np.load(filepath)

        s.layers.append(name='start_' + name,
                        layer=neuroglancer.AnnotationLayer(voxelSize=res))
        start = s.layers[-1].annotations

        s.layers.append(name='end_' + name,
                        layer=neuroglancer.AnnotationLayer(voxelSize=res))
        end = s.layers[-1].annotations
        id = -1
        for data in results:
            if float(data[5]) > weight_th:
                id += 1
                start.append(
                    neuroglancer.LineAnnotation(
                        id=id,
                        point_a=data[2][::-1],
                        point_b=data[3][::-1]))
                end.append(
                    neuroglancer.PointAnnotation(
                        id=id,
                        point=data[2][::-1]))

def show_grad(file_path, name, vec_lec=1, downsample_fac=1):
    global viewer
    global res
    h5file = h5py.File(file_path, 'r')
    for key in h5file.keys():
        line_id = 0
        data = h5file[key]
        with viewer.txn() as s:
            s.layers.append(name=name + '_' + key + '_end',
                            layer=neuroglancer.AnnotationLayer(voxelSize=res))
            s.layers.append(name=name + '_' + key + '_start',
                            layer=neuroglancer.AnnotationLayer(voxelSize=res))
            line = s.layers[-2].annotations
            point = s.layers[-1].annotations
            for i in range(data.shape[0]):
                line.append(
                    neuroglancer.LineAnnotation(
                        id=line_id,
                        point_a=downsample_fac*data[i, 0:3],
                        point_b=downsample_fac*data[i, 0:3] + vec_lec*data[i, 3:6]
                    ))
                point.append(
                    neuroglancer.PointAnnotation(
                        id=line_id + 500000,
                        point=downsample_fac*data[i, 0:3]
                    ))

                line_id += 1


def show_grad_field(file_path, name, seg_file=None, seg_id=None, vec_lec=1, upsample_fac=1, sparsity=15, bloat=0):
    global viewer
    global res
    h5file = h5py.File(file_path, 'r')

    if seg_file is not None:
        h5file_seg = h5py.File(seg_file, 'r')
        segs = np.asarray(h5file_seg['main'])
        h5file_seg.close()
        mask = (segs == seg_id)
    else:
        mask = None

    for key in h5file.keys():
        line_id = 0
        data = h5file[key]

        if mask is None:
            locations = np.transpose(np.nonzero((data[0] != 0) | (data[1] != 0) | (data[2] != 0)))
        else:
            if bloat > 0:
                mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3, 3)), iterations=int(bloat))
            locations = np.transpose(np.nonzero((mask)))

        locations = locations[::sparsity]
        with viewer.txn() as s:
            s.layers.append(name=name + '_' + key + '_end',
                            layer=neuroglancer.AnnotationLayer(voxelSize=res))
            s.layers.append(name=name + '_' + key + '_start',
                            layer=neuroglancer.AnnotationLayer(voxelSize=res))
            line = s.layers[-2].annotations
            point = s.layers[-1].annotations
            for i in range(locations.shape[0]):
                loc =tuple(locations[i])
                line.append(
                    neuroglancer.LineAnnotation(
                        id=line_id,
                        point_a=upsample_fac*locations[i, ::-1],
                        point_b=upsample_fac*locations[i, ::-1] + vec_lec*np.array([data[(2,) + loc], data[(1,) + loc], data[(0,) + loc]])
                    ))
                point.append(
                    neuroglancer.PointAnnotation(
                        id=line_id + 500000,
                        point=upsample_fac*locations[i, ::-1]
                    ))

                line_id += 1


def show_points(points_file, name, resolution=None):
    global viewer
    global res
    r = resolution if resolution else res
    h5file = h5py.File(points_file, 'r')
    with viewer.txn() as s:
        for key in h5file.keys():
            s.layers.append(name=name,
                            layer=neuroglancer.AnnotationLayer(voxelSize=r))
            points_layer = s.layers[-1].annotations

            points_ar = np.array(h5file[key])
            points = [neuroglancer.PointAnnotation(point=points_ar[i, ::-1], id=i) for i in range(points_ar.shape[0])]
            points_layer.extend(points)

def show_ends_junctions(points_file, name):
    global viewer
    global res
    with h5py.File(points_file, 'r') as h5file:
        with viewer.txn() as s:
            s.layers.append(name='ends-' + name,
                            layer=neuroglancer.AnnotationLayer(voxelSize=res))
            ends = s.layers[-1].annotations

            s.layers.append(name='junc-' + name,
                            layer=neuroglancer.AnnotationLayer(voxelSize=res))
            junctions = s.layers[-1].annotations
            id = -1
            for key in h5file.keys():
                data = np.asarray(h5file[key])
                if key[0] == 'e': point = ends
                else: point = junctions
                for j in range(data.shape[0]):
                    id += 1
                    point.append(neuroglancer.PointAnnotation(point=data[j, ::-1], id=id))

def show_array(array, name, bounds=None, resolution=None):
    global viewer
    global res
    if resolution is None:
        r = res
    else:
        r = resolution

    if bounds is not None:
        array = array[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]

    with viewer.txn() as s:
        s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=array,
                voxel_size=r,
            ))

def show_with_junctions(data_path, name, resolution=None):
    global viewer
    global res
    if resolution is None:
        r = res
    else:
        r = resolution

    h5file = h5py.File(data_path, 'r')
    seg_vol = (np.array(h5file['vol'])).astype(np.uint32)
    seg_sz = seg_vol.shape
    junctions = (np.array(h5file['junctions'])).astype(np.uint32)
    junctions_vol = np.zeros_like(seg_vol, dtype=np.uint32)
    print(junctions)
    ww = (2, 2, 2)
    for j in range(junctions.shape[0]):
        pt = junctions[j]
        junctions_vol[max(0, pt[0] - ww[0]): min(seg_sz[0], pt[0] + ww[0] + 1), \
        max(0, pt[1] - ww[1]): min(seg_sz[1], pt[1] + ww[1] + 1), \
        max(0, pt[2] - ww[2]): min(seg_sz[2], pt[2] + ww[2] + 1)] = 1

    with viewer.txn() as s:
        s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=seg_vol,
                voxel_size=r,
            ))

        s.layers.append(
            name=name + '_junctions',
            layer=neuroglancer.LocalVolume(
                data=junctions_vol,
                voxel_size=r,
            ))

        s.layers.append(
            name=name + '_merged',
            layer=neuroglancer.LocalVolume(
                data=(seg_vol > 0).astype(np.uint32),
                voxel_size=r,
            ))

def show_zebrafinch(res):
    global viewer
    Dd='precomputed://gs://j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata_realigned'
    with viewer.txn() as s:
        s.layers['zf-im'] = neuroglancer.ImageLayer(source=Dd)

class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, vertices, edges):
        super(SkeletonSource, self).__init__()

        self.vertex_attributes['color'] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32,
            num_components=1,
        )

        self.vertices = vertices
        self.edges = edges

    def get_skeleton(self, i):
        i = str(i)
        if i not in self.vertices.keys():
            return None
        else:
            color = np.full(self.vertices[i].shape[0], float(i)/len(self.vertices), dtype=np.float32)
            return neuroglancer.skeleton.Skeleton(
                vertex_positions=self.vertices[i],
                edges=self.edges[i],
                vertex_attributes=dict(color=color))


class SkeletonEndPointSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self, vertices, edges):
        super(SkeletonEndPointSource, self).__init__()

        self.vertex_attributes['color'] = neuroglancer.skeleton.VertexAttributeInfo(
            data_type=np.float32,
            num_components=1,
        )

        self.vertices = vertices
        self.edges = edges

    def get_skeleton(self, i):
        i = str(i)
        if i not in self.vertices.keys():
            return None
        else:
            color = np.full(self.vertices[i].shape[0], float(i)/len(self.vertices), dtype=np.float32)
            return neuroglancer.skeleton.Skeleton(
                vertex_positions=self.vertices[i][-1:, :],
                edges=np.array([0], dtype=np.uint16),
                vertex_attributes=dict(color=color[-1:]))


def show_skeleton(filepath, name='skeletons', resolution=None, show_ends=False):
    global viewer
    global res
    r = resolution if resolution else res
    
    vertices = {}
    edges = {}        
    if filepath[-3:] == '.h5':
        with h5py.File(filepath, 'r') as hf:
            for g in hf.keys():
                vertices[g] = np.asarray(hf.get(g)['vertices'])[:, ::-1] * np.array(r)
                if 'edges' in hf.get(g).keys():
                    edges[g] = np.asarray(hf.get(g)['edges'])
                else:
                    edges[g] = np.array([], dtype=np.uint16)
    else:
        with open(filepath, 'rb') as phandle: paths = pickle.load(phandle)

        for g, data in paths.items():
            vertices[str(g)] = data['vertices'][:, ::-1] * np.array(r)
            edges[str(g)] = data['edges'] if 'edges' in data.keys() else np.array([], dtype=np.uint16)

    skeletons = SkeletonSource(vertices, edges)

    if show_ends is True: skeletonEndpoints = SkeletonEndPointSource(vertices, edges)

    with viewer.txn() as s:
        s.layers.append(
            name=name,
            layer=neuroglancer.SegmentationLayer(
                source=neuroglancer.LocalVolume(data=np.zeros((1,1,1)),
                                                voxel_size=r,
                                                skeletons=skeletons),
                skeleton_shader='void main() { emitRGB(colormapJet(color[0])); }',
                selected_alpha=0,
                not_selected_alpha=0,
            ))

        if show_ends is True:
            s.layers.append(
                name=name + 'Ends',
                layer=neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(data=np.zeros((1,1,1)),
                                                    voxel_size=r,
                                                    skeletons=skeletonEndpoints),
                    skeleton_shader='void main() { emitRGB(colormapJet(color[0])); }',
                    selected_alpha=0,
                    not_selected_alpha=0,
                ))

def read_pkl(filepath):
    with open(filepath, 'rb') as phandle: 
        data = pickle.load(phandle)
    return data

ip='localhost' # or public IP of the machine for sharable display
port=18779 # change to an unused port number
neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
viewer=neuroglancer.Viewer()

#### SNEMI #####
res = [6, 6, 30]
# D0 = '/n/pfister_lab2/Lab/alok/snemi/'
# show(D0 + 'train_image.h5', 'im', is_image=True)
# show(D0 + 'skeleton/train_labels_separated_disjointed_removedGlial.h5', 'gt-seg', is_image=False)
# show('/n/pfister_lab2/Lab/alok/results/snemi/test_context_L1_22000_14000_denseSupervision/1x/splitted.h5', 'split', is_image=False)
# show('/n/pfister_lab2/Lab/alok/snemi/skeleton/skeleton.h5', 'gt-skeleton')
# show_grad_field('/n/pfister_lab2/Lab/alok/snemi/skeleton/grad_distance.h5', 'gt_grad', D0 + 'skeleton/train_labels_separated_disjointed_removedGlial.h5', 401, sparsity=1000, vec_lec=2.0)
# show_grad_field('/n/pfister_lab2/Lab/alok/results/snemi/snemi_complete_AllAug/gradient_0.h5', 'result_grad', D0 + 'skeleton/train_labels_separated_disjointed_removedGlial.h5', 401, sparsity=1000, vec_lec=2.0)
# # show('/n/pfister_lab2/Lab/alok/snemi/skeleton/temp/(0)seg_0_distance.h5', 'distanceTx', is_image=True, normalize=True)
# show_grad_field('/n/pfister_lab2/Lab/alok/snemi/skeleton/temp/(0)seg_0_grad_distance.h5', 'grad', D0+'train_labels_separated_disjointed.h5', seg_id=14, sparsity=600, vec_lec=2)

# Validation
# show('/n/pfister_lab2/Lab/alok/snemi/skeleton/val/val_image_half.h5', 'im', is_image=True)
# show('/n/pfister_lab2/Lab/alok/snemi/skeleton/val/val_labels_half_relabelled.h5', 'val', is_image=False)

# Zebra Finch Skeletons
# res = [9, 9, 20]
# show_zebrafinch(res)
# show_skeleton('/n/pfister_lab2/Lab/vcg_connectomics/zebraFinch/skel/valid_12/skeletons.h5', 'skel')
# show_skeleton('/n/pfister_lab2/Lab/vcg_connectomics/zebraFinch/skel/valid_12/skeletonsData.h5', 'skel-interpolated')
# show_skeleton('/n/pfister_lab2/Lab/vcg_connectomics/zebraFinch/skel/valid_12/skeletonsData1.h5', 'skel-context')

##### JWR #####
# res = [120, 128, 128]
# show('/home/alok/pfister_lab2/donglai/data/JWR/snow_cell/cell128nm/seg_jz/yl_den_11.h5', 'jwr', res)
#train_data = '/home/alok/data/falseMerge/train/train.h5'
#show_with_junctions(train_data, 'train', res)


# res = [6, 6, 30]
# show('/home/alok/hp03/donglai/code/cerebellum/alok_data/vol0-0_im.h5', 'p7_01', True)
# show('/home/alok/hp03/donglai/code/cerebellum/alok_data/vol0-0_seg_all.h5', 'p7_01_all')
# show('/home/alok/hp03/donglai/code/cerebellum/alok_data/vol0-0_seg_bigcell.h5', 'p7_01_bigcell')
# show('/home/alok/nag_v0.h5', 'nag')
# show('/home/alok/hp03/alok/p7/vol0-0_seg_combined.h5', 'p7_combined')


### CREMI ###
# res = [4, 4, 40]
# im = '/home/alok/rcHome/pytorch_connectomics/scripts/im_0, 31, 456, 1185_.h5'
# labels = '/home/alok/rcHome/pytorch_connectomics/scripts/label_0, 31, 456, 1185_.h5'
# show(im, 'CREMI', True)
# show(labels, 'CREMI')

# ### Mitochondria ###
# res = [6, 6, 30]
# basepath = '/n/pfister_lab2/Lab/vcg_connectomics/human/roi466/mito/cell47/'
# show(basepath + 'vol1_im_8nm.h5', 'img', True)
# show(basepath + 'vol1_mito_8nm.h5', 'mito')
# show(basepath + 'vol1_cell_8nm.h5', 'cell')
# print(viewer)

### JWR MITO ###
# res = [8, 8, 40]
# basepath = '/n/pfister_lab2/Lab/vcg_connectomics/JWR15/data_8x8x3um/vol5/'
# show(basepath + 'im_df.h5', 'img', True)
# show(basepath + 'mito_gt.h5', 'mito')
# show(basepath + 'mito_gt.h5', 'mito')
# show('/home/alok/repositories/pytorch_connectomics/scripts/outputs/mito/affinity_0.h5', 'aff', True)
# show('/home/alok/repositories/pytorch_connectomics/scripts/outputs/mito/0_0.050000_0.995000_800_0.200000_600_0.900000_1.h5', 'res', True)


### JWR 128x128x120 ###
# res = [128, 128, 120]
# show('/n/pfister_lab2/Lab/alok/JWR15/im_64nm_y2_iso.h5', 'img', True)
# show('/n/pfister_lab2/Lab/alok/JWR15/segmentations/64nm/cell_68_iso.h5', 'cell2')
# show('/n/pfister_lab2/Lab/alok/JWR15/segmentations/64nm/cell_68_iso_shrunk_2_355_onlyLarge.h5', 'cell2_shrunk')

#small Vol

#test chunk
# bs = [0, 0, 0]
# be = [800, 500, 800]

# Train Chunk
# bs = [374, 1242, 0]
# be = [1053, 3025, 1059]

# bounds =[[bs[0], be[0]], [bs[1], be[1]], [bs[2], be[2]]]
# show('/n/pfister_lab2/Lab/alok/JWR15/im_64nm_y2_iso.h5', 'img', bounds=bounds, is_image=True)
# show('/n/pfister_lab2/Lab/alok/JWR15/segmentations/64nm/cell_68_iso.h5', 'seg-gt', bounds=bounds, is_image=False)
# show('/n/home11/averma/pytorch_connectomics/scripts/outputs/JWRNeuron/mask_raw0.h5', 'prob-output', is_image=True)
# show('/n/home11/averma/pytorch_connectomics/scripts/outputs/JWRNeuron/mask_0.h5', 'seg-output', is_image=False)
# show_points('/n/home11/averma/pytorch_connectomics/scripts/outputs/JWRNeuron/prediction_points0.h5', np.array(be) - np.array(bs), 'center_pos')



#Parallel fibers P7
# res = [6, 6, 30]

# bs = [0, 2300, 2900]
# be = [500, 5000, 5200]
# bounds =[[bs[0], be[0]], [bs[1], be[1]], [bs[2], be[2]]]
# show('/n/pfister_lab2/Lab/alok/p7/im_6nm.h5', 'img', bounds=bounds, is_image=True)
# show('/n/pfister_lab2/Lab/alok/p7/segmentations/6nm_0.h5', 'seg-gt', bounds=bounds, is_image=False)
# show('/n/home11/averma/pytorch_connectomics/scripts/outputs/JWRNeuron/mask_raw0.h5', 'prob-output', is_image=True)
# show('/n/home11/averma/pytorch_connectomics/scripts/outputs/JWRNeuron/mask_0.h5', 'seg-output', is_image=False)
# show_points('/n/home11/averma/pytorch_connectomics/scripts/outputs/JWRNeuron/prediction_points0.h5', np.array(be) - np.array(bs), 'center_pos')

# Train volume Old P7
# show('/n/pfister_lab2/Lab/alok/p7/segmentations/partial/im_0.h5', 'img', is_image=True)
# show('/n/pfister_lab2/Lab/alok/p7/segmentations/partial/seg_0_separated_split_removedNonSkel.h5', 'seg-gt', is_image=False)

#Test/val Volume Old P7
# res = [6, 6, 30]
# bounds =[[0, 994], [2570, 3570], [4233, 6233]]
# show('/n/pfister_lab2/Lab/alok/p7/im_6nm.h5', 'img', bounds=bounds, is_image=True)
# show('/n/pfister_lab2/Lab/alok/results/p7/RotBlurEtcEtc/cc_div_direction_0.h5', 'result-div', is_image=False)

# Train volume New P7
# res = [6, 6, 30]
# show('/n/pfister_lab2/Lab/alok/p7_new/segmentations/partial/im_0.h5', 'im', is_image=True)
# show('/n/pfister_lab2/Lab/alok/p7_new/segmentations/partial/seg_0_separated_rem_border.h5', 'seg-gt', is_image=False)
# show('/n/pfister_lab2/Lab/alok/p7_new/segmentations/partial/seg_0_skeleton_spline.h5', 'skel_spine', is_image=False)


#liver
# res=[5,5,5]
# show('/n/pfister_lab2/Lab/alok/liver/resampled/labels/0.h5', 'labels', is_image=0) 
# show('/n/pfister_lab2/Lab/alok/results/liver/liver_0/skel_large_merged.h5', 'skel-merged', is_image=0)
# show_vec_2('/n/pfister_lab2/Lab/alok/results/liver/liver_0/end_directions.h5', 2, 'directions')


print(viewer)
