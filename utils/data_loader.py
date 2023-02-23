"""
    caGAN project data loader
    todo: revise
"""
import os.path

import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from skimage.measure import block_reduce

from utils.fcns import prctile_norm, fix_path, reorder


def data_loader(images_path, data_path, gt_path, ny, nx, nz,
                batch_size, norm_flag=1, scale=2, wf_weight=0, wf_path=None):
    if wf_weight == 0:
        return data_loader_multi_channel_3d(images_path, data_path, gt_path, ny, nx, nz,
                                            batch_size, norm_flag, scale)
    else:
        return data_loader_multi_channel_3d_wf(images_path, data_path, wf_path, gt_path, ny, nx, nz,
                                               batch_size, norm_flag, scale)


def data_loader_multi_channel_3d(images_path, data_path, gt_path, ny, nx, nz,
                                 batch_size, norm_flag=1, scale=2, wf=0):
    """

    :param images_path:
    :param data_path:
    :param gt_path:
    :param ny:
    :param nx:
    :param nz:
    :param batch_size:
    :param norm_flag:
    :param scale:
    :param wf:
    :return:
    """
    batch_images_path = np.random.choice(images_path, size=batch_size)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        cur_img = tiff.imread(path)

        path = path.replace('\\', '/')
        cur_gt_path = path.replace(data_path, gt_path)
        cur_gt = tiff.imread(cur_gt_path)

        if norm_flag:
            cur_img = prctile_norm(np.array(cur_img))
            cur_gt = prctile_norm(np.array(cur_gt))
        else:
            cur_img = np.array(cur_img) / 65535
            cur_gt = np.array(cur_gt) / 65535

        image_batch.append(cur_img)
        gt_batch.append(cur_gt)

    image_batch = np.array(image_batch)
    gt_batch = np.array(gt_batch)
    nslice = image_batch.shape[1]
    image_batch = np.reshape(image_batch, (batch_size, nslice // nz, nz, ny, nx),
                             order='F').transpose((0, 3, 4, 2, 1))
    gt_batch = gt_batch.reshape((batch_size, nz, ny * scale, nx * scale, 1),
                                order='F').transpose((0, 2, 3, 1, 4))

    if wf == 1:
        image_batch = np.mean(image_batch, 4)
        for b in range(batch_size):
            image_batch[b, :, :, :] = prctile_norm(image_batch[b, :, :, :])
        image_batch = image_batch[:, :, :, np.newaxis]

    return image_batch, gt_batch


def data_loader_multi_channel_3d_wf(images_path, data_path, wf_path, gt_path, ny, nx, nz,
                                    batch_size, norm_flag=1, scale=2, wf=0):
    data_path = fix_path(data_path)
    gt_path = fix_path(gt_path)
    images_path = [fix_path(image_path) for image_path in images_path]
    batch_images_path = np.random.choice(images_path, size=batch_size)
    image_batch = []
    wf_batch = []
    gt_batch = []
    for path in batch_images_path:
        cur_img = tiff.imread(path)

        nslice = cur_img.shape[0]
        nchannels = int(nslice / nz)
        # cur_wf = tiff.imread(path.replace(data_path, wf_path))
        # WideField is the sum of angles and phases in each z patch
        cur_wf = reorder(cur_img)
        cur_wf = block_reduce(cur_wf, block_size=(nchannels, 1, 1),
                              func=np.mean, cval=np.mean(cur_wf))

        wf_shape = np.shape(cur_wf)

        plt.figure()
        # plot widefield images to make sure there are no fringes
        for i in range(wf_shape[0]):
            plt.subplot(wf_shape[0], 1, i + 1)
            plt.imshow(cur_wf[i, :, :])
        if not os.path.exists(wf_path):
            os.mkdir(wf_path)
        img_name = path.split('/')[-1].split('.')[0]
        plt.savefig(wf_path + img_name + '.png')

        cur_gt = tiff.imread(path.replace(data_path, gt_path))

        if norm_flag:
            cur_img = prctile_norm(np.array(cur_img))
            cur_wf = prctile_norm(np.array(cur_wf))
            cur_gt = prctile_norm(np.array(cur_gt))
        else:
            cur_img = np.array(cur_img) / 65535
            cur_wf = np.array(cur_wf) / 65535
            cur_gt = np.array(cur_gt) / 65535

        image_batch.append(cur_img)
        wf_batch.append(cur_wf)
        gt_batch.append(cur_gt)

    image_batch = np.array(image_batch)
    wf_batch = np.array(wf_batch)
    gt_batch = np.array(gt_batch)

    image_batch = np.reshape(image_batch, (batch_size, nslice // nz, nz, ny, nx),
                             order='F').transpose((0, 3, 4, 2, 1))
    wf_batch = wf_batch.reshape((batch_size, nz, ny, nx, 1),
                                order='F').transpose((0, 2, 3, 1, 4))

    gt_batch = gt_batch.reshape((batch_size, nz, ny * scale, nx * scale, 1),
                                order='F').transpose((0, 2, 3, 1, 4))

    if wf == 1:
        image_batch = np.mean(image_batch, 4)
        for b in range(batch_size):
            image_batch[b, :, :, :] = prctile_norm(image_batch[b, :, :, :])
        image_batch = image_batch[:, :, :, np.newaxis]

    return image_batch, wf_batch, gt_batch



