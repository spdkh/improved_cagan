"""
    author: SPDKH
    todo: complete
"""
import os
import glob
import numpy as np
from data.data import Data
import matplotlib.pyplot as plt
from utils import fcns
from skimage.measure import block_reduce
from matplotlib import pyplot as plt
import tifffile as tiff

from utils.fcns import prctile_norm, fix_path, reorder


class FixedCell(Data):
    def __init__(self, args):
        Data.__init__(self, args)
        self.data_groups = {'train': 'training',
                            'test': 'testing',
                            'val': 'validation'}

        self.data_types = {'x': 'rawdata', 'y': 'gt'}
        self.args.data_dir = fcns.fix_path(self.args.data_dir)
        input_dir = os.path.join(self.args.data_dir,
                                 self.data_groups['train'],
                                 self.data_types['y'])
        in_sample_dir = os.path.join(input_dir,
                                     os.listdir(input_dir)[0])
        self.input_dim = self.load_sample(in_sample_dir)
        print('input', self.input_dim)
        self.output_dim = [self.input_dim[0] * self.args.scale_factor,
                           self.input_dim[1] * self.args.scale_factor,
                           self.input_dim[2], 1]
        print('output', self.output_dim)

        save_weights_name = 'SIM_fixed_cell'

        self.save_weights_path = os.path.join(self.args.checkpoint_dir,
                                              save_weights_name)

        if not os.path.exists(self.save_weights_path):
            os.mkdir(self.save_weights_path)

        self.sample_path = os.path.join(self.save_weights_path, 'sampled_img')

        self.log_path = os.path.join(self.save_weights_path, 'graph')
        self.data_dirs = dict()

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        for data_group in self.data_groups.keys():
            self.data_dirs[data_group] = os.path.join(self.args.data_dir,
                                                      self.data_groups[data_group])
            for data_type in self.data_types.keys():
                self.data_dirs[data_type + data_group] =\
                    os.path.join(self.data_dirs[data_group],
                                 self.data_types[data_type])
        print(self.data_dirs)

    def data_loader(self,
                    mode, it,
                    batch_size,
                    norm_flag,
                    scale,
                    wf_weight=0):
        """

        Parameters
        ----------
        mode: str
            options: "train" or "test" or "val"
        batch_size
        norm_flag
        wf_weight

        Returns
        -------

        todo 1: data loader only works with #samples >= #epochs: batches should reset after reaching the end of the samples
        todo 1.5: only training samples equal to the number of epochs * batch_size are loaded not all the data
        todo 2: assign wf = 0 changes to wf > 0 function too
        """

        images_names = os.listdir(self.data_dirs['x' + mode])
        gt_names = os.listdir(self.data_dirs['y' + mode])
        # print(gt_names)
        images_names.sort()
        gt_names.sort()

        # train_wf_path = self.args.data_dir + '/training_wf/'
        # print(images_names)
        if wf_weight == 0:
            return self.data_loader_multi_channel_3d(images_names,
                                                     gt_names,
                                                     self.data_dirs['x' + mode],
                                                     self.data_dirs['y' + mode],
                                                     it,
                                                     batch_size,
                                                     norm_flag,
                                                     scale)
        else:
            return self.data_loader_multi_channel_3d_wf(images_names,
                                                        gt_names,
                                                        self.data_dirs[mode],
                                                        '',
                                                        it,
                                                        batch_size,
                                                        norm_flag,
                                                        scale)

    def data_loader_multi_channel_3d(self, images_names, gt_names, x_path, y_path,
                                     it, batch_size, norm_flag=1, scale=2):
        """
        :param images_names:
        :param data_path:
        :param batch_size:
        :param norm_flag:
        :param scale:
        :param wf:
        :return:
        """
        # print(images_names)
        # print(batch_size)
        # batch_images_path = np.random.choice(images_names,
        #                                      size=batch_size)
        batch_images_path = images_names[it:batch_size + it]
        gt_images_path = gt_names[it:batch_size + it]
        image_batch = []
        gt_batch = []
        for i in range(len(batch_images_path)):
            cur_img = tiff.imread(os.path.join(x_path,
                                               batch_images_path[i]))

            cur_gt = tiff.imread(os.path.join(y_path,
                                              gt_images_path[i]))

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
        image_batch = np.reshape(image_batch,
                                 (batch_size,
                                  nslice // self.input_dim[2],
                                  self.input_dim[2],
                                  self.input_dim[1],
                                  self.input_dim[0]),
                                  order='F').transpose((0, 3, 4, 2, 1))
        gt_batch = gt_batch.reshape((batch_size,
                                     self.input_dim[2],
                                     self.input_dim[1] * scale,
                                     self.input_dim[0] * scale,
                                     1),
                                     order='F').transpose((0, 2, 3, 1, 4))

        return image_batch, gt_batch

    def data_loader_multi_channel_3d_wf(self, images_path, data_path, wf_path, gt_path, ny, nx, nz,
                                        it, batch_size, norm_flag=1, scale=2, wf=0):
        data_path = fix_path(data_path)
        gt_path = fix_path(gt_path)
        images_path = [fix_path(image_path) for image_path in images_path]
        batch_images_path = np.random.choice(images_path,
                                             size=batch_size)
        image_batch = []
        wf_batch = []
        gt_batch = []
        for path in batch_images_path:
            cur_img = tiff.imread(path)

            nslice = cur_img.shape[0]
            nchannels = int(nslice / self.input_dim[2])
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