"""
    author: SPDKH
"""
import os
import glob
import numpy as np
from data.data import Data
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from utils import fcns
from skimage.measure import block_reduce
from matplotlib import pyplot as plt
import tifffile as tiff

from utils.fcns import prctile_norm, fix_path, reorder


class FairSIM(Data):
    def __init__(self, args):
        Data.__init__(self, args)
        self.data_groups = {'train': 'training',
                            'test': 'testing',
                            'val': 'validation'}

        self.data_types = {'x': 'raw_data', 'y': 'gt'}
        self.args.data_dir = fcns.fix_path(self.args.data_dir)
        input_dir = os.path.join(self.args.data_dir,
                                 self.data_groups['train'],
                                 self.data_types['x'])
        output_dir = os.path.join(self.args.data_dir,
                                  self.data_groups['train'],
                                  self.data_types['y'])
        in_sample_dir = os.path.join(input_dir,
                                     os.listdir(input_dir)[0])
        self.input_dim = self.load_sample(in_sample_dir, 1)
        print('input', self.input_dim)
        out_sample_dir = os.path.join(output_dir,
                                      os.listdir(output_dir)[0])

        self.output_dim = self.load_sample(out_sample_dir, 1)
        print('output', self.output_dim)

        save_weights_name = 'FairSIM'

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
                self.data_dirs[data_type + data_group] = \
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


        todo: plot wf and make sure it is fine
        """

        images_names = os.listdir(self.data_dirs['x' + mode])
        gt_names = os.listdir(self.data_dirs['y' + mode])
        # print(gt_names)
        images_names.sort()
        gt_names.sort()

        x_path = self.data_dirs['x' + mode]
        y_path = self.data_dirs['y' + mode]
        # train_wf_path = self.args.data_dir + '/training_wf/'
        # print(images_names)
        batch_images_path = images_names[it:batch_size + it]
        gt_images_path = gt_names[it:batch_size + it]

        image_batch = []
        gt_batch = []
        wf_batch = []
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

        if wf_weight > 0:
            for cur_img in image_batch:
                cur_wf = reorder(cur_img)
                cur_wf = block_reduce(cur_wf,
                                      block_size=(self.input_dim[3], 1, 1),
                                      func=np.sum,
                                      cval=np.sum(cur_wf))

                # wf_shape = np.shape(cur_wf)

                if norm_flag:
                    cur_wf = prctile_norm(np.array(cur_wf))
                else:
                    cur_wf = np.array(cur_wf) / 65535

                wf_batch.append(cur_wf)

            wf_batch = np.array(wf_batch)

            wf_batch = wf_batch.reshape((batch_size,
                                         self.input_dim[2],
                                         self.input_dim[1],
                                         self.input_dim[0],
                                         1),
                                        order='F').transpose((0, 2, 3, 1, 4))

        return image_batch, gt_batch, wf_batch
