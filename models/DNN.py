"""
    author: SPDKH
    todo: complete
"""
# -*- coding: utf-8 -*-
from __future__ import division
import os
import glob
import re
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, add, multiply

import matplotlib.pyplot as plt
from utils.data_loader import load_sample, data_loader


class DNN(ABC):
    """
        Abstract class for DNN architectures
    """

    def __init__(self, args):
        """
            todo: what other info needed from load_sample?
            todo: write load_sample
            todo: figure out what is z_dim in CGAN
        """
        self.model = Model()
        self.args = args

        self.input_dim = load_sample(self.args.dataset_dir)
        self.output_dim = [self.input_dim[:1] * self.args.scale, self.input_dim[2:]]

        self.input = Input(self.input_dim)
        self.optimizer =

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    # --------------------------------------------------------------------------------
    #                             Sample and validate
    # --------------------------------------------------------------------------------
    def validate(self, epoch, sample=0):
        """
        todo: complete
        :param epoch:
        :param sample:
        :return:
        """
        patch_y, patch_x, patch_z = self.input_dim
        validate_path = glob.glob(validate_images_path + '*')
        validate_path.sort()
        if sample == 1:
            validate_path = np.random.choice(validate_path, size=1)
        elif self.args.validate_num < validate_path.__len__():
            validate_path = validate_path[0:self.args.validate_num]

        mses, nrmses, psnrs, ssims, uqis = [], [], [], [], []
        imgs, imgs_gt, output = [], [], []
        for path in validate_path:
            [imgs, imgs_gt] = \
                data_loader([path], validate_images_path, validate_gt_path,
                            patch_y, patch_x, patch_z, 1,
                            norm_flag=self.args.norm_flag)

            output = self.model.predict(imgs)
            # predict generates [1, x, y, z, 1]
            # It is converted to [x, y, z] below
            output = np.reshape(output, (patch_x * 2, patch_y * 2, patch_z))

            output_proj = np.max(output, 2)

            gt_proj = np.max(np.reshape(imgs_gt, (patch_x * 2, patch_y * 2, patch_z)), 2)
            mses, nrmses, psnrs, ssims, uqis = img_comp(gt_proj, output_proj, mses, nrmses, psnrs, ssims, uqis)

        if sample == 0:
            # if best, save weights.best
            self.model.save_weights(save_weights_path + 'weights_latest.h5')

            if min(validate_nrmse) > np.mean(nrmses):
                self.model.save_weights(save_weights_path + 'weights_best.h5')
                print(self.model.summary)

            validate_nrmse.append(np.mean(nrmses))
            curlr_g = lr_controller_g.on_epoch_end(epoch, np.mean(nrmses))
            curlr_d = lr_controller_d.on_epoch_end(epoch, np.mean(nrmses))
            self.write_log(writer, val_names[0], np.mean(mses), epoch)
            self.write_log(writer, val_names[1], np.mean(ssims), epoch)
            self.write_log(writer, val_names[2], np.mean(psnrs), epoch)
            self.write_log(writer, val_names[3], np.mean(nrmses), epoch)
            self.write_log(writer, val_names[4], np.mean(uqis), epoch)
            self.write_log(writer, 'lr_g', curlr_g, epoch)
            self.write_log(writer, 'lr_d', curlr_d, epoch)

        else:
            imgs = np.mean(imgs, 4)
            plt.figure(figsize=(22, 6))

            # figures equal to the number of z patches in columns
            for j in range(patch_z):
                output_results = {'Raw Input': imgs[0, :, :, j],
                                  'Super Resolution Output': output[:, :, j],
                                  'Ground Truth': imgs_gt[0, :, :, j, 0]}
                plt.title('Z = ' + str(j))
                for i, (label, img) in enumerate(output_results.items()):
                    # first row: input image average of angles and phases
                    # second row: resulting output
                    # third row: ground truth
                    plt.subplot(3, patch_z, j + patch_z * i + 1)
                    plt.ylabel(label)
                    plt.imshow(img, cmap=plt.get_cmap('hot'))

                    plt.gca().axes.yaxis.set_ticklabels([])
                    plt.gca().axes.xaxis.set_ticklabels([])
                    plt.gca().axes.yaxis.set_ticks([])
                    plt.gca().axes.xaxis.set_ticks([])
                    plt.colorbar()

            plt.savefig(sample_path + '%d.png' % epoch)  # Save sample results
            plt.close("all")  # Close figures to avoid memory leak

    @abstractmethod
    def visualize_results(self, epoch):
        pass

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.args.dnn_type, self.args.dataset_name,
            self.args.batch_size)

    def save(self, checkpoint_dir, step):
        """
        todo: test
        Parameters
        ----------
        checkpoint_dir
        step

        Returns
        -------
        callback
        """
        checkpoint_dir = os.path.join(checkpoint_dir,
                                      self.model_dir,
                                      self.args.dnn_type)
        checkpoint_path = os.path.join(checkpoint_dir,
                                       "-{epoch:04d}.ckpt")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_freq=step * self.args.batch_size)

        self.model.save_weights(checkpoint_path.format(epoch=0))
        return cp_callback

    def load(self, checkpoint_dir):
        """
        todo: add a function to generate the final checkpoint dir
        todo: test
        source: https://www.tensorflow.org/tutorials/keras/save_and_load

        Parameters
        ----------
        checkpoint_dir

        Returns
        -------

        """
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        if checkpoint_dir:
            ckpt_name = os.path.basename(checkpoint_dir)
            ckpt = self.model.load_weights(latest)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def write_log(self, writer, names, logs, batch_no):
        """
        todo: test
        Parameters
        ----------
        names
        logs
        batch_no
        """
        with writer.as_default():
            tf.summary.scalar(names, logs, step=batch_no)
            writer.flush()
