# -*- coding: utf-8 -*-
"""
todo: write
"""
import datetime
import os
import glob

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dense, Flatten, Input, add, multiply
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from sewar.full_ref import uqi
from skimage.metrics import mean_squared_error as compare_mse, \
    normalized_root_mse as compare_nrmse, \
    peak_signal_noise_ratio as compare_psnr, \
    structural_similarity as compare_ssim
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

from utils.lr_controller import ReduceLROnPlateau
from data.fixed_cell import FixedCell
from models.GAN import GAN
from models.binary_classification import discriminator
from models.super_resolution import rcan


class CAGAN(GAN):
    def __init__(self, args):
        GAN.__init__(self, args)
        print('CAGAN')
        self.data = FixedCell(self.args)
        self.g_input = Input(self.data.input_dim)
        self.d_input = Input(self.data.output_dim)

        self.g_output = None
        self.d_output = None

        self.writer = None

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        optimizer_d = self.args.d_opt
        optimizer_g = self.args.g_opt

    def build_model(self):
        # --------------------------------------------------------------------------------
        #                              define combined model
        # --------------------------------------------------------------------------------
        self.disc, self.frozen_d = self.discriminator()

        self.gen = self.generator()
        # print(self.gen.summary())

        fake_hp = self.gen(inputs=self.g_input)
        judge = self.frozen_d(fake_hp)
        label = np.zeros(self.args.batch_size)

        # last fake hp
        gen_loss = self.generator_loss(judge)
        # gen_total_loss, gen_gan_loss, gen_l1_loss =\
        # self.generator_loss(judge, fake_hp, self.g_output)
        # disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        self.gen.compile(loss=loss_mse_ssim_3d,
                         optimizer=self.args.g_opt)
        # if weight_wf_loss > 0:
        #     combined = Model(input_lp, [judge, fake_hp, fake_hp])
        #     loss_wf = create_psf_loss(psf)
        #     combined.compile(loss=['binary_crossentropy', loss_mse_ssim_3d, loss_wf],
        #                      optimizer=optimizer_g,
        #                      loss_weights=[0.1, 1, weight_wf_loss])  # 0.1 1
        # else:
        #     combined = Model(input_lp, [judge, fake_hp])
        #     combined.compile(loss=['binary_crossentropy', loss_mse_ssim_3d],
        #                      optimizer=optimizer_g,
        #                      loss_weights=[0.1, 1])  # 0.1 1

        self.lr_controller_g = ReduceLROnPlateau(model=self.gen,
                                                 factor=self.args.lr_decay_factor,
                                                 patience=10,
                                                 mode='min',
                                                 min_delta=1e-3,
                                                 cooldown=0,
                                                 min_learning_rate=self.args.g_start_lr * 0.1,
                                                 verbose=1)
        self.lr_controller_d = ReduceLROnPlateau(model=self.disc,
                                                 factor=self.args.lr_decay_factor,
                                                 patience=10,
                                                 mode='min',
                                                 min_delta=1e-3,
                                                 cooldown=0,
                                                 min_learning_rate=self.args.d_start_lr * 0.1,
                                                 verbose=1)

    def train(self):
        # label
        start_time = datetime.datetime.now()
        gloss_record = []
        dloss_record = []

        batch_size_d = round(self.args.batch_size / 2)
        valid_d = np.ones(batch_size_d).reshape((batch_size_d, 1))
        fake_d = np.zeros(batch_size_d).reshape((batch_size_d, 1))
        valid = np.ones(self.args.batch_size).reshape((self.args.batch_size, 1))
        fake = np.zeros(self.args.batch_size).reshape((self.args.batch_size, 1))

        train_names = ['Generator_loss', 'Discriminator_loss']

        for it in range(self.args.epoch):
            # ------------------------------------
            #         train generator
            # ------------------------------------
            for i in range(self.args.train_generator_times):
                input_g, gt_g = \
                    self.data.data_loader('train',
                                          self.args.batch_size,
                                          self.args.norm_flag,
                                          self.args.scale_factor)

                loss_generator = self.gen.train_on_batch(input_g, gt_g)
                gloss_record.append(loss_generator)

            # ------------------------------------
            #         train discriminator
            # ------------------------------------
            for i in range(self.args.train_discriminator_times):
                input_d, gt_d = \
                    self.data.data_loader('train',
                                          batch_size_d,
                                          self.args.norm_flag,
                                          self.args.scale_factor)

                fake_input_d = self.gen.predict(input_d)

                # discriminator loss separate for real/fake:
                # https://stackoverflow.com/questions/49988496/loss-functions-in-gans

                loss_discriminator = self.disc.train_on_batch(gt_d, valid_d)
                loss_discriminator += self.disc.train_on_batch(fake_input_d, fake_d)
                dloss_record.append(loss_discriminator[0])

            elapsed_time = datetime.datetime.now() - start_time
            print("%d epoch: time: %s, d_loss = %.5s, d_acc = %.5s, g_loss = %s" % (
                it + 1, elapsed_time, loss_discriminator[0], loss_discriminator[1], loss_generator))

            if (it + 1) % self.args.sample_interval == 0:
                self.validate(it + 1, sample=1)

            if (it + 1) % self.args.validate_interval == 0:
                self.validate(it + 1, sample=0)
                self.write_log(self.writer, train_names[0], np.mean(gloss_record), it + 1)
                self.write_log(self.writer, train_names[1], np.mean(dloss_record), it + 1)
                gloss_record = []
                dloss_record = []

    def validate(self, epoch, sample=0):
        """
                todo: complete
                :param epoch:
                :param sample:
                :return:
        """

        # initialization

        self.lr_controller_g.on_train_begin()
        self.lr_controller_d.on_train_begin()
        validate_nrmse = [np.Inf]

        # -------------------------------------------------------------------
        #                       about Tensor Board
        # -------------------------------------------------------------------
        self.writer = tf.summary.create_file_writer(self.data.log_path)
        val_names = ['val_MSE',
                     'val_SSIM',
                     'val_PSNR',
                     'val_NRMSE',
                     'val_UQI']

        patch_y, patch_x, patch_z, _ = self.data.input_dim
        validate_path = glob.glob(self.data.data_dirs['val'] + '*')
        validate_path.sort()
        # print('_______________', validate_path)
        # if sample == 1:
        #     validate_path = np.random.choice(validate_path, size=1)
        # elif self.args.validate_num < validate_path.__len__():
        #     validate_path = validate_path[0:self.args.validate_num]

        # save_weights_name = model_name + '-SIM_' + data_name
        #
        # save_weights_path = os.path.join(self.args.checkpoint_dir, save_weights_name)
        # sample_path = save_weights_path + 'sampled_img/'
        #
        # if not os.path.exists(save_weights_path):
        #     os.mkdir(save_weights_path)
        # if not os.path.exists(sample_path):
        #     os.mkdir(sample_path)

        mses, nrmses, psnrs, ssims, uqis = [], [], [], [], []
        imgs, imgs_gt, output = [], [], []
        # for path in validate_path:
        imgs, imgs_gt = \
            self.data.data_loader('val',
                                  self.args.batch_size,
                                  self.args.norm_flag,
                                  self.args.scale_factor)

        outputs = self.gen.predict(imgs)
        for output, img_gt in zip(outputs, imgs_gt):
            # predict generates [1, x, y, z, 1]
            # It is converted to [x, y, z] below
            # print(np.shape(imgs))
            output = np.reshape(output,
                                self.data.output_dim[:-1])

            output_proj = np.max(output, 2)

            gt_proj = np.max(np.reshape(img_gt,
                                        self.data.output_dim[:-1]),
                             2)
            mses, nrmses, psnrs, ssims, uqis = \
                self.img_comp(gt_proj,
                              output_proj,
                              mses,
                              nrmses,
                              psnrs,
                              ssims,
                              uqis)

        if sample == 0:
            # if best, save weights.best
            self.gen.save_weights(self.data.save_weights_path +
                                  'weights_gen_latest.h5')
            self.disc.save_weights(self.data.save_weights_path +
                                   'weights_disc_latest.h5')

            if min(validate_nrmse) > np.mean(nrmses):
                self.gen.save_weights(self.data.save_weights_path +
                                      'weights_gen_best.h5')
                self.disc.save_weights(self.data.save_weights_path +
                                       'weights_disc_best.h5')

            validate_nrmse.append(np.mean(nrmses))
            curlr_g = self.lr_controller_g.on_epoch_end(epoch, np.mean(nrmses))
            curlr_d = self.lr_controller_d.on_epoch_end(epoch, np.mean(nrmses))
            self.write_log(self.writer, val_names[0], np.mean(mses), epoch)
            self.write_log(self.writer, val_names[1], np.mean(ssims), epoch)
            self.write_log(self.writer, val_names[2], np.mean(psnrs), epoch)
            self.write_log(self.writer, val_names[3], np.mean(nrmses), epoch)
            self.write_log(self.writer, val_names[4], np.mean(uqis), epoch)
            self.write_log(self.writer, 'lr_g', curlr_g, epoch)
            self.write_log(self.writer, 'lr_d', curlr_d, epoch)

        else:

            # imgs = np.mean(imgs[0], 4)
            plt.figure(figsize=(22, 6))

            # figures equal to the number of z patches in columns
            for j in range(patch_z):
                output_results = {'Raw Input': imgs[0, :, :, j, 0],
                                  'Super Resolution Output': outputs[0, :, :, j, 0],
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

            plt.savefig(self.data.sample_path + '%d.png' % epoch)  # Save sample results
            plt.close("all")  # Close figures to avoid memory leak

    def prctile_norm(self, x_in, min_prc=0, max_prc=100):
        """

        :param x_in:
        :param min_prc:
        :param max_prc:
        :return: output
        """
        output = (x_in - np.percentile(x_in, min_prc)) / (np.percentile(x_in, max_prc)
                                                          - np.percentile(x_in, min_prc) + 1e-7)
        output[output > 1] = 1
        output[output < 0] = 0
        return output

    def img_comp(self,
                 gt, pr,
                 mses=None, nrmses=None,
                 psnrs=None, ssims=None,
                 uqis=None):
        """
        :param gt:
        :param pr:
        :param mses:
        :param nrmses:
        :param psnrs:
        :param ssims:
        :param uqis
        :return:
        """
        if ssims is None:
            ssims = []
        if psnrs is None:
            psnrs = []
        if nrmses is None:
            nrmses = []
        if mses is None:
            mses = []
        if uqis is None:
            uqis = []

        gt, pr = np.squeeze(gt), np.squeeze(pr)
        gt = gt.astype(np.float32)
        if gt.ndim == 2:
            num = 1
            gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
            pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
        else:
            num = np.size(gt, 0)

        for i in range(num):
            mses.append(compare_mse(self.prctile_norm(np.squeeze(gt[i])),
                                    self.prctile_norm(np.squeeze(pr[i]))))
            nrmses.append(compare_nrmse(self.prctile_norm(np.squeeze(gt[i])),
                                        self.prctile_norm(np.squeeze(pr[i]))))
            psnrs.append(compare_psnr(self.prctile_norm(np.squeeze(gt[i])),
                                      self.prctile_norm(np.squeeze(pr[i]))))
            ssims.append(compare_ssim(self.prctile_norm(np.squeeze(gt[i])),
                                      self.prctile_norm(np.squeeze(pr[i]))))
            uqis.append(uqi(self.prctile_norm(np.squeeze(pr[i])),
                            self.prctile_norm(np.squeeze(gt[i]))))
        return mses, nrmses, psnrs, ssims, uqis

    def discriminator(self):
        self.d_output = discriminator(self.d_input)

        disc = Model(inputs=self.d_input,
                     outputs=self.d_output)
        disc.compile(loss='binary_crossentropy',
                     optimizer=self.args.d_opt,
                     metrics=['accuracy'])

        frozen_disc = Model(inputs=disc.inputs, outputs=disc.outputs)
        frozen_disc.trainable = False
        return disc, frozen_disc

    def generator(self):
        self.g_output = rcan(self.g_input)

        gen = Model(inputs=self.g_input,
                    outputs=self.g_output)

        return gen

    def generator_loss(self,
                       disc_generated_output):
        def gen_loss(y_true, y_pred):
            alpha = 1
            beta = 10
            beta = 1
            gan_loss = self.loss_object(tf.ones_like(disc_generated_output),
                                        disc_generated_output)

            # # Mean absolute error
            l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
            new_loss = loss_mse_ssim_3d(y_true, y_pred)

            total_gen_loss = gan_loss +\
                             (alpha * l1_loss) +\
                             (beta * new_loss)
            return total_gen_loss  # , gan_loss, l1_loss

        return gen_loss


def loss_mse_ssim_3d(y_true, y_pred):
    ssim_para = 1e-1  # 1e-2
    mse_para = 1

    # SSIM loss and MSE loss
    x = K.permute_dimensions(y_true, (0, 4, 1, 2, 3))
    y = K.permute_dimensions(y_pred, (0, 4, 1, 2, 3))
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mse_loss = mse_para * K.mean(K.square(y - x))

    return mse_loss + ssim_loss
