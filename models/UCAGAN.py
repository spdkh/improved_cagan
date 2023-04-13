# -*- coding: utf-8 -*-
"""
"""
from models.CAGAN import CAGAN
from models.super_resolution import rcan, srcnn

import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
import visualkeras


class UCAGAN(CAGAN):
    """

    """

    def __init__(self, args, unrolling_its=1):
        CAGAN.__init__(self, args)

    # # @title Physics-Guided Unrolling gamma = 0.0005
    # gamma = 0.001
    # original_input = layers.Input(shape=(test_x.shape[1], test_x.shape[2], 1))
    # net_input = original_input
    #
    # for iteration in range(NUM_ITER):
    #     x = srcnn(net_input)
    #
    #     ## physics-guided
    #
    #     x = tf.add(net_input, x)
    #     print(x.shape)
    #     y = tf.nn.conv2d(x, kernel_transpose,
    #                      strides=1,
    #                      padding='SAME')
    #     ya = tf.multiply(y, gamma)
    #     x = tf.add(x, ya)
    #     F = tf.signal.fft2d(tf.cast(x,
    #                                 tf.complex64,
    #                                 name=None),
    #                         name=None)
    #     x = tf.multiply(F, 1 / (1 + gamma * K_norm ** 2))
    #     x = tf.cast(tf.signal.ifft2d(x,
    #                                  name=None),
    #                 tf.float32,
    #                 name=None)
    #     gamma /= 2
    #     net_input = x
    #
    # # Auto encoder
    # model_pg = Model(original_input, x)
    # print(model_pg.summary())

    def generator(self, g_input):
        x = rcan(g_input,
                 n_rcab=self.args.n_RCAB,
                 n_res_group=self.args.n_ResGroup,
                 channel=self.args.n_channel)
        initial_x = x

        kernel_T = self.data.psf.transpose(0, 2, 1, 3)
        # print(kernel_T.shape)
        K_norm = tf.norm(tf.signal.fft3d(self.data.psf))
        print('K norm:', K_norm)
        # plt.imshow(self.data.psf)
        # plt.colorbar()
        # kernel_T = self.data.psf[:, :, :]
        # kernel_transpose = np.expand_dims(kernel_T, axis=0)
        # kernel_transpose = kernel_T.reshape(1,
        #                                     kernel_T.shape[0],
        #                                     kernel_T.shape[1],
        #                                     kernel_T.shape[2],
        #                                     kernel_T.shape[3])
        gamma = self.args.gamma

        for iteration in range(self.args.unrolling_iter):
            x = rcan(x, scale=1,
                     n_rcab=self.args.n_RCAB,
                     n_res_group=self.args.n_ResGroup,
                     channel=self.args.n_channel)
            # x = x[:, :, :, :, 0]
            x = tf.add(initial_x, x)

            y = self.conv3d(initial_x, kernel_T)
            ya = tf.multiply(y, gamma)
            x = tf.add(x, ya)
            F = tf.signal.fft3d(tf.cast(x,
                                        tf.complex64,
                                        name=None),
                                name=None)
            x = tf.multiply(F, 1 / (1 + gamma * K_norm ** 2))
            x = tf.cast(tf.signal.ifft3d(x,
                                         name=None),
                        tf.float32,
                        name=None)
            # x = np.expand_dims(x, axis=-1)
            gamma /= 2

        self.g_output = x

        gen = Model(inputs=self.g_input,
                    outputs=self.g_output)
        # tf.keras.utils.plot_model(gen, to_file='Unrolled_generator.png', show_shapes=True, dpi=64, rankdir='LR')

        # font = ImageFont.truetype("C:\\Windows\\Fonts\\Times New Roman.ttf", 32)  # using comic sans is strictly prohibited!
        visualkeras.layered_view(gen, legend=True)  # font is optional!
        visualkeras.layered_view(gen, to_file='Unrolled_generator.png')  # write to disk
        return gen

    def conv3d(self, x, psf):
        psf = np.expand_dims(psf, axis=0)
        if psf.shape[3] > x.shape[3]:
            psf = psf[:, :, :,
                  psf.shape[3] // 2 - (x.shape[3] - 1) // 2:
                  psf.shape[3] // 2 + (x.shape[3] - 1) // 2 + 1,
                  :]

        return tf.nn.conv3d(x,
                            psf,
                            strides=[1] * 5,
                            padding='SAME')
