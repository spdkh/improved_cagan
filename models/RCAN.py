# -*- coding: utf-8 -*-
# Copyright 2022 The Improved caGAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the Residual Channel Attention Networks.
Residual Channel Attention Networks (RCANs) were originally proposed in:
[1] Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., & Fu, Y. (2018). Image super-resolution using very deep residual
    channel attention networks. In Proceedings of the European conference on computer vision (ECCV) (pp. 286-301).

Typical use:

    from models.RCAN import RCAN

"""
from abc import ABC

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, UpSampling3D, LeakyReLU, Lambda
from tensorflow.keras.layers import Input, add, multiply
from tensorflow.keras.models import Model

from DNN import DNN


class RCAN(DNN, ABC):

    def __init__(self, model, args):
        super(RCAN, self).__init__(model, args)
        self.input_shape = (args.patch_y, args.patch_x, args.patch_z, args.input_channels)
        self.n_ResGroup = args.n_ResGroup
        self.n_RCAB = args.n_RCAB

    @staticmethod
    def GlobalAveragePooling3d(layer_in):
        return tf.reduce_mean(layer_in, axis=(1, 2, 3), keepdims=True)

    @staticmethod
    def CALayer(X, channel, reduction=16):
        W = Lambda(RCAN.GlobalAveragePooling3d)(X)
        W = Conv3D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
        W = Conv3D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
        mul = multiply([X, W])
        return mul

    @staticmethod
    def RCAB(X, channel):
        conv = X
        for _ in range(2):
            conv = Conv3D(channel, kernel_size=3, padding='same')(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
        att = RCAN.CALayer(conv, channel, reduction=16)
        output = add([att, X])
        return output

    @staticmethod
    def ResidualGroup(X, channel, n_RCAB):
        conv = X
        for _ in range(n_RCAB):
            conv = RCAN.RCAB(conv, channel)
        return conv

    def build_model(self, channel=64):
        """

        :param channel: number of fixed channels in Conv layer
        :type channel: int
        :return: model
        :rtype: Model
        """
        inputs = Input(self.input_shape)
        conv = Conv3D(channel, kernel_size=3, padding='same')(inputs)

        for _ in range(self.n_ResGroup):
            conv = RCAN.ResidualGroup(conv, channel=channel, n_RCAB=self.n_RCAB)

        up = UpSampling3D(size=(2, 2, 1))(conv)
        conv = Conv3D(channel, kernel_size=3, padding='same')(up)
        conv = LeakyReLU(alpha=0.2)(conv)
        conv = Conv3D(1, kernel_size=3, padding='same')(conv)
        output = LeakyReLU(alpha=0.2)(conv)

        model = Model(inputs=inputs, outputs=output)

        return model
