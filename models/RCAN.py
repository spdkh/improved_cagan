# -*- coding: utf-8 -*-
"""
todo: write
"""


from models import DNN
from tensorflow.keras.layers import Conv3D, UpSampling3D, LeakyReLU, Lambda, ReLU
from utils.common import global_average_pooling3d, conv_block3d


class RCAN(GAN):
    """
    todo: check the parameters from the paper
    todo: make important parameters variable
    """
    def __init__(self, args):
        DNN.__init__(self, args)

    def ca_layer(self, channel, reduction=16):
        conv = Lambda(global_average_pooling3d)(self.input)
        conv = Conv3D(channel // reduction,
                      kernel_size=1,
                      activation='relu',
                      padding='same')(conv)
        conv = Conv3D(channel,
                      kernel_size=1,
                      activation='sigmoid',
                      padding='same')(conv)
        mul = multiply([self.input, conv])
        return mul

    def rcab(self, channel):
        conv = self.input
        for _ in range(2):
            conv = Conv3D(channel, kernel_size=3, padding='same')(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
        att = ca_layer(conv, channel, reduction=16)
        output = add([att, X])
        return output

    def res_group(X, channel, n_RCAB):
        conv = X
        for _ in range(n_RCAB):
            conv = RCAB(conv, channel)
        return conv