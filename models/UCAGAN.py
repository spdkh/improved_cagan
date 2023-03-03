# -*- coding: utf-8 -*-
"""
"""
from models.CAGAN import CAGAN
from models.super_resolution import rcan, srcnn

from tensorflow.keras.models import Model
import tensorflow as tf


class UCAGAN(CAGAN):
    """

    """

    def __init__(self, args, unrolling_its=1):
        CAGAN.__init__(self, args)

    def generator(self, g_input):
        x = rcan(g_input)
        initial_x = x

        for iteration in range(self.args.unrolling_iter):
            x = rcan(x, scale=1)
            x = tf.add(initial_x, x)

        self.g_output = x

        gen = Model(inputs=self.g_input,
                    outputs=self.g_output)
        tf.keras.utils.plot_model(gen, show_shapes=True, dpi=64)
        return gen
