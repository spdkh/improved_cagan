# -*- coding: utf-8 -*-
"""
todo: write
"""

from GAN import GAN


class RCAN(GAN):
    def discriminator(self, x, is_training=True, reuse=False):
        pass

    def generator(self, z, is_training=True, reuse=False):
        pass

    def build_model(self):
        pass

    def train(self):
        pass

    def save(self, checkpoint_dir, step):
        pass

    def load(self, checkpoint_dir):
        pass

    def __init__(self, args):
        super(RCAN, self).__init__(args)
