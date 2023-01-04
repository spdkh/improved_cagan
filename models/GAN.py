# -*- coding: utf-8 -*-
"""
todo: complete
"""

from abc import abstractmethod
import tensorflow as tf
from models.DNN import DNN


class GAN(DNN):
    """
        Abstract class for any GAN architecture
    """
    def __init__(self, args):
        DNN.__init__(self, args)

    def discriminator(self):
        pass

    def generator(self):
        pass

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output),
                                     disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output),
                                          disc_generated_output)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def generator_loss(self, fake_output):
        return self.loss_object(tf.ones_like(fake_output),
                                fake_output)

    def build_model(self):
        pass

    def train(self):
        pass

    def visualize_results(self, epoch):
        pass

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        pass

    def load(self, checkpoint_dir):
        pass
