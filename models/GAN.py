# -*- coding: utf-8 -*-
import DNN


class GAN(DNN):
    """
        Abstract class for any GAN architecture
    """

    def __init__(self, args):
        DNN.__init__(self, args)

    @abstractmethod
    def discriminator(self, x, is_training=True, reuse=False):
        pass

    @abstractmethod
    def generator(self, z, is_training=True, reuse=False):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def visualize_results(self, epoch):
        pass

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    @abstractmethod
    def save(self, checkpoint_dir, step):
        pass

    @abstractmethod
    def load(self, checkpoint_dir):
        pass
