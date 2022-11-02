"""

"""
# -*- coding: utf-8 -*-
from __future__ import division
from abc import ABC, abstractmethod
from utils.data_loader import load_sample


class DNN(ABC):
    """
        Abstract class for DNN architectures
    """

    def __init__(self, args):
        """
            todo: what other info needed from load_sample
            todo: write load_sample
        """
        self.args = args

        self.input_dim = load_sample(self.args.dataset_dir)
        self.output_dim = [self.input_dim[:1] * self.args.scale, self.input_dim[2:]]

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
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
