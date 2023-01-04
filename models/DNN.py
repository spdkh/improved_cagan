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

from data.data import Data
from utils.data_loader import data_loader


class DNN(ABC):
    """
        Abstract class for DNN architectures
    """

    def __init__(self, args):
        """
            todo: what other info needed from load_sample?
            todo: write load_sample
            todo: figure out what is z_dim in CGAN
            todo: fix optimizer
        """
        self.model = Model()
        self.args = args

        self.data = Data(self.args)

        self.optimizer = self.args.g_opt

        data_name = self.args.data_dir.split('/')[-1]

        super().__init__()

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    # --------------------------------------------------------------------------------
    #                             Sample and validate
    # --------------------------------------------------------------------------------
    @abstractmethod
    def validate(self, epoch, sample=0):
        pass

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
