# Copyright 2023 The Improved caGAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains the code for prediction using trained models. 
    author: SPDKH
    date: Nov 2, 2023
    updated: Mazhar on June 05, 2023
    
Example usage:
    conda activate tf_gpu
    python -m predict_3d --dnn_type RCAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 2 --n_RCAB 10 --n_channel 16 --epoch 500 --start_lr 1e-3 --lr_decay_factor 0.5 --alpha 0 --beta 0.05 --mae_loss 1 --mse_loss 0 --unrolling_iter 0 --model_weights "C:\Users\unrolled_caGAN\Desktop\mazhar_Unrolled caGAN project\checkpoint\FixedCell_RCAN_17-04-2023_time0257weights_best.h5" --test_dir D:\Data\FixedCell\PFA_eGFP\cropped3d_128_3\testing
"""
import argparse
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Input, add, multiply

from models.SRGAN import SRGAN
from models.CAGAN import CAGAN
from models.UCAGAN import UCAGAN
from models.CGAN import CGAN
from models.DNN import DNN
from models.RCAN import RCAN
from models.URCAN import URCAN

import imageio
import os
from models import *
from utils.fcns import prctile_norm
import tifffile as tiff
from data.data import Data

from models.super_resolution import rcan

from utils.config import parse_args


def main():
    """
        Main Predicting Function
    """
    # parse arguments
    args = parse_args()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model_fns = {'CAGAN': CAGAN,
                 'SRGAN': SRGAN,
                 'CGAN': CGAN,
                 'DNN': DNN,
                 'UCAGAN': UCAGAN,
                 'RCAN': RCAN,
                 'URCAN': URCAN}

    # declare instance for GAN
    dnn = model_fns[args.dnn_type](args)

    # build graph
    dnn.build_model()

    # data = Data(args)
    # dnn = tf.keras.models.load_model(args.model_weights)
    dnn.model.load_weights(args.model_weights)

    output_name = 'output_' + args.dnn_type + '-'
    test_images_path = args.test_dir
    output_dir = args.result_dir + '\\' + output_name

    # --------------------------------------------------------------------------------
    #                              glob test data path
    # --------------------------------------------------------------------------------

    img_path = [test_images_path]
    img_path.sort()
    n_channel = 15

    print(img_path)
    img = tiff.imread(img_path[0])
    shape = img.shape
    input_y, input_x = shape[1], shape[2]
    input_z = shape[0] // n_channel
    output_dir = output_dir + 'SIM'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Processing ' + test_images_path + '...')
    im_count = 0
    for curp in img_path:
        print(curp)
        img = tiff.imread(curp)
        img = np.array(img).reshape((n_channel, input_z, input_y, input_x),
                                    order='F').transpose((2, 3, 1, 0))
        img = img[np.newaxis, :]

        img = prctile_norm(img)
        pr = prctile_norm(np.squeeze(dnn.model.predict(img)))

        outName = curp.replace(test_images_path, output_dir)

        pr = np.transpose(65535 * pr, (2, 0, 1)).astype('uint16')
        tiff.imwrite(outName, pr, dtype='uint16')


if __name__ == '__main__':
    main()
